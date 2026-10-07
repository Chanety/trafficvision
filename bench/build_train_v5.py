"""构建第五版数据集：把 VisDrone 三轮车子集按平衡比例并入，补上第 9 类 tricycle。

背景：
  · train_v4（473 训练 / 116 验证）里三轮车实例为 0，模型根本没有这一类的监督。
  · datasets\\visdrone_tricycle 已转换好：训练 2262 张（8058 个三轮车框）、
    验证 399 张（1577 个三轮车框），但全部是俯拍机位。
  · 若把 2262 张全并进训练集，俯拍数据会占 473+2262=2735 张里的 83%，
    街面数据被淹没——因此对 VisDrone 训练帧做等间隔抽样，默认取 500 张，
    使街面:俯拍 ≈ 473:500，接近 1:1。

产物：
  · datasets\\train_v5   训练集 = train_v4 训练集 + 抽样 VisDrone 训练帧；
                        验证集 = train_v4 验证集（116 张，保持不变，
                        以保证与 v3/v4 的指标可直接比较）
  · datasets\\visdrone_tricycle\\val9.yaml
                        用 VisDrone 自带的 399 张验证帧专门测 tricycle 的 AP
                        （这 399 张从不进入训练）

类别顺序固定 9 类，tricycle 追加在第 9 位，不影响已有 0~7 的标注：
  0 car 1 truck 2 bus 3 motorcycle 4 bicycle 5 person 6 sign 7 bollard 8 tricycle

用法：python bench/build_train_v5.py [--visdrone 500]
"""

from __future__ import annotations

import collections
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "datasets"
V4 = BASE / "train_v4"
VIS = BASE / "visdrone_tricycle"
V5 = BASE / "train_v5"
NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "sign", "bollard", "tricycle"]


def val_videos() -> set[str]:
    """训练集必须排除来自验证视频的帧（数据卫生红线）。"""
    return {p.stem.split("_f")[0] for p in (V4 / "labels" / "val").glob("*.txt")}


def count_classes(label_dir: Path) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    for label in label_dir.glob("*.txt"):
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                counter[int(parts[0])] += 1
    return counter


def main() -> None:
    target = 500
    if "--visdrone" in sys.argv:
        target = int(sys.argv[sys.argv.index("--visdrone") + 1])

    blocked = val_videos()
    vis_train = sorted((VIS / "images").glob("visdrone_train_*.jpg"))
    vis_val = sorted((VIS / "images").glob("visdrone_val_*.jpg"))
    if not vis_train:
        print(f"{VIS} 下没找到 VisDrone 图片，先跑 bench\\convert_visdrone.py")
        return

    step = max(1, len(vis_train) // target)
    sampled = vis_train[::step][:target]

    if V5.exists():
        shutil.rmtree(V5)
    counts: collections.Counter = collections.Counter()
    skipped = 0

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (V5 / kind / split).mkdir(parents=True, exist_ok=True)
        for image in (V4 / "images" / split).iterdir():
            shutil.copyfile(image, V5 / "images" / split / image.name)
            shutil.copyfile(V4 / "labels" / split / (image.stem + ".txt"),
                            V5 / "labels" / split / (image.stem + ".txt"))
            counts[split] += 1

    # 俯拍补充：VisDrone 训练帧按等间隔抽样后并入训练集
    for image in sampled:
        if image.stem in blocked:
            skipped += 1
            continue
        label = VIS / "labels" / (image.stem + ".txt")
        if not label.exists():
            continue
        shutil.copyfile(image, V5 / "images" / "train" / image.name)
        shutil.copyfile(label, V5 / "labels" / "train" / label.name)
        counts["train_visdrone"] += 1

    (V5 / "data.yaml").write_text(
        f"path: {V5.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)),
        encoding="utf-8",
    )

    # 三轮车专项验证：只测 VisDrone 自带验证帧（从未参与训练）
    list_path = VIS / "val9_list.txt"
    list_path.write_text("\n".join(str(p) for p in vis_val) + "\n", encoding="utf-8")
    (VIS / "val9.yaml").write_text(
        f"path: {VIS.as_posix()}\ntrain: val9_list.txt\nval: val9_list.txt\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)),
        encoding="utf-8",
    )

    train_dist = count_classes(V5 / "labels" / "train")
    val_dist = count_classes(V5 / "labels" / "val")
    vis_val_dist: collections.Counter = collections.Counter()
    for image in vis_val:
        label = VIS / "labels" / (image.stem + ".txt")
        if label.exists():
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    vis_val_dist[int(parts[0])] += 1
    print(f"训练集 {counts['train']} 张（街面 {counts['train'] - counts['train_visdrone']}"
          f" + 俯拍 {counts['train_visdrone']}，抽样步长 {step}，排除验证视频 {skipped} 帧）")
    print(f"验证集 {counts['val']} 张（与 train_v4 完全一致，保证指标可比）")
    print("训练集类别分布：" + "  ".join(f"{NAMES[i]}={train_dist.get(i, 0)}"
                                        for i in range(len(NAMES))))
    print("验证集类别分布：" + "  ".join(f"{NAMES[i]}={val_dist.get(i, 0)}"
                                        for i in range(len(NAMES))))
    print(f"data.yaml  -> {V5 / 'data.yaml'}")
    print(f"三轮车验证 -> {VIS / 'val9.yaml'}（{len(vis_val)} 张验证帧，"
          f"共 {vis_val_dist.get(8, 0)} 个三轮车框）")


if __name__ == "__main__":
    main()
