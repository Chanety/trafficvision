"""构建第九版数据集：10 类口径（合并 + 路灯 + 信号灯 + 非机动车驾驶员）。

新类别表（顺序即标签编号，与 trafficvision.config.MERGED_CLASSES 一致）：
    0 car  1 truck  2 bus  3 nonmotor  4 person  5 sign  6 bollard
    7 roadlamp  8 trafficlight  9 rider

数据来源与映射：
    · 新素材 `datasets/videoset_02_label2` —— **已经是 10 类，直接沿用**（无需映射）
    · 旧街面 `datasets/train_v4`（9 类）—— 3/4/8→nonmotor，5→person，6→sign，7→bollard
    · 俯拍 `datasets/visdrone_tricycle`（9 类）—— 同上映射；roadlamp/trafficlight/rider 无旧数据

注意：旧数据里没有 7/8/9 三类，所以这三类只由新素材（videoset_02）提供监督信号。
图片用硬链接，不额外占盘；只有标签重新映射后另存。

产物 `datasets/train_v9/`：
    images/train,labels/train   新素材 267 + 旧街面 473 + 俯拍 250（+ 新素材过采样）
    images/val,labels/val       新素材留出的 3 段共 65 帧
    street/                     旧街面 116 帧（与 v1~v8 同口径，转成 10 类）
    aerial/                     俯拍 399 帧

用法：python bench/build_train_v9.py [--visdrone 250] [--repeat 3]
"""

from __future__ import annotations

import os
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trafficvision.config import MERGED_CLASSES, MERGE_MAP  # noqa: E402

BASE = ROOT / "datasets"
NEW = BASE / "videoset_02_label2"        # 已是 10 类
OLD_TRAIN = BASE / "train_v4"            # 9 类
STREET_VAL = BASE / "train_v5"           # 9 类
VIS = BASE / "visdrone_tricycle"         # 9 类
OUT = BASE / "train_v9"

NAMES = list(MERGED_CLASSES)
VAL_VIDEOS = {"VID_20261004_112844", "VID_20261004_113037", "VID_20261004_114022"}


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def convert_9_to_10(src: Path, dst: Path, remap: bool) -> None:
    """把 9 类标签映射成 10 类；remap=False 时表示源标签已是 10 类，原样复制。"""
    lines = []
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            old = int(float(parts[0]))
            new = MERGE_MAP.get(old) if remap else old
            if new is None or not (0 <= new < len(NAMES)):
                continue
            lines.append(" ".join([str(new)] + parts[1:5]))
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def add_sample(src_image: Path, src_label: Path, sub: str, remap: bool) -> str | None:
    if not src_image.exists() or not src_label.exists():
        return None
    image_dst = OUT / "images" / sub / src_image.name
    label_dst = OUT / "labels" / sub / (src_image.stem + ".txt")
    link_or_copy(src_image, image_dst)
    convert_9_to_10(src_label, label_dst, remap)
    return str(image_dst)


def main() -> None:
    global NEW, OUT
    visdrone_n, repeat = 250, 3
    if "--visdrone" in sys.argv:
        visdrone_n = int(sys.argv[sys.argv.index("--visdrone") + 1])
    if "--repeat" in sys.argv:
        repeat = int(sys.argv[sys.argv.index("--repeat") + 1])
    # 允许换成清洗后的数据源 / 换输出目录（例如 label3 -> train_v9b）
    if "--new-src" in sys.argv:
        NEW = Path(sys.argv[sys.argv.index("--new-src") + 1])
    if "--out" in sys.argv:
        OUT = Path(sys.argv[sys.argv.index("--out") + 1])

    new_images = sorted((NEW / "images").glob("*.jpg"))
    train_new = [p for p in new_images if p.stem.split("_f")[0] not in VAL_VIDEOS]
    val_new = [p for p in new_images if p.stem.split("_f")[0] in VAL_VIDEOS]
    old_images = sorted((OLD_TRAIN / "images" / "train").glob("*.jpg"))
    street_val = sorted((STREET_VAL / "images" / "val").glob("*.jpg"))
    vis_train = sorted((VIS / "images").glob("visdrone_train_*.jpg"))
    step = max(1, len(vis_train) // visdrone_n)
    vis_pick = vis_train[::step][:visdrone_n]
    vis_val = sorted((VIS / "images").glob("visdrone_val_*.jpg"))

    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("train", "val", "street", "aerial"):
        (OUT / "images" / sub).mkdir(parents=True)
        (OUT / "labels" / sub).mkdir(parents=True)

    train_paths: list[str] = []
    for image in train_new:
        p = add_sample(image, NEW / "labels" / (image.stem + ".txt"), "train", remap=False)
        if p:
            train_paths.append(p)
    train_new_count = len(train_paths)
    train_paths *= repeat
    for image in old_images:
        p = add_sample(image, OLD_TRAIN / "labels" / "train" / (image.stem + ".txt"),
                       "train", remap=True)
        if p:
            train_paths.append(p)
    for image in vis_pick:
        p = add_sample(image, VIS / "labels" / (image.stem + ".txt"), "train", remap=True)
        if p:
            train_paths.append(p)

    val_paths = [p for p in (add_sample(i, NEW / "labels" / (i.stem + ".txt"), "val", False)
                             for i in val_new) if p]
    street_paths = [p for p in (add_sample(i, STREET_VAL / "labels" / "val" / (i.stem + ".txt"),
                                           "street", True) for i in street_val) if p]
    aerial_paths = [p for p in (add_sample(i, VIS / "labels" / (i.stem + ".txt"), "aerial", True)
                                for i in vis_val) if p]

    (OUT / "train.txt").write_text("\n".join(train_paths) + "\n", encoding="utf-8")
    names_block = "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    for fname, val in (("data.yaml", "images/val"),
                       ("street_val.yaml", "images/street"),
                       ("aerial_val.yaml", "images/aerial")):
        (OUT / fname).write_text(
            f"path: {OUT.as_posix()}\ntrain: " + ("train.txt" if fname == "data.yaml" else val)
            + f"\nval: {val}\n" + names_block, encoding="utf-8")
    (OUT / "classes.txt").write_text("\n".join(NAMES) + "\n", encoding="utf-8")

    def dist(sub: str) -> str:
        c: Counter = Counter()
        for label in (OUT / "labels" / sub).glob("*.txt"):
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    c[int(parts[0])] += 1
        return "  ".join(f"{NAMES[i]}={c.get(i, 0)}" for i in range(len(NAMES)))

    print("类别映射：motorcycle/bicycle/tricycle → nonmotor；新增 roadlamp/trafficlight/rider")
    print(f"训练条目 {len(train_paths)}（新素材 {train_new_count}×{repeat} + "
          f"旧街面 {len(old_images)} + 俯拍 {len(vis_pick)}）")
    print(f"验证：新素材 {len(val_paths)} / 旧街面 {len(street_paths)} / 俯拍 {len(aerial_paths)}")
    print("训练集分布：" + dist("train"))
    print("新素材验证分布：" + dist("val"))
    print(f"data.yaml -> {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
