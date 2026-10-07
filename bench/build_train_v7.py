"""构建第七版数据集：类别合并为 7 类，摩托车/自行车/三轮车 → 非机动车（nonmotor）。

新类别表（顺序即标签编号）：
    0 car  1 truck  2 bus  3 nonmotor  4 person  5 sign  6 bollard

旧 9 类 → 新 7 类的映射：
    0→0 1→1 2→2 3→3 4→3 5→4 6→5 7→6 8→3

为什么内部名用 nonmotor 而不是"非机动车"：画面标签是 OpenCV 的 putText 画的，
它不支持中文字形，写中文会渲染成方块。中文显示留给上层界面做映射。

图片不复制内容，用硬链接（同盘秒建、不占额外空间）；只有标签重新映射后另存。

产物 datasets/train_v7/：
    images/train,labels/train   新素材 267 + 旧街面 473 + 俯拍 250 = 990 张
    images/val,labels/val       新素材留出的 3 段共 65 帧（自动标注真值）
    street/                     旧街面 116 帧（与 v1~v6 同口径，改成 7 类）
    aerial/                     俯拍 399 帧（看合并后的非机动车在俯拍上的表现）
    data.yaml / street_val.yaml / aerial_val.yaml

用法：python bench/build_train_v7.py [--visdrone 250] [--repeat 3]
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "datasets"
NEW = BASE / "videoset_02_label"
OLD_TRAIN = BASE / "train_v4"
STREET_VAL = BASE / "train_v5"
VIS = BASE / "visdrone_tricycle"
OUT = BASE / "train_v7"

NAMES = ["car", "truck", "bus", "nonmotor", "person", "sign", "bollard"]
OLD_TO_NEW = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 5, 7: 6, 8: 3}
VAL_VIDEOS = {"VID_20261004_112844", "VID_20261004_113037", "VID_20261004_114022"}


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def convert_label(src: Path, dst: Path) -> None:
    """把 9 类标签重映射成 7 类；越界类别丢弃。"""
    lines = []
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            new_id = OLD_TO_NEW.get(int(float(parts[0])))
            if new_id is None:
                continue
            lines.append(" ".join([str(new_id)] + parts[1:5]))
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def add_sample(src_image: Path, src_label: Path, sub: str) -> str | None:
    if not src_image.exists() or not src_label.exists():
        return None
    image_dst = OUT / "images" / sub / src_image.name
    label_dst = OUT / "labels" / sub / (src_image.stem + ".txt")
    link_or_copy(src_image, image_dst)
    convert_label(src_label, label_dst)
    return str(image_dst)


def main() -> None:
    visdrone_n = 250
    repeat = 3
    if "--visdrone" in sys.argv:
        visdrone_n = int(sys.argv[sys.argv.index("--visdrone") + 1])
    if "--repeat" in sys.argv:
        repeat = int(sys.argv[sys.argv.index("--repeat") + 1])

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
        path = add_sample(image, NEW / "labels" / (image.stem + ".txt"), "train")
        if path:
            train_paths.append(path)
    train_new_count = len(train_paths)
    train_paths *= repeat                                    # 新素材过采样
    for image in old_images:
        path = add_sample(image, OLD_TRAIN / "labels" / "train" / (image.stem + ".txt"), "train")
        if path:
            train_paths.append(path)
    for image in vis_pick:
        path = add_sample(image, VIS / "labels" / (image.stem + ".txt"), "train")
        if path:
            train_paths.append(path)

    val_paths = [p for p in (add_sample(i, NEW / "labels" / (i.stem + ".txt"), "val")
                             for i in val_new) if p]
    street_paths = [p for p in (add_sample(i, STREET_VAL / "labels" / "val" / (i.stem + ".txt"),
                                           "street") for i in street_val) if p]
    aerial_paths = [p for p in (add_sample(i, VIS / "labels" / (i.stem + ".txt"), "aerial")
                                for i in vis_val) if p]

    (OUT / "train.txt").write_text("\n".join(train_paths) + "\n", encoding="utf-8")
    names_block = "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: train.txt\nval: images/val\n" + names_block,
        encoding="utf-8")
    (OUT / "street_val.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/street\nval: images/street\n" + names_block,
        encoding="utf-8")
    (OUT / "aerial_val.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/aerial\nval: images/aerial\n" + names_block,
        encoding="utf-8")
    (OUT / "classes.txt").write_text("\n".join(NAMES) + "\n", encoding="utf-8")

    from collections import Counter
    def dist(sub: str) -> str:
        counter: Counter = Counter()
        for label in (OUT / "labels" / sub).glob("*.txt"):
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    counter[int(parts[0])] += 1
        return "  ".join(f"{NAMES[i]}={counter.get(i, 0)}" for i in range(len(NAMES)))

    print(f"合并映射：motorcycle/bicycle/tricycle → nonmotor")
    print(f"训练样本 {len(train_paths)} 条（新素材 {train_new_count}×{repeat} + "
          f"旧街面 {len(old_images)} + 俯拍 {len(list((OUT / 'images' / 'train').glob('visdrone_*')))}）")
    print(f"新素材验证 {len(val_paths)} 帧 / 旧街面验证 {len(street_paths)} 帧 / "
          f"俯拍验证 {len(aerial_paths)} 帧")
    print("训练集类别分布：" + dist("train"))
    print("新素材验证分布：" + dist("val"))
    print(f"data.yaml -> {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
