"""把 VisDrone 标注转换成我们的 YOLO 格式，并只保留含三轮车的图片（省空间）。

VisDrone 标注格式（每行）：
    bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion
类别（1 起）：1 行人 2 人群 3 自行车 4 汽车 5 面包车 6 卡车
              7 三轮车 8 三轮棚车 9 公交 10 摩托车
其中 score=0 表示忽略，category=0 表示忽略区域。

我们的类别顺序：
    0 car  1 truck  2 bus  3 motorcycle  4 bicycle  5 person  6 sign  7 bollard  8 tricycle

空间控制：默认只复制"含三轮车（7）或三轮棚车（8）"的图片；
完整 VisDrone 约 2 GB，这样通常只需 300~500 MB。

用法：
    python bench/convert_visdrone.py <VisDrone根目录> <输出目录> [--all] [--limit N]
    根目录下需有 VisDrone2019-DET-train/ 与 VisDrone2019-DET-val/（含 annotations/ 与 images/）
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

OUR_NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person",
             "sign", "bollard", "tricycle"]

# VisDrone 类别 → 我们的类别编号；None 表示不保留
VISDRONE_MAP = {
    1: 5,      # pedestrian  → person
    2: 5,      # people      → person
    3: 4,      # bicycle     → bicycle
    4: 0,      # car         → car
    5: 0,      # van         → car（小型厢式车按 car 处理）
    6: 1,      # truck       → truck
    7: 8,      # tricycle    → tricycle
    8: 8,      # awning-tricycle → tricycle（三轮棚车归三轮车）
    9: 2,      # bus         → bus
    10: 3,     # motor       → motorcycle
}
RARE_CATEGORIES = {7, 8}      # 有三轮车才保留该图


def parse_annotation(path: Path):
    """返回 [(category, x, y, w, h)]，已过滤忽略项。"""
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) < 6:
            continue
        try:
            x, y, w, h = (float(v) for v in parts[:4])
            score = float(parts[4])
            category = int(float(parts[5]))
        except ValueError:
            continue
        if score <= 0 or category == 0 or w <= 1 or h <= 1:
            continue          # score=0 / 忽略区域 / 退化框
        boxes.append((category, x, y, w, h))
    return boxes


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        return
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    keep_all = "--all" in sys.argv
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    splits = [d for d in root.iterdir() if d.is_dir() and "VisDrone" in d.name]
    if not splits:
        print(f"未在 {root} 下找到 VisDrone2019-DET-* 目录")
        return

    written = 0
    stats: dict[int, int] = {}
    for split_dir in splits:
        ann_dir = split_dir / "annotations"
        img_dir = split_dir / "images"
        if not ann_dir.exists() or not img_dir.exists():
            continue
        for ann in sorted(ann_dir.glob("*.txt")):
            boxes = parse_annotation(ann)
            if not boxes:
                continue
            has_rare = any(cat in RARE_CATEGORIES for cat, *_ in boxes)
            if not has_rare and not keep_all:
                continue
            image = img_dir / f"{ann.stem}.jpg"
            if not image.exists():
                continue
            lines = []
            # VisDrone 的坐标基于原图；读取图片尺寸做归一化
            from PIL import Image
            with Image.open(image) as im:
                width, height = im.size
            for category, x, y, w, h in boxes:
                ours = VISDRONE_MAP.get(category)
                if ours is None:
                    continue
                cx = (x + w / 2) / width
                cy = (y + h / 2) / height
                lines.append(f"{ours} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}")
                stats[ours] = stats.get(ours, 0) + 1
            if not lines:
                continue
            name = f"visdrone_{split_dir.name.split('-')[-1]}_{ann.stem}"
            shutil.copyfile(image, out / "images" / f"{name}.jpg")
            (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n",
                                                        encoding="utf-8")
            written += 1
            if limit and written >= limit:
                break
            if written % 200 == 0:
                print(f"  已转换 {written} 张")
        if limit and written >= limit:
            break

    (out / "classes.txt").write_text("\n".join(OUR_NAMES) + "\n", encoding="utf-8")
    print(f"\n共转换 {written} 张图片 -> {out}")
    for cid, n in sorted(stats.items()):
        print(f"    {OUR_NAMES[cid]:<11} {n}")
    print("（默认只保留含三轮车的图片，用 --all 可转换全部）")


if __name__ == "__main__":
    main()
