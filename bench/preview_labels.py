"""生成自动标注预览图：把若干张标注集的框画出来，方便先看质量。

用法：python bench/preview_labels.py <标注集目录> [数量] [输出图]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trafficvision.config import CLASS_COLORS, DEFAULT_COLOR  # noqa: E402


def draw(image_path: Path, label_path: Path, names: list[str]):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            name = names[cid] if cid < len(names) else str(cid)
            color = CLASS_COLORS.get(name, DEFAULT_COLOR)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, name, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    return img


def main() -> None:
    dataset = Path(sys.argv[1])
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else dataset / "preview.jpg"

    names = (dataset / "classes.txt").read_text(encoding="utf-8").split()
    images = sorted((dataset / "images").glob("*.jpg"))
    if not images:
        print("没有图片")
        return

    # 优先挑标注最密的帧，这样预览能反映真实工作内容
    def boxes_of(img_path: Path) -> int:
        label_path = dataset / "labels" / (img_path.stem + ".txt")
        if not label_path.exists():
            return 0
        return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())

    ranked = sorted(images, key=boxes_of, reverse=True)
    # 保证来自不同视频：按视频名去重
    seen: set[str] = set()
    picks = []
    for img_path in ranked:
        key = img_path.stem.split("_f")[0]
        if key in seen:
            continue
        seen.add(key)
        picks.append(img_path)
        if len(picks) >= count:
            break
    tiles = []
    for img_path in picks:
        label_path = dataset / "labels" / (img_path.stem + ".txt")
        tile = draw(img_path, label_path, names)
        if tile is None:
            continue
        tile = cv2.resize(tile, (640, int(640 * tile.shape[0] / tile.shape[1])))
        tiles.append(tile)

    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    tile_h = max(t.shape[0] for t in tiles)
    sheet = np.full((rows * tile_h, cols * 640, 3), 255, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * tile_h:r * tile_h + tile.shape[0], c * 640:(c + 1) * 640] = tile
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"预览图 -> {out_path}   共 {len(tiles)} 张")


if __name__ == "__main__":
    main()
