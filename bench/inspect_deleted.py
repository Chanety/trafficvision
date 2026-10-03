"""把"被人工删掉的框"裁出来拼成对照图，看清模型到底把什么误判成了目标。

用法：python bench/inspect_deleted.py <人工标注目录> <自动对照目录> <输出目录> [每类数量]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.diff_analysis import load, match  # noqa: E402

CLASS_NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
TILE_W, TILE_H = 160, 160


def main() -> None:
    human_dir = Path(sys.argv[1])
    auto_dir = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    out_dir.mkdir(parents=True, exist_ok=True)

    per_class: dict[int, list[np.ndarray]] = {i: [] for i in range(len(CLASS_NAMES))}
    for label_path in sorted((human_dir / "labels").glob("*.txt")):
        if label_path.name == "classes.txt":
            continue
        auto_path = auto_dir / "labels" / label_path.name
        if not auto_path.exists():
            continue
        auto = load(auto_path)
        human = load(label_path)
        _, deleted, _ = match(auto, human)
        if not deleted:
            continue
        image_path = human_dir / "images" / (label_path.stem + ".jpg")
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        for box in deleted:
            cls = int(box[0])
            if len(per_class[cls]) >= limit:
                continue
            cx, cy, bw, bh = box[1], box[2], box[3], box[4]
            x1 = max(0, int((cx - bw / 2) * w) - 4)
            y1 = max(0, int((cy - bh / 2) * h) - 4)
            x2 = min(w, int((cx + bw / 2) * w) + 4)
            y2 = min(h, int((cy + bh / 2) * h) + 4)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crop = image[y1:y2, x1:x2]
            per_class[cls].append(cv2.resize(crop, (TILE_W, TILE_H)))

    for cls, crops in per_class.items():
        if not crops:
            continue
        cols = 8
        rows = (len(crops) + cols - 1) // cols
        sheet = np.full((rows * TILE_H, cols * TILE_W, 3), 40, dtype=np.uint8)
        for i, crop in enumerate(crops):
            r, c = divmod(i, cols)
            sheet[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = crop
        # 每行之间画分隔线，便于观察
        for r in range(1, rows):
            cv2.line(sheet, (0, r * TILE_H), (cols * TILE_W, r * TILE_H), (90, 90, 90), 1)
        out = out_dir / f"deleted_{CLASS_NAMES[cls]}.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{CLASS_NAMES[cls]:<12} 被删 {len(crops)} 个样本 -> {out.name}")


if __name__ == "__main__":
    main()
