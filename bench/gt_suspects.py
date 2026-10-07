"""把"最可疑的真值框"自动挑出来拼成一页，人工扫一眼即可定性。

判别依据（详见 bench/gt_signature.py 的说明）：
  · 模型能力不足 → 漏检集中在**小**目标；
  · 真值脏 → 会出现"漏检的框反而更大"的反常方向。

所以本脚本挑的是：
  **尺寸不小、但 v7 和 v8s 两个模型都认不出来**的标注框——
  这种框要么是标错了，要么是标注规则不符（例如骑车人被标成 person）。
  把它们按尺寸从大到小拼成一页，人工只需扫这一页（不需要逐帧看 332 帧）。

用法：python bench/gt_suspects.py [类别=person] [最少高度px=30] [页数=24]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from fusion_experiment import CLASSES, load_gt  # noqa: E402

IMAGES = ROOT / "datasets" / "train_v7" / "images" / "val"
LABELS = ROOT / "datasets" / "train_v7" / "labels" / "val"
MODELS = {
    "v7": ROOT / "outputs" / "train" / "v7" / "weights" / "best.pt",
    "v8s": ROOT / "outputs" / "train" / "v8s" / "weights" / "best.pt",
}
OUT = ROOT / "outputs" / "gt_suspects"


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    import cv2
    from ultralytics import YOLO

    cls_name = sys.argv[1] if len(sys.argv) > 1 else "person"
    min_h = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    page = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    cid_want = CLASSES.index(cls_name)
    models = {tag: YOLO(str(p)) for tag, p in MODELS.items()}
    OUT.mkdir(parents=True, exist_ok=True)

    suspects = []
    for image in sorted(IMAGES.glob("*.jpg")):
        frame = cv2.imread(str(image))
        h, w = frame.shape[:2]
        gt = [g for g in load_gt(LABELS / (image.stem + ".txt"), w, h) if g[0] == cid_want]
        if not gt:
            continue
        consensus = None
        for tag, model in models.items():
            res = model.predict(frame, imgsz=1280, conf=0.25, device=0, verbose=False)[0]
            boxes = []
            if res.boxes is not None:
                for i in range(len(res.boxes)):
                    if int(res.boxes.cls[i]) == cid_want:
                        boxes.append([float(v) for v in res.boxes.xyxy[i]])
            found = {gi for gi, (_c, gbox) in enumerate(gt)
                     if any(iou(gbox, b) >= 0.5 for b in boxes)}
            consensus = found if consensus is None else (consensus & found)
        for gi, (_c, gbox) in enumerate(gt):
            height = gbox[3] - gbox[1]
            if height >= min_h and gi not in consensus:
                suspects.append({"image": image, "frame": frame, "box": gbox, "h": height})

    suspects.sort(key=lambda s: -s["h"])
    print(f"类别 {cls_name}：高度 ≥{min_h:.0f}px 且两个模型都不认的标注框 {len(suspects)} 个"
          f"（占该类别标注 {len(suspects) / max(1, sum(1 for i in IMAGES.glob('*.jpg'))):.0f} 个/帧量级）")
    if not suspects:
        print("没有可疑框")
        return

    columns, cell = 6, 300
    rows = []
    for i in range(0, min(len(suspects), page), columns):
        cells = []
        for s in suspects[i:i + columns]:
            x1, y1, x2, y2 = (int(v) for v in s["box"])
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            px, py = int(bw * 0.8), int(bh * 0.8)
            H, W = s["frame"].shape[:2]
            crop = s["frame"][max(0, y1 - py):min(H, y2 + py),
                              max(0, x1 - px):min(W, x2 + px)].copy()
            if crop.size == 0:
                continue
            # 在裁片上画出标注框，方便看清"标的是什么"
            cx1, cy1 = max(0, x1 - (x1 - px) if x1 - px > 0 else x1), max(0, y1 - py)
            cv2.rectangle(crop, (x1 - max(0, x1 - px), y1 - max(0, y1 - py)),
                          (x2 - max(0, x1 - px), y2 - max(0, y1 - py)), (0, 0, 255), 2)
            scale = cell / max(1, crop.shape[0])
            crop = cv2.resize(crop, None, fx=scale, fy=scale)
            canvas = np.zeros((cell + 26, cell, 3), dtype=np.uint8)
            ch = min(cell, crop.shape[0])
            cw = min(cell, crop.shape[1])
            canvas[26:26 + ch, :cw] = crop[:ch, :cw]
            cv2.putText(canvas, f"{s['image'].stem[-6:]} 高{int(s['h'])}px",
                        (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(canvas)
        if cells:
            while len(cells) < columns:
                cells.append(np.zeros_like(cells[0]))
            rows.append(np.hstack(cells))
    sheet = np.vstack(rows)
    path = OUT / f"{cls_name}_suspects.png"
    cv2.imwrite(str(path), sheet)
    print(f"可疑框一览（红框=标注框，模型认不出来）-> {path}")
    print("判读提示：如果这些红框里大多是骑车人/反光/背景杂物，就说明是真值问题；")
    print("          如果都是模糊的远处小人，那才是模型能力问题。")


if __name__ == "__main__":
    main()
