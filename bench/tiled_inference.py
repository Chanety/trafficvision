"""切图推理实验（SAHI 思路）：把高清帧切成有重叠的小块分别检测，再合并。

动机：我们的弱项是小目标（行人 AP50 0.462、卡车 0.285）。
整图提高分辨率到 1600 已经证明无效（docs/11），但"切图"不同：
1920x1080 整图送进 1280 网络是**下采样**，而 960x540 的小块送进 1280 是**上采样**，
等效于把小目标放大 2 倍再看。文献见 docs/15（SAHI, arXiv:2202.06934）。

合并用开源 WBF（vendor/wbf/，MIT）——把每个切块当作一个"模型"，
再加上整图那一路（SAHI 的 with-full-inference 做法）。

用法：python bench/tiled_inference.py [conf] [每边切块数] [重叠比例]
例：  python bench/tiled_inference.py 0.001 2 0.2
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "bench"))
from wbf.ensemble_boxes_wbf import weighted_boxes_fusion  # noqa: E402
from fusion_experiment import CLASSES, average_precision, load_gt  # noqa: E402

WEIGHTS = ROOT / "outputs" / "train" / "v7" / "weights" / "best.pt"
SETS = {
    "新素材65帧": ROOT / "datasets" / "train_v7" / "images" / "val",
    "旧街面116帧": ROOT / "datasets" / "train_v7" / "images" / "street",
}


def tiles_for(w: int, h: int, n: int, overlap: float):
    tw, th = w // n, h // n
    step_x, step_y = int(tw * (1 - overlap)), int(th * (1 - overlap))
    boxes = []
    for i in range(n):
        for j in range(n):
            x1 = min(i * step_x, max(0, w - tw))
            y1 = min(j * step_y, max(0, h - th))
            boxes.append((x1, y1, min(w, x1 + tw), min(h, y1 + th)))
    return boxes


def detections(model, frame, imgsz: int, conf: float):
    res = model.predict(frame, imgsz=imgsz, conf=conf, device=0, verbose=False)[0]
    out = []
    if res.boxes is not None:
        for i in range(len(res.boxes)):
            out.append(([float(v) for v in res.boxes.xyxy[i]],
                        float(res.boxes.conf[i]), int(res.boxes.cls[i])))
    return out


def nms(cands: list[tuple[list[float], float, int]], iou_thr: float = 0.5):
    """类别感知的贪心 NMS（SAHI 的标准后处理用的就是这个，不是 WBF）。"""
    import math

    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    kept = []
    for box, score, cid in sorted(cands, key=lambda c: -c[1]):
        if any(cid == k[2] and iou(box, k[0]) > iou_thr for k in kept):
            continue
        kept.append((box, score, cid))
    return kept


def main() -> None:
    import cv2
    from ultralytics import YOLO

    conf = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    overlap = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    imgsz = 1280
    model = YOLO(str(WEIGHTS))
    print(f"切图推理实验：{n}×{n} 切块，重叠 {overlap:.0%}，conf={conf}，imgsz={imgsz}")

    for set_name, images_dir in SETS.items():
        images = sorted(images_dir.glob("*.jpg"))
        labels_dir = images_dir.parent.parent / "labels" / images_dir.name
        plain_records, tiled_records = [], []
        gt_count: dict[int, int] = {}

        for image in images:
            frame = cv2.imread(str(image))
            h, w = frame.shape[:2]
            gt = load_gt(labels_dir / (image.stem + ".txt"), w, h)
            for cid, _ in gt:
                gt_count[cid] = gt_count.get(cid, 0) + 1

            full = detections(model, frame, imgsz, conf)
            for box, score, cid in full:
                plain_records.append({"cls": cid, "score": score, "box": box,
                                      "gt": gt, "image": image.stem})

            merged: list[tuple[list[float], float, int]] = list(full)
            edge_margin = 2.0
            for (x1, y1, x2, y2) in tiles_for(w, h, n, overlap):
                tile = frame[y1:y2, x1:x2]
                for box, score, cid in detections(model, tile, imgsz, conf):
                    # 贴到切块边缘的框多半是被切断的半个目标，SAHI 的标准做法是丢弃
                    tw, th = x2 - x1, y2 - y1
                    if (box[0] <= edge_margin or box[1] <= edge_margin
                            or box[2] >= tw - edge_margin or box[3] >= th - edge_margin):
                        continue
                    merged.append(([box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1],
                                   score, cid))
            for box, score, cid in nms(merged, 0.5):
                tiled_records.append({"cls": cid, "score": score, "box": box,
                                      "gt": gt, "image": image.stem})

        print(f"\n=== {set_name}（{len(images)} 帧）===")
        print(f"{'方案':<12}{'mAP50':>9}" + "".join(f"{c:>10}" for c in CLASSES))
        for tag, records in (("整图1280", plain_records), ("切图合并", tiled_records)):
            aps = average_precision(records, gt_count)
            mean = float(np.mean(list(aps.values()))) if aps else float("nan")
            print(f"{tag:<12}{mean:>9.4f}"
                  + "".join(f"{aps.get(c, float('nan')):>10.3f}" for c in CLASSES))


if __name__ == "__main__":
    main()
