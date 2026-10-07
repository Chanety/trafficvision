"""模型融合实验：v7（yolo11n·链式）与 v8s（yolo11s·从零）能否优势互补。

背景：v7 召回高（新素材 R=0.620）、v8s 精确率高（P=0.673）且预判误差更低。
两个模型架构不同、训练历史不同，错检模式有一定互补性——这正是**加权框融合（WBF）**
的用武之地：把两个模型的框按 IoU 聚类后加权平均，通常能同时抬精确率和召回率。

WBF 实现来自开源项目 ZFTurbo/Weighted-Boxes-Fusion（MIT，见 vendor/wbf/LICENSE）。

自评口径：脚本内置一个 AP50 评估器（全点插值，与 COCO/ultralytics 同口径），
并先用"v7 单独"这一栏去对齐 ultralytics 自己算出的数（新素材 0.6371），
对得上才说明后面融合的对比可信。

用法：python bench/fusion_experiment.py [conf] [iou_thr]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from wbf.ensemble_boxes_wbf import weighted_boxes_fusion  # noqa: E402

MODELS = {
    "v7": ROOT / "outputs" / "train" / "v7" / "weights" / "best.pt",
    "v8s": ROOT / "outputs" / "train" / "v8s" / "weights" / "best.pt",
}
SETS = {
    "新素材65帧": ROOT / "datasets" / "train_v7" / "images" / "val",
    "旧街面116帧": ROOT / "datasets" / "train_v7" / "images" / "street",
}
CLASSES = ["car", "truck", "bus", "nonmotor", "person", "sign", "bollard"]


def load_gt(label_path: Path, w: int, h: int) -> list[tuple[int, list[float]]]:
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        out.append((cid, [(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def average_precision(records: list[dict], n_gt: dict[int, int], iou_thr: float = 0.5) -> dict:
    """records: [{'cls','score','box','gt':[(cls,box)],'image':key}]，返回 {类别: AP}。"""
    by_image: dict[str, list] = {}
    for r in records:
        by_image.setdefault(r["image"], []).append(r)
    # 为每张图维护已匹配的 GT
    used = {key: set() for key in by_image}
    for r in sorted(records, key=lambda x: -x["score"]):
        best_iou, best_idx = 0.0, -1
        for idx, (cid, box) in enumerate(r["gt"]):
            if cid != r["cls"] or (r["image"], idx) in used[r["image"]]:
                continue
            score = iou(box, r["box"])
            if score > best_iou:
                best_iou, best_idx = score, idx
        if best_iou >= iou_thr and best_idx >= 0:
            used[r["image"]].add((r["image"], best_idx))
            r["tp"] = 1
        else:
            r["tp"] = 0

    aps = {}
    for cid, name in enumerate(CLASSES):
        # 必须按分数降序：PR 曲线的累积 TP/FP 依赖这个顺序
        preds = sorted((r for r in records if r["cls"] == cid),
                       key=lambda r: -r["score"])
        gt_count = n_gt.get(cid, 0)
        if gt_count == 0:
            continue
        if not preds:
            aps[name] = 0.0
            continue
        tp = np.array([r["tp"] for r in preds], dtype=float)
        fp = 1.0 - tp
        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        recall = tp_cum / gt_count
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        # 全点插值
        mrec = np.concatenate([[0.0], recall, [1.0]])
        mpre = np.concatenate([[1.0], precision, [0.0]])
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        aps[name] = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return aps


def main() -> None:
    from ultralytics import YOLO

    conf = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    iou_thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.55
    imgsz = 1280
    models = {tag: YOLO(str(path)) for tag, path in MODELS.items()}
    print(f"融合实验：conf={conf}, WBF-IoU={iou_thr}, imgsz={imgsz}")

    for set_name, images_dir in SETS.items():
        images = sorted(images_dir.glob("*.jpg"))
        labels_dir = images_dir.parent.parent / "labels" / images_dir.name
        per_model = {tag: [] for tag in models}
        fused_records, gt_count = [], {}

        for image in images:
            import cv2
            frame = cv2.imread(str(image))
            h, w = frame.shape[:2]
            gt = load_gt(labels_dir / (image.stem + ".txt"), w, h)
            for cid, _box in gt:
                gt_count[cid] = gt_count.get(cid, 0) + 1

            boxes_list, scores_list, labels_list = [], [], []
            for tag, model in models.items():
                res = model.predict(frame, imgsz=imgsz, conf=conf, device=0, verbose=False)[0]
                boxes, scores, labels = [], [], []
                if res.boxes is not None and len(res.boxes):
                    for i in range(len(res.boxes)):
                        x1, y1, x2, y2 = (float(v) for v in res.boxes.xyxy[i])
                        score = float(res.boxes.conf[i])
                        cid = int(res.boxes.cls[i])
                        boxes.append([x1, y1, x2, y2])
                        scores.append(score)
                        labels.append(cid)
                        per_model[tag].append({"cls": cid, "score": score,
                                               "box": [x1, y1, x2, y2], "gt": gt,
                                               "image": image.stem})
                norm = [[b[0] / w, b[1] / h, b[2] / w, b[3] / h] for b in boxes]
                boxes_list.append(norm)
                scores_list.append(scores)
                labels_list.append(labels)

            fb, fs, fl = weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                               iou_thr=iou_thr, skip_box_thr=0.0)
            for box, score, cid in zip(fb, fs, fl):
                fused_records.append({"cls": int(cid), "score": float(score),
                                      "box": [box[0] * w, box[1] * h, box[2] * w, box[3] * h],
                                      "gt": gt, "image": image.stem})

        print(f"\n=== {set_name}（{len(images)} 帧）===")
        print(f"{'方案':<10}{'mAP50':>9}{'汽车':>8}{'卡车':>8}{'公交':>8}"
              f"{'非机动车':>10}{'行人':>8}{'标志牌':>9}{'路桩':>8}")
        for tag, records in list(per_model.items()) + [("融合", fused_records)]:
            aps = average_precision(records, gt_count)
            mean = float(np.mean(list(aps.values()))) if aps else float("nan")
            print(f"{tag:<10}{mean:>9.4f}"
                  + "".join(f"{aps.get(c, float('nan')):>8.3f}" if len(c) <= 3
                            else f"{aps.get(c, float('nan')):>10.3f}" for c in CLASSES))


if __name__ == "__main__":
    main()
