"""真值质量审计：不靠人工逐帧看，用逻辑规则自动判断"指标低是模型问题还是真值问题"。

三条互相独立的证据链：

  1. **规则违反检测**：项目规则是"骑车人只标载具、不标 person"。
     一个 person 标注框若被某个两轮车框大面积覆盖，它就是违规标注。
     两轮车框不必来自真值——用**模型检测**，这样即使真值里漏了车也能揪出来。
  2. **跨模型共识**：v7 与 v8s 都认为"这里有两轮车盖着这个人"才算违规，
     避免单个模型的误检把结论带偏。
  3. **误差方向**：模型能力不足时漏检集中在**小**目标；
     真值脏时，被漏掉的框反而偏**大**（实测 46px vs 命中 21px）。

决定性验证：把被判定违规的 person 框从真值里剔除后重算 AP——
如果 AP 从 0.46 升到 0.7 附近，就等于证明"低分来自真值，不来自模型"。

为对照，同一个审计也跑一遍**人工复核过的旧街面验证集**：
如果那边几乎不触发违规判定，说明规则不是"见人就判违规"，判断有效。

用法：python bench/gt_quality_audit.py [覆盖率阈值=0.3] [检出置信门槛=0.25]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from fusion_experiment import CLASSES, average_precision, load_gt  # noqa: E402

SETS = {
    "新素材65帧(自动标注)": ROOT / "datasets" / "train_v7" / "images" / "val",
    "旧街面116帧(人工复核)": ROOT / "datasets" / "train_v7" / "images" / "street",
}
MODELS = {
    "v7": ROOT / "outputs" / "train" / "v7" / "weights" / "best.pt",
    "v8s": ROOT / "outputs" / "train" / "v8s" / "weights" / "best.pt",
}
PERSON = CLASSES.index("person")
# 7 类方案里两轮/三轮都合并成 nonmotor；同时兼容旧 9 类标签里的对应类名
TWO_WHEEL = {CLASSES.index(name) for name in ("nonmotor", "bicycle", "motorcycle", "tricycle")
             if name in CLASSES}


def coverage(inner, outer) -> float:
    """inner 框被 outer 框覆盖的面积比例。"""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def key(stem: str, box) -> tuple:
    return (stem, round(box[0], 1), round(box[1], 1), round(box[2], 1), round(box[3], 1))


def main() -> None:
    import cv2
    from ultralytics import YOLO

    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    conf_gate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    models = {tag: YOLO(str(p)) for tag, p in MODELS.items()}
    print(f"真值质量审计：两轮车覆盖率 ≥{thr:.0%}（v7 与 v8s 都满足才判违规），"
          f"两轮车检出置信 ≥{conf_gate}\n")

    for set_name, images_dir in SETS.items():
        labels_dir = images_dir.parent.parent / "labels" / images_dir.name
        images = sorted(images_dir.glob("*.jpg"))
        gt_all: dict[str, list] = {}
        preds: list[dict] = []          # v7 的预测（用于算 AP）
        flagged: set[tuple] = set()
        person_boxes: list[tuple[str, list[float]]] = []

        for image in images:
            frame = cv2.imread(str(image))
            h, w = frame.shape[:2]
            gt = load_gt(labels_dir / (image.stem + ".txt"), w, h)
            gt_all[image.stem] = gt
            person_boxes += [(image.stem, b) for cid, b in gt if cid == PERSON]

            votes = {tag: [] for tag in models}
            for tag, model in models.items():
                res = model.predict(frame, imgsz=1280, conf=0.001, device=0, verbose=False)[0]
                boxes = []
                if res.boxes is not None:
                    for i in range(len(res.boxes)):
                        cid = int(res.boxes.cls[i])
                        score = float(res.boxes.conf[i])
                        box = [float(v) for v in res.boxes.xyxy[i]]
                        boxes.append((cid, score, box))
                        if tag == "v7":
                            preds.append({"cls": cid, "score": score, "box": box,
                                          "image": image.stem})
                votes[tag] = [(cid, box) for cid, score, box in boxes if score >= conf_gate]

            for cid, box in gt:
                if cid != PERSON:
                    continue
                hits = [any(c in TWO_WHEEL and coverage(box, ob) >= thr for c, ob in votes[tag])
                        for tag in models]
                if all(hits):
                    flagged.add(key(image.stem, box))

        gt_count: dict[int, int] = {}
        for stem, items in gt_all.items():
            for cid, _b in items:
                gt_count[cid] = gt_count.get(cid, 0) + 1
        raw_records = [dict(p, gt=gt_all[p["image"]]) for p in preds]
        aps_raw = average_precision(raw_records, gt_count)

        clean_all = {stem: [(cid, b) for cid, b in items
                            if not (cid == PERSON and key(stem, b) in flagged)]
                     for stem, items in gt_all.items()}
        clean_count: dict[int, int] = {}
        for stem, items in clean_all.items():
            for cid, _b in items:
                clean_count[cid] = clean_count.get(cid, 0) + 1
        clean_records = [dict(p, gt=clean_all[p["image"]]) for p in preds]
        aps_clean = average_precision(clean_records, clean_count)

        n_person = len(person_boxes)
        heights = [b[3] - b[1] for _s, b in person_boxes]
        flag_h = [b[3] - b[1] for s, b in person_boxes if key(s, b) in flagged]
        print(f"=== {set_name} ===")
        print(f"  行人标注 {n_person} 个（框高中位 {np.median(heights):.0f} px）"
              f"，判定为「两轮车盖着人」的违规框 {len(flagged)} 个"
              f"（{len(flagged) / max(1, n_person) * 100:.0f}%）")
        if flag_h:
            print(f"  违规框高度中位 {np.median(flag_h):.0f} px")
        print(f"  行人 AP50：原始真值 {aps_raw.get('person', float('nan')):.3f}"
              f" → 剔除违规框后 {aps_clean.get('person', float('nan')):.3f}")
        print(f"  全类 mAP50：{np.mean(list(aps_raw.values())):.3f}"
              f" → {np.mean(list(aps_clean.values())):.3f}\n")


if __name__ == "__main__":
    main()
