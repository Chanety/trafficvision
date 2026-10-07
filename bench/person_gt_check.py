"""行人类别真值体检：把"模型与标注对不上"的行人框挑出来，做成图给人工看。

背景：同一个模型在旧街面验证集上行人 AP50 是 0.72，在新素材上只有 0.46。
新素材的验证真值来自**自动标注**，所以差距很可能是标注质量问题，而不是模型。

本脚本把两类可疑样本可视化：
  · FN（标注有、模型没检出）：可能是**幽灵标注**（把背景杂物/反光当成人）；
  · FP（模型检出、标注没有）：可能是**漏标注**（真有人但没标）。

同时输出统计（数量、框高分布），并挑出行人最多的若干帧组成一个**专项复核集**，
让人工只花几分钟就能判断问题出在哪一边。

用法：python bench/person_gt_check.py [conf] [专项复核集帧数]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "datasets" / "train_v7" / "images" / "val"
LABELS = ROOT / "datasets" / "train_v7" / "labels" / "val"
WEIGHTS = ROOT / "outputs" / "train" / "v7" / "weights" / "best.pt"
OUT_DIR = ROOT / "outputs" / "person_gt_check"
SUBSET_DIR = ROOT / "datasets" / "person_check"
CLASSES = ["car", "truck", "bus", "nonmotor", "person", "sign", "bollard"]
PERSON = 4


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(label: Path, w: int, h: int):
    out = []
    if not label.exists():
        return out
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        out.append((cid, [(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def main() -> None:
    import cv2
    from ultralytics import YOLO

    conf = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    subset_n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))

    fn_items, fp_items, per_image = [], [], []
    total_gt = total_tp = 0
    heights_fn, heights_tp = [], []

    for image in sorted(IMAGES.glob("*.jpg")):
        frame = cv2.imread(str(image))
        h, w = frame.shape[:2]
        gt = load_gt(LABELS / (image.stem + ".txt"), w, h)
        gt_person = [g for g in gt if g[0] == PERSON]
        res = model.predict(frame, imgsz=1280, conf=conf, device=0, verbose=False)[0]
        preds = []
        if res.boxes is not None:
            for i in range(len(res.boxes)):
                if int(res.boxes.cls[i]) != PERSON:
                    continue
                preds.append(([float(v) for v in res.boxes.xyxy[i]], float(res.boxes.conf[i])))

        used = set()
        n_tp = 0
        for gi, (_cid, gbox) in enumerate(gt_person):
            best, best_i = 0.0, -1
            for pi, (pbox, _s) in enumerate(preds):
                if pi in used:
                    continue
                score = iou(gbox, pbox)
                if score > best:
                    best, best_i = score, pi
            if best >= 0.5:
                used.add(best_i)
                n_tp += 1
                heights_tp.append(gbox[3] - gbox[1])
            else:
                fn_items.append((image, frame, gbox))
                heights_fn.append(gbox[3] - gbox[1])
        for pi, (pbox, score) in enumerate(preds):
            if pi not in used:
                fp_items.append((image, frame, pbox, score))

        total_gt += len(gt_person)
        total_tp += n_tp
        per_image.append((image.stem, len(gt_person), len(preds), n_tp))

    n_fn, n_fp = len(fn_items), len(fp_items)
    print(f"行人类别体检（新素材验证集 65 帧，v7@1280，conf={conf}）")
    print(f"  标注框 {total_gt} 个 → 命中 {total_tp} 个，"
          f"漏检 {n_fn} 个（{(n_fn / max(1, total_gt) * 100):.0f}%）")
    print(f"  模型检出但标注没有的：{n_fp} 个")
    if heights_tp:
        print(f"  命中框高度中位 {np.median(heights_tp):.0f} px")
    if heights_fn:
        print(f"  漏检框高度中位 {np.median(heights_fn):.0f} px")

    # 拼图：上面是"标注有但模型没检出"，下面是"模型检出但标注没有"
    def montage(items, tag, is_fn: bool, columns: int = 6, cell: int = 260):
        rows = []
        for i in range(0, min(len(items), 12), columns):
            cells = []
            for item in items[i:i + columns]:
                image, frame, box = item[0], item[1], item[2]
                pad = 0.6
                x1, y1, x2, y2 = (int(v) for v in box)
                bw, bh = max(1, x2 - x1), max(1, y2 - y1)
                px, py = int(bw * pad), int(bh * pad)
                H, W = frame.shape[:2]
                crop = frame[max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(W, x2 + px)]
                if crop.size == 0:
                    continue
                scale = cell / max(1, crop.shape[0])
                crop = cv2.resize(crop, None, fx=scale, fy=scale)
                if crop.shape[1] > cell:
                    crop = crop[:, :cell]
                canvas = np.zeros((cell + 26, cell, 3), dtype=np.uint8)
                canvas[26:26 + crop.shape[0], :crop.shape[1]] = crop
                text = f"{tag} {image.stem[-6:]} 高{bh}px"
                if not is_fn:
                    text += f" conf{item[3]:.2f}"
                cv2.putText(canvas, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cells.append(canvas)
            if cells:
                while len(cells) < columns:
                    cells.append(np.zeros_like(cells[0]))
                rows.append(np.hstack(cells))
        return np.vstack(rows) if rows else None

    for items, tag, is_fn, fname in (
            (fn_items, "FN-标注有/模型没检出", True, "fn_gt_only.png"),
            (fp_items, "FP-模型检出/标注没有", False, "fp_pred_only.png")):
        img = montage(items, tag, is_fn)
        if img is not None:
            cv2.imwrite(str(OUT_DIR / fname), img)
            print(f"  样例图 -> {OUT_DIR / fname}")

    # 专项复核集：挑行人最多的若干帧（标注重、问题最集中）
    per_image.sort(key=lambda r: (-r[1], r[0]))
    chosen = [r[0] for r in per_image[:subset_n]]
    for sub in ("images", "labels"):
        (SUBSET_DIR / sub).mkdir(parents=True, exist_ok=True)
    for stem in chosen:
        shutil.copyfile(IMAGES / f"{stem}.jpg", SUBSET_DIR / "images" / f"{stem}.jpg")
        shutil.copyfile(LABELS / f"{stem}.txt", SUBSET_DIR / "labels" / f"{stem}.txt")
    (SUBSET_DIR / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    rows = [f"{s}.jpg" for s in chosen]
    (SUBSET_DIR / "review_order.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(json.dumps({
        "total_gt": total_gt, "tp": total_tp, "fn": n_fn, "fp": n_fp,
        "fn_median_height": float(np.median(heights_fn)) if heights_fn else None,
        "tp_median_height": float(np.median(heights_tp)) if heights_tp else None,
        "per_image": per_image,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n专项复核集（{len(chosen)} 帧，行人标注最密集）-> {SUBSET_DIR}")
    print(f"统计明细 -> {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
