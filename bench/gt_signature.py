"""真值质量判别规则：用"误差方向"自动区分"模型不行"还是"真值脏"。

背后逻辑（这是同类任务里通用的判别思路）：

  · 如果是**模型能力不足**（看得不够远、不够小），漏检应当集中在**小目标**上，
    即：漏检框的尺寸 < 命中框的尺寸；这类问题靠加数据/加分辨率能缓解。
  · 如果是**真值脏**（把非目标标成了目标、或规则不符的标注），
    那些"错的框"往往不是小目标，模型自然不会在它上面输出 → 漏检框反而偏大，
    出现"漏检框 > 命中框"的反常方向；这类问题加数据也没用，只能修标注。

因此可以定义一个自动判别量：

    误差方向比 R = 漏检框中位尺寸 / 命中框中位尺寸
    R < 1  → 正常（模型对小目标吃力）
    R ≳ 1  → 疑似真值脏（越大于 1 越可疑）

再用"人工复核过的验证集"做对照基线，因为那里的真值可信。

用法：python bench/gt_signature.py [类别=person] [模型=v7]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from fusion_experiment import CLASSES, load_gt  # noqa: E402

SETS = {
    "新素材65帧（自动标注）": ROOT / "datasets" / "train_v7" / "images" / "val",
    "旧街面116帧（人工复核）": ROOT / "datasets" / "train_v7" / "images" / "street",
}
MODEL = ROOT / "outputs" / "train" / "v7" / "weights" / "best.pt"
IMGSZ = 1280
CONF = 0.25
IOU_THR = 0.5


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
    cid_want = CLASSES.index(cls_name)
    model = YOLO(str(MODEL))
    print(f"真值质量指纹（类别 {cls_name}，{MODEL.name}@{IMGSZ}，conf={CONF}）\n")
    print(f"{'验证集':<24}{'标注':>6}{'命中':>6}{'漏检':>6}{'误报':>6}"
          f"{'命中框高':>10}{'漏检框高':>10}{'误差方向比':>12}{'判定':>10}")

    for set_name, images_dir in SETS.items():
        labels_dir = images_dir.parent.parent / "labels" / images_dir.name
        gt_h, fn_h = [], []
        n_gt = n_tp = n_fp = 0
        for image in sorted(images_dir.glob("*.jpg")):
            frame = cv2.imread(str(image))
            h, w = frame.shape[:2]
            gt = [g for g in load_gt(labels_dir / (image.stem + ".txt"), w, h)
                  if g[0] == cid_want]
            res = model.predict(frame, imgsz=IMGSZ, conf=CONF, device=0, verbose=False)[0]
            preds = []
            if res.boxes is not None:
                for i in range(len(res.boxes)):
                    if int(res.boxes.cls[i]) == cid_want:
                        preds.append([float(v) for v in res.boxes.xyxy[i]])
            used = set()
            for _c, gbox in gt:
                n_gt += 1
                best, bi = 0.0, -1
                for pi, pbox in enumerate(preds):
                    if pi in used:
                        continue
                    s = iou(gbox, pbox)
                    if s > best:
                        best, bi = s, pi
                if best >= IOU_THR:
                    used.add(bi)
                    n_tp += 1
                    gt_h.append(gbox[3] - gbox[1])
                else:
                    fn_h.append(gbox[3] - gbox[1])
            n_fp += sum(1 for pi in range(len(preds)) if pi not in used)

        tp_med = float(np.median(gt_h)) if gt_h else float("nan")
        fn_med = float(np.median(fn_h)) if fn_h else float("nan")
        ratio = fn_med / tp_med if gt_h and fn_h else float("nan")
        verdict = "正常" if ratio < 1.0 else ("可疑" if ratio < 1.4 else "很可能真值脏")
        print(f"{set_name:<24}{n_gt:>6}{n_tp:>6}{len(fn_h):>6}{n_fp:>6}"
              f"{tp_med:>10.0f}{fn_med:>10.0f}{ratio:>12.2f}{verdict:>12}")

    print("\n判别说明：误差方向比 = 漏检框高中位 / 命中框高中位。")
    print("  人工复核集是可信基线：若它的比值 <1（漏检的是更小的目标），")
    print("  而自动标注集的比值明显 >1，就说明后者的低分主要来自真值，而不是模型。")


if __name__ == "__main__":
    main()
