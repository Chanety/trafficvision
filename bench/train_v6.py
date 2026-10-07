"""第六版微调：以新素材（videoset_02）为主体，在 v5 基础上继续训练。

与之前几版的区别：
  · 起点是 v5 的 best.pt，不是原始 yolo11n —— 保住已学会的三轮车与街面能力；
  · 训练数据一半以上来自新素材（域完全匹配用户实际场景）；
  · 训练完成后在同一批 116 帧街面验证集、新素材验证集、VisDrone 三轮车集上分别评估。

用法：python bench/train_v6.py [epochs]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "train_v6" / "data.yaml"
STREET_VAL = ROOT / "datasets" / "train_v5" / "data.yaml"        # 116 帧旧街面验证集
TRICYCLE_VAL = ROOT / "datasets" / "visdrone_tricycle" / "val9.yaml"
BASE_WEIGHTS = ROOT / "outputs" / "train" / "v5" / "weights" / "best.pt"


def report(label: str, metrics) -> None:
    print(f"\n{label}: mAP50 = {metrics.box.map50:.4f}   mAP50-95 = {metrics.box.map:.4f}")
    index = list(getattr(metrics.box, "ap_class_index", []))
    if not index:
        index = list(range(len(metrics.box.ap50)))
    for slot, class_id in enumerate(index):
        name = metrics.names[int(class_id)]
        print(f"  {str(name):<11} AP50 = {metrics.box.ap50[slot]:.4f}")


def main() -> None:
    from ultralytics import YOLO

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    if not BASE_WEIGHTS.exists():
        raise SystemExit(f"找不到 v5 权重：{BASE_WEIGHTS}")
    print(f"数据配置: {DATA}\n起点权重: {BASE_WEIGHTS}")
    model = YOLO(str(BASE_WEIGHTS))
    model.train(
        data=str(DATA),
        epochs=epochs,
        imgsz=960,
        batch=8,
        device=0,
        workers=4,
        project=str(ROOT / "outputs" / "train"),
        name="v6",
        exist_ok=True,
        patience=20,
        plots=True,
    )

    v6 = ROOT / "outputs" / "train" / "v6" / "weights" / "best.pt"
    model = YOLO(str(v6))
    report("新素材验证集（65 帧，自动标注真值 → 偏乐观）",
           model.val(data=str(DATA), split="val", imgsz=960, device=0, verbose=False,
                     project=str(ROOT / "outputs" / "eval"), name="v6_new", exist_ok=True))
    report("旧街面验证集（116 帧，与 v1~v5 同口径）",
           model.val(data=str(STREET_VAL), split="val", imgsz=960, device=0, verbose=False,
                     project=str(ROOT / "outputs" / "eval"), name="v6_street", exist_ok=True))
    report("三轮车专项（VisDrone 399 帧俯拍）",
           model.val(data=str(TRICYCLE_VAL), split="val", imgsz=960, device=0, verbose=False,
                     project=str(ROOT / "outputs" / "eval"), name="v6_tricycle", exist_ok=True))
    print(f"\n权重 -> {v6}")


if __name__ == "__main__":
    main()
