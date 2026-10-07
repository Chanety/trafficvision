"""第七版微调：类别合并为 7 类（摩托车/自行车/三轮车 → 非机动车 nonmotor）。

起点是 v6 的 best.pt。类别数从 9 变 7 时，检测头会被重建（骨干与颈部权重仍然继承），
所以这一版相当于"用已经学会的特征去适配新类别定义"。

用法：python bench/train_v7.py [epochs]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "train_v7" / "data.yaml"
STREET_VAL = ROOT / "datasets" / "train_v7" / "street_val.yaml"
AERIAL_VAL = ROOT / "datasets" / "train_v7" / "aerial_val.yaml"
BASE_WEIGHTS = ROOT / "outputs" / "train" / "v6" / "weights" / "best.pt"


def report(label: str, metrics) -> None:
    print(f"\n{label}: mAP50 = {metrics.box.map50:.4f}   mAP50-95 = {metrics.box.map:.4f}")
    index = list(getattr(metrics.box, "ap_class_index", []))
    if not index:
        index = list(range(len(metrics.box.ap50)))
    for slot, class_id in enumerate(index):
        print(f"  {str(metrics.names[int(class_id)]):<11} AP50 = {metrics.box.ap50[slot]:.4f}")


def main() -> None:
    from ultralytics import YOLO

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    if not BASE_WEIGHTS.exists():
        raise SystemExit(f"找不到 v6 权重：{BASE_WEIGHTS}")
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
        name="v7",
        exist_ok=True,
        patience=20,
        plots=True,
    )

    model = YOLO(str(ROOT / "outputs" / "train" / "v7" / "weights" / "best.pt"))
    report("新素材验证集（65 帧，自动标注真值）",
           model.val(data=str(DATA), split="val", imgsz=960, device=0, verbose=False,
                     project=str(ROOT / "outputs" / "eval"), name="v7_new", exist_ok=True))
    report("旧街面验证集（116 帧，7 类口径）",
           model.val(data=str(STREET_VAL), split="val", imgsz=960, device=0, verbose=False,
                     project=str(ROOT / "outputs" / "eval"), name="v7_street", exist_ok=True))
    report("俯拍验证集（399 帧，7 类口径）",
           model.val(data=str(AERIAL_VAL), split="val", imgsz=960, device=0, verbose=False,
                     project=str(ROOT / "outputs" / "eval"), name="v7_aerial", exist_ok=True))
    print(f"\n权重 -> {ROOT / 'outputs' / 'train' / 'v7' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
