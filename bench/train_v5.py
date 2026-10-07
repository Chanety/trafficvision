"""第五版微调：在 train_v4 基础上并入 VisDrone 三轮车数据，补齐第 9 类 tricycle。

超参与 train_v4.py 完全一致（imgsz=960, batch=8, 100 epochs, patience=25），
这样 v4 与 v5 的差异只来自数据，指标可比。

用法：python bench/train_v5.py [epochs]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "train_v5" / "data.yaml"
TRICYCLE_DATA = ROOT / "datasets" / "visdrone_tricycle" / "val9.yaml"
BASE_WEIGHTS = ROOT / "trafficlab-3d" / "repo" / "models" / "yolo11n.pt"


def main() -> None:
    from ultralytics import YOLO

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(f"数据配置: {DATA}")
    model = YOLO(str(BASE_WEIGHTS))
    model.train(
        data=str(DATA),
        epochs=epochs,
        imgsz=960,
        batch=8,
        device=0,
        workers=4,
        project=str(ROOT / "outputs" / "train"),
        name="v5",
        exist_ok=True,
        patience=25,
        plots=True,
    )

    print("\n验证结果（第五版，街面 116 帧验证集，与 v3/v4 同口径）:")
    metrics = model.val(data=str(DATA), split="val", imgsz=960, device=0, verbose=False,
                        project=str(ROOT / "outputs" / "eval"), name="v5", exist_ok=True)
    print(f"  mAP50    = {metrics.box.map50:.4f}")
    print(f"  mAP50-95 = {metrics.box.map:.4f}")
    for i, name in enumerate(metrics.names.values()):
        if i < len(metrics.box.ap50):
            print(f"  {name:<11} AP50 = {metrics.box.ap50[i]:.4f}")

    print("\n三轮车专项验证（VisDrone 399 张验证帧，域外俯拍）:")
    tri = model.val(data=str(TRICYCLE_DATA), split="val", imgsz=960, device=0, verbose=False,
                    project=str(ROOT / "outputs" / "eval"), name="v5_tricycle", exist_ok=True)
    print(f"  mAP50    = {tri.box.map50:.4f}")
    print(f"  mAP50-95 = {tri.box.map:.4f}")
    for i, name in enumerate(tri.names.values()):
        if i < len(tri.box.ap50) and name == "tricycle":
            print(f"  tricycle  AP50 = {tri.box.ap50[i]:.4f}  AP50-95 = {tri.box.ap[i]:.4f}")


if __name__ == "__main__":
    main()
