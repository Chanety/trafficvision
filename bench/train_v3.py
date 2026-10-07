"""第二版微调：在人工复核样本基础上加入稀有类补充数据。

对比对象是第一版（仅 219 张人工复核样本），重点看 bicycle / motorcycle 的变化。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "train_v2" / "data.yaml"
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
        name="v3",
        exist_ok=True,
        patience=25,
        plots=True,
    )
    metrics = model.val()
    print("验证结果（第二版）:")
    print(f"  mAP50    = {metrics.box.map50:.4f}")
    print(f"  mAP50-95 = {metrics.box.map:.4f}")
    for i, name in enumerate(metrics.names.values()):
        if i < len(metrics.box.ap50):
            print(f"  {name:<11} AP50 = {metrics.box.ap50[i]:.4f}")


if __name__ == "__main__":
    main()
