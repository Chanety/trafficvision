"""微调第一版检测模型。

以 yolo11n 为起点，在人工复核后的 219 张图上微调 8 类；
验证集按视频划分（3 段视频完全不参与训练），结果写入 runs/ 目录。

用法：python bench/train_v1.py [epochs]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "train_v1" / "data.yaml"
BASE = ROOT / "trafficlab-3d" / "repo" / "models" / "yolo11n.pt"


def main() -> None:
    from ultralytics import YOLO

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(f"基础权重: {BASE}")
    print(f"数据配置: {DATA}")
    model = YOLO(str(BASE))
    model.train(
        data=str(DATA),
        epochs=epochs,
        imgsz=960,          # 小目标多，分辨率不宜过低
        batch=8,
        device=0,
        workers=4,
        project=str(ROOT / "outputs" / "train"),
        name="v1",
        exist_ok=True,
        patience=25,
        cache=False,
        plots=True,
    )
    metrics = model.val()
    print("验证结果:")
    print(f"  mAP50    = {metrics.box.map50:.4f}")
    print(f"  mAP50-95 = {metrics.box.map:.4f}")
    for i, name in enumerate(metrics.names.values()):
        if i < len(metrics.box.ap50):
            print(f"  {name:<11} AP50 = {metrics.box.ap50[i]:.4f}")


if __name__ == "__main__":
    main()
