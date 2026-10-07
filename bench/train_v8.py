"""第八版：换更大检测模型，验证"小目标能不能靠模型容量救回来"。

为什么还要训一版 n：
  v7 是在 v4→v5→v6 链式微调的基础上来的，而 yolo11s 只能从 COCO 预训练重新开始。
  为了把"模型更大"和"训练历史不同"这两个因素分开，这里同时从零训一版 yolo11n
  （v8n）作为对照，配方与 v8s 完全一致。

batch 保持 8，与 v7 的配方一致（实测 yolo11s 在 imgsz=960、batch=8 下
显存约 4.7GB，8GB 卡放得下）。

用法：python bench/train_v8.py n|s [epochs]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "train_v7" / "data.yaml"
STREET_VAL = ROOT / "datasets" / "train_v7" / "street_val.yaml"
AERIAL_VAL = ROOT / "datasets" / "train_v7" / "aerial_val.yaml"


def report(label: str, metrics) -> None:
    print(f"\n{label}: mAP50 = {metrics.box.map50:.4f}   mAP50-95 = {metrics.box.map:.4f}"
          f"   P = {metrics.box.mp:.3f}   R = {metrics.box.mr:.3f}")
    index = list(getattr(metrics.box, "ap_class_index", [])) or list(range(len(metrics.box.ap50)))
    for slot, class_id in enumerate(index):
        print(f"  {str(metrics.names[int(class_id)]):<11} AP50 = {metrics.box.ap50[slot]:.4f}")


def main() -> None:
    from ultralytics import YOLO

    size = (sys.argv[1] if len(sys.argv) > 1 else "s").lower()
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    if size not in {"n", "s", "m"}:
        raise SystemExit("模型规格只能是 n / s / m")
    weights = ROOT / "trafficlab-3d" / "repo" / "models" / f"yolo11{size}.pt"
    if not weights.exists():
        raise SystemExit(f"找不到预训练权重：{weights}")
    name = f"v8{size}"
    print(f"数据 {DATA}\n起点权重 {weights.name}\n实验名 {name}，{epochs} 轮，batch=8")
    model = YOLO(str(weights))
    model.train(
        data=str(DATA), epochs=epochs, imgsz=960, batch=8, device=0, workers=4,
        project=str(ROOT / "outputs" / "train"), name=name, exist_ok=True,
        patience=20, plots=True,
    )

    best = ROOT / "outputs" / "train" / name / "weights" / "best.pt"
    model = YOLO(str(best))
    # 推理分辨率用产品默认的 1280（实测 1280 优于 960）
    report("新素材验证集（65 帧，imgsz 1280）",
           model.val(data=str(DATA), split="val", imgsz=1280, device=0, verbose=False,
                     project=str(ROOT / "outputs" / "eval"), name=f"{name}_new", exist_ok=True))
    report("旧街面验证集（116 帧，imgsz 1280）",
           model.val(data=str(STREET_VAL), split="val", imgsz=1280, device=0, verbose=False,
                     project=str(ROOT / "outputs" / "eval"), name=f"{name}_street", exist_ok=True))
    report("俯拍验证集（399 帧，imgsz 1280）",
           model.val(data=str(AERIAL_VAL), split="val", imgsz=1280, device=0, verbose=False,
                     project=str(ROOT / "outputs" / "eval"), name=f"{name}_aerial", exist_ok=True))


if __name__ == "__main__":
    main()
