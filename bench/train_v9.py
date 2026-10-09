"""第九版微调：10 类口径（合并 + 路灯 + 信号灯 + 非机动车驾驶员）。

起点是 v7 的 best.pt（7 类）。类别数从 7 变 10 时检测头会被重建，
骨干与颈部权重仍然继承——相当于"用已经学到的特征去适配新的类别定义"。

三轮验证集都用 10 类口径，且与 v1~v8 的验证帧完全相同，便于横比：
    data.yaml（新素材 65 帧）/ street_val.yaml（旧街面 116 帧）/ aerial_val.yaml（俯拍 399 帧）

用法：python bench/train_v9.py [epochs]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _arg(name: str, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


# 支持用 --data 指到别的数据集目录（例如清洗后的 train_v9b），--name 指定输出名
_DATA_DIR = Path(_arg("--data", str(ROOT / "datasets" / "train_v9")))
DATA = _DATA_DIR / "data.yaml"
STREET_VAL = _DATA_DIR / "street_val.yaml"
AERIAL_VAL = _DATA_DIR / "aerial_val.yaml"
OUT_NAME = _arg("--name", "v9")
# 起点权重：默认 v7（7 类链式微调）；也可用 --base 指到 v9 续训
BASE_WEIGHTS = Path(_arg("--base", str(ROOT / "outputs" / "train" / "v7" / "weights" / "best.pt")))
RESUME = "--resume" in sys.argv


def report(label: str, metrics) -> None:
    print(f"\n{label}: mAP50 = {metrics.box.map50:.4f}   mAP50-95 = {metrics.box.map:.4f}")
    index = list(getattr(metrics.box, "ap_class_index", []))
    if not index:
        index = list(range(len(metrics.box.ap50)))
    for slot, class_id in enumerate(index):
        print(f"  {str(metrics.names[int(class_id)]):<13} AP50 = {metrics.box.ap50[slot]:.4f}")


def main() -> None:
    from ultralytics import YOLO

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 120
    if not BASE_WEIGHTS.exists():
        raise SystemExit(f"找不到 v7 权重：{BASE_WEIGHTS}")
    print(f"数据配置: {DATA}\n起点权重: {BASE_WEIGHTS}\n输出名: {OUT_NAME}\n"
          f"轮数: {epochs}\n续训: {RESUME}", flush=True)

    last = ROOT / "outputs" / "train" / OUT_NAME / "weights" / "last.pt"
    model = YOLO(str(last if (RESUME and last.exists()) else BASE_WEIGHTS))
    model.train(
        data=str(DATA),
        epochs=epochs,
        imgsz=960,
        batch=8,
        device=0,
        workers=4,
        project=str(ROOT / "outputs" / "train"),
        name=OUT_NAME,
        exist_ok=True,
        patience=25,
        plots=True,
        resume=bool(RESUME and last.exists()),
    )

    best = ROOT / "outputs" / "train" / OUT_NAME / "weights" / "best.pt"
    model = YOLO(str(best))
    for tag, path in (("新素材验证集（65 帧）", DATA),
                      ("旧街面验证集（116 帧）", STREET_VAL),
                      ("俯拍验证集（399 帧）", AERIAL_VAL)):
        report(f"{tag} · 10 类口径",
               model.val(data=str(path), split="val", imgsz=1280, device=0, verbose=False,
                         project=str(ROOT / "outputs" / "eval"),
                         name=f"{OUT_NAME}_{path.stem}", exist_ok=True))
    print(f"\n权重 -> {best}")


if __name__ == "__main__":
    main()
