"""第五版加入后的公平对比：同样的 116 帧街面验证集 + VisDrone 三轮车专项验证。

注意口径：
  · v1/v3/v4 是 8 类模型，用 train_v4/data.yaml 评测；
    v5 是 9 类模型，用 train_v5/data.yaml 评测（验证集图片完全相同）。
  · 为了可比，表里额外给出"共有 8 类"的 mAP50 均值（不含 tricycle），
    这是 v1~v5 唯一可以直接横比的口径。
  · tricycle 只在 VisDrone 的 399 张验证帧上单独测（街面验证集没有三轮车真值）。

用法：python bench/eval_compare_v5.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STREET_VAL = ROOT / "datasets" / "train_v5" / "data.yaml"
OLD_VAL = ROOT / "datasets" / "train_v4" / "data.yaml"
TRICYCLE_VAL = ROOT / "datasets" / "visdrone_tricycle" / "val9.yaml"
SHARED = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "sign", "bollard"]

MODELS = {
    "v1_人工复核219": (ROOT / "outputs" / "train" / "v1" / "weights" / "best.pt", OLD_VAL),
    "v3_加补充370": (ROOT / "outputs" / "train" / "v3" / "weights" / "best.pt", OLD_VAL),
    "v4_扩充473": (ROOT / "outputs" / "train" / "v4" / "weights" / "best.pt", OLD_VAL),
    "v5_加三轮973": (ROOT / "outputs" / "train" / "v5" / "weights" / "best.pt", STREET_VAL),
}


def per_class(metrics) -> dict[str, float]:
    """把 AP 数组正确对回类别名。

    坑：metrics.box.ap50 只包含"验证集里有真值"的类别，顺序由 ap_class_index 给出，
    直接 enumerate(metrics.names) 会把后面的类名整体错位（本项目验证集没有 sign 真值，
    于是 bollard 的 AP 被写成了 sign）。
    """
    index = list(getattr(metrics.box, "ap_class_index", []))
    if not index:
        index = list(range(len(metrics.box.ap50)))
    values = {}
    for slot, class_id in enumerate(index):
        name = metrics.names[int(class_id)] if int(class_id) in metrics.names \
            else metrics.names[str(int(class_id))]
        values[str(name)] = float(metrics.box.ap50[slot])
    return values


def main() -> None:
    from ultralytics import YOLO

    rows = {}
    for label, (weights, data) in MODELS.items():
        if not weights.exists():
            print(f"跳过 {label}（找不到 {weights}）")
            continue
        model = YOLO(str(weights))
        metrics = model.val(data=str(data), split="val", imgsz=960, device=0, verbose=False,
                            project=str(ROOT / "outputs" / "eval"), name=label.split("_")[0],
                            exist_ok=True)
        values = per_class(metrics)
        shared = [values[c] for c in SHARED if c in values]
        rows[label] = {"per_class": values, "mAP50_shared8": float(np.mean(shared))}
        print(f"{label}: 8 类均值 {rows[label]['mAP50_shared8']:.4f}"
              f"（官方 mAP50 = {metrics.box.map50:.4f}，含该类表全部类别）")

    print("\n=== 街面 116 帧验证集（8 类共有口径，唯一可横比）===")
    print(f"{'版本':<18}" + "".join(f"{c:>11}" for c in SHARED) + f"{'8类均值':>10}")
    for label, row in rows.items():
        line = f"{label:<18}"
        for c in SHARED:
            value = row["per_class"].get(c)
            line += f"{value:>11.4f}" if value is not None else f"{'-':>11}"
        line += f"{row['mAP50_shared8']:>10.4f}"
        print(line)

    v5 = MODELS.get("v5_加三轮973")
    if v5 and v5[0].exists():
        model = YOLO(str(v5[0]))
        tri = model.val(data=str(TRICYCLE_VAL), split="val", imgsz=960, device=0, verbose=False,
                        project=str(ROOT / "outputs" / "eval"), name="v5_tricycle", exist_ok=True)
        values = per_class(tri)
        print("\n=== 三轮车专项（VisDrone 399 张验证帧，俯拍域外）===")
        print(f"  tricycle AP50 = {values.get('tricycle', float('nan')):.4f}"
              f"   mAP50(全部 9 类) = {tri.box.map50:.4f}")
        print("  说明：该验证集是俯拍机位，只用于判断三轮车这一类是否学到，"
              "不代表街面表现。")
    else:
        print("\n（v5 权重还没生成，训练完成后再跑本脚本即可）")


if __name__ == "__main__":
    main()
