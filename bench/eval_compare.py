"""把多个版本的权重放在同一个验证集上重新评测，保证对比公平。

验证集变化后，旧版本的指标不能直接拿来比较，必须在同一验证集上重跑。

用法：python bench/eval_compare.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "train_v4" / "data.yaml"
MODELS = {
    "v1_仅人工复核219": ROOT / "outputs" / "train" / "v1" / "weights" / "best.pt",
    "v3_加干净补充370": ROOT / "outputs" / "train" / "v3" / "weights" / "best.pt",
    "v4_扩充后473": ROOT / "outputs" / "train" / "v4" / "weights" / "best.pt",
}


def main() -> None:
    from ultralytics import YOLO

    results = {}
    for name, weights in MODELS.items():
        if not weights.exists():
            print(f"跳过 {name}（找不到权重）")
            continue
        model = YOLO(str(weights))
        metrics = model.val(data=str(DATA), split="val", imgsz=960, device=0,
                            verbose=False, project=str(ROOT / "outputs" / "eval"),
                            name=name.split("_")[0], exist_ok=True)
        per_class = {}
        for i, cname in enumerate(metrics.names.values()):
            if i < len(metrics.box.ap50):
                per_class[cname] = float(metrics.box.ap50[i])
        results[name] = {"mAP50": float(metrics.box.map50),
                         "mAP50-95": float(metrics.box.map),
                         "per_class": per_class}
        print(f"{name}: mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")

    classes = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "sign", "bollard"]
    header = f"{'版本':<20}" + "".join(f"{c:>11}" for c in classes) + f"{'mAP50':>9}"
    print("\n" + header)
    for name, r in results.items():
        row = f"{name:<20}"
        for c in classes:
            value = r["per_class"].get(c)
            row += f"{value:>11.4f}" if value is not None else f"{'-':>11}"
        row += f"{r['mAP50']:>9.4f}"
        print(row)


if __name__ == "__main__":
    main()
