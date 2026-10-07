"""模型横向对比：把多个权重在同样的三个验证集上重跑，输出可比表格。

三个验证集：
  · 新素材 65 帧（videoset_02 留出的 3 段）—— 目标域
  · 旧街面 116 帧（与 v1~v7 同口径）
  · 俯拍 399 帧（VisDrone val，7 类口径）

用法：python bench/eval_compare_models.py [imgsz] [权重1=名称1] [权重2=名称2] ...
默认对比 v7(yolo11n 链式微调) / v8n(yolo11n 从零) / v8s(yolo11s 从零)。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "outputs" / "train"
SETS = {
    "新素材": ROOT / "datasets" / "train_v7" / "data.yaml",
    "旧街面": ROOT / "datasets" / "train_v7" / "street_val.yaml",
    "俯拍": ROOT / "datasets" / "train_v7" / "aerial_val.yaml",
}
DEFAULT = {
    "v7_yolo11n链式": TRAIN / "v7" / "weights" / "best.pt",
    "v8n_yolo11n从零": TRAIN / "v8n" / "weights" / "best.pt",
    "v8s_yolo11s从零": TRAIN / "v8s" / "weights" / "best.pt",
}


def main() -> None:
    from ultralytics import YOLO

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    imgsz = 1280
    models = dict(DEFAULT)
    if args:
        try:
            imgsz = int(args[0])
            args = args[1:]
        except ValueError:
            pass
    if args:
        models = {}
        for item in args:
            path, _, label = item.partition("=")
            models[label or Path(path).stem] = Path(path)

    print(f"推理分辨率 {imgsz}")
    for set_name, data in SETS.items():
        print(f"\n=== {set_name} ===")
        print(f"{'模型':<20}{'mAP50':>9}{'mAP50-95':>10}{'精确率':>9}{'召回率':>9}"
              f"{'行人':>8}{'非机动车':>10}{'汽车':>8}")
        for label, weights in models.items():
            if not weights.exists():
                print(f"{label:<20}（找不到权重，跳过）")
                continue
            m = YOLO(str(weights)).val(data=str(data), split="val", imgsz=imgsz, device=0,
                                       verbose=False, project=str(ROOT / "outputs" / "eval"),
                                       name=f"{label.split('_')[0]}_{set_name}", exist_ok=True)
            index = list(getattr(m.box, "ap_class_index", [])) or list(range(len(m.box.ap50)))
            per = {str(m.names[int(c)]): float(m.box.ap50[k]) for k, c in enumerate(index)}
            print(f"{label:<20}{m.box.map50:>9.4f}{m.box.map:>10.4f}"
                  f"{m.box.mp:>9.3f}{m.box.mr:>9.3f}"
                  f"{per.get('person', float('nan')):>8.3f}"
                  f"{per.get('nonmotor', float('nan')):>10.3f}"
                  f"{per.get('car', float('nan')):>8.3f}")


if __name__ == "__main__":
    main()
