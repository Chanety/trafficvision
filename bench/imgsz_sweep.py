"""推理分辨率扫描：小目标（行人、自行车、远处车辆）能不能靠提高 imgsz 救回来。

背景：检测端最弱的是小目标（自行车 AP50 只有 0.06 量级）。
开源里针对小目标的成熟做法是提高输入分辨率（或 SAHI 切图推理），
而提高分辨率是零依赖、只花算力的办法，先验证它值不值。

用法：python bench/imgsz_sweep.py [权重] [尺寸列表逗号分隔]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "新素材验证(65帧)": ROOT / "datasets" / "train_v7" / "data.yaml",
    "旧街面验证(116帧)": ROOT / "datasets" / "train_v7" / "street_val.yaml",
}


def main() -> None:
    from ultralytics import YOLO

    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "outputs" / "train" / "v7" / "weights" / "best.pt"
    sizes = [int(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2
                              else ["960", "1280", "1600"])]

    model = YOLO(str(weights))
    print(f"权重 {weights.name}，尺寸 {sizes}")
    for name, data in DATASETS.items():
        print(f"\n=== {name} ===")
        print(f"{'imgsz':>7}{'mAP50':>9}{'mAP50-95':>10}{'精确率':>9}{'召回率':>9}"
              f"{'行人AP':>9}{'非机动车AP':>11}")
        for size in sizes:
            m = model.val(data=str(data), split="val", imgsz=size, device=0, verbose=False,
                          project=str(ROOT / "outputs" / "eval"),
                          name=f"imgsz{size}_{'new' if '新' in name else 'street'}",
                          exist_ok=True)
            index = list(getattr(m.box, "ap_class_index", [])) or list(range(len(m.box.ap50)))
            per = {str(m.names[int(c)]): float(m.box.ap50[k]) for k, c in enumerate(index)}
            print(f"{size:>7}{m.box.map50:>9.4f}{m.box.map:>10.4f}"
                  f"{m.box.mp:>9.3f}{m.box.mr:>9.3f}"
                  f"{per.get('person', float('nan')):>9.3f}"
                  f"{per.get('nonmotor', float('nan')):>11.3f}")


if __name__ == "__main__":
    main()
