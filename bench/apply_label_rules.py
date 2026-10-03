"""按"先载具、后行人"的规则清理标注：删掉与载具重叠的"行人"框。

规则依据（用户提出的标注规范）：
  骑车人、驾驶位里的人等，其"人"的身份已由载具表达，
  再单独标一个 person 框会与载具框重叠，给模型自相矛盾的监督信号。

判定条件（同时满足才删）：
  1. 该 person 框中心落在某个载具框内
  2. 该 person 框被载具框覆盖的面积比例 ≥ --ratio（默认 0.5）
  3. 该载具框面积不小于 person 框的 --min-scale 倍（避免小车吞掉大行人）

用法：
    python bench/apply_label_rules.py <标注集目录>            # 只统计，不改文件
    python bench/apply_label_rules.py <标注集目录> --apply    # 实际修改（自动备份）
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

CLASS_NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
VEHICLE_IDS = {0, 1, 2, 3, 4}
PERSON_ID = 5


def load_boxes(path: Path) -> list[list[float]]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append([float(v) for v in parts])
    return boxes


def to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    _, cx, cy, w, h = box
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def overlap_ratio(person: list[float], vehicle: list[float]) -> tuple[float, bool]:
    """返回（person 被 vehicle 覆盖的面积比例，person 中心是否在 vehicle 内）。"""
    px1, py1, px2, py2 = to_xyxy(person)
    vx1, vy1, vx2, vy2 = to_xyxy(vehicle)
    ix1, iy1 = max(px1, vx1), max(py1, vy1)
    ix2, iy2 = min(px2, vx2), min(py2, vy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0, False
    inter = (ix2 - ix1) * (iy2 - iy1)
    person_area = max(1e-9, (px2 - px1) * (py2 - py1))
    cx, cy = person[1], person[2]
    inside = vx1 <= cx <= vx2 and vy1 <= cy <= vy2
    return inter / person_area, inside


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--apply", action="store_true", help="实际修改标注文件（会先备份）")
    ap.add_argument("--ratio", type=float, default=0.5, help="被载具覆盖比例阈值")
    ap.add_argument("--min-scale", type=float, default=0.8, help="载具框面积至少为行人框的多少倍")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    labels_dir = dataset / "labels"
    files = sorted(labels_dir.glob("*.txt"))

    removed_total = 0
    kept_person = 0
    affected_files = 0
    ratio_hist: list[float] = []

    for path in files:
        boxes = load_boxes(path)
        if not boxes:
            continue
        keep: list[list[float]] = []
        removed_here = 0
        for box in boxes:
            if int(box[0]) != PERSON_ID:
                keep.append(box)
                continue
            best = 0.0
            drop = False
            for other in boxes:
                if int(other[0]) not in VEHICLE_IDS:
                    continue
                ratio, inside = overlap_ratio(box, other)
                if ratio <= 0:
                    continue
                p_area = box[3] * box[4]
                v_area = other[3] * other[4]
                if inside and ratio >= args.ratio and v_area >= p_area * args.min_scale:
                    drop = True
                    best = max(best, ratio)
            if drop:
                removed_here += 1
                ratio_hist.append(best)
            else:
                keep.append(box)
                kept_person += 1

        if removed_here:
            affected_files += 1
            removed_total += removed_here
            if args.apply:
                backup = dataset / "labels_backup_before_rules"
                backup.mkdir(exist_ok=True)
                if not (backup / path.name).exists():
                    shutil.copyfile(path, backup / path.name)
                path.write_text(
                    "\n".join(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
                              for b in keep) + ("\n" if keep else ""),
                    encoding="utf-8",
                )

    print(f"扫描标注文件 {len(files)} 个")
    print(f"删除与载具重叠的行人框：{removed_total} 个（涉及 {affected_files} 张图）")
    print(f"保留的行人框：{kept_person} 个")
    if ratio_hist:
        arr = sorted(ratio_hist)
        print(f"被删框的覆盖比例：最小 {arr[0]:.2f}  中位 {arr[len(arr)//2]:.2f}  最大 {arr[-1]:.2f}")
    print("模式：" + ("已实际修改（原文件备份在 labels_backup_before_rules/）" if args.apply else "仅统计，未修改文件"))


if __name__ == "__main__":
    main()
