"""差异分析：逐框对比"纯自动标注"与"人工复核后标注"。

产出四类差异，用于定位自动标注的系统性问题：
  · 被删框（人工判定为误检）
  · 新增框（自动标注漏检）
  · 改类框（类别混淆，输出混淆矩阵）
  · 匹配框的位置偏差（IoU 与尺寸比）

用法：python bench/diff_analysis.py <人工标注目录> <自动对照目录> [输出json]
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

CLASS_NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]


def load(path: Path) -> list[list[float]]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append([float(v) for v in parts])
    return boxes


def xyxy(box):
    _, cx, cy, w, h = box
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = xyxy(a)
    bx1, by1, bx2, by2 = xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(1e-9, area_a + area_b - inter)


def match(auto: list, human: list, threshold: float = 0.5):
    """贪心匹配：按 IoU 从高到低配对。"""
    pairs = []
    for i, a in enumerate(auto):
        for j, h in enumerate(human):
            value = iou(a, h)
            if value >= threshold:
                pairs.append((value, i, j))
    pairs.sort(reverse=True)
    used_a, used_h, matched = set(), set(), []
    for value, i, j in pairs:
        if i in used_a or j in used_h:
            continue
        used_a.add(i); used_h.add(j)
        matched.append((value, i, j))
    deleted = [a for i, a in enumerate(auto) if i not in used_a]
    added = [h for j, h in enumerate(human) if j not in used_h]
    return matched, deleted, added


def main() -> None:
    human_dir = Path(sys.argv[1]) / "labels"
    auto_dir = Path(sys.argv[2]) / "labels"
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    stats = {
        "auto_total": 0, "human_total": 0,
        "kept": 0, "deleted": 0, "added": 0, "class_changed": 0,
        "iou": [], "size_ratio": [],
    }
    per_class = {name: collections.Counter() for name in CLASS_NAMES}
    confusion = collections.Counter()          # (自动类别 -> 人工类别)
    deleted_by_class = collections.Counter()
    added_by_class = collections.Counter()
    per_image = []

    files = sorted(p for p in human_dir.glob("*.txt") if p.name != "classes.txt")
    for path in files:
        auto_path = auto_dir / path.name
        if not auto_path.exists():
            continue
        auto = load(auto_path)
        human = load(path)
        stats["auto_total"] += len(auto)
        stats["human_total"] += len(human)

        matched, deleted, added = match(auto, human)
        stats["kept"] += len(matched)
        stats["deleted"] += len(deleted)
        stats["added"] += len(added)
        per_image.append((path.stem, len(auto), len(human), len(deleted), len(added)))

        for value, i, j in matched:
            a, h = auto[i], human[j]
            ca, ch = int(a[0]), int(h[0])
            per_class[CLASS_NAMES[ca]]["matched"] += 1
            stats["iou"].append(value)
            stats["size_ratio"].append((h[3] * h[4]) / max(1e-9, a[3] * a[4]))
            if ca != ch:
                stats["class_changed"] += 1
                confusion[(CLASS_NAMES[ca], CLASS_NAMES[ch])] += 1
        for a in deleted:
            deleted_by_class[CLASS_NAMES[int(a[0])]] += 1
            per_class[CLASS_NAMES[int(a[0])]]["deleted"] += 1
        for h in added:
            added_by_class[CLASS_NAMES[int(h[0])]] += 1
            per_class[CLASS_NAMES[int(h[0])]]["added"] += 1

    def pct(x, base):
        return f"{x / base * 100:.1f}%" if base else "-"

    ious = sorted(stats["iou"])
    ratios = sorted(stats["size_ratio"])
    print(f"对比图片 {len(per_image)} 张")
    print(f"自动标注框 {stats['auto_total']} 个 → 人工复核后 {stats['human_total']} 个")
    print(f"  保留（IoU≥0.5 匹配） {stats['kept']} 个")
    print(f"  被删（人工判为误检）  {stats['deleted']} 个（占自动框 {pct(stats['deleted'], stats['auto_total'])}）")
    print(f"  新增（自动漏检）      {stats['added']} 个（占人工框 {pct(stats['added'], stats['human_total'])}）")
    print(f"  改了类别              {stats['class_changed']} 个")
    if ious:
        print(f"  匹配框 IoU：中位 {ious[len(ious)//2]:.3f}  P25 {ious[len(ious)//4]:.3f}  P75 {ious[3*len(ious)//4]:.3f}")
    if ratios:
        print(f"  人工框/自动框 面积比：中位 {ratios[len(ratios)//2]:.2f}")

    print(f"\n{'类别':<12}{'自动':>7}{'保留':>7}{'被删':>7}{'删除率':>9}{'新增':>7}{'改类':>7}")
    for name in CLASS_NAMES:
        row = per_class[name]
        auto_n = row["matched"] + row["deleted"]
        changed = sum(v for (a, b), v in confusion.items() if a == name)
        print(f"{name:<12}{auto_n:>7}{row['matched']:>7}{row['deleted']:>7}"
              f"{pct(row['deleted'], auto_n):>9}{row['added']:>7}{changed:>7}")

    print("\n--- 类别改动最多的组合（自动 → 人工）---")
    for (a, b), v in confusion.most_common(12):
        print(f"  {a:<11} → {b:<11} {v}")

    worst = sorted(per_image, key=lambda r: -(r[3] + r[4]))[:10]
    print("\n--- 改动最多的 10 张图 ---")
    print(f"{'图片':<40}{'自动':>6}{'人工':>6}{'删':>5}{'增':>5}")
    for stem, na, nh, nd, nadd in worst:
        print(f"{stem:<40}{na:>6}{nh:>6}{nd:>5}{nadd:>5}")

    if out_path:
        payload = {
            "summary": {k: v for k, v in stats.items() if k not in ("iou", "size_ratio")},
            "iou_median": ious[len(ious) // 2] if ious else None,
            "size_ratio_median": ratios[len(ratios) // 2] if ratios else None,
            "per_class": {k: dict(v) for k, v in per_class.items()},
            "confusion": {f"{a}->{b}": v for (a, b), v in confusion.items()},
            "deleted_by_class": dict(deleted_by_class),
            "added_by_class": dict(added_by_class),
            "per_image": per_image,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告 -> {out_path}")


if __name__ == "__main__":
    main()
