"""标注规则一致性审计：专门查"规则执行不规范"造成的两类问题。

背景（用户 2026-10-09 抽检反馈）：
  对自留前 40 帧逐张检查后，残留问题只有两类，且用户判断"很大程度是
  操作不规范的误判，不是什么问题"：
    ① **边缘路桩**    —— 贴着画面边缘、被裁掉一半的静态目标
    ② **行人与非机动车驾驶员** —— rider 与 person 的边界没执行一致

规则本身没问题，是**执行不一致**。本脚本把这两类在数据里找出来，
给出清单与计数，供人工快速过一遍或直接批量修正。

用法：
    python bench/audit_label_rules.py --data datasets/videoset_02_label3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trafficvision.config import MERGED_CLASSES, STATIC_CLASSES  # noqa: E402

EDGE_PX = 3.0        # 判定"贴边"的像素阈值


def read(path: Path, w: int, h: int):
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            box = [(cx - bw / 2) * w, (cy - bh / 2) * h,
                   (cx + bw / 2) * w, (cy + bh / 2) * h]
            out.append((cid, box))
    return out


def _h_overlap(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    narrow = min(a[2] - a[0], b[2] - b[0])
    return ix / narrow if narrow > 1e-9 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets" / "videoset_02_label3"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--fix", action="store_true",
                    help="顺便修掉可确定的违规：①a 框裁回画面内；②a person 改判 rider。"
                         "改动前会把原文件备份到 <data>/_before_rulefix/")
    args = ap.parse_args()

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    by_name = {Path(m["image"]).stem: m for m in manifest}

    out_of_bounds = []      # ①a 框超出画面（违反"只框可见部分"）
    edge_touch = []         # ①b 只是贴边，仍在画面内（正常）
    rider_suspect = []      # ② 该是 rider 却标成 person
    rider_orphan = []       # ②b 标成 rider 但附近没有任何两三轮载具
    person_total = rider_total = 0
    per_video = Counter()

    for img in sorted((data / "images").glob("*.jpg")):
        meta = by_name.get(img.stem)
        w = (meta or {}).get("w", 1920)
        h = (meta or {}).get("h", 1080)
        boxes = read(data / "labels" / f"{img.stem}.txt", w, h)

        for cid, b in boxes:
            name = MERGED_CLASSES[cid]
            # ① 静态类贴边
            if name in STATIC_CLASSES:
                oob = (b[0] < -0.5 or b[1] < -0.5 or b[2] > w + 0.5 or b[3] > h + 0.5)
                touch = (b[0] <= EDGE_PX or b[1] <= EDGE_PX
                         or b[2] >= w - EDGE_PX or b[3] >= h - EDGE_PX)
                if oob or touch:
                    item = {
                        "frame": img.stem, "cls": name,
                        "box": [round(v) for v in b],
                        "side": "左" if b[0] <= EDGE_PX else "上" if b[1] <= EDGE_PX
                                else "右" if b[2] >= w - EDGE_PX else "下",
                    }
                    (out_of_bounds if oob else edge_touch).append(item)
                    per_video[img.stem.rsplit("_f", 1)[0]] += 1
            if name == "person":
                person_total += 1
            if name == "rider":
                rider_total += 1

        # ② 人骑在车上却标成 person（用产品的骑乘判定规则复核）
        persons = [b for cid, b in boxes if MERGED_CLASSES[cid] == "person"]
        vehicles = [b for cid, b in boxes if MERGED_CLASSES[cid] in
                    ("nonmotor", "motorcycle", "bicycle", "tricycle")]
        for p in persons:
            for v in vehicles:
                if _h_overlap(p, v) < 0.40:
                    continue
                vh = v[3] - v[1]
                if not (v[1] - 0.20 * vh <= p[3] <= v[3] + 0.05 * vh):
                    continue
                if p[1] >= v[3]:
                    continue
                rider_suspect.append({
                    "frame": img.stem,
                    "person_box": [round(x) for x in p],
                    "vehicle_box": [round(x) for x in v],
                })
                break

        # ②b 标成 rider 却找不到任何两三轮载具（可能是把行人误标成 rider）
        riders = [b for cid, b in boxes if MERGED_CLASSES[cid] == "rider"]
        for r in riders:
            if not any(_h_overlap(r, v) >= 0.40 for v in vehicles):
                rider_orphan.append({"frame": img.stem, "rider_box": [round(x) for x in r]})

    report = {
        "data": str(data),
        "frames": len(list((data / "images").glob("*.jpg"))),
        "out_of_bounds": {"count": len(out_of_bounds), "items": out_of_bounds[:200]},
        "edge_touch": {"count": len(edge_touch), "items": edge_touch[:200]},
        "rider_suspect": {"count": len(rider_suspect), "items": rider_suspect[:200]},
        "rider_orphan": {"count": len(rider_orphan), "items": rider_orphan[:200]},
        "person_boxes": person_total,
        "rider_boxes": rider_total,
    }
    out = Path(args.out) if args.out else data / "rule_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.fix:
        backup = data / "_before_rulefix"
        backup.mkdir(exist_ok=True)
        touched = []
        for img in sorted((data / "images").glob("*.jpg")):
            meta = by_name.get(img.stem)
            w = (meta or {}).get("w", 1920)
            h = (meta or {}).get("h", 1080)
            lbl = data / "labels" / f"{img.stem}.txt"
            boxes = read(lbl, w, h)
            if not boxes:
                continue
            changed = False
            fixed = []
            for cid, b in boxes:
                nb = [max(0.0, min(float(w), b[0])), max(0.0, min(float(h), b[1])),
                      max(0.0, min(float(w), b[2])), max(0.0, min(float(h), b[3]))]
                if nb != b:
                    changed = True
                fixed.append([float(cid), nb])
            # 人骑在车上却标 person → 改判 rider。
            # 注意：labelrules.classify_riders 吃的是 YOLO 归一化格式，
            # 这里用的是像素框，所以直接复用同一套几何判据。
            rider_id = MERGED_CLASSES.index("rider")
            person_id = MERGED_CLASSES.index("person")
            vehicles = [b for c, b in fixed if MERGED_CLASSES[int(c)] in
                        ("nonmotor", "motorcycle", "bicycle", "tricycle")]
            n_changed = 0
            for item in fixed:
                if int(item[0]) != person_id or not vehicles:
                    continue
                p = item[1]
                for v in vehicles:
                    vh = v[3] - v[1]
                    if _h_overlap(p, v) < 0.40:
                        continue
                    if not (v[1] - 0.20 * vh <= p[3] <= v[3] + 0.05 * vh):
                        continue
                    if p[1] >= v[3]:
                        continue
                    item[0] = float(rider_id)
                    n_changed += 1
                    break
            if n_changed:
                changed = True
            if not changed:
                continue
            if not (backup / lbl.name).exists():
                (backup / lbl.name).write_text(lbl.read_text(encoding="utf-8"), encoding="utf-8")
            lines = []
            for cid, b in fixed:
                x1, y1, x2, y2 = b
                lines.append(f"{int(cid)} {((x1 + x2) / 2) / w:.6f} {((y1 + y2) / 2) / h:.6f} "
                             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
            touched.append(lbl.name)
        print(f"\n[已修正] {len(touched)} 个标注文件（原文件备份在 {backup.name}/）")
        for t in touched[:10]:
            print(f"   {t}")

    print(f"数据集：{data.name}   共 {report['frames']} 帧\n")
    print("① 静态目标贴边问题")
    print(f"   ①a 框超出画面（违反'出画只框可见部分'）：{len(out_of_bounds)} 个")
    print(f"   ①b 仅贴边但仍在画面内（正常）：{len(edge_touch)} 个")
    print(f"   合计涉及 {len(per_video)} 段视频")
    for k, v in per_video.most_common(8):
        print(f"      {k}  {v} 个")
    if out_of_bounds:
        print("   前 5 例：")
        for it in out_of_bounds[:5]:
            print(f"      {it['frame']}  {it['cls']}  贴{it['side']}边  box={it['box']}")

    print(f"\n② 该是 rider 却标成 person")
    print(f"   person 框共 {person_total} 个，rider 框共 {rider_total} 个")
    print(f"   ②a 疑似漏改（人骑在车上却标 person）：{len(rider_suspect)} 个")
    print(f"   ②b 标成 rider 但附近没有两三轮载具：{len(rider_orphan)} 个")
    if rider_suspect:
        print("   前 5 例：")
        for it in rider_suspect[:5]:
            print(f"      {it['frame']}  person={it['person_box']}  在载具 {it['vehicle_box']} 上")

    print(f"\n明细：{out}")
    print("\n说明：这两类都是**规则执行不一致**，不是模型能力问题。")
    print("      ② 可以用 classify_riders 批量修正；① 建议人工确认哪些真的该保留。")


if __name__ == "__main__":
    main()
