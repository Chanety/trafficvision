"""按类别确定"过小目标"门槛：用时序一致性做免标注的代价最小化。

背景（用户 2026-10-08）：
  "过小目标不识别" 与 "行人 AP 提到 0.70" 不能同时成立——统一的最小尺寸门槛
  会把 13.9% 的行人检测一起丢掉。用户要求：
    · 行人和车辆用**独立**的标准；
    · 行人与非机动车应当**适当宽容**；
    · 最好有一种**粗方法**，以最小的综合开销决定该怎么筛。

方法（不需要人工标注）：
  真目标在相邻帧会稳定复现，噪声不会。于是用"**时序一致性**"当"是不是真目标"的代理：

    1. 低阈值跑检测（不做尺寸过滤）；
    2. 相邻帧之间按类别 + IoU≥0.3 贪心配对，配上的记为"稳定"；
    3. 按目标尺寸分桶，统计每桶的稳定率；
    4. 用代价模型扫门槛：
         cost(S) = w_miss[类] × 被丢掉的稳定框数 + w_keep × 保留下来的不稳定框数
       取 cost 最小的 S 作为该类的建议门槛。

  w_miss 是"漏掉真目标"的代价权重：行人/非机动车给高（更宽容，门槛更低），
  车辆给低（漏一个不致命），静态结构居中。

用法：
    python bench/min_size_sweep.py --videos 3 --frames 150
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 漏掉真目标的代价权重（越大越宽容，门槛越低）
MISS_WEIGHT = {
    "person": 4.0,
    "nonmotor": 3.0,
    "rider": 4.0,
    "bicycle": 3.0,
    "motorcycle": 3.0,
    "tricycle": 3.0,
    "bollard": 2.0,
    "sign": 2.0,
    "roadlamp": 1.5,
    "trafficlight": 1.5,
    "car": 1.0,
    "truck": 1.0,
    "bus": 1.0,
}
DEFAULT_MISS = 1.5
KEEP_COST = 1.0          # 保留一个不稳定框的代价

# 候选门槛（按"短边"像素）
GRID = [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40]


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "datasets" / "videoset_02_origin"))
    ap.add_argument("--weights", default=str(ROOT / "outputs" / "train" / "v7" / "weights" / "best.pt"))
    ap.add_argument("--videos", type=int, default=3, help="取前 N 段视频")
    ap.add_argument("--frames", type=int, default=150, help="每段取多少帧")
    ap.add_argument("--conf", type=float, default=0.10, help="低阈值跑，尽量拿到候选")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--match-iou", type=float, default=0.30)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "min_size_sweep.json"))
    args = ap.parse_args()

    from ultralytics import YOLO

    video_dir = Path(args.dir)
    vids = sorted(video_dir.glob("*.mp4"))[: args.videos]
    if not vids:
        raise SystemExit(f"没找到视频：{video_dir}")

    model = YOLO(args.weights)
    names = {i: n for i, n in enumerate(model.names)}

    # 每个类 → 每档尺寸的 [总数, 稳定数]
    hist: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    matched_total = 0
    det_total = 0

    for v in vids:
        cap = cv2.VideoCapture(str(v))
        frames = []
        while len(frames) < args.frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if len(frames) < 2:
            continue
        print(f"  {v.name}: {len(frames)} 帧")

        res = model.predict(frames, imgsz=args.imgsz, conf=args.conf, iou=0.5,
                            device=0, verbose=False, batch=8)
        per = []
        for r in res:
            boxes = []
            for b in r.boxes:
                nm = r.names[int(b.cls)]
                xyxy = [float(x) for x in b.xyxy[0]]
                boxes.append((nm, xyxy))
            per.append(boxes)

        used = set()
        for t in range(len(per) - 1):
            for nm, box in per[t]:
                det_total += 1
                side = min(box[2] - box[0], box[3] - box[1])
                bucket = next((g for g in GRID if side < g), GRID[-1])
                # 与下一帧同类框配对
                best, best_i = 0.0, None
                for j, (nm2, box2) in enumerate(per[t + 1]):
                    if nm2 != nm or (t + 1, j) in used:
                        continue
                    v2 = iou(box, box2)
                    if v2 > best:
                        best, best_i = v2, j
                stable = best >= args.match_iou
                if stable:
                    used.add((t + 1, best_i))
                    matched_total += 1
                hist[nm][bucket][0] += 1
                hist[nm][bucket][1] += int(stable)

    print(f"\n检测框 {det_total} 个，其中相邻帧可复现 {matched_total} 个"
          f"（{matched_total / max(1, det_total) * 100:.1f}%）\n")

    # ---- 逐类报告稳定率 + 代价最小化 ----
    report = {}
    for cls in sorted(hist, key=lambda c: -sum(v[0] for v in hist[c].values())):
        rows = hist[cls]
        total = sum(v[0] for v in rows.values())
        if total < 30:
            continue
        print(f"=== {cls}  （共 {total} 框） ===")
        print("  短边<  数量   稳定率")
        for g in GRID:
            n, st = rows.get(g, [0, 0])
            if n:
                print(f"  {g:>4}  {n:>5}  {st / n * 100:5.1f}%")

        w_miss = MISS_WEIGHT.get(cls, DEFAULT_MISS)
        # 每个阈值 S：被丢掉的稳定框 = 所有桶<S 的稳定数；保留的不稳定框 = 所有桶≥S 的不稳定数
        best = None
        curve = []
        for S in GRID:
            dropped_stable = sum(rows.get(g, [0, 0])[1] for g in GRID if g < S)
            kept_unstable = sum(rows.get(g, [0, 0])[0] - rows.get(g, [0, 0])[1]
                                for g in GRID if g >= S)
            cost = w_miss * dropped_stable + KEEP_COST * kept_unstable
            curve.append((S, round(cost, 1), dropped_stable, kept_unstable))
            if best is None or cost < best[1]:
                best = (S, cost, dropped_stable, kept_unstable)
        print(f"  → 代价最小门槛 S* = {best[0]}px"
              f"（漏掉稳定框 {best[2]} 个，保留不稳定框 {best[3]} 个，"
              f"cost={best[1]:.0f}，w_miss={w_miss}）")
        report[cls] = {
            "total": total,
            "buckets": {str(g): rows.get(g, [0, 0]) for g in GRID if rows.get(g)},
            "miss_weight": w_miss,
            "suggested_min_side": best[0],
            "curve": [{"min_side": c[0], "cost": c[1],
                       "dropped_stable": c[2], "kept_unstable": c[3]} for c in curve],
        }
        print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"videos": [v.name for v in vids], "frames_per_video": args.frames,
         "conf": args.conf, "grid": GRID, "match_iou": args.match_iou,
         "det_total": det_total, "stable_total": matched_total,
         "classes": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细：{args.out}")


if __name__ == "__main__":
    main()
