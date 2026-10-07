"""同一批目标的配对抖动对比：把"两个模型看到的目标集合不同"这个混淆因素去掉。

问题：jitter_moving_only.py 里 A（开放词汇）与 B（微调 v3）的运动轨迹来自
各自检测结果，B 多检出大量摩托，运动子集完全不同，比出来的抖动占比不可比。

做法：
  1. 逐帧把 A 与 B 的检测框按 IoU >= 阈值 做贪心匹配（跨模型配对同一个物理目标）；
  2. 统计配对共现次数，保留"互相是对方最佳配对"且共现 >= 最短观测帧数的配对；
  3. 对每一对配对，分别在两个模型里算该轨迹的抖动 RMS 与每帧位移，然后做配对比较
     （同一目标上 B 的抖动比 A 高还是低）。

只吃 predict_compare.py 已经生成的 JSON，不需要重新跑推理。

用法：python bench/jitter_matched.py <输出目录> [IoU阈值] [最短帧数]
例：  python bench/jitter_matched.py outputs/predict_compare
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

FILE_RE = re.compile(r"^(?P<label>.+)_(?P<video>VID_\d+_\d+)\.json$")


def load(path: Path) -> list[list[dict]]:
    return json.loads(path.read_text(encoding="utf-8"))["frames"]


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def track_of(frames: list[list[dict]], track_id: int) -> list[tuple[float, float]]:
    pts = []
    for frame in frames:
        for obj in frame:
            if obj["id"] == track_id:
                pts.append((float(obj["centroid"][0]), float(obj["centroid"][1])))
                break
    return pts


def jitter_step(points: list[tuple[float, float]], window: int = 9) -> tuple[float, float] | None:
    if len(points) < window:
        return None
    arr = np.array(points, dtype=float)
    steps = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    half = window // 2
    jitters = []
    for c in range(half, len(arr) - half):
        seg = arr[c - half:c + half + 1]
        t = np.arange(len(seg), dtype=float)
        for dim in (0, 1):
            resid = seg[:, dim] - np.polyval(np.polyfit(t, seg[:, dim], 2), t)
            jitters.append(float(np.sqrt(np.mean(resid**2))))
    return float(np.mean(jitters)), float(np.mean(steps))


def match_video(frames_a: list[list[dict]], frames_b: list[list[dict]], iou_thr: float):
    """返回 {(aid,bid): 共现帧数} 与 {(aid,bid): 类别一致帧数}。"""
    co: Counter = Counter()
    same_cls: Counter = Counter()
    for fa, fb in zip(frames_a, frames_b):
        if not fa or not fb:
            continue
        pairs = []
        for obj_a in fa:
            for obj_b in fb:
                score = iou(obj_a["bbox"], obj_b["bbox"])
                if score >= iou_thr:
                    pairs.append((score, obj_a, obj_b))
        pairs.sort(key=lambda p: -p[0])
        used_a, used_b = set(), set()
        for _, obj_a, obj_b in pairs:
            if obj_a["id"] in used_a or obj_b["id"] in used_b:
                continue
            used_a.add(obj_a["id"])
            used_b.add(obj_b["id"])
            key = (obj_a["id"], obj_b["id"])
            co[key] += 1
            if obj_a["cls"] == obj_b["cls"]:
                same_cls[key] += 1
    return co, same_cls


def main() -> None:
    out_dir = Path(sys.argv[1])
    iou_thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    min_obs = int(sys.argv[3]) if len(sys.argv) > 3 else 9

    files: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in sorted(out_dir.glob("*.json")):
        match = FILE_RE.match(path.name)
        if match:
            files[match.group("video")][match.group("label")] = path
    videos = {v: m for v, m in files.items() if len(m) == 2}
    if not videos:
        print(f"{out_dir} 下需要成对的 <模型A>_<视频>.json 与 <模型B>_<视频>.json")
        return

    labels = sorted(next(iter(videos.values())).keys())
    label_a, label_b = labels[0], labels[1]
    print(f"对比：{label_a}  vs  {label_b}    （IoU>={iou_thr}，配对共现>={min_obs} 帧）")

    rows = []
    class_agree = class_total = 0
    for video, mapping in sorted(videos.items()):
        frames_a = load(mapping[label_a])
        frames_b = load(mapping[label_b])
        co, same_cls = match_video(frames_a, frames_b, iou_thr)

        # 只保留互为最佳、且共现够长的配对，避免"一个 A 框轮配多个 B 框"
        best_b: dict[int, tuple[int, int]] = {}
        best_a: dict[int, tuple[int, int]] = {}
        for (aid, bid), n in co.items():
            if n < min_obs:
                continue
            if bid not in best_b or n > best_b[bid][1]:
                best_b[bid] = (aid, n)
            if aid not in best_a or n > best_a[aid][1]:
                best_a[aid] = (bid, n)
        kept = [((aid, bid), n) for (aid, bid), n in co.items()
                if n >= min_obs and best_b[bid][0] == aid and best_a[aid][0] == bid]

        for (aid, bid), n in kept:
            stats_a = jitter_step(track_of(frames_a, aid))
            stats_b = jitter_step(track_of(frames_b, bid))
            if stats_a is None or stats_b is None:
                continue
            class_total += 1
            class_agree += int(same_cls[(aid, bid)] >= n * 0.8)
            rows.append({"video": video, "aid": aid, "bid": bid, "obs": n,
                         "j_a": stats_a[0], "s_a": stats_a[1],
                         "j_b": stats_b[0], "s_b": stats_b[1]})

    if not rows:
        print("没有找到足够长的配对轨迹")
        return

    j_a = np.array([r["j_a"] for r in rows])
    s_a = np.array([r["s_a"] for r in rows])
    j_b = np.array([r["j_b"] for r in rows])
    s_b = np.array([r["s_b"] for r in rows])
    print(f"配对轨迹 {len(rows)} 条（横跨 {len(videos)} 段视频），"
          f"类别一致率 {class_agree / class_total:.3f}\n")
    print(f"{'模型':<18}{'抖动RMS(px)':>13}{'每帧位移(px)':>14}{'抖动占比':>10}")
    print(f"{label_a:<18}{j_a.mean():>13.2f}{s_a.mean():>14.2f}{j_a.mean() / s_a.mean() * 100:>9.1f}%")
    print(f"{label_b:<18}{j_b.mean():>13.2f}{s_b.mean():>14.2f}{j_b.mean() / s_b.mean() * 100:>9.1f}%")

    ratio = j_b / np.maximum(j_a, 1e-6)
    print(f"\n配对差值（{label_b} - {label_a}）：抖动 "
          f"{np.median(j_b - j_a):+.2f} px（中位），位移 {np.median(s_b - s_a):+.2f} px")
    print(f"抖动比 {label_b}/{label_a}：中位 {np.median(ratio):.2f}，"
          f"均值 {ratio.mean():.2f}；B 更抖的配对占 {(ratio > 1).mean() * 100:.0f}%")


if __name__ == "__main__":
    main()
