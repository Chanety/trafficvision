"""误差归因：把预判误差按类别、轨迹长度、目标尺寸分组统计，找出主要失误来源。

用法：python bench/error_breakdown.py <批处理输出目录>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def analyze(payload: dict, horizon: float = 1.0) -> list[dict]:
    frames = payload["frames"]
    info = payload["video_info"]
    fps = info["fps"] or 30.0
    times = [i / fps for i in range(len(frames))]
    by_frame = [{o["id"]: o for o in f} for f in frames]
    height, width = info["height"], info["width"]
    area = float(height * width)

    # 轨迹长度（用于分组）
    track_len: dict[int, int] = {}
    for f in frames:
        for o in f:
            if o["id"] > 0:
                track_len[o["id"]] = track_len.get(o["id"], 0) + 1

    samples = []
    t_arr = np.array(times, dtype=float)
    for t, frame in enumerate(frames):
        target = times[t] + horizon
        i1 = int(np.searchsorted(t_arr, target))
        if i1 >= len(frames):
            continue
        i0 = max(0, i1 - 1)
        span = times[i1] - times[i0]
        frac = (target - times[i0]) / span if span > 1e-9 else 0.0
        for obj in frame:
            if not obj.get("predictable"):
                continue
            pred = obj["preds"].get(str(horizon))
            if pred is None:
                continue
            tid = obj["id"]
            a = by_frame[i0].get(tid)
            b = by_frame[i1].get(tid)
            if a is None or b is None:
                continue
            truth = (a["centroid"][0] + (b["centroid"][0] - a["centroid"][0]) * frac,
                     a["centroid"][1] + (b["centroid"][1] - a["centroid"][1]) * frac)
            pc = ((pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0)
            error = float(np.hypot(pc[0] - truth[0], pc[1] - truth[1]))
            # 该时间梯度内的真实位移，用于换算相对误差
            disp = float(np.hypot(truth[0] - obj["centroid"][0], truth[1] - obj["centroid"][1]))
            x1, y1, x2, y2 = obj["bbox"]
            ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / area
            samples.append({
                "cls": obj["cls"],
                "error": error,
                "disp": disp,
                "rel": error / disp if disp > 5.0 else None,
                "len": track_len.get(tid, 0),
                "size": "small" if ratio < 0.002 else ("medium" if ratio < 0.02 else "large"),
                "speed": obj.get("speed_px_s", 0.0),
                "video": Path(payload["video"]).name,
            })
    return samples


def group(samples: list[dict], key: str) -> None:
    buckets: dict[str, list[float]] = {}
    rels: dict[str, list[float]] = {}
    for s in samples:
        buckets.setdefault(str(s[key]), []).append(s["error"])
        if s.get("rel") is not None:
            rels.setdefault(str(s[key]), []).append(s["rel"])
    print(f"\n--- 按 {key} 分组 ---")
    print(f"{'分组':<14}{'样本数':>8}{'平均误差':>10}{'中位':>9}{'P90':>9}{'相对误差中位':>13}")
    for name, errs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        arr = np.array(errs)
        rel_median = np.median(rels[name]) * 100 if rels.get(name) else float("nan")
        print(f"{name:<14}{arr.size:>8}{arr.mean():>10.1f}{np.median(arr):>9.1f}"
              f"{np.percentile(arr, 90):>9.1f}{rel_median:>12.1f}%")


def length_bucket(n: int) -> str:
    if n < 10:
        return "轨迹<10帧"
    if n < 30:
        return "轨迹10-30帧"
    return "轨迹>30帧"


def main() -> None:
    out_dir = Path(sys.argv[1])
    exclude = sys.argv[2:] if len(sys.argv) > 2 else []
    all_samples: list[dict] = []
    for path in sorted(out_dir.glob("VID_*.json")):
        if any(token in path.name for token in exclude):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        all_samples.extend(analyze(payload))
    if exclude:
        print(f"已排除：{', '.join(exclude)}")

    if not all_samples:
        print("没有可用样本")
        return
    for s in all_samples:
        s["length_bucket"] = length_bucket(s["len"])
        s["speed_bucket"] = "慢速<50像素/秒" if s["speed"] < 50 else (
            "中速50-150" if s["speed"] < 150 else "快速>150")

    errors = np.array([s["error"] for s in all_samples])
    print(f"总样本 {errors.size}，平均误差 {errors.mean():.1f} px，"
          f"中位 {np.median(errors):.1f} px，P90 {np.percentile(errors, 90):.1f} px")
    group(all_samples, "cls")
    group(all_samples, "length_bucket")
    group(all_samples, "size")
    group(all_samples, "speed_bucket")


if __name__ == "__main__":
    main()
