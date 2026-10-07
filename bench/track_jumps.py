"""量化轨迹跳变：跟踪器串轨（ID switch）的直接证据。

与预判误差无关，纯粹看轨迹本身：如果一条轨迹相邻两帧的中心点跳了很远，
而它平时的步长很小，那多半是跟到了另一个目标（串轨）或者检测框抖了。

判定：某帧步长 > max(40 px, 4 × 该轨迹步长中位数) 记一次跳变。

用法：python bench/track_jumps.py <目录1=名称1> <目录2=名称2> ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def analyze_dir(out_dir: Path) -> dict:
    tracks = 0
    lengths: list[int] = []
    jumps = 0
    jump_tracks = 0
    frames_total = 0
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames_total += len(payload["frames"])
        series: dict[int, list[tuple[int, float, float, float]]] = {}
        for i, frame in enumerate(payload["frames"]):
            for obj in frame:
                # 只统计真实观测：遮挡续推的框是推算出来的，会把"跳变"抹平，不能计入
                if obj["id"] > 0 and not obj.get("coasting"):
                    h = obj["bbox"][3] - obj["bbox"][1]
                    series.setdefault(obj["id"], []).append(
                        (i, obj["centroid"][0], obj["centroid"][1], h))
        for pts in series.values():
            if len(pts) < 4:
                continue
            tracks += 1
            lengths.append(len(pts))
            pts.sort()
            steps = [np.hypot(pts[k][1] - pts[k - 1][1], pts[k][2] - pts[k - 1][2])
                     for k in range(1, len(pts))]
            median = float(np.median(steps))
            threshold = max(40.0, 4.0 * median)
            n_jump = sum(1 for s in steps if s > threshold)
            if n_jump:
                jumps += n_jump
                jump_tracks += 1
    return {"tracks": tracks, "jumps": jumps, "jump_tracks": jump_tracks,
            "frames": frames_total,
            "median_len": float(np.median(lengths)) if lengths else 0.0}


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    print(f"{'配置':<24}{'轨迹数':>8}{'跳变次数':>10}{'涉事轨迹':>10}{'涉事占比':>10}"
          f"{'每千帧跳变':>12}{'轨迹中位长':>11}")
    for item in sys.argv[1:]:
        path, _, label = item.partition("=")
        stats = analyze_dir(Path(path))
        share = stats["jump_tracks"] / max(1, stats["tracks"]) * 100
        per_k = stats["jumps"] / max(1, stats["frames"]) * 1000
        print(f"{(label or Path(path).name):<24}{stats['tracks']:>8}{stats['jumps']:>10}"
              f"{stats['jump_tracks']:>10}{share:>9.1f}%{per_k:>12.2f}{stats['median_len']:>11.0f}")


if __name__ == "__main__":
    main()
