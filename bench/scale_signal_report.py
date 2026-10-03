"""诊断工具：统计轨迹里可用于"维度判定"的干净样本比例与尺度变化幅度。

用法：python bench/scale_signal_report.py outputs/cv_v2.json
"""

from __future__ import annotations

import collections
import json
import sys

import numpy as np


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/cv_v2.json"
    data = json.load(open(path, encoding="utf-8"))
    width = data["video_info"]["width"]
    height = data["video_info"]["height"]

    tracks: dict[int, list[tuple[float, bool]]] = collections.defaultdict(list)
    for frame in data["frames"]:
        for obj in frame:
            if obj["id"] <= 0:
                continue
            x1, y1, x2, y2 = obj["bbox"]
            clean = not (x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1)
            tracks[obj["id"]].append((y2 - y1, clean))

    print(f"画面 {width}x{height}")
    header = f"{'id':>3} {'总样本':>7} {'干净样本':>9} {'干净占比':>9} {'干净样本框高变化':>17}"
    print(header)
    for tid, samples in sorted(tracks.items(), key=lambda kv: -len(kv[1])):
        if len(samples) < 5:
            continue
        clean = [h for h, ok in samples if ok]
        if len(clean) >= 3:
            variation = (max(clean) - min(clean)) / float(np.mean(clean)) * 100
            variation_text = f"{variation:.1f}%"
        else:
            variation_text = "样本不足"
        print(f"{tid:>3} {len(samples):>7} {len(clean):>9} "
              f"{len(clean) / len(samples) * 100:>8.0f}% {variation_text:>17}")


if __name__ == "__main__":
    main()
