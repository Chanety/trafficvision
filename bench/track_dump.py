"""把指定轨迹的原始位置序列打印出来，用于判断"拐弯"是真机动还是 ID 跳变。

用法：python bench/track_dump.py <某个视频的json> <轨迹ID> [打印间隔]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    path = Path(sys.argv[1])
    want_ids = {int(x) for x in sys.argv[2].split(",")}
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    payload = json.loads(path.read_text(encoding="utf-8"))
    fps = payload["video_info"]["fps"] or 30.0
    seq: dict[int, list[tuple[int, float, float, float, float]]] = {}
    for i, frame in enumerate(payload["frames"]):
        for obj in frame:
            if obj["id"] in want_ids:
                x1, y1, x2, y2 = obj["bbox"]
                seq.setdefault(obj["id"], []).append(
                    (i, obj["centroid"][0], obj["centroid"][1], x2 - x1, y2 - y1)
                )

    for tid, rows in seq.items():
        print(f"=== 轨迹 {tid}（{path.name}），共 {len(rows)} 帧，帧率 {fps:.1f} ===")
        print(f"{'帧':>6}{'x':>9}{'y':>9}{'框宽':>8}{'框高':>8}{'Δx':>8}{'Δy':>8}{'步长':>8}")
        prev = None
        for k, (i, x, y, w, h) in enumerate(rows):
            if k % step and k != len(rows) - 1:
                continue
            if prev is None:
                dx = dy = 0.0
                step_len = 0.0
            else:
                dx, dy = x - prev[0], y - prev[1]
                step_len = float(np.hypot(dx, dy))
            print(f"{i:>6}{x:>9.1f}{y:>9.1f}{w:>8.1f}{h:>8.1f}{dx:>8.1f}{dy:>8.1f}{step_len:>8.1f}")
            prev = (x, y)


if __name__ == "__main__":
    main()
