"""对比两套跟踪器的真实产出：不只比"比例"，而是比"绝对可预判目标数"。

覆盖率（可预判比例）提升可能只是分母变小造成的假象，
必须同时看"每帧能真正给出预判的目标数量"。

用法：python bench/tracker_compare.py <bytetrack汇总.json> <tracktrack汇总.json> [排除关键词]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load(path: str) -> dict:
    return {r["video"]: r for r in json.loads(Path(path).read_text(encoding="utf-8"))}


def main() -> None:
    base = load(sys.argv[1])
    other = load(sys.argv[2])
    exclude = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"{'视频':<14}{'目标/帧(A)':>11}{'目标/帧(B)':>11}"
          f"{'可预判/帧(A)':>13}{'可预判/帧(B)':>13}"
          f"{'轨迹(A)':>9}{'轨迹(B)':>9}{'误差(A)':>9}{'误差(B)':>9}")

    rows = []
    for key in sorted(base):
        if key not in other or (exclude and exclude in key):
            continue
        a, b = base[key], other[key]
        a_pred = a["objects_per_frame"] * a["predictable_ratio"]
        b_pred = b["objects_per_frame"] * b["predictable_ratio"]
        rows.append((a, b, a_pred, b_pred))
        print(f"{key.replace('VID_20261003_', ''):<14}"
              f"{a['objects_per_frame']:>11}{b['objects_per_frame']:>11}"
              f"{a_pred:>13.2f}{b_pred:>13.2f}"
              f"{a['tracks']:>9}{b['tracks']:>9}"
              f"{str(a['err_1s_mean']):>9}{str(b['err_1s_mean']):>9}")

    if not rows:
        print("没有可对比的数据")
        return

    mean = lambda idx: float(np.mean([r[idx] for r in rows if r[idx] is not None]))  # noqa: E731
    print()
    print("排除 %s 后，共 %d 段：" % (exclude or "无", len(rows)))
    print("  平均目标数/帧     : A %.2f   B %.2f   (%+.1f%%)"
          % (mean(0) if False else float(np.mean([r[0]['objects_per_frame'] for r in rows])),
             float(np.mean([r[1]['objects_per_frame'] for r in rows])),
             (float(np.mean([r[1]['objects_per_frame'] for r in rows]))
              / float(np.mean([r[0]['objects_per_frame'] for r in rows])) - 1) * 100))
    print("  平均可预判目标/帧 : A %.2f   B %.2f   (%+.1f%%)"
          % (float(np.mean([r[2] for r in rows])), float(np.mean([r[3] for r in rows])),
             (float(np.mean([r[3] for r in rows])) / float(np.mean([r[2] for r in rows])) - 1) * 100))
    print("  平均轨迹条数      : A %.1f   B %.1f   (%+.1f%%)"
          % (float(np.mean([r[0]['tracks'] for r in rows])),
             float(np.mean([r[1]['tracks'] for r in rows])),
             (float(np.mean([r[1]['tracks'] for r in rows]))
              / float(np.mean([r[0]['tracks'] for r in rows])) - 1) * 100))


if __name__ == "__main__":
    main()
