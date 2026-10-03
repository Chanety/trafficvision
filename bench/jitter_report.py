"""抖动分解：区分"检测框抖动"与"真实非匀速运动"对预判误差的贡献。

做法：对每条轨迹的中心点序列，在滑动窗口内做二次拟合，
残差 RMS 代表高频抖动（检测/关联噪声），
与每帧真实位移量对比，即可判断速度估计的噪声水平。

用法：python bench/jitter_report.py <批处理输出目录> [排除关键词] [窗口帧数]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def track_series(payload: dict) -> dict[int, list[tuple[int, float, float]]]:
    series: dict[int, list[tuple[int, float, float]]] = {}
    for i, frame in enumerate(payload["frames"]):
        for obj in frame:
            if obj["id"] > 0:
                series.setdefault(obj["id"], []).append(
                    (i, obj["centroid"][0], obj["centroid"][1])
                )
    return series


def main() -> None:
    out_dir = Path(sys.argv[1])
    exclude = sys.argv[2] if len(sys.argv) > 2 else None
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 15

    all_jitter, all_step = [], []
    per_video = []
    for path in sorted(out_dir.glob("VID_*.json")):
        if exclude and exclude in path.name:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        series = track_series(payload)
        jitters, steps = [], []
        for pts in series.values():
            if len(pts) < window:
                continue
            arr = np.array([[p[1], p[2]] for p in pts], dtype=float)
            steps.extend(np.linalg.norm(np.diff(arr, axis=0), axis=1).tolist())
            half = window // 2
            for c in range(half, len(arr) - half):
                seg = arr[c - half:c + half + 1]
                t = np.arange(len(seg), dtype=float)
                for dim in (0, 1):
                    coef = np.polyfit(t, seg[:, dim], 2)
                    resid = seg[:, dim] - np.polyval(coef, t)
                    jitters.append(float(np.sqrt(np.mean(resid ** 2))))
        if jitters:
            per_video.append((path.name, float(np.mean(jitters)), float(np.mean(steps))))
            all_jitter.extend(jitters)
            all_step.extend(steps)

    if not all_jitter:
        print("数据不足")
        return

    jitter = float(np.mean(all_jitter))
    step = float(np.mean(all_step))
    print(f"样本视频 {len(per_video)} 段（排除 {exclude or '无'}），窗口 {window} 帧")
    print(f"轨迹单轴抖动 RMS : {jitter:.2f} px")
    print(f"每帧平均位移     : {step:.2f} px")
    print(f"抖动/位移 比值   : {jitter / step * 100:.1f}%")
    print("\n--- 逐视频 ---")
    print(f"{'视频':<26}{'抖动RMS':>10}{'每帧位移':>10}{'抖动占比':>10}")
    for name, j, s in per_video:
        print(f"{name.replace('VID_20261003_', ''):<26}{j:>10.2f}{s:>10.2f}"
              f"{(j / s * 100 if s else 0):>9.1f}%")


if __name__ == "__main__":
    main()
