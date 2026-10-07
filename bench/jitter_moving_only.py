"""抖动重算（只统计运动目标）：修正"停放车辆/路桩"把抖动占比拉高的污染。

原 jitter_report.py 把静止目标的轨迹也算进去：静止目标每帧位移接近 0，
分母被拉小，抖动/位移比值虚高，无法反映"运动目标的轨迹是否干净"。

本脚本在批处理输出（run_clip 的 out_json）上按速度分档统计：
    · 静止   < 0.5 px/帧
    · 慢速   0.5 ~ 2 px/帧
    · 中速   2 ~ 8 px/帧
    · 快速   >= 8 px/帧
对每档给出轨迹单轴抖动 RMS、每帧位移、抖动占比，并单独给出"运动子集
（中速+快速）"的口径，便于跨模型比较。

用法：
    python bench/jitter_moving_only.py <输出目录> [最低速度 px/帧]
例：
    python bench/jitter_moving_only.py outputs/predict_compare
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

WINDOW = 15
FILE_RE = re.compile(r"^(?P<label>.+)_VID_\d+_\d+\.json$")


def track_series(frames: list[list[dict]]) -> dict[int, list[tuple[float, float]]]:
    series: dict[int, list[tuple[float, float]]] = {}
    for frame in frames:
        for obj in frame:
            if obj.get("id", 0) > 0:
                series.setdefault(obj["id"], []).append(
                    (float(obj["centroid"][0]), float(obj["centroid"][1]))
                )
    return series


def track_stats(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """返回 (单轴抖动RMS, 每帧平均位移)，轨迹太短返回 None。"""
    if len(points) < WINDOW:
        return None
    arr = np.array(points, dtype=float)
    steps = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    half = WINDOW // 2
    jitters = []
    for c in range(half, len(arr) - half):
        seg = arr[c - half:c + half + 1]
        t = np.arange(len(seg), dtype=float)
        for dim in (0, 1):
            resid = seg[:, dim] - np.polyval(np.polyfit(t, seg[:, dim], 2), t)
            jitters.append(float(np.sqrt(np.mean(resid**2))))
    return float(np.mean(jitters)), float(np.mean(steps))


def band_of(step: float) -> str:
    if step < 0.5:
        return "静止"
    if step < 2.0:
        return "慢速"
    if step < 8.0:
        return "中速"
    return "快速"


BANDS = ["静止", "慢速", "中速", "快速"]


def main() -> None:
    out_dir = Path(sys.argv[1])
    min_step = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

    grouped: dict[str, dict[str, list]] = {}
    for path in sorted(out_dir.glob("*.json")):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        label = match.group("label")
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames = payload["frames"]
        entry = grouped.setdefault(label, {"tracks": [], "coverage": [], "classes": {}})
        pred_objs = objs = 0
        for points in track_series(frames).values():
            stats = track_stats(points)
            if stats is not None:
                entry["tracks"].append(stats)
        for frame in frames:
            for obj in frame:
                objs += 1
                if obj.get("predictable"):
                    pred_objs += 1
                entry["classes"][obj["cls"]] = entry["classes"].get(obj["cls"], 0) + 1
        if objs:
            entry["coverage"].append(pred_objs / objs)

    if not grouped:
        print(f"{out_dir} 下没有找到 <模型>_VID_*.json 输出")
        return

    print(f"目录：{out_dir}")
    print(f"判据：轨迹每帧平均位移 >= {min_step:.1f} px 视为运动目标"
          f"（滑动窗口 {WINDOW} 帧二次拟合残差）\n")

    for label, entry in grouped.items():
        tracks = entry["tracks"]
        print(f"=== {label}（{len(tracks)} 条长轨迹）===")
        print(f"{'速度档':<8}{'轨迹数':>8}{'抖动RMS(px)':>13}{'每帧位移(px)':>14}{'抖动占比':>10}")
        moving = []
        for band in BANDS:
            subset = [t for t in tracks if band_of(t[1]) == band]
            if band == "中速" or band == "快速":
                moving.extend(subset)
            if not subset:
                print(f"{band:<8}{0:>8}{'-':>13}{'-':>14}{'-':>10}")
                continue
            j = float(np.mean([t[0] for t in subset]))
            s = float(np.mean([t[1] for t in subset]))
            print(f"{band:<8}{len(subset):>8}{j:>13.2f}{s:>14.2f}{j / s * 100:>9.1f}%")

        if moving:
            j = float(np.mean([t[0] for t in moving]))
            s = float(np.mean([t[1] for t in moving]))
            share = len(moving) / len(tracks) * 100
            print(f"{'运动子集':<8}{len(moving):>8}{j:>13.2f}{s:>14.2f}{j / s * 100:>9.1f}%"
                  f"   （占长轨迹 {share:.0f}%）")
        overall_j = float(np.mean([t[0] for t in tracks]))
        overall_s = float(np.mean([t[1] for t in tracks]))
        print(f"{'全部轨迹':<8}{len(tracks):>8}{overall_j:>13.2f}{overall_s:>14.2f}"
              f"{overall_j / overall_s * 100:>9.1f}%")
        print(f"可预判覆盖率（逐段平均）：{float(np.mean(entry['coverage'])):.3f}"
              f"  共 {len(entry['coverage'])} 段")
        print()

    print("=== 运动子集横向对比 ===")
    print(f"{'模型':<16}{'运动轨迹':>10}{'抖动RMS':>10}{'每帧位移':>10}{'抖动占比':>10}")
    for label, entry in grouped.items():
        moving = [t for t in entry["tracks"] if t[1] >= min_step]
        if not moving:
            print(f"{label:<16}{0:>10}{'-':>10}{'-':>10}{'-':>10}")
            continue
        j = float(np.mean([t[0] for t in moving]))
        s = float(np.mean([t[1] for t in moving]))
        print(f"{label:<16}{len(moving):>10}{j:>10.2f}{s:>10.2f}{j / s * 100:>9.1f}%")


if __name__ == "__main__":
    main()
