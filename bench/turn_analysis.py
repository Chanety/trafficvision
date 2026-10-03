"""转弯行为统计：这批素材里到底有多少真正在转弯/掉头的轨迹。

没有足够的机动样本，任何"转弯感知"的改进都无法验证，所以先量化。

做法：对每条轨迹用滑动窗口估计航向角序列，统计航向累计变化量、净变化量与最大转弯率。

用法：python bench/turn_analysis.py <批处理输出目录> [排除关键词] [窗口秒数]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def track_series(payload: dict) -> dict[int, list[tuple[float, float, float, float, float]]]:
    """返回 {轨迹ID: [(时间秒, x, y, 宽, 高), ...]}。"""
    fps = payload["video_info"]["fps"] or 30.0
    series: dict[int, list[tuple[float, float, float, float, float]]] = {}
    for i, frame in enumerate(payload["frames"]):
        for obj in frame:
            if obj["id"] > 0:
                t = obj.get("t", i / fps)
                x1, y1, x2, y2 = obj["bbox"]
                series.setdefault(obj["id"], []).append(
                    (float(t), obj["centroid"][0], obj["centroid"][1], x2 - x1, y2 - y1)
                )
    return series


def heading_series(pts, window_s: float, min_disp: float = 8.0):
    """用窗口位移估计航向角（度），返回 [(时间, 航向角, 速度)]。"""
    arr = np.array([[p[1], p[2]] for p in pts], dtype=float)
    times = np.array([p[0] for p in pts], dtype=float)
    out = []
    j = 0
    for i in range(len(pts)):
        while j < i and times[i] - times[j] > window_s:
            j += 1
        if i == j:
            continue
        d = arr[i] - arr[j]
        dist = float(np.linalg.norm(d))
        dt = times[i] - times[j]
        # 位移太小时方向完全被噪声主导（静止目标抖动会让航向乱翻），必须门控
        if dist < min_disp or dt <= 0:
            continue
        angle = float(np.degrees(np.arctan2(d[1], d[0])))
        out.append((times[i], angle, dist / dt))
    return out


def quality_ok(pts, w, h, frame_area: float) -> bool:
    """与 trafficvision.motion.TrackMotion.quality_ok 同口径的离线版门控。"""
    import numpy as _np

    w = _np.asarray(w, dtype=float)
    h = _np.asarray(h, dtype=float)
    if h.size < 5 or h.mean() <= 1 or w.mean() <= 1:
        return False
    aspect = float(w.mean() / h.mean())
    if not (0.2 <= aspect <= 6.0):
        return False
    dw = _np.abs(_np.diff(w)) / _np.maximum(w[:-1], 1.0)
    dh = _np.abs(_np.diff(h)) / _np.maximum(h[:-1], 1.0)
    if dw.size and float(_np.median(dw)) > 0.05:
        return False
    if dh.size and float(_np.median(dh)) > 0.05:
        return False
    if frame_area and float(w.mean() * h.mean() / frame_area) > 0.30:
        return False
    return True


def main() -> None:
    out_dir = Path(sys.argv[1])
    exclude = sys.argv[2] if len(sys.argv) > 2 else None
    window_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    straight, moderate, sharp, uturn = 0, 0, 0, 0
    turn_rates = []
    details = []
    total = 0
    rejected = 0

    for path in sorted(out_dir.glob("VID_*.json")):
        if exclude and exclude in path.name:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        info = payload["video_info"]
        frame_area = float(info["width"] * info["height"])
        for tid, pts in track_series(payload).items():
            if len(pts) < 20:
                continue
            # 质量门控：剔除退化框与 ID 跳变造成的假轨迹
            if not quality_ok(pts, [p[3] for p in pts], [p[4] for p in pts], frame_area):
                rejected += 1
                continue
            hs = heading_series(pts, window_s)
            if len(hs) < 10:
                continue
            # 只统计真正在移动的轨迹：中位速度需超过 20 像素/秒
            median_speed = float(np.median([h[2] for h in hs]))
            if median_speed < 20.0:
                continue
            total += 1
            angles = np.array([h[1] for h in hs])
            times = np.array([h[0] for h in hs])
            diffs = np.diff(angles)
            diffs = (diffs + 180) % 360 - 180        # 归一化到 ±180
            dts = np.diff(times)
            rates = np.where(dts > 1e-6, np.abs(diffs) / np.maximum(dts, 1e-6), 0.0)
            cumulative = float(np.sum(np.abs(diffs)))
            net = float(abs((angles[-1] - angles[0] + 180) % 360 - 180))
            peak_rate = float(np.percentile(rates, 90)) if rates.size else 0.0
            turn_rates.append(peak_rate)
            details_window = None  # 占位避免误用

            if cumulative < 15:
                straight += 1
            elif cumulative < 60:
                moderate += 1
                details.append((path.name, tid, cumulative, net, peak_rate, len(pts)))
            elif cumulative < 150:
                sharp += 1
                details.append((path.name, tid, cumulative, net, peak_rate, len(pts)))
            else:
                uturn += 1
                details.append((path.name, tid, cumulative, net, peak_rate, len(pts)))

    print(f"窗口 {window_s}s，排除 {exclude or '无'}，可用轨迹 {total} 条")
    print(f"（质量门控剔除退化/跳变轨迹 {rejected} 条）")
    print(f"  基本直线（累计航向变化 <15°）  : {straight:>4} 条  {straight/max(1,total)*100:5.1f}%")
    print(f"  中等转弯（15° ~ 60°）          : {moderate:>4} 条  {moderate/max(1,total)*100:5.1f}%")
    print(f"  明显转弯（60° ~ 150°）         : {sharp:>4} 条  {sharp/max(1,total)*100:5.1f}%")
    print(f"  掉头/大角度（>150°）           : {uturn:>4} 条  {uturn/max(1,total)*100:5.1f}%")
    if turn_rates:
        arr = np.array(turn_rates)
        print(f"\n转弯率（度/秒）P50/P90/P99: {np.percentile(arr,50):.1f} / "
              f"{np.percentile(arr,90):.1f} / {np.percentile(arr,99):.1f}")

    if details:
        print(f"\n--- 明显转弯以上（前 20 条）---")
        print(f"{'视频':<12}{'轨迹':>6}{'累计变化':>10}{'净变化':>9}{'转弯率P90':>11}{'帧数':>7}")
        for name, tid, cum, net, rate, n in sorted(details, key=lambda x: -x[2])[:20]:
            print(f"{name.replace('VID_20261003_','').replace('.json',''):<12}{tid:>6}"
                  f"{cum:>10.0f}{net:>9.0f}{rate:>11.1f}{n:>7}")


if __name__ == "__main__":
    main()
