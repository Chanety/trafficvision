"""位移增益标定：检验预判位移是否存在系统性偏差，并做交叉验证。

对每个样本，比较"预判位移向量"与"真实位移向量"：
    g* = argmin Σ|g·p - t|²  =  Σ(p·t) / Σ(p·p)
g* > 1 表示系统性低估位移，< 1 表示高估。

为避免自证，增益在"留一视频"方式下交叉验证：用其余视频标定，在留出的视频上检验。

用法：python bench/gain_calibration.py <批处理输出目录> [排除关键词]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def collect(payload: dict, horizon: float = 1.0):
    frames = payload["frames"]
    info = payload["video_info"]
    fps = info["fps"] or 30.0
    times = [i / fps for i in range(len(frames))]
    t_arr = np.array(times, dtype=float)
    by_frame = [{o["id"]: o for o in f} for f in frames]

    out = []
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
            a, b = by_frame[i0].get(tid), by_frame[i1].get(tid)
            if a is None or b is None:
                continue
            truth = np.array([a["centroid"][0] + (b["centroid"][0] - a["centroid"][0]) * frac,
                              a["centroid"][1] + (b["centroid"][1] - a["centroid"][1]) * frac])
            now = np.array(obj["centroid"])
            p = np.array([(pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0]) - now
            out.append((p, truth - now))
    return out


def gain_of(pairs) -> float:
    num = sum(float(p @ t) for p, t in pairs)
    den = sum(float(p @ p) for p, t in pairs)
    return num / den if den > 1e-9 else 1.0


def errors(pairs, gain: float = 1.0):
    if not pairs:
        return np.array([0.0])
    return np.array([float(np.linalg.norm(gain * p - t)) for p, t in pairs])


def main() -> None:
    out_dir = Path(sys.argv[1])
    exclude = sys.argv[2] if len(sys.argv) > 2 else None
    per_video = {}
    for path in sorted(out_dir.glob("VID_*.json")):
        if exclude and exclude in path.name:
            continue
        pairs = collect(json.loads(path.read_text(encoding="utf-8")))
        if pairs:
            per_video[path.name] = pairs

    all_pairs = [pair for pairs in per_video.values() for pair in pairs]
    g_all = gain_of(all_pairs)
    base = errors(all_pairs, 1.0)
    fixed = errors(all_pairs, g_all)
    print(f"样本 {len(all_pairs)}，视频 {len(per_video)} 段（排除 {exclude or '无'}）")
    print(f"最优位移增益 g* = {g_all:.3f}    （>1 表示系统性低估位移）")
    print(f"修正前：平均 {base.mean():.1f} px  中位 {np.median(base):.1f}")
    print(f"修正后：平均 {fixed.mean():.1f} px  中位 {np.median(fixed):.1f}"
          f"   → 平均误差下降 {(1 - fixed.mean() / base.mean()) * 100:.1f}%")

    print("\n--- 留一视频交叉验证（用其余视频标定，在留出视频上检验）---")
    print(f"{'视频':<26}{'g*':>8}{'原始误差':>10}{'修正后':>10}{'改进':>9}")
    gains, improvements = [], []
    for name, pairs in per_video.items():
        others = [q for n, ps in per_video.items() if n != name for q in ps]
        g = gain_of(others)
        before = errors(pairs, 1.0).mean()
        after = errors(pairs, g).mean()
        gains.append(g)
        improvements.append((1 - after / before) * 100)
        print(f"{name.replace('VID_20261003_', ''):<26}{g:>8.3f}"
              f"{before:>10.1f}{after:>10.1f}{(1 - after / before) * 100:>8.1f}%")
    print(f"\n留一验证：平均改进 {np.mean(improvements):.1f}%，"
          f"其中 {sum(1 for x in improvements if x > 0)}/{len(improvements)} 段变好")


if __name__ == "__main__":
    main()
