"""机动门控阈值扫描：用已跑好的 JSON（含 turn_deg_s / maneuver_dev_px）离线比较不同门控。

门控 = "机动中的目标不发布该梯度预判"。它必然牺牲覆盖率，所以要看性价比：
拦掉多少预判、换来多少误差下降、以及显著的错判（>100px）少了多少。

用法：python bench/maneuver_gate_sweep.py <预判JSON目录> [梯度=1.0]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def samples(out_dir: Path, horizon: float) -> list[dict]:
    rows = []
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames = payload["frames"]
        times = []
        last = 0.0
        for f in frames:
            if f:
                last = float(f[0]["t"])
            times.append(last)
        t_arr = np.array(times, dtype=float)
        truth_map = [{o["id"]: o for o in f if not o.get("coasting")} for f in frames]
        for t, frame in enumerate(frames):
            target = t_arr[t] + horizon
            i1 = int(np.searchsorted(t_arr, target))
            if i1 >= len(frames):
                continue
            i0 = max(0, i1 - 1)
            span = t_arr[i1] - t_arr[i0]
            frac = (target - t_arr[i0]) / span if span > 1e-9 else 0.0
            for obj in frame:
                key = f"{horizon:g}"
                pred = obj["preds"].get(key) or obj["preds"].get(str(horizon))
                if pred is None or not obj.get("predictable"):
                    continue
                tid = obj["id"]
                a, b = truth_map[i0].get(tid), truth_map[i1].get(tid)
                if a is not None and b is not None:
                    truth = (a["centroid"][0] + (b["centroid"][0] - a["centroid"][0]) * frac,
                             a["centroid"][1] + (b["centroid"][1] - a["centroid"][1]) * frac)
                elif a is not None and frac < 0.5:
                    truth = tuple(a["centroid"])
                elif b is not None and frac >= 0.5:
                    truth = tuple(b["centroid"])
                else:
                    continue
                pc = ((pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0)
                rows.append({
                    "err": float(np.hypot(pc[0] - truth[0], pc[1] - truth[1])),
                    "turn": obj.get("turn_deg_s"),
                    "dev": obj.get("maneuver_dev_px", 0.0),
                    "conf": obj.get("confidence", "high"),
                })
    return rows


def summarize(rows: list[dict], keep: np.ndarray, label: str) -> None:
    err = np.array([r["err"] for r in rows])
    kept = err[keep]
    removed = (~keep).sum()
    print(f"{label:<34}{removed:>8}{removed / len(rows) * 100:>9.1f}%"
          f"{np.median(kept):>11.1f}{kept.mean():>11.1f}"
          f"{np.percentile(kept, 90):>11.1f}{np.mean(kept > 100) * 100:>12.1f}%")


def main() -> None:
    out_dir = Path(sys.argv[1])
    horizon = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    rows = samples(out_dir, horizon)
    if not rows:
        print("没有样本")
        return
    turn = np.array([r["turn"] if r["turn"] is not None else 0.0 for r in rows])
    dev = np.array([r["dev"] for r in rows])

    print(f"样本 {len(rows)}，梯度 {horizon:g}s")
    header = (f"{'门控方案':<34}{'拦掉':>8}{'占比':>10}{'剩余中位':>11}{'剩余均值':>11}"
              f"{'剩余P90':>11}{'>100px占比':>13}")
    print(header)
    print("-" * len(header))
    summarize(rows, np.ones(len(rows), dtype=bool), "不门控（现状）")
    for thr in (15, 25, 40, 60):
        summarize(rows, np.abs(turn) < thr, f"仅按转角 |ω|≥{thr}°/s")
    for budget in (25, 50, 100):
        summarize(rows, dev < budget, f"仅按偏离 dev≥{budget}px")
    for thr, budget in ((25, 50), (40, 100)):
        summarize(rows, (np.abs(turn) < thr) & (dev < budget),
                  f"转角≥{thr}°/s 或 偏离≥{budget}px")


if __name__ == "__main__":
    main()
