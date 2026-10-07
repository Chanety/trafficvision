"""跟踪器横向对比：把若干份预判 JSON 放在一起，比"连续性 / ID 跳变 / 可预判数量 / 误差"。

只看覆盖率会骗人（分母变小比例就上去了），所以这里同时看**绝对数量**：
每帧可预判目标数、轨迹条数、轨迹长度，以及误差总量里有多少来自 ID 跳变样本。

用法：python bench/tracker_bench.py <目录1=名称1> <目录2=名称2> ...
例：  python bench/tracker_bench.py outputs/diag02_bytetrack=ByteTrack默认 \
          outputs/diag02_btlong=ByteTrack长缓冲
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_diagnostics import analyze  # noqa: E402


def evaluate(out_dir: Path, label: str, horizon: float = 1.0) -> dict:
    rows, frames_total, predictable, track_frames = [], 0, 0, {}
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(analyze(payload, horizon))
        frames_total += len(payload["frames"])
        for frame in payload["frames"]:
            for obj in frame:
                if obj.get("predictable"):
                    predictable += 1
                if obj["id"] > 0:
                    track_frames[obj["id"]] = track_frames.get(obj["id"], 0) + 1

    errs = np.array([r["error"] for r in rows])
    gap = np.array([r["gap_frames"] for r in rows])
    ratio = np.array([r["implied_ratio"] for r in rows])
    continuous = gap == 0
    anomaly = ratio > 2.0
    lengths = np.array(list(track_frames.values())) if track_frames else np.array([0])
    return {
        "label": label,
        "samples": len(rows),
        "err_median": float(np.median(errs)),
        "err_mean": float(errs.mean()),
        "err_p90": float(np.percentile(errs, 90)),
        "cont_share": float(continuous.mean() * 100),
        "cont_median": float(np.median(errs[continuous])) if continuous.any() else float("nan"),
        "anomaly_share": float(anomaly.mean() * 100),
        "anomaly_mass": float(errs[anomaly].sum() / errs.sum() * 100),
        "anomaly_median": float(np.median(errs[anomaly])) if anomaly.any() else float("nan"),
        "predictable_per_frame": predictable / max(1, frames_total),
        "tracks": len(track_frames),
        "track_median": float(np.median(lengths)),
    }


def main() -> None:
    specs = []
    for item in sys.argv[1:]:
        path, _, label = item.partition("=")
        specs.append((Path(path), label or Path(path).name))
    if not specs:
        print(__doc__)
        return

    results = [evaluate(path, label) for path, label in specs]
    header = (f"{'配置':<18}{'样本':>7}{'中位误差':>10}{'均值':>9}{'P90':>9}"
              f"{'连续占比':>10}{'连续中位':>10}{'异常样本':>10}{'异常占比':>10}"
              f"{'异常中位':>10}{'可预判/帧':>11}{'轨迹数':>8}{'轨迹中位':>10}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['label']:<18}{r['samples']:>7}{r['err_median']:>10.1f}{r['err_mean']:>9.1f}"
              f"{r['err_p90']:>9.1f}{r['cont_share']:>9.1f}%{r['cont_median']:>10.1f}"
              f"{r['anomaly_share']:>9.1f}%{r['anomaly_mass']:>9.0f}%{r['anomaly_median']:>10.1f}"
              f"{r['predictable_per_frame']:>11.2f}{r['tracks']:>8}{r['track_median']:>10.0f}")
    print("\n说明：异常样本 = 窗口内真实位移换算速度 > 预测时刻估计速度的 2 倍，"
          "是 ID 跳变/串轨的典型特征；它占比很小但吃掉大量误差。")


if __name__ == "__main__":
    main()
