"""IMM（交互式多模型）实验：能不能比产品现在的匀速卡尔曼更准。

背景：昨天验证了"硬信转角估计"（圆弧外推）反而更差——转角估计噪声太大。
IMM 是这个问题在文献里的标准解法：并行跑多个运动模型（这里用两个不同过程噪声的
匀速模型），按各自的似然自动加权，不需要显式相信任何单一转角估计。
算法来自开源项目 filterpy（MIT，见 vendor/filterpy/）。

对比口径：完全配对的同一批样本——同一视频、同一目标、同一时刻，
产品已经算出的 1 秒预判 vs 用同一条观测序列喂给 IMM 得到的 1 秒预判。

用法：python bench/imm_experiment.py <预判JSON目录> [梯度=1.0]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from filterpy import IMMEstimator  # noqa: E402
from filterpy.minimal_kf import MinimalKalmanFilter  # noqa: E402

MEAS_STD = 4.0          # 与产品一致：检测中心噪声 4 px
ACCEL_CALM = 400.0      # 与产品一致：匀速模型过程噪声
ACCEL_MANEUVER = 1200.0 # 机动模型：过程噪声放大 9 倍


def q_matrix(accel_std: float, dt: float) -> np.ndarray:
    q = accel_std ** 2
    d2, d3, d4 = dt * dt, dt ** 3, dt ** 4
    return q * np.array([[d4 / 4, 0.0, d3 / 2, 0.0],
                         [0.0, d4 / 4, 0.0, d3 / 2],
                         [d3 / 2, 0.0, d2, 0.0],
                         [0.0, d3 / 2, 0.0, d2]])


def f_matrix(dt: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dt, 0.0],
                     [0.0, 1.0, 0.0, dt],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def make_kf(accel_std: float, dt: float):
    kf = MinimalKalmanFilter(dim_x=4, dim_z=2)
    kf.F = f_matrix(dt)
    kf.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    kf.R = np.eye(2) * MEAS_STD ** 2
    kf.Q = q_matrix(accel_std, dt)
    kf.P = np.diag([MEAS_STD ** 2, MEAS_STD ** 2, 400.0 ** 2, 400.0 ** 2])
    kf.x = np.zeros((4, 1))
    return kf


def build_tracks(payload: dict):
    """返回 {id: [(t, cx, cy, h), ...]} 与产品样本 {(id, t): (pred_center, truth, turn)}。"""
    frames = payload["frames"]
    times, last = [], 0.0
    for f in frames:
        if f:
            last = float(f[0]["t"])
        times.append(last)
    t_arr = np.array(times, dtype=float)

    tracks: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    truth_map = []
    for f in frames:
        bucket = {}
        for o in f:
            if o.get("coasting"):
                continue
            bucket[o["id"]] = o
            tracks[o["id"]].append((float(o["t"]), float(o["centroid"][0]),
                                    float(o["centroid"][1]),
                                    float(o["bbox"][3] - o["bbox"][1])))
        truth_map.append(bucket)

    samples = {}
    for t, frame in enumerate(frames):
        target = t_arr[t] + 1.0
        i1 = int(np.searchsorted(t_arr, target))
        if i1 >= len(frames):
            continue
        i0 = max(0, i1 - 1)
        span = t_arr[i1] - t_arr[i0]
        frac = (target - t_arr[i0]) / span if span > 1e-9 else 0.0
        for obj in frame:
            pred = obj["preds"].get("1.0") or obj["preds"].get("1")
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
            samples[(tid, round(float(obj["t"]), 3))] = (
                ((pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0),
                truth, obj.get("turn_deg_s"))
    return tracks, samples


def run_imm(points: list[tuple[float, float, float, float]], horizon: float = 1.0,
            trans: float = 0.05, mu=(0.8, 0.2)):
    """把一条轨迹喂给 IMM，返回 {时刻: (预测中心, 机动模型概率)}。"""
    out = {}
    if len(points) < 2:
        return out
    dt0 = max(1e-3, points[1][0] - points[0][0])
    kf_calm = make_kf(ACCEL_CALM, dt0)
    kf_man = make_kf(ACCEL_MANEUVER, dt0)
    imm = IMMEstimator([kf_calm, kf_man], mu,
                       np.array([[1 - trans, trans], [trans, 1 - trans]]))
    prev_t = None
    for (t, cx, cy, _h) in points:
        dt = dt0 if prev_t is None else max(1e-3, t - prev_t)
        for kf in (kf_calm, kf_man):
            kf.F = f_matrix(dt)
            kf.Q = q_matrix(ACCEL_CALM if kf is kf_calm else ACCEL_MANEUVER, dt)
        if prev_t is None:
            for kf in (kf_calm, kf_man):
                kf.x = np.array([[cx], [cy], [0.0], [0.0]])
                kf.P = np.diag([MEAS_STD ** 2, MEAS_STD ** 2, 400.0 ** 2, 400.0 ** 2])
        else:
            imm.predict()
        imm.update(np.array([[cx], [cy]]))
        x = imm.x
        out[round(t, 3)] = (float(x[0, 0] + x[2, 0] * horizon),
                            float(x[1, 0] + x[3, 0] * horizon),
                            float(imm.mu[1]))
        prev_t = t
    return out


def run_single_kf(points: list[tuple[float, float, float, float]], accel: float,
                  horizon: float = 1.0) -> dict:
    """单模型匀速卡尔曼（与产品同结构，过程噪声可调），用于扫参数。"""
    out = {}
    if len(points) < 2:
        return out
    dt0 = max(1e-3, points[1][0] - points[0][0])
    kf = make_kf(accel, dt0)
    prev_t = None
    for (t, cx, cy, _h) in points:
        dt = dt0 if prev_t is None else max(1e-3, t - prev_t)
        kf.F = f_matrix(dt)
        kf.Q = q_matrix(accel, dt)
        if prev_t is None:
            kf.x = np.array([[cx], [cy], [0.0], [0.0]])
        else:
            kf.predict()
        kf.update(np.array([[cx], [cy]]))
        x = kf.x
        out[round(t, 3)] = (float(x[0, 0] + x[2, 0] * horizon),
                            float(x[1, 0] + x[3, 0] * horizon), 0.0)
        prev_t = t
    return out


def run_adaptive_kf(points: list[tuple[float, float, float, float]], accel: float = 400.0,
                    horizon: float = 1.0, gain: float = 6.0, lo: float = 0.25,
                    hi: float = 6.0) -> dict:
    """自适应卡尔曼：按归一化新息平方（NIS）在线调整过程噪声。

    动机：固定 Q 有两难——Q 小则中位误差好但 P90 差（跟不上机动）。
    这里用 Sage-Husa 式的思路：新息明显大于理论值时把 Q 放大（跟随），
    明显偏小时把 Q 缩小（平滑）。
    """
    out = {}
    if len(points) < 2:
        return out
    dt0 = max(1e-3, points[1][0] - points[0][0])
    scale = 1.0
    kf = make_kf(accel, dt0)
    prev_t = None
    for (t, cx, cy, _h) in points:
        dt = dt0 if prev_t is None else max(1e-3, t - prev_t)
        q_std = accel * scale
        kf.F = f_matrix(dt)
        kf.Q = q_matrix(q_std, dt)
        if prev_t is None:
            kf.x = np.array([[cx], [cy], [0.0], [0.0]])
        else:
            kf.predict()
        # 先算新息与理论协方差，再更新
        z = np.array([[cx], [cy]])
        innov = z - kf.H @ kf.x
        S = kf.H @ kf.P @ kf.H.T + kf.R
        nis = float((innov.T @ np.linalg.inv(S) @ innov).item())
        target = 2.0                      # 二维观测下 NIS 的期望值
        if nis > 2.0 * target:
            scale = min(hi, scale * (1.0 + gain / 10.0))
        elif nis < 0.5 * target:
            scale = max(lo, scale * (1.0 - gain / 20.0))
        kf.update(z)
        x = kf.x
        out[round(t, 3)] = (float(x[0, 0] + x[2, 0] * horizon),
                            float(x[1, 0] + x[3, 0] * horizon), scale)
        prev_t = t
    return out


def sweep_accel(out_dir: Path) -> None:
    """过程噪声扫描：产品的匀速 KF 用 accel_std=400，看它是不是最优。"""
    values = (100, 200, 400, 800, 1600, 3200)
    results: dict[str, list[float]] = {str(v): [] for v in values}
    results["自适应Q"] = []
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        tracks, samples = build_tracks(payload)
        per_track = {tid: {str(v): run_single_kf(points, v) for v in values}
                     for tid, points in tracks.items()}
        for tid, points in tracks.items():
            per_track[tid]["自适应Q"] = run_adaptive_kf(points)
        for (key, (_pred, truth, _turn)) in samples.items():
            for v in results:
                got = per_track[key[0]][v].get(key[1])
                if got is None:
                    continue
                results[v].append(float(np.hypot(got[0] - truth[0], got[1] - truth[1])))
    n = len(results[str(values[0])])
    print(f"过程噪声（accel_std）扫描，配对样本 {n}")
    print(f"{'accel_std':>10}{'中位':>9}{'均值':>9}{'P90':>9}")
    for v in results:
        arr = np.array(results[v])
        mark = "  ← 产品当前值" if v == "400" else ("  ← 自适应" if v == "自适应Q" else "")
        print(f"{v:>10}{np.median(arr):>9.1f}{arr.mean():>9.1f}"
              f"{np.percentile(arr, 90):>9.1f}{mark}")


def main() -> None:
    out_dir = Path(sys.argv[1])
    trans = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    if "--sweep-q" in sys.argv:
        sweep_accel(out_dir)
        return
    rows = []
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        tracks, samples = build_tracks(payload)
        for tid, points in tracks.items():
            imm_pred = run_imm(points, 1.0, trans)
            for (key, (prod_pred, truth, turn)) in samples.items():
                if key[0] != tid:
                    continue
                got = imm_pred.get(key[1])
                if got is None:
                    continue
                prod_err = float(np.hypot(prod_pred[0] - truth[0], prod_pred[1] - truth[1]))
                imm_err = float(np.hypot(got[0] - truth[0], got[1] - truth[1]))
                rows.append({"prod": prod_err, "imm": imm_err, "p_man": got[2],
                             "turn": turn, "cls": None})
    if not rows:
        print("没有配对样本")
        return
    prod = np.array([r["prod"] for r in rows])
    imm = np.array([r["imm"] for r in rows])
    print(f"配对样本 {len(rows)}（IMM 转移概率 {trans}，机动模型过程噪声 {ACCEL_MANEUVER:.0f}）")
    print(f"{'':<16}{'中位':>9}{'均值':>9}{'P90':>9}")
    print(f"{'产品(匀速KF)':<16}{np.median(prod):>9.1f}{prod.mean():>9.1f}{np.percentile(prod, 90):>9.1f}")
    print(f"{'IMM':<16}{np.median(imm):>9.1f}{imm.mean():>9.1f}{np.percentile(imm, 90):>9.1f}")
    print(f"配对差值（IMM−产品）：中位 {np.median(imm - prod):+.2f} px，"
          f"均值 {(imm - prod).mean():+.2f} px，IMM 更准的样本占 {np.mean(imm < prod) * 100:.0f}%")

    turn = np.array([abs(r["turn"]) if r["turn"] is not None else 0.0 for r in rows])
    print(f"\n{'分档':<18}{'样本':>7}{'产品中位':>10}{'IMM 中位':>10}{'IMM 更准占比':>13}{'机动模型概率':>13}")
    for name, lo, hi in (("直行 <5°/s", 0, 5), ("缓弯 5~15", 5, 15),
                         ("转弯 15~40", 15, 40), ("急转 >40", 40, 1e9)):
        mask = (turn >= lo) & (turn < hi)
        if not mask.any():
            continue
        p_man = np.mean([r["p_man"] for r, m in zip(rows, mask) if m])
        print(f"{name:<18}{mask.sum():>7}{np.median(prod[mask]):>10.1f}"
              f"{np.median(imm[mask]):>10.1f}{np.mean(imm[mask] < prod[mask]) * 100:>12.0f}%"
              f"{p_man:>13.2f}")


if __name__ == "__main__":
    main()
