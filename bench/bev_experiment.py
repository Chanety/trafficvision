"""实验 C：地面平面（BEV）运动建模——在"米"空间做外推，再投回画面。

动机：像素空间的运动模型有个天生缺陷——同一个车速，远处的车每秒走几个像素，
近处的车每秒走几百像素。一个固定的过程噪声（Q）没法同时适配两者，
这正是我们反复遇到的"中位误差 vs P90 尾部"两难。
把画面映射到地面平面后，"匀速"就是物理意义上的匀速，与远近无关。

文献：Traffic-Net (arXiv:2109.09205)、Camera-to-Satellite Homography
Rectification (arXiv:2601.16124)，见 docs/15。

标定不靠人工：用经典的"目标框高度 ∝ 1/距离"关系，
对所有检测框做 高度 ≈ a·y + b 的线性拟合，地平线 y_h = -b/a。
（本项目 pipeline 里的 `_estimate_horizon` 用的是同一原理。）

映射（针孔相机 + 平地面，尺度取 1）：
    Z = 1 / (v - y_h)              深度（相对单位）
    X = (u - cx) / (v - y_h)       横向（相对单位）
测量噪声按雅可比传过去：R_bev = J·diag(σu², σv²)·Jᵀ，
否则远处目标的观测噪声会被错误地放大/缩小。

用法：python bench/bev_experiment.py <预判JSON目录> [梯度=1.0]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MEAS_STD = 4.0        # 与产品一致：检测中心噪声 4 px
ACCEL_CALM = 400.0    # 像素空间的过程噪声（用于对照）
MIN_DEPTH_PX = 25.0   # 离地平线太近的目标深度不可信，跳过


def horizon_from_boxes(observations: list[tuple[float, float, float, float, str]],
                       frame_h: float) -> float | None:
    """用 框高 ≈ a·y + b 拟合地平线（框高随 y 减小，到地平线为 0）。

    必须只用**同类目标**（这里用汽车）：不同类别的物理高度差太多，
    混在一起拟合会得到荒谬的结果（实测 y_h=2191px，跑到画面外）。
    再按 y 分箱取中位高度，压制离群框。
    """
    # 关键：用**框底边**（地面接触点）而不是框中心——车在路上，接触点才在地面平面上
    cars = [(o[2], o[3]) for o in observations if o[4] == "car" and o[3] > 3.0]
    if len(cars) < 40:
        cars = [(o[2], o[3]) for o in observations
                if o[4] in ("car", "truck", "bus") and o[3] > 3.0]
    if len(cars) < 40:
        return None
    ys = np.array([c[0] for c in cars])
    hs = np.array([c[1] for c in cars])
    edges = np.linspace(ys.min(), ys.max(), 13)
    bx, by = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (ys >= lo) & (ys < hi)
        if mask.sum() >= 5:
            bx.append(float(np.median(ys[mask])))
            by.append(float(np.median(hs[mask])))
    if len(bx) < 4:
        return None
    a, b = np.polyfit(np.array(bx), np.array(by), 1)
    # 用地面接触点后，越靠近画面下方（y 越大）目标越大 → 斜率必须为正；
    # 地平线就是框高外推到 0 的位置。
    if a <= 1e-6:
        return None
    y_h = float(-b / a)
    # 合理性检查：地平线必须在画面内或略上方（本项目场景实测在 500~600px 附近）
    if not (-0.5 * frame_h <= y_h <= frame_h * 0.95):
        return None
    return y_h


def bev_jacobian(u: float, v: float, y_h: float, cx: float) -> np.ndarray:
    d = max(MIN_DEPTH_PX, v - y_h)
    z = 1.0 / d
    # 行 = (X, Z)，列 = (u, v)
    return np.array([[z, -(u - cx) * z * z],
                     [0.0, -z * z]])


def run_bev(points: list[tuple[float, float, float, float]], y_h: float, cx: float,
            horizon_s: float = 1.0, accel_bev: float = 0.02) -> dict:
    """在 BEV 空间做匀速卡尔曼；accel_bev 是 BEV 单位下的过程噪声。"""
    out = {}
    if len(points) < 2:
        return out
    dt0 = max(1e-3, points[1][0] - points[0][0])
    x = np.zeros((4, 1))
    P = np.diag([1e-4, 1e-4, 1e-3, 1e-3])
    R = np.eye(2) * MEAS_STD ** 2
    prev_t = None
    started = False
    for (t, u, v, _h, _cls) in points:
        if v - y_h < MIN_DEPTH_PX:
            prev_t = t
            continue
        dt = dt0 if prev_t is None else max(1e-3, t - prev_t)
        z = 1.0 / (v - y_h)
        zx = (u - cx) * z
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                      [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        q = accel_bev ** 2
        Q = q * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                          [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                          [dt ** 3 / 2, 0, dt ** 2, 0],
                          [0, dt ** 3 / 2, 0, dt ** 2]])
        J = bev_jacobian(u, v, y_h, cx)
        R_bev = J @ R @ J.T + np.eye(2) * 1e-9
        if not started:
            x = np.array([[zx], [z], [0.0], [0.0]])
            P = np.diag([1e-4, 1e-4, 1e-3, 1e-3])
            started = True
        else:
            x = F @ x
            P = F @ P @ F.T + Q
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        y = np.array([[zx], [z]]) - H @ x
        S = H @ P @ H.T + R_bev
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(4) - K @ H) @ P
        # 外推后投回画面
        Zp = x[1, 0] + x[3, 0] * horizon_s
        Xp = x[0, 0] + x[2, 0] * horizon_s
        if Zp <= 1e-6:
            prev_t = t
            continue
        depth = 1.0 / Zp
        out[round(t, 3)] = (cx + Xp * depth, y_h + depth)
        prev_t = t
    return out


def build(payload: dict):
    frames = payload["frames"]
    info = payload["video_info"]
    cx = info["width"] / 2.0
    times, last = [], 0.0
    for f in frames:
        if f:
            last = float(f[0]["t"])
        times.append(last)
    t_arr = np.array(times, dtype=float)
    tracks: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    truth_map, samples = [], {}
    for f in frames:
        bucket = {}
        for o in f:
            if o.get("coasting"):
                continue
            bucket[o["id"]] = o
            # 地面接触点 = 框底边中点（BEV 映射与地平线拟合都用它）
            tracks[o["id"]].append((float(o["t"]),
                                    (float(o["bbox"][0]) + float(o["bbox"][2])) / 2.0,
                                    float(o["bbox"][3]),
                                    float(o["bbox"][3] - o["bbox"][1]), o["cls"]))
        truth_map.append(bucket)
    for t, frame in enumerate(frames):
        target = t_arr[t] + 1.0
        i1 = int(np.searchsorted(t_arr, target))
        if i1 >= len(frames):
            continue
        i0 = max(0, i1 - 1)
        span = t_arr[i1] - t_arr[i0]
        frac = (target - t_arr[i0]) / span if span > 1e-9 else 0.0
        for obj in frame:
            pred = obj["preds"].get("1.0")
            if pred is None or not obj.get("predictable"):
                continue
            tid = obj["id"]
            a, b = truth_map[i0].get(tid), truth_map[i1].get(tid)
            if a is not None and b is not None:
                # 真值也用地面接触点（框底边中点），与两种方法的口径一致
                ax = (a["bbox"][0] + a["bbox"][2]) / 2.0
                bx = (b["bbox"][0] + b["bbox"][2]) / 2.0
                truth = (ax + (bx - ax) * frac, a["bbox"][3] + (b["bbox"][3] - a["bbox"][3]) * frac)
            elif a is not None and frac < 0.5:
                truth = ((a["bbox"][0] + a["bbox"][2]) / 2.0, a["bbox"][3])
            elif b is not None and frac >= 0.5:
                truth = ((b["bbox"][0] + b["bbox"][2]) / 2.0, b["bbox"][3])
            else:
                continue
            samples[(tid, round(float(obj["t"]), 3))] = (
                # 产品那一路也取"预判框的底边中点"，保证两方口径一致
                ((pred[0] + pred[2]) / 2.0, pred[3]), truth,
                float(obj.get("speed_px_s", 0.0)))
    return tracks, samples, cx


def main() -> None:
    out_dir = Path(sys.argv[1])
    rows = []
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        tracks, samples, cx = build(payload)
        allobs = [p for pts in tracks.values() for p in pts]
        y_h = horizon_from_boxes(allobs, float(payload["video_info"]["height"]))
        if y_h is None:
            print(f"  {path.name}: 地平线拟合失败，跳过")
            continue
        depth_change = {}
        for tid, pts in tracks.items():
            if len(pts) >= 5:
                vs = [p[2] for p in pts]
                depth_change[tid] = abs(vs[-1] - vs[0])
        for tid, pts in tracks.items():
            bev = run_bev(pts, y_h, cx)
            for (key, (prod, truth, speed)) in samples.items():
                if key[0] != tid:
                    continue
                got = bev.get(key[1])
                if got is None:
                    continue
                rows.append({
                    "video": path.name, "y_h": y_h,
                    "prod": float(np.hypot(prod[0] - truth[0], prod[1] - truth[1])),
                    "bev": float(np.hypot(got[0] - truth[0], got[1] - truth[1])),
                    "dy": depth_change.get(tid, 0.0), "speed": speed,
                })
        print(f"  {path.name}: 地平线 y_h = {y_h:.0f} px，累计配对 {len(rows)}")

    if not rows:
        print("没有配对样本")
        return
    prod = np.array([r["prod"] for r in rows])
    bev = np.array([r["bev"] for r in rows])
    print(f"\n配对样本 {len(rows)}")
    print(f"{'方案':<14}{'中位':>9}{'均值':>9}{'P90':>9}")
    print(f"{'产品(像素KF)':<14}{np.median(prod):>9.1f}{prod.mean():>9.1f}"
          f"{np.percentile(prod, 90):>9.1f}")
    print(f"{'BEV(地面平面)':<14}{np.median(bev):>9.1f}{bev.mean():>9.1f}"
          f"{np.percentile(bev, 90):>9.1f}")
    print(f"配对差值：中位 {np.median(bev - prod):+.2f} px，均值 {(bev - prod).mean():+.2f} px，"
          f"BEV 更准的样本占 {np.mean(bev < prod) * 100:.0f}%")

    dy = np.array([r["dy"] for r in rows])
    print(f"\n按'窗口内深度变化'分档（BEV 理论上就该在这类目标上赢）：")
    print(f"{'深度变化':<16}{'样本':>7}{'产品中位':>10}{'BEV中位':>10}{'BEV更准占比':>13}")
    for name, lo, hi in (("几乎没变 <20px", 0, 20), ("中等 20~60px", 20, 60),
                         ("明显 >60px", 60, 1e9)):
        mask = (dy >= lo) & (dy < hi)
        if not mask.any():
            continue
        print(f"{name:<16}{mask.sum():>7}{np.median(prod[mask]):>10.1f}"
              f"{np.median(bev[mask]):>10.1f}"
              f"{np.mean(bev[mask] < prod[mask]) * 100:>12.0f}%")


if __name__ == "__main__":
    main()
