"""机动分析：转弯/急加速的样本误差有多大，以及"圆弧外推"能不能救回来。

背景：上一轮把串轨（ID 跳变）用防护解决后，剩下的误差大头是"目标真的在机动"：
匀速卡尔曼假设目标直线匀速，遇到转弯、变道、急加速就会系统性偏离。

本脚本做两件事（都只用过去帧，保持因果）：
  1. 按机动强度分档统计预判误差——看转弯到底吃掉多少误差；
  2. 在同一批样本上对比两种外推：
       · 匀速（当前产品用的）
       · 恒速恒转角 CTRV 圆弧外推（用最近一小段轨迹估转角速度）
     如果 CTRV 在转弯档明显更准，就值得写进产品。

用法：python bench/maneuver_analysis.py <预判JSON目录> [梯度=1.0] [估计窗口秒=0.6]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

GAINS = (0.0, 0.2, 0.3, 0.5, 0.7, 1.0)


def unwrap(delta: float) -> float:
    """把角度差折到 (-pi, pi]。"""
    while delta > np.pi:
        delta -= 2 * np.pi
    while delta <= -np.pi:
        delta += 2 * np.pi
    return delta


def ctrv_predict(history: list[tuple[float, float, float]], horizon: float,
                 window: float = 0.6, gain: float = 1.0) -> tuple[float, float] | None:
    """用最近 window 秒的轨迹估 heading / 转角速度 / 速度，再做圆弧外推。"""
    if len(history) < 4:
        return None
    t_now = history[-1][0]
    seg = [h for h in history if h[0] >= t_now - window]
    if len(seg) < 4:
        return None
    pts = np.array([[h[1], h[2]] for h in seg], dtype=float)
    ts = np.array([h[0] for h in seg], dtype=float)

    steps = np.diff(pts, axis=0)
    dist = np.linalg.norm(steps, axis=1)
    # 只有真正在动的样本才能估计转角：静止目标的中心点是噪声，航向随机，
    # 硬估出来的"转角速度"能到几十度/秒（上一版就吃了这个亏）
    good = dist > 0.5
    if good.sum() < 3:
        return None
    # 窗口内总位移太小时，航向完全被检测噪声主导，估出来的转角不可信
    if float(np.sum(dist[good])) < 20.0:
        return None
    headings = np.arctan2(steps[good, 1], steps[good, 0])
    t_mid = ts[:-1][good]

    # 用最小二乘拟合 heading 随时间的变化率（转角速度 ω）
    A = np.vstack([t_mid - t_mid.mean(), np.ones_like(t_mid)]).T
    unwrapped = np.unwrap(headings)
    omega, _ = np.linalg.lstsq(A, unwrapped, rcond=None)[0]
    omega = float(omega) * float(gain)   # 阻尼：gain<1 表示只采纳一部分估计转角
    heading_now = float(unwrapped[-1] + omega * (t_now - t_mid[-1]))
    speed = float(np.sum(dist[good]) / max(1e-3, ts[-1] - ts[0]))

    # 沿圆弧小步递推外推（比解析式好写，等价且不易出符号错误）
    steps_n = max(1, int(abs(omega) * horizon / 0.05))
    dt_small = horizon / steps_n
    x, y = pts[-1]
    theta = heading_now
    for _ in range(steps_n):
        theta += omega * dt_small
        x += speed * dt_small * np.cos(theta)
        y += speed * dt_small * np.sin(theta)
    return float(x), float(y), float(np.degrees(omega))


def analyze(payload: dict, horizon: float, window: float) -> list[dict]:
    frames = payload["frames"]
    times = []
    last = 0.0
    for f in frames:
        if f:
            last = float(f[0]["t"])
        times.append(last)
    times_arr = np.array(times, dtype=float)

    obs: dict[int, list[tuple[float, float, float]]] = {}
    truth_map: list[dict[int, dict]] = []
    for i, frame in enumerate(frames):
        bucket: dict[int, dict] = {}
        for obj in frame:
            if obj.get("coasting"):
                continue
            bucket[obj["id"]] = obj
            obs.setdefault(obj["id"], []).append(
                (float(obj["t"]), float(obj["centroid"][0]), float(obj["centroid"][1])))
        truth_map.append(bucket)

    rows = []
    for t, frame in enumerate(frames):
        target = times_arr[t] + horizon
        i1 = int(np.searchsorted(times_arr, target))
        if i1 >= len(frames):
            continue
        i0 = max(0, i1 - 1)
        span = times_arr[i1] - times_arr[i0]
        frac = (target - times_arr[i0]) / span if span > 1e-9 else 0.0
        for obj in frame:
            key = f"{horizon:g}"
            pred = obj["preds"].get(key) or obj["preds"].get(str(horizon))
            if pred is None or not obj.get("predictable"):
                continue
            tid = obj["id"]
            a = truth_map[i0].get(tid)
            b = truth_map[i1].get(tid)
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
            cv_err = float(np.hypot(pc[0] - truth[0], pc[1] - truth[1]))

            history = [h for h in obs.get(tid, []) if h[0] <= obj["t"] + 1e-6]
            ctrv = ctrv_predict(history, horizon, window)
            ctrv_err = None
            omega_deg = None
            if ctrv is not None:
                ctrv_err = float(np.hypot(ctrv[0] - truth[0], ctrv[1] - truth[1]))
                omega_deg = ctrv[2]
            # 阻尼转角：只采纳 gain 比例的估计转角
            damped = {}
            for gain in GAINS:
                if gain == 1.0:
                    damped[gain] = ctrv_err
                    continue
                alt = ctrv_predict(history, horizon, window, gain=gain)
                damped[gain] = (float(np.hypot(alt[0] - truth[0], alt[1] - truth[1]))
                                if alt is not None else None)
            rows.append({"cls": obj["cls"], "cv_err": cv_err, "ctrv_err": ctrv_err,
                         "omega": omega_deg, "speed": float(obj.get("speed_px_s", 0.0)),
                         "damped": damped, "video": Path(payload["video"]).name})
    return rows


def main() -> None:
    out_dir = Path(sys.argv[1])
    horizon = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    window = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    min_speed = float(sys.argv[4]) if len(sys.argv) > 4 else 120.0
    rows: list[dict] = []
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        rows.extend(analyze(json.loads(path.read_text(encoding="utf-8")), horizon, window))
    if not rows:
        print("没有样本")
        return

    cv = np.array([r["cv_err"] for r in rows])
    speed = np.array([r["speed"] for r in rows])
    omega = np.array([r["omega"] if r["omega"] is not None else np.nan for r in rows])
    have = ~np.isnan(omega)
    print(f"样本 {len(rows)}（其中 {have.sum()} 个能估出转角速度），梯度 {horizon:g}s，"
          f"估计窗口 {window:g}s")
    print(f"匀速外推误差：中位 {np.median(cv):.1f} px，均值 {cv.mean():.1f} px\n")

    # 只看真的在动的目标：静止目标的航向是噪声，混进来会把分档搅乱
    moving = have & (speed >= min_speed)
    print(f"其中速度 ≥{min_speed:g} px/s 的移动目标：{moving.sum()} 个"
          f"（占全部样本 {moving.mean() * 100:.1f}%）\n")

    bins = [("直行 |ω|<5°/s", 0, 5), ("缓弯 5~15°/s", 5, 15),
            ("转弯 15~40°/s", 15, 40), ("急转 >40°/s", 40, 1e9)]
    print(f"{'分档':<18}{'样本':>7}{'占比':>7}{'匀速中位':>10}{'圆弧中位':>10}"
          f"{'圆弧更好占比':>13}{'占误差总量':>11}")
    total = cv.sum()
    for name, lo, hi in bins:
        mask = moving & (np.abs(omega) >= lo) & (np.abs(omega) < hi)
        if not mask.any():
            continue
        sub_cv = cv[mask]
        sub_ct = np.array([r["ctrv_err"] for r, m in zip(rows, mask)
                           if m and r["ctrv_err"] is not None])
        better = float(np.mean(sub_ct < sub_cv)) * 100 if sub_ct.size == sub_cv.size else float("nan")
        print(f"{name:<18}{mask.sum():>7}{mask.mean() * 100:>6.1f}%"
              f"{np.median(sub_cv):>10.1f}"
              f"{(np.median(sub_ct) if sub_ct.size else float('nan')):>10.1f}"
              f"{better:>12.0f}%{sub_cv.sum() / total * 100:>10.0f}%")

    mask = have & (np.abs(omega) >= 15)
    if mask.any():
        sub_cv = cv[mask]
        sub_ct = np.array([r["ctrv_err"] for r, m in zip(rows, mask)
                           if m and r["ctrv_err"] is not None])
        if sub_ct.size:
            print(f"\n转弯以上合计（|ω|≥15°/s）：{mask.sum()} 个样本，"
                  f"匀速中位 {np.median(sub_cv):.1f} px → 圆弧中位 {np.median(sub_ct):.1f} px")
            print(f"  圆弧更准的样本占 {np.mean(sub_ct < sub_cv) * 100:.0f}%，"
                  f"误差均值 {sub_cv.mean():.1f} → {sub_ct.mean():.1f} px")

    # ---- 阻尼转角扫描：转得越狠，越该采纳多少估计转角？ ----
    print("\n阻尼转角扫描（表内为误差中位数 px，加粗即该档最优）：")
    header = f"{'分档':<18}" + "".join(f"{('gain=' + str(g)):>12}" for g in GAINS)
    print(header)
    for name, lo, hi in bins:
        mask = moving & (np.abs(omega) >= lo) & (np.abs(omega) < hi)
        if not mask.any():
            continue
        cells = []
        for gain in GAINS:
            vals = [r["damped"][gain] for r, m in zip(rows, mask)
                    if m and r["damped"].get(gain) is not None]
            cells.append(np.median(vals) if vals else float("nan"))
        best = int(np.nanargmin(cells))
        line = f"{name:<18}"
        for k, value in enumerate(cells):
            text = f"{value:.1f}"
            line += f"{('*' + text + '*'):>12}" if k == best else f"{text:>12}"
        print(line + f"   （最优 gain={GAINS[best]}）")

    # ---- 可信度门控：机动目标不给长梯度预判，会去掉多少误差？ ----
    print("\n机动门控方案（对 |ω|≥阈值的移动目标，取消 1 秒预判，只留 0.5s 以内）：")
    print(f"{'阈值':<22}{'被拦截样本':>10}{'占全部':>8}{'占移动目标':>11}"
          f"{'剩余中位':>10}{'剩余均值':>10}{'>100px 占比':>12}")
    base_bad = float(np.mean(cv > 100)) * 100
    print(f"{'不门控（现状）':<22}{0:>10}{0.0:>7.1f}%{0.0:>10.1f}%"
          f"{np.median(cv):>10.1f}{cv.mean():>10.1f}{base_bad:>11.1f}%")
    for thr in (15, 25, 40):
        gate = moving & (np.abs(omega) >= thr)
        keep = ~gate
        if not keep.any():
            continue
        kept = cv[keep]
        print(f"{'|ω|≥' + str(thr) + '°/s':<22}{gate.sum():>10}"
              f"{gate.mean() * 100:>7.1f}%{gate.sum() / moving.sum() * 100:>10.1f}%"
              f"{np.median(kept):>10.1f}{kept.mean():>10.1f}"
              f"{np.mean(kept > 100) * 100:>11.1f}%")

    # ---- 偏离预算门控：转弯 h 秒后偏离直线 (v/ω)(1-cos(ωh))，超过预算就不给该梯度 ----
    rad = np.abs(np.radians(omega))
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.where(rad > 1e-6,
                       (speed / np.maximum(rad, 1e-6)) * (1.0 - np.cos(rad * horizon)),
                       0.5 * speed * rad * horizon ** 2)
    dev = np.where(have, dev, 0.0)
    corr = float(np.corrcoef(dev[moving], cv[moving])[0, 1]) if moving.sum() > 10 else float("nan")
    print(f"\n偏离预算门控（转弯在 {horizon:g}s 后偏离直线的像素量，与误差相关性 {corr:.2f}）：")
    print(f"{'预算':<22}{'被拦截样本':>10}{'占移动目标':>11}{'剩余中位':>10}"
          f"{'剩余均值':>10}{'>100px 占比':>12}")
    for budget in (15, 25, 40, 60):
        gate = moving & (dev > budget)
        keep = ~gate
        kept = cv[keep]
        print(f"{f'偏离>{budget}px':<22}{gate.sum():>10}"
              f"{gate.sum() / moving.sum() * 100:>10.1f}%{np.median(kept):>10.1f}"
              f"{kept.mean():>10.1f}{np.mean(kept > 100) * 100:>11.1f}%")


if __name__ == "__main__":
    main()
