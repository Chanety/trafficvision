"""逐单位 / 逐帧 / 逐类别的预判误差分布（含运动方向上的正负号）。

目的：mAP50 这类聚合指标把误差压成一个数，看不出"误差从哪来、长什么样"。
本脚本把预判误差拆回到**每一条轨迹、每一帧、每一个类别**，并把误差在
**运动方向**上保留正负号，便于判断：

  * 误差是"少数单位/少数帧"的问题，还是普遍性问题（看集中度与 ICC）；
  * 分布是重尾还是近高斯（决定用均值还是中位数、要不要门控）；
  * 沿运动方向是系统性超前还是滞后（决定要不要做增益/偏置校正）。

符号约定（重要）
----------------
对每一帧的目标，取它**自身最近若干帧的观测位移方向**作为参考方向 û（只用当前帧
及之前的数据，因果）。令 Δ = 实测未来中心 − 预判中心，则

    e_along = Δ · û      >0：实测比预判更靠前 → 预判**滞后 / 不足（短打）**
                         <0：实测比预判更靠后 → 预判**超前 / 过头（长打）**
    e_cross = Δ · n̂      n̂ = (-û_y, û_x)，即画面坐标下顺时针旋转 90°
                         >0：实测偏在运动方向的**右手侧**（画面顺时针侧）

真值口径：与其他评测一致，取自**同一跟踪器的未来帧观测**（线性插值），
属于自证式真值——它衡量的是"相对跟踪器自身未来观测的误差"，不是绝对真值。
遮挡续推（coasting）出来的位置永远不当真值。

用法：
    python bench/error_distribution.py [输入目录] [输出目录]
默认输入 outputs/nms_recheck（6 段 × 300 帧，NMS 修复后口径），
默认输出 outputs/error_dist/。
"""

from __future__ import annotations

import csv
import json
import sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

HORIZONS = (0.1, 0.5, 1.0, 2.0)
HEAD_WINDOW = 10         # 估算参考方向最多回溯多少帧（自适应取够用的最短基线）
MIN_DIR_PX = 8.0         # 基线位移达到该值即认为方向可信
MIN_DIR_HARD = 4.0       # 兜底下限：连该值都达不到才算"方向不可辨"
MOVING_PX_S = 30.0       # 判定"运动单位"的速度门槛（px/s）

CLASS_ZH = {
    "car": "汽车", "truck": "卡车", "bus": "公交", "nonmotor": "非机动车",
    "person": "行人", "sign": "标志牌", "bollard": "路桩",
    "roadlamp": "路灯", "trafficlight": "信号灯", "rider": "驾驶员",
}
ORDER = ["car", "bus", "truck", "nonmotor", "person", "sign", "bollard"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------- 读取

def load_video(fp: Path):
    payload = json.loads(fp.read_text(encoding="utf-8"))
    frames = payload["frames"]
    fps = float(payload["video_info"]["fps"])
    n = len(frames)

    times = np.full(n, np.nan)
    for i, frame in enumerate(frames):
        if frame:
            times[i] = frame[0]["t"]
    known = np.where(~np.isnan(times))[0]
    if known.size == 0:
        times = np.arange(n) / fps
    else:
        if np.isnan(times[0]):
            times[: known[0]] = times[known[0]] - (known[0] - np.arange(known[0])) / fps
        for i in range(n):
            if np.isnan(times[i]):
                prev = known[known < i]
                if prev.size:
                    times[i] = times[prev[-1]] + (i - prev[-1]) / fps
                else:
                    times[i] = times[known[0]]

    # 真值池：续推出来的位置是推算值，绝不能当真值
    truth_pool = [{o["id"]: o["centroid"] for o in f if not o.get("coasting")}
                  for f in frames]
    # 方向估计只用观测位置
    hist: dict[int, list[int]] = defaultdict(list)
    for i, pool in enumerate(truth_pool):
        for tid in pool:
            hist[tid].append(i)
    return payload, frames, times, truth_pool, dict(hist)


def truth_at(truth_pool, times, t_arr, i_from, tid, h):
    """复刻 pipeline._evaluate 的真值取法（线性插值 + 就近兜底）。"""
    target = times[i_from] + h
    i1 = int(np.searchsorted(t_arr, target))
    if i1 >= len(truth_pool):
        return None
    i0 = max(0, i1 - 1)
    span = times[i1] - times[i0]
    frac = (target - times[i0]) / span if span > 1e-9 else 0.0
    a = truth_pool[i0].get(tid)
    b = truth_pool[i1].get(tid)
    if a is not None and b is not None:
        return (a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac)
    if a is not None and frac < 0.5:
        return a
    if b is not None and frac >= 0.5:
        return b
    return None


# --------------------------------------------------------------------------- 计算

def collect(input_dir: Path):
    rows: list[dict] = []
    for fp in sorted(input_dir.glob("*.json")):
        payload, frames, times, truth_pool, hist = load_video(fp)
        video = Path(payload["video"]).stem
        n = len(frames)
        for i, frame in enumerate(frames):
            if not frame:
                continue
            for obj in frame:
                if not obj.get("predictable"):
                    continue
                preds = obj.get("preds") or {}
                if all(preds.get(str(h)) is None for h in HORIZONS):
                    continue
                tid = obj["id"]
                c_now = obj["centroid"]

                # 参考方向：用观测历史，不用续推位置。
                # 自适应基线：在 [i-HEAD_WINDOW, i-1] 内从最早开始找，取第一个
                # 位移达到 MIN_DIR_PX 的观测——这样既保住了长基线（方向稳），
                # 又不会像"只取最近一帧"那样把大量低速目标误判成方向不可辨。
                u = None
                past = hist.get(tid)
                if past:
                    k = bisect_left(past, i) - 1
                    lo = bisect_left(past, i - HEAD_WINDOW)
                    if k >= 0 and lo <= k:
                        best = None
                        fallback = None
                        for idx in range(lo, k + 1):
                            prev = truth_pool[past[idx]].get(tid)
                            if prev is None:
                                continue
                            dx, dy = c_now[0] - prev[0], c_now[1] - prev[1]
                            norm = float(np.hypot(dx, dy))
                            if fallback is None:
                                fallback = (dx, dy, norm)
                            if norm >= MIN_DIR_PX:
                                best = (dx, dy, norm)
                                break
                        if best is None and fallback is not None and fallback[2] >= MIN_DIR_HARD:
                            best = fallback
                        if best is not None:
                            dx, dy, norm = best
                            u = (dx / norm, dy / norm)

                box_h = float(obj["bbox"][3] - obj["bbox"][1])
                for h in HORIZONS:
                    pred = preds.get(str(h))
                    if pred is None:
                        continue
                    truth = truth_at(truth_pool, times, times, i, tid, h)
                    if truth is None:
                        continue
                    pc = ((pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0)
                    dx, dy = truth[0] - pc[0], truth[1] - pc[1]
                    mag = float(np.hypot(dx, dy))
                    # 实际位移 / 预判位移（沿参考方向投影），用于增益诊断
                    a_true = a_pred = None
                    if u is not None:
                        e_along = dx * u[0] + dy * u[1]
                        e_cross = dx * (-u[1]) + dy * u[0]
                        a_true = (truth[0] - c_now[0]) * u[0] + (truth[1] - c_now[1]) * u[1]
                        a_pred = (pc[0] - c_now[0]) * u[0] + (pc[1] - c_now[1]) * u[1]
                    else:
                        e_along = e_cross = None
                    rows.append({
                        "video": video, "frame": i, "t": float(times[i]), "id": tid,
                        "cls": obj["cls"], "h": h,
                        "coasting": bool(obj.get("coasting")),
                        "speed": float(obj.get("speed_px_s") or 0.0),
                        "turn": obj.get("turn_deg_s"), "maneuver": bool(obj.get("maneuver")),
                        "box_h": box_h, "conf": float(obj.get("conf") or 0.0),
                        "e_along": e_along, "e_cross": e_cross, "mag": mag,
                        "a_true": a_true, "a_pred": a_pred,
                    })
        print(f"  {fp.name}: 累计 {len(rows)} 行")
    return rows


# --------------------------------------------------------------------------- 统计

def stats(a: np.ndarray) -> dict:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    q = np.percentile(a, [25, 50, 75, 90, 95, 99])
    m, s = float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0
    sk = float(((a - m) ** 3).mean() / s ** 3) if s > 0 else 0.0
    ku = float(((a - m) ** 4).mean() / s ** 4) if s > 0 else 0.0
    return {
        "n": int(a.size), "mean": m, "std": s, "q25": float(q[0]), "median": float(q[1]),
        "q75": float(q[2]), "p90": float(q[3]), "p95": float(q[4]), "p99": float(q[5]),
        "skew": sk, "kurt": ku, "mean_over_median": (m / float(q[1]) if q[1] else float("nan")),
    }


def concentration(vals: np.ndarray, groups: np.ndarray, top=(0.01, 0.05, 0.10)) -> dict:
    """误差总量集中在多少比例的样本/单位上。vals 为 |误差|，groups 为所属单位标识。"""
    out = {}
    v = np.sort(vals)[::-1]
    total = v.sum()
    for p in top:
        k = max(1, int(round(p * v.size)))
        out[f"top{p:.0%}_samples_share"] = float(v[:k].sum() / total) if total else 0.0
    # 按单位：先按组求和再排序
    agg: dict = defaultdict(float)
    for g, x in zip(groups, vals):
        agg[g] += x
    arr = np.array(sorted(agg.values(), reverse=True))
    for p in top:
        k = max(1, int(round(p * arr.size)))
        out[f"top{p:.0%}_units_share"] = float(arr[:k].sum() / arr.sum()) if arr.sum() else 0.0
    out["n_units"] = int(arr.size)
    return out


def icc(vals: np.ndarray, groups: np.ndarray) -> dict:
    """轨迹间 vs 轨迹内的方差分解。ICC 高 → 误差是"单位级"的系统偏差。"""
    agg: dict = defaultdict(list)
    for g, x in zip(groups, vals):
        agg[g].append(x)
    grand = vals.mean()
    between = 0.0
    within = 0.0
    n_tot = 0
    k = 0
    for g, xs in agg.items():
        xs = np.asarray(xs, dtype=float)
        if xs.size < 2:
            continue
        k += 1
        between += xs.size * (xs.mean() - grand) ** 2
        within += ((xs - xs.mean()) ** 2).sum()
        n_tot += xs.size
    if k == 0 or (between + within) == 0:
        return {"icc": float("nan"), "tracks": k}
    return {"icc": float(between / (between + within)), "tracks": k,
            "var_between": float(between / max(1, k - 1)),
            "var_within": float(within / max(1, n_tot - k))}


def gain_fit(a_true: np.ndarray, a_pred: np.ndarray) -> dict:
    """最小二乘最优位移增益 g*：最优解 g*=Σ(a_t·a_p)/Σ(a_p²)。"""
    mask = np.isfinite(a_true) & np.isfinite(a_pred) & (np.abs(a_pred) > 1e-6)
    at, ap = a_true[mask], a_pred[mask]
    if at.size < 10:
        return {"n": int(at.size)}
    g = float((at * ap).sum() / (ap * ap).sum())
    base = float(np.abs(at - ap).mean())
    tuned = float(np.abs(at - g * ap).mean())
    ratio = at / np.where(np.abs(ap) > 1e-6, ap, np.nan)
    ratio = ratio[np.isfinite(ratio)]
    return {"n": int(at.size), "gain": g, "mae_base": base, "mae_tuned": tuned,
            "improve_pct": (base - tuned) / base * 100 if base else float("nan"),
            "ratio_median": float(np.median(ratio)) if ratio.size else float("nan"),
            "ratio_q25": float(np.percentile(ratio, 25)) if ratio.size else float("nan"),
            "ratio_q75": float(np.percentile(ratio, 75)) if ratio.size else float("nan")}


def calibration_fit(a_true: np.ndarray, a_pred: np.ndarray) -> dict:
    """带截距的标定拟合 a_true = g·a_pred + c。

    斜率 g<1 → 速度量级被高估（乘性）；截距 c≠0 → 与位移无关的恒定偏置（加性）。
    两者对应完全不同的修正手段，必须分开看。
    """
    mask = np.isfinite(a_true) & np.isfinite(a_pred) & (np.abs(a_pred) > 1e-6)
    at, ap = a_true[mask], a_pred[mask]
    if at.size < 50:
        return {"n": int(at.size)}
    A = np.vstack([ap, np.ones_like(ap)]).T
    coef, *_ = np.linalg.lstsq(A, at, rcond=None)
    g, c = float(coef[0]), float(coef[1])
    pred = A @ coef
    ss_res = float(((at - pred) ** 2).sum())
    ss_tot = float(((at - at.mean()) ** 2).sum())
    return {"n": int(at.size), "slope": g, "intercept": c,
            "r2": 1 - ss_res / ss_tot if ss_tot else float("nan")}


def time_offset(samples: list[dict], h: float) -> dict:
    """把像素偏置换算成等效时间偏置（秒）：e_along / 速度。

    该值随梯度**基本不变** → 是时间轴错位（预测时刻对不上）；
    随梯度**线性增长** → 是速度量级高估（乘性缩放）。
    """
    vals = []
    for r in samples:
        if r["h"] != h or r["e_along"] is None:
            continue
        v = r["speed"]
        if v > 20:
            vals.append(r["e_along"] / v)
    a = np.array(vals)
    if a.size < 20:
        return {"n": int(a.size)}
    return {"n": int(a.size), "median_s": float(np.median(a)),
            "q25_s": float(np.percentile(a, 25)), "q75_s": float(np.percentile(a, 75)),
            "mean_s": float(a.mean()), "h": h}


def wrong_fraction(samples: list[dict], h: float, cls: str | None = None) -> dict:
    """错误位移占比 k = e_along / (速度 × 梯度)。

    k=1 表示该目标外推出去的位移**整个都是错的**（典型是静止结构被赋予虚假速度）；
    k=0.2 表示外推位移里只有两成不到是错的。
    """
    vals = []
    for r in samples:
        if r["h"] != h or r["e_along"] is None or r["speed"] <= 20:
            continue
        if cls is not None and r["cls"] != cls:
            continue
        vals.append(r["e_along"] / (r["speed"] * h))
    a = np.array(vals)
    if a.size < 20:
        return {"n": int(a.size)}
    return {"n": int(a.size), "k_median": float(np.median(a)),
            "k_abs_median": float(np.median(np.abs(a))),
            "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75))}


def bias_persistence(rows, h: float = 1.0, min_n: int = 20):
    """轨迹级偏差在时间上是否稳定：把每条轨迹的样本按时间对半分，
    比较前半段与后半段的中位偏差。稳定 → 可做在线逐单位校正。
    """
    by_track: dict = defaultdict(list)
    for r in rows:
        if r["h"] == h and r["e_along"] is not None:
            by_track[(r["video"], r["id"])].append((r["frame"], r["e_along"]))
    first, second, n_used = [], [], 0
    for xs in by_track.values():
        if len(xs) < min_n:
            continue
        xs.sort()
        half = len(xs) // 2
        first.append(float(np.median([v for _, v in xs[:half]])))
        second.append(float(np.median([v for _, v in xs[half:]])))
        n_used += 1
    first, second = np.asarray(first), np.asarray(second)
    if first.size < 10:
        return {"n_tracks": int(first.size)}
    same_sign = float((np.sign(first) == np.sign(second)).mean())
    corr = float(np.corrcoef(first, second)[0, 1])
    return {"n_tracks": n_used, "corr": corr, "same_sign_frac": same_sign,
            "first": first, "second": second}


# --------------------------------------------------------------------------- 绘图

def _note(ax, txt, loc=0.02):
    ax.text(loc, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))


def _boxplot(ax, data, labels=None, **kw):
    """兼容新旧 matplotlib：labels -> tick_labels。"""
    try:
        return ax.boxplot(data, tick_labels=labels, **kw)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kw)


def fig_along_overall(rows, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, h in zip(axes.ravel(), HORIZONS):
        a = np.array([r["e_along"] for r in rows
                      if r["h"] == h and r["e_along"] is not None])
        if a.size == 0:
            continue
        st = stats(a)
        lim = float(np.percentile(np.abs(a), 99))
        ax.hist(a, bins=80, range=(-lim, lim), color="#4C78A8", alpha=0.85)
        ax.axvline(0, color="0.3", lw=1)
        ax.axvline(st["median"], color="#E45756", lw=1.6, ls="--")
        ax.axvline(st["mean"], color="#F58518", lw=1.6, ls=":")
        pos = float((a > 0).mean() * 100)
        _note(ax, f"n={st['n']}\n中位 {st['median']:.1f}px\n均值 {st['mean']:.1f}px\n"
                  f"P90 |e| {np.percentile(np.abs(a), 90):.1f}px\n"
                  f"偏度 {st['skew']:.2f}  峰度 {st['kurt']:.1f}\n"
                  f"滞后(>0) 占比 {pos:.0f}%")
        ax.set_title(f"梯度 {h:g} s", fontsize=11)
        ax.set_xlabel("沿运动方向误差 e_along (px)　正=滞后　负=超前")
        ax.set_ylabel("样本数")
    fig.suptitle("图1　沿运动方向的有符号误差分布（红虚线=中位，橙点线=均值）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "fig01_along_by_horizon.png", dpi=180)
    plt.close(fig)


def fig_by_class(rows, key, fname, title, xlabel, out: Path, h=1.0):
    data, labels = [], []
    for cls in ORDER:
        a = np.array([r[key] for r in rows
                      if r["h"] == h and r["cls"] == cls and r[key] is not None])
        if a.size >= 20:
            data.append(a)
            labels.append(f"{CLASS_ZH.get(cls, cls)}\n(n={a.size})")
    if not data:
        return
    fig, ax = plt.subplots(figsize=(10, 5.2))
    lo = float(np.percentile(np.concatenate(data), 1))
    hi = float(np.percentile(np.concatenate(data), 99))
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    parts = ax.violinplot(data, showextrema=False, widths=0.85)
    for b in parts["bodies"]:
        b.set_facecolor("#4C78A8")
        b.set_alpha(0.35)
    bp = _boxplot(ax, data, widths=0.22, showfliers=False, patch_artist=True,
                  medianprops=dict(color="#E45756", lw=1.8))
    for p in bp["boxes"]:
        p.set_facecolor("white")
    ax.axhline(0, color="0.4", lw=1)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(xlabel)
    txt = []
    for cls, a in zip([c for c in ORDER if np.array([r[key] for r in rows if r["h"] == h
                                                     and r["cls"] == c and r[key] is not None]).size >= 20],
                      data):
        txt.append(f"{CLASS_ZH.get(cls, cls)}: 中位 {np.median(a):+.1f}")
    _note(ax, "　".join(txt))
    fig.tight_layout()
    fig.savefig(out / fname, dpi=180)
    plt.close(fig)


def fig_mag_by_class(rows, out: Path):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
    for ax, h in zip(axes, HORIZONS):
        data, labels = [], []
        for cls in ORDER:
            a = np.array([r["mag"] for r in rows if r["h"] == h and r["cls"] == cls])
            if a.size >= 20:
                data.append(a)
                labels.append(CLASS_ZH.get(cls, cls))
        _boxplot(ax, data, showfliers=False, widths=0.6,
                 medianprops=dict(color="#E45756", lw=1.6))
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=60, fontsize=8)
        ax.set_title(f"{h:g} s", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("误差距离 |Δ| (px)")
    fig.suptitle("图2　误差幅值按类别分布（各梯度，箱线=四分位，红=中位）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig02_mag_by_class.png", dpi=180)
    plt.close(fig)


def fig_unit_level(rows, out: Path, h=1.0):
    per_track: dict = defaultdict(list)
    for r in rows:
        if r["h"] == h and r["e_along"] is not None:
            per_track[(r["video"], r["id"])].append(r["e_along"])
    med, sd, sizes = [], [], []
    for xs in per_track.values():
        if len(xs) >= 5:
            xs = np.asarray(xs)
            med.append(np.median(xs))
            sd.append(xs.std(ddof=1))
            sizes.append(len(xs))
    if not med:
        return
    med, sd = np.asarray(med), np.asarray(sd)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].hist(med, bins=60, color="#4C78A8")
    axes[0].axvline(0, color="0.3")
    axes[0].set_title("每条轨迹的 e_along 中位数")
    axes[0].set_xlabel("px"); axes[0].set_ylabel("轨迹数")
    _note(axes[0], f"轨迹数 {med.size}\n中位 {np.median(med):.1f}px\n"
                   f"IQR [{np.percentile(med, 25):.1f}, {np.percentile(med, 75):.1f}]")
    axes[1].hist(sd, bins=60, color="#72B7B2")
    axes[1].set_title("每条轨迹内 e_along 的标准差")
    axes[1].set_xlabel("px"); axes[1].set_ylabel("轨迹数")
    _note(axes[1], f"中位 {np.median(sd):.1f}px")
    axes[2].scatter(med, sd, s=14, alpha=0.5, color="#F58518")
    axes[2].set_title("单位级偏差 vs 单位内离散")
    axes[2].set_xlabel("轨迹中位偏差 (px)"); axes[2].set_ylabel("轨迹内标准差 (px)")
    fig.suptitle(f"图3　误差是单位级还是帧级（梯度 {h:g} s）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig03_unit_level.png", dpi=180)
    plt.close(fig)


def fig_gain(rows, out: Path, h=1.0):
    ap = np.array([r["a_pred"] for r in rows if r["h"] == h and r["a_pred"] is not None])
    at = np.array([r["a_true"] for r in rows if r["h"] == h and r["a_true"] is not None])
    if ap.size < 50:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    lim = float(np.percentile(np.abs(ap), 99))
    sub = (np.abs(ap) <= lim) & (np.abs(at) <= lim)
    axes[0].scatter(ap[sub], at[sub], s=5, alpha=0.15, color="#4C78A8")
    axes[0].plot([0, lim], [0, lim], color="0.4", lw=1, label="理想 y=x")
    g = float((at * ap).sum() / (ap * ap).sum())
    cal = calibration_fit(at, ap)
    xs = np.array([0.0, lim])
    axes[0].plot(xs, g * xs, color="#E45756", lw=1.6, ls="--",
                 label=f"过原点增益 {g:.3f}")
    if "slope" in cal:
        axes[0].plot(xs, cal["slope"] * xs + cal["intercept"], color="#F58518", lw=2.0,
                     label=f"带截距 a_t={cal['slope']:.3f}·a_p{cal['intercept']:+.1f}")
    axes[0].set_xlim(0, lim); axes[0].set_ylim(min(-lim * 0.2, 0), lim)
    axes[0].legend(fontsize=9)
    axes[0].set_xlabel("预判位移（沿运动方向, px）")
    axes[0].set_ylabel("实测位移（沿运动方向, px）")
    if "slope" in cal:
        _note(axes[0], f"斜率 {cal['slope']:.3f}（<1=速度高估）\n"
                       f"截距 {cal['intercept']:+.1f} px（≠0=恒定偏置）\n"
                       f"R² {cal['r2']:.3f}")

    ratio = at / np.where(np.abs(ap) > 1e-6, ap, np.nan)
    ratio = ratio[np.isfinite(ratio)]
    ratio = ratio[(ratio > -1) & (ratio < 3)]
    axes[1].hist(ratio, bins=90, color="#54A24B")
    axes[1].axvline(1.0, color="0.3", lw=1, label="1.0")
    axes[1].axvline(np.median(ratio), color="#E45756", lw=1.8, ls="--",
                    label=f"中位 {np.median(ratio):.3f}")
    axes[1].legend(fontsize=9)
    axes[1].set_xlabel("实测位移 / 预判位移")
    axes[1].set_ylabel("样本数")
    axes[1].set_title("比值分布：1.0 左侧=预判过头")
    # 偏差是否随预判位移等比放大（乘性）还是恒定（加性）
    qs = np.percentile(np.abs(ap), np.linspace(0, 100, 11))
    cx, cy = [], []
    for a, b in zip(qs[:-1], qs[1:]):
        sel = (np.abs(ap) >= a) & (np.abs(ap) <= b)
        if sel.sum() >= 10:
            cx.append(float(np.mean(np.abs(ap[sel]))))
            cy.append(float(np.median((at - ap)[sel])))
    axes[2].plot(cx, cy, "o-", color="#B279A2")
    axes[2].axhline(0, color="0.5", lw=1)
    axes[2].set_xlabel("预判位移绝对值 (px)")
    axes[2].set_ylabel("该档内 e_along 中位数 (px)")
    axes[2].set_title("偏差随位移增长：直线=乘性，水平=加性")
    fig.suptitle(f"图4　位移尺度诊断（梯度 {h:g} s）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig04_gain.png", dpi=180)
    plt.close(fig)


def fig_time_offset(rows, out: Path):
    """等效时间偏置：像素偏差 ÷ 自身速度，看它随梯度是恒定还是增长。"""
    hs = list(HORIZONS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    med = [time_offset(rows, h).get("median_s", np.nan) for h in hs]
    q25 = [time_offset(rows, h).get("q25_s", np.nan) for h in hs]
    q75 = [time_offset(rows, h).get("q75_s", np.nan) for h in hs]
    axes[0].fill_between(hs, np.abs(q25), np.abs(q75), color="#4C78A8", alpha=0.25,
                         label="四分位区间（取绝对值）")
    axes[0].plot(hs, np.abs(med), "o-", color="#E45756", label="中位")
    axes[0].axhline(0, color="0.5", lw=1)
    axes[0].set_xlabel("梯度 (s)"); axes[0].set_ylabel("等效时间偏置 |e_along| / 速度 (s)")
    axes[0].set_title("恒定 → 时间轴错位；随梯度增长 → 速度量级高估")
    axes[0].legend(fontsize=9)
    for cls in ORDER:
        ys = []
        for h in hs:
            vals = [r["e_along"] / r["speed"] for r in rows
                    if r["h"] == h and r["cls"] == cls and r["e_along"] is not None
                    and r["speed"] > 20]
            ys.append(abs(float(np.median(vals))) if len(vals) >= 20 else np.nan)
        if np.isfinite(ys).sum() >= 2:
            axes[1].plot(hs, ys, "o-", label=CLASS_ZH.get(cls, cls), lw=1.4, ms=4)
    axes[1].set_xlabel("梯度 (s)"); axes[1].set_ylabel("等效时间偏置中位 (s)")
    axes[1].set_title("分类型：偏置的量级与随梯度的走势")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("图11　等效时间偏置诊断（正=预判超前）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig11_time_offset.png", dpi=180)
    plt.close(fig)


def fig_bias_persistence(rows, out: Path, h=1.0):
    res = bias_persistence(rows, h)
    if "first" not in res:
        return
    first, second = res["first"], res["second"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lim = float(np.percentile(np.concatenate([np.abs(first), np.abs(second)]), 98))
    axes[0].scatter(first, second, s=22, alpha=0.6, color="#4C78A8")
    axes[0].axhline(0, color="0.5", lw=1)
    axes[0].axvline(0, color="0.5", lw=1)
    lo, hi = -lim, lim
    axes[0].plot([lo, hi], [lo, hi], color="#E45756", lw=1.4, ls="--", label="完全稳定 y=x")
    axes[0].set_xlim(lo, hi); axes[0].set_ylim(lo, hi)
    axes[0].set_xlabel(f"轨迹前半段 e_along 中位 (px)")
    axes[0].set_ylabel(f"轨迹后半段 e_along 中位 (px)")
    axes[0].set_title("逐轨迹偏差：前半段 vs 后半段")
    axes[0].legend(fontsize=9)
    _note(axes[0], f"轨迹数 {res['n_tracks']}\n相关系数 {res['corr']:+.3f}\n"
                   f"前后同号比例 {res['same_sign_frac'] * 100:.0f}%")
    # 错误位移占比 k
    hs = list(HORIZONS)
    for cls in ORDER:
        ys = []
        for hh in hs:
            w = wrong_fraction(rows, hh, cls)
            ys.append(w.get("k_abs_median", np.nan))
        if np.isfinite(ys).sum() >= 2:
            axes[1].plot(hs, ys, "o-", label=CLASS_ZH.get(cls, cls), lw=1.5, ms=4)
    axes[1].axhline(1.0, color="0.5", lw=1, ls=":")
    axes[1].text(0.15, 1.02, "k=1：外推位移全部是错的", fontsize=8, color="0.35")
    axes[1].set_xlabel("梯度 (s)"); axes[1].set_ylabel("错误位移占比 k = |e_along| / (速度×梯度)")
    axes[1].set_title("分类型：外推位移里有多大比例是错的")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle(f"图12　逐单位偏差的稳定性与错误位移占比（梯度 {h:g} s）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig12_bias_persistence.png", dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------- 按类别独立成页

def _hist_panel(ax, a, title, xlabel, color="#4C78A8"):
    a = a[np.isfinite(a)]
    if a.size == 0:
        ax.set_title(title); return
    st = stats(a)
    lim = float(np.percentile(np.abs(a), 99)) or 1.0
    ax.hist(a, bins=60, range=(-lim, lim), color=color, alpha=0.88)
    ax.axvline(0, color="0.35", lw=1)
    ax.axvline(st["median"], color="#E45756", lw=1.7, ls="--")
    ax.axvline(st["mean"], color="#F58518", lw=1.5, ls=":")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    _note(ax, f"n={st['n']}\n中位 {st['median']:+.1f}\n均值 {st['mean']:+.1f}\n"
              f"P90|·| {np.percentile(np.abs(a), 90):.1f}\n偏度 {st['skew']:+.2f}\n"
              f"峰度 {st['kurt']:.1f}")


def fig_class_sheets(rows, out: Path):
    """每个类别一张独立图页：四梯度的有符号误差分布 + 横向 + 幅值 + 单位级 + 帧级。"""
    cls_dir = out / "按类别"
    cls_dir.mkdir(parents=True, exist_ok=True)
    md = ["# 按类别独立的误差分布统计", "",
          "每个类别一张图页（`按类别/<类别>.png`）；下表是同一批数字。",
          "符号：e_along>0 滞后（短打），<0 超前（长打）。", ""]
    for cls in ORDER:
        sub = [r for r in rows if r["cls"] == cls]
        if len(sub) < 100:
            continue
        zh = CLASS_ZH.get(cls, cls)
        fig, axes = plt.subplots(2, 4, figsize=(19, 8.6))

        # 第一行：四个梯度的 e_along 各自独立
        for ax, h in zip(axes[0], HORIZONS):
            a = np.array([r["e_along"] for r in sub
                          if r["h"] == h and r["e_along"] is not None])
            _hist_panel(ax, a, f"{zh}　梯度 {h:g} s　e_along 分布",
                        "e_along (px)　正=滞后　负=超前")

        # 第二行
        ax = axes[1][0]
        for h, c in zip(HORIZONS, ["#4C78A8", "#54A24B", "#F58518", "#B279A2"]):
            a = np.array([r["e_cross"] for r in sub
                          if r["h"] == h and r["e_cross"] is not None])
            a = a[np.isfinite(a)]
            if a.size < 20:
                continue
            lim = float(np.percentile(np.abs(a), 99)) or 1.0
            ax.hist(a, bins=60, range=(-lim, lim), histtype="step", lw=1.3,
                    color=c, label=f"{h:g} s")
        ax.axvline(0, color="0.35", lw=1)
        ax.set_title("横向误差 e_cross（四梯度叠加）", fontsize=10)
        ax.set_xlabel("e_cross (px)　正=运动方向右手侧", fontsize=9)
        ax.legend(fontsize=8)

        ax = axes[1][1]
        data = [np.array([r["mag"] for r in sub if r["h"] == h]) for h in HORIZONS]
        _boxplot(ax, data, [f"{h:g} s" for h in HORIZONS], showfliers=False,
                 medianprops=dict(color="#E45756", lw=1.6))
        ax.set_title("误差幅值 |Δ| 分梯度", fontsize=10)
        ax.set_ylabel("px", fontsize=9)
        ax.set_ylim(bottom=0)

        ax = axes[1][2]
        per_track: dict = defaultdict(list)
        for r in sub:
            if r["h"] == 1.0 and r["e_along"] is not None:
                per_track[(r["video"], r["id"])].append(r["e_along"])
        med = np.array([np.median(v) for v in per_track.values() if len(v) >= 5])
        if med.size:
            st = stats(med)
            if med.size >= 12:
                lim = float(np.percentile(np.abs(med), 99)) or 1.0
                ax.hist(med, bins=60, range=(-lim, lim), color="#54A24B", alpha=0.9)
            else:
                # 单位太少时直方图退化成几根细柱，改用散点把每个单位画清楚
                rng = np.random.default_rng(0)
                ax.scatter(med, rng.uniform(0.2, 0.8, med.size), s=60,
                           color="#54A24B", alpha=0.85, edgecolor="white")
                ax.set_ylim(0, 1)
                ax.set_yticks([])
            ax.axvline(0, color="0.35", lw=1)
            ax.axvline(st["median"], color="#E45756", lw=1.7, ls="--")
            _note(ax, f"单位数 {st['n']}\n中位 {st['median']:+.1f}px\n"
                      f"IQR [{st['q25']:+.1f}, {st['q75']:+.1f}]")
        ax.set_title("每个运动单位的偏差（1 s，逐轨迹中位）", fontsize=10)
        ax.set_xlabel("轨迹中位 e_along (px)", fontsize=9)

        ax = axes[1][3]
        for video in sorted({r["video"] for r in sub}):
            fr = sorted({r["frame"] for r in sub if r["video"] == video and r["h"] == 1.0})
            if not fr:
                continue
            acc = defaultdict(list)
            for r in sub:
                if r["video"] == video and r["h"] == 1.0:
                    acc[r["frame"]].append(r["mag"])
            ax.plot(fr, [float(np.mean(acc[f])) for f in fr], lw=1.0,
                    label=video[-6:], alpha=0.85)
        ax.set_title("逐帧平均误差（1 s，每段一条）", fontsize=10)
        ax.set_xlabel("帧号", fontsize=9)
        ax.set_ylabel("px", fontsize=9)
        ax.legend(fontsize=7, ncol=2)

        fig.suptitle(f"{zh}（{cls}）　预判误差分布总览　"
                     f"样本 {len(sub)}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(cls_dir / f"{zh}_{cls}.png", dpi=170)
        plt.close(fig)

        md += [f"## {zh}（{cls}）", "",
               "| 梯度 | n | \\|Δ\\| 中位 | \\|Δ\\| P90 | e_along 中位 | e_along 均值 | "
               "e_cross 中位 | 滞后占比 | 偏度 | 峰度 | 错误位移占比 k |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for h in HORIZONS:
            s = [r for r in sub if r["h"] == h]
            am = np.array([r["mag"] for r in s])
            ae = np.array([r["e_along"] for r in s if r["e_along"] is not None])
            ac = np.array([r["e_cross"] for r in s if r["e_cross"] is not None])
            w = wrong_fraction(rows, h, cls)
            if am.size == 0:
                continue
            md.append(
                f"| {h:g} s | {am.size} | {np.median(am):.1f} | "
                f"{np.percentile(am, 90):.1f} | "
                + (f"{np.median(ae):+.1f} | {ae.mean():+.1f} | " if ae.size else "— | — | ")
                + (f"{np.median(ac):+.1f} | " if ac.size else "— | ")
                + (f"{(ae > 0).mean() * 100:.0f}% | {stats(ae)['skew']:+.2f} | "
                   f"{stats(ae)['kurt']:.1f} |" if ae.size else "— | — | — |")
                + (f" {w['k_abs_median']:.2f} |" if "k_abs_median" in w else " — |"))
        pers = bias_persistence(sub, 1.0, min_n=10)
        if "corr" in pers:
            md += ["", f"- 逐单位偏差稳定性（1 s，{pers['n_tracks']} 条轨迹）："
                       f"相关性 {pers['corr']:+.3f}，前后同号 {pers['same_sign_frac'] * 100:.0f}%"]
        md.append("")
    (cls_dir / "统计.md").write_text("\n".join(md), encoding="utf-8")


def fig_vs_horizon(rows, out: Path):
    hs = list(HORIZONS)
    med, p90, alng, skew = [], [], [], []
    for h in hs:
        mg = np.array([r["mag"] for r in rows if r["h"] == h])
        e = np.array([r["e_along"] for r in rows if r["h"] == h and r["e_along"] is not None])
        med.append(np.median(mg)); p90.append(np.percentile(mg, 90))
        alng.append(np.median(e)); skew.append(stats(e)["skew"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].plot(hs, med, "o-", label="中位 |Δ|")
    axes[0].plot(hs, p90, "s--", label="P90 |Δ|")
    axes[0].plot(hs, med, "o", color="#4C78A8")
    axes[0].set_xlabel("梯度 (s)"); axes[0].set_ylabel("px"); axes[0].legend(fontsize=9)
    axes[0].set_title("误差随梯度增长")
    ref = np.array(med) / med[0] * hs[0]
    axes[0].plot(hs, [med[0] * h / hs[0] for h in hs], ":", color="0.5",
                 label="线性外推参考")
    axes[0].legend(fontsize=9)
    axes[1].plot(hs, alng, "o-", color="#E45756")
    axes[1].axhline(0, color="0.5", lw=1)
    axes[1].set_xlabel("梯度 (s)"); axes[1].set_ylabel("沿运动方向中位误差 (px)")
    axes[1].set_title("滞后/超前随梯度变化（正=滞后）")
    axes[2].plot(hs, skew, "o-", color="#B279A2")
    axes[2].axhline(0, color="0.5", lw=1)
    axes[2].set_xlabel("梯度 (s)"); axes[2].set_ylabel("偏度")
    axes[2].set_title("分布偏斜随梯度变化")
    fig.suptitle("图5　梯度依赖：误差是否只是速度乘以时间", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig05_vs_horizon.png", dpi=180)
    plt.close(fig)


def fig_tail(rows, out: Path, h=1.0):
    e = np.array([r["e_along"] for r in rows if r["h"] == h and r["e_along"] is not None])
    mg = np.array([r["mag"] for r in rows if r["h"] == h])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    s = np.sort(mg)[::-1]
    p = np.arange(1, s.size + 1) / s.size
    axes[0].loglog(s, p, color="#4C78A8", lw=1.4)
    axes[0].set_xlabel("误差距离 (px)"); axes[0].set_ylabel("P(误差 > x)")
    axes[0].set_title("幅值 CCDF（对数-对数，直线=重尾）")
    # 高斯 vs 拉普拉斯拟合对比
    from math import log, pi, sqrt
    b = float(np.mean(np.abs(e - np.median(e))))
    ll_lap = float(np.sum(-np.log(2 * b) - np.abs(e - np.median(e)) / b)) if b > 0 else -np.inf
    sd = float(e.std(ddof=1))
    ll_gau = float(np.sum(-0.5 * np.log(2 * pi * sd ** 2) - (e - e.mean()) ** 2 / (2 * sd ** 2)))
    axes[1].hist(e, bins=100, density=True, color="#BAB0AC", alpha=0.8)
    xs = np.linspace(e.min(), e.max(), 400)
    axes[1].plot(xs, np.exp(-np.abs(xs - np.median(e)) / b) / (2 * b), color="#E45756",
                 lw=1.8, label=f"拉普拉斯 (b={b:.1f})")
    axes[1].plot(xs, np.exp(-(xs - e.mean()) ** 2 / (2 * sd ** 2)) / (sd * sqrt(2 * pi)),
                 color="#4C78A8", lw=1.8, label=f"高斯 (σ={sd:.1f})")
    axes[1].legend(fontsize=9)
    axes[1].set_xlabel("e_along (px)"); axes[1].set_ylabel("密度")
    axes[1].set_title("分布形态：拉普拉斯 vs 高斯")
    _note(axes[1], f"对数似然\n拉普拉斯 {ll_lap:,.0f}\n高斯 {ll_gau:,.0f}\n"
                   f"（越大越好）\n峰度 {stats(e)['kurt']:.1f}")
    fig.suptitle(f"图6　尾部与分布形态（梯度 {h:g} s）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig06_tail_shape.png", dpi=180)
    plt.close(fig)


def fig_factors(rows, out: Path, h=1.0):
    sub = [r for r in rows if r["h"] == h and not r["coasting"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # 速度分档
    sp = np.array([r["speed"] for r in sub])
    mg = np.array([r["mag"] for r in sub])
    edges = [0, 60, 120, 200, 320, 500, 1e9]
    names = ["<60", "60~120", "120~200", "200~320", "320~500", ">500"]
    groups = [[] for _ in names]
    for x, y in zip(sp, mg):
        for k in range(len(names)):
            if edges[k] <= x < edges[k + 1]:
                groups[k].append(y)
                break
    _boxplot(axes[0], [g for g in groups if g],
             [n for n, g in zip(names, groups) if g], showfliers=False,
             medianprops=dict(color="#E45756"))
    axes[0].set_xlabel("目标速度 (px/s)"); axes[0].set_ylabel("误差距离 |Δ| (px)")
    axes[0].set_title("速度")
    # 转角
    tn = np.array([r["turn"] if r["turn"] is not None else np.nan for r in sub], dtype=float)
    edges = [0, 5, 10, 20, 40, 1e9]
    names = ["<5", "5~10", "10~20", "20~40", ">40"]
    groups = [[] for _ in names]
    for x, y in zip(tn, mg):
        if not np.isfinite(x):
            continue
        for k in range(len(names)):
            if edges[k] <= x < edges[k + 1]:
                groups[k].append(y)
                break
    _boxplot(axes[1], [g for g in groups if g],
             [n for n, g in zip(names, groups) if g], showfliers=False,
             medianprops=dict(color="#E45756"))
    axes[1].set_xlabel("转角速度 (°/s)"); axes[1].set_title("机动")
    # 框高（近远）
    bh = np.array([r["box_h"] for r in sub])
    qs = np.percentile(bh, [0, 20, 40, 60, 80, 100])
    groups, names = [], []
    for k in range(5):
        lo, hi = qs[k], qs[k + 1]
        sel = (bh >= lo) & (bh <= hi) if k == 4 else (bh >= lo) & (bh < hi)
        groups.append(mg[sel])
        names.append(f"{lo:.0f}~{hi:.0f}")
    _boxplot(axes[2], groups, names, showfliers=False,
             medianprops=dict(color="#E45756"))
    axes[2].set_xlabel("目标框高 (px)——越大越近")
    axes[2].set_title("远近 / 尺度")
    fig.suptitle(f"图7　三类因素对误差的影响（{h:g} s，已排除续推样本）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig07_factors.png", dpi=180)
    plt.close(fig)


def fig_per_frame(rows, out: Path, h=1.0):
    per_frame: dict = defaultdict(list)
    for r in rows:
        if r["h"] == h:
            per_frame[(r["video"], r["frame"])].append(r["mag"])
    videos = sorted({v for v, _ in per_frame})
    pick = videos[:3] if len(videos) >= 3 else videos
    fig, axes = plt.subplots(len(pick), 1, figsize=(11, 2.4 * len(pick)), sharex=False)
    if len(pick) == 1:
        axes = [axes]
    for ax, v in zip(axes, pick):
        fr = sorted(f for vv, f in per_frame if vv == v)
        ys = [float(np.mean(per_frame[(v, f)])) for f in fr]
        p90 = [float(np.percentile(per_frame[(v, f)], 90)) for f in fr]
        ax.plot(fr, ys, lw=1.1, color="#4C78A8", label="帧内平均 |Δ|")
        ax.plot(fr, p90, lw=0.9, color="#F58518", alpha=0.7, label="帧内 P90")
        ax.set_title(v, fontsize=10)
        ax.set_ylabel("px")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("帧号")
    fig.suptitle(f"图8　逐帧误差时间序列（{h:g} s）——突发还是普遍", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig08_per_frame.png", dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------- 主流程

def main() -> None:
    in_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "nms_recheck"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs" / "error_dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输入 {in_dir}\n输出 {out_dir}")
    rows = collect(in_dir)
    print(f"总样本 {len(rows)}")

    # 原始长表
    with (out_dir / "rows.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fig_along_overall(rows, out_dir)
    fig_by_class(rows, "e_along", "fig09_along_by_class.png",
                 "图9　沿运动方向的有符号误差按类别（1 s）",
                 "e_along (px)　正=滞后　负=超前", out_dir)
    fig_by_class(rows, "e_cross", "fig10_cross_by_class.png",
                 "图10　垂直运动方向的有符号误差按类别（1 s）",
                 "e_cross (px)　正=运动方向右手侧", out_dir)
    fig_mag_by_class(rows, out_dir)
    fig_unit_level(rows, out_dir)
    fig_gain(rows, out_dir)
    fig_vs_horizon(rows, out_dir)
    fig_tail(rows, out_dir)
    fig_factors(rows, out_dir)
    fig_per_frame(rows, out_dir)
    fig_time_offset(rows, out_dir)
    fig_bias_persistence(rows, out_dir)
    fig_class_sheets(rows, out_dir)

    # ---- 文字结论
    lines = ["# 预判误差分布分析（逐单位 / 逐帧 / 逐类别，含方向正负号）", "",
             f"数据源：`{in_dir}`（6 段 × 300 帧）", "",
             "符号：e_along>0 表示预判滞后（短打）；e_along<0 表示预判超前（长打）。", ""]
    for h in HORIZONS:
        sub = [r for r in rows if r["h"] == h]
        ok = [r for r in sub if r["e_along"] is not None]
        lines += [f"## 方向可辨覆盖率（梯度 {h:g} s）：{len(ok)}/{len(sub)} "
                  f"= {len(ok) / len(sub) * 100:.1f}%", ""]
        lines.append("| 类别 | 样本 | 方向可用 | 可用率 | 方向可用样本的速度中位 (px/s) |")
        lines.append("|---|---|---|---|---|")
        for cls in ORDER:
            a = [r for r in sub if r["cls"] == cls]
            b = [r for r in a if r["e_along"] is not None]
            if not a:
                continue
            sp = np.median([r["speed"] for r in b]) if b else float("nan")
            lines.append(f"| {CLASS_ZH.get(cls, cls)} | {len(a)} | {len(b)} | "
                         f"{len(b) / len(a) * 100:.0f}% | {sp:.0f} |")
        lines.append("")
        sp_all = np.array([r["speed"] for r in sub])
        lines.append(f"全部样本速度分位：P10 {np.percentile(sp_all, 10):.0f} / "
                     f"中位 {np.median(sp_all):.0f} / P90 {np.percentile(sp_all, 90):.0f} px/s")
        lines.append("")
    for h in HORIZONS:
        sub = [r for r in rows if r["h"] == h]
        e = np.array([r["e_along"] for r in sub if r["e_along"] is not None])
        mg = np.array([r["mag"] for r in sub])
        st, sm = stats(e), stats(mg)
        if st["n"] == 0:
            continue
        lines += [f"## 梯度 {h:g} s", "",
                  f"- 样本 {sm['n']}，其中方向可用 {st['n']}（静止/方向不可辨 "
                  f"{sm['n'] - st['n']}）",
                  f"- 误差幅值 |Δ|：中位 {sm['median']:.1f} px，均值 {sm['mean']:.1f} px，"
                  f"P90 {sm['p90']:.1f}，P99 {sm['p99']:.1f}，均值/中位 {sm['mean_over_median']:.2f}",
                  f"- 沿运动方向 e_along：中位 {st['median']:+.1f} px，均值 {st['mean']:+.1f} px，"
                  f"IQR [{st['q25']:+.1f}, {st['q75']:+.1f}]，偏度 {st['skew']:+.2f}，峰度 {st['kurt']:.1f}",
                  f"- 滞后占比 {(e > 0).mean() * 100:.1f}%，|e_along| 中位 "
                  f"{np.median(np.abs(e)):.1f} px",
                  f"- ICC（轨迹间方差占比）{icc(e, [(r['video'], r['id']) for r in sub if r['e_along'] is not None])['icc']:.3f}",
                  f"- 增益拟合 {gain_fit(np.array([r['a_true'] for r in sub if r['a_true'] is not None]), np.array([r['a_pred'] for r in sub if r['a_pred'] is not None]))}",
                  ""]
        # 遮挡续推样本单独看：那里的"当前位置"是推算出来的，不能与实测混在一起
        for tag, sel in (("实测帧", False), ("遮挡续推帧", True)):
            s2 = [r for r in sub if r["coasting"] is sel]
            if not s2:
                continue
            m2, e2 = np.array([r["mag"] for r in s2]), np.array(
                [r["e_along"] for r in s2 if r["e_along"] is not None])
            lines.append(f"- [{tag}] n={len(s2)}，|Δ| 中位 {np.median(m2):.1f} px，"
                         + (f"e_along 中位 {np.median(e2):+.1f} px，滞后占比 "
                            f"{(e2 > 0).mean() * 100:.0f}%" if e2.size else "e_along 无"))
        lines.append("")
        lines.append("| 类别 | n | \\|Δ\\| 中位 | \\|Δ\\| P90 | e_along 中位 | e_cross 中位 | 滞后占比 | 偏度 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for cls in ORDER:
            a = [r for r in sub if r["cls"] == cls]
            if len(a) < 20:
                continue
            am = np.array([r["mag"] for r in a])
            ae = np.array([r["e_along"] for r in a if r["e_along"] is not None])
            ac = np.array([r["e_cross"] for r in a if r["e_cross"] is not None])
            lines.append(
                f"| {CLASS_ZH.get(cls, cls)} | {len(a)} | {np.median(am):.1f} | "
                f"{np.percentile(am, 90):.1f} | "
                + (f"{np.median(ae):+.1f} | " if ae.size else "— | ")
                + (f"{np.median(ac):+.1f} | " if ac.size else "— | ")
                + (f"{(ae > 0).mean() * 100:.0f}% | {stats(ae)['skew']:+.2f} |"
                   if ae.size else "— | — |"))
        lines.append("")
    lines += ["## 集中度（梯度 1 s）", ""]
    sub = [r for r in rows if r["h"] == 1.0]
    mg = np.array([r["mag"] for r in sub])
    lines.append(f"- 按样本：{concentration(mg, [r['video'] for r in sub])}")
    lines.append(f"- 按轨迹：{concentration(mg, [(r['video'], r['id']) for r in sub])}")
    lines.append(f"- 按帧：{concentration(mg, [(r['video'], r['frame']) for r in sub])}")
    lines += ["", "## 偏差性质诊断（决定用哪种修正手段）", "",
              "| 梯度 | 带截距标定斜率 | 截距 (px) | R² | 过原点增益 | 过原点增益可改进 | 等效时间偏置中位 (s) |",
              "|---|---|---|---|---|---|---|"]
    for h in HORIZONS:
        s = [r for r in rows if r["h"] == h]
        at = np.array([r["a_true"] for r in s if r["a_true"] is not None])
        ap = np.array([r["a_pred"] for r in s if r["a_pred"] is not None])
        cal = calibration_fit(at, ap)
        gf = gain_fit(at, ap)
        to = time_offset(rows, h)
        if "slope" not in cal:
            continue
        lines.append(f"| {h:g} s | {cal['slope']:.3f} | {cal['intercept']:+.1f} | "
                     f"{cal['r2']:.3f} | {gf['gain']:.3f} | {gf['improve_pct']:+.2f}% | "
                     f"{to.get('median_s', float('nan')):+.3f} |")
    lines += ["", "读法：**斜率<1** 表示沿运动方向的位移被系统性放大（速度高估）；"
                  "**截距≠0** 表示存在与位移无关的恒定偏移；"
                  "**等效时间偏置随梯度基本不变** → 时间轴错位，"
                  "**随梯度线性增长** → 速度尺度问题。", ""]
    lines += ["", "## 错误位移占比 k = |e_along| / (速度 × 梯度)", "",
              "k=1 表示该目标外推出去的位移整个都是错的；k 越小说明外推量级越接近实际。", "",
              "| 类别 | 0.1 s | 0.5 s | 1 s | 2 s |", "|---|---|---|---|---|"]
    for cls in ORDER:
        row = [f"| {CLASS_ZH.get(cls, cls)} "]
        for h in HORIZONS:
            w = wrong_fraction(rows, h, cls)
            row.append(f"| {w.get('k_abs_median', float('nan')):.2f} ")
        lines.append("".join(row) + "|")
    lines.append("")
    pers = bias_persistence(rows, 1.0)
    if "corr" in pers:
        lines += ["## 逐轨迹偏差的稳定性（1 s）", "",
                  f"- 参与统计的轨迹 {pers['n_tracks']} 条（每条 ≥20 个样本）",
                  f"- 前半段与后半段偏差的相关系数 {pers['corr']:+.3f}",
                  f"- 前后半段同号比例 {pers['same_sign_frac'] * 100:.0f}%",
                  "",
                  "相关系数高、同号比例高 → 偏差是**该单位自身的稳定属性**，"
                  "可用在线逐单位校正；接近 0 则说明前半段的偏差预测不了后半段，"
                  "只能靠更稳健的估计器或门控。", ""]
    (out_dir / "分析结论.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n图与数据已写入 {out_dir}")


if __name__ == "__main__":
    main()
