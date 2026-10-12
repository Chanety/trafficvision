"""预判误差的分布模型：给出「偏差参数 θ」与「异常度 A」两套判据原料。

## 为什么用不对称拉普拉斯分布（ALD）

实测（`bench/error_distribution.py`）：误差幅值分布的峰度 20~30、均值/中位 1.76~1.81、
拉普拉斯拟合的对数似然比高斯高约 3078 nats。也就是说误差**不是**正态，而是重尾。
ALD 是"正态的重尾推广 + 一个不对称参数"：尺度参数控制展宽，不对称参数正好承担偏差。

## 分布函数（本文件使用的定义）

记误差 x（本文件一律指沿运动方向的 e_along）。密度

    f(x; mu, s, kappa) = 1/(s(kappa + 1/kappa)) * exp( -(x-mu)*kappa/s )    x >= mu
                       = 1/(s(kappa + 1/kappa)) * exp(  (x-mu)/(kappa*s) )   x <  mu

分布函数

    F(x) = 1 - exp( -(x-mu)*kappa/s ) / (1 + kappa^2)                 x >= mu
         = kappa^2 / (1 + kappa^2) * exp( (x-mu)/(kappa*s) )          x <  mu

kappa=1 时退化为对称拉普拉斯（即"重尾版正态"）；kappa<1 表示**负侧更重**，
kappa>1 表示正侧更重。

## 参数分工（按用户口径）

  * **mu**：位置参数。**只作展示**，不参与判据。
  * **theta_a = mu / s**：标准化偏移（效应量）——"整体偏了多少个标准差"。
  * **theta_b = -ln kappa**：不对称度——theta_b>0 表示负侧（超前/长打）更重。
  * **s**：尺度（展宽）。

两个 theta 都是"展示偏差"的参数，拿哪一个、怎么组合成得分由使用者决定。

## 异常度（第二个判据量）

  1. **自身拟合残差 A**：该组数据用 ALD 拟合后，模型分布函数与经验分布函数的最大偏差
     （KS 距离）。A 越大说明"用一个位置+尺度+不对称的故事"越讲不通。
  2. **可比异常度 A\\* = A*sqrt(n)**：去掉样本量的影响，使不同 n 的组之间可比
     （渐近服从 Kolmogorov 分布，alpha=0.05 临界值 1.36，alpha=0.01 为 1.63）。
  3. **逐样本异常度** a_i = 2*|F_ref(x_i) - 0.5| 属于 [0,1)：该样本在参照分布里有多极端；
     以及标准化残差 z_i = (x_i - mu_ref)/s_ref。参照分布取"该类别 x 该梯度"的池化拟合，
     因此样本量很小的帧级/单位级也能算。

用法：python bench/error_dist_model.py [rows.csv] [输出目录]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (0.1, 0.5, 1.0, 2.0)
CLASS_ZH = {
    "car": "汽车", "truck": "卡车", "bus": "公交", "nonmotor": "非机动车",
    "person": "行人", "sign": "标志牌", "bollard": "路桩",
}
ORDER = ["car", "bus", "truck", "nonmotor", "person", "sign", "bollard"]
COLORS = dict(zip(ORDER, ["#4C78A8", "#F58518", "#54A24B", "#E45756",
                          "#B279A2", "#8C6D31", "#72B7B2"]))
MIN_OWN_FIT = 15          # 自身拟合所需最少样本

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _note(ax, txt, loc=0.02):
    ax.text(loc, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.92))


# --------------------------------------------------------------------------- 分布

def ald_cdf(x, mu, s, kappa):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    hi = x >= mu
    out[hi] = 1.0 - np.exp(-(x[hi] - mu) * kappa / s) / (1.0 + kappa ** 2)
    out[~hi] = (kappa ** 2 / (1.0 + kappa ** 2)) * np.exp((x[~hi] - mu) / (kappa * s))
    return np.clip(out, 0.0, 1.0)


def ald_pdf(x, mu, s, kappa):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    hi = x >= mu
    norm = 1.0 / (s * (kappa + 1.0 / kappa))
    out[hi] = norm * np.exp(-(x[hi] - mu) * kappa / s)
    out[~hi] = norm * np.exp((x[~hi] - mu) / (kappa * s))
    return out


def ald_ppf(p, mu, s, kappa):
    """ALD 分位数（闭式）。"""
    p = np.asarray(p, dtype=float)
    split = kappa ** 2 / (1.0 + kappa ** 2)
    hi = p >= split
    out = np.empty_like(p)
    out[hi] = mu - (s / kappa) * np.log(np.maximum((1 - p[hi]) * (1 + kappa ** 2), 1e-300))
    out[~hi] = mu + kappa * s * np.log(np.maximum(p[~hi] * (1 + kappa ** 2) / kappa ** 2, 1e-300))
    return out


def fit_ald(x):
    """分位数估计（稳健、闭式）：mu=中位，kappa 由上下四分位宽度比给出。

    推导：ALD 上半侧为均值 kappa*s 的指数分布、下半侧为均值 s/kappa 的指数分布，
    故 (q75-mu)/(mu-q25) = kappa^2。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return None
    mu = float(np.median(x))
    q25, q75 = (float(v) for v in np.percentile(x, [25, 75]))
    q05, q95 = (float(v) for v in np.percentile(x, [5, 95]))
    up, dn = q75 - mu, mu - q25
    if up <= 0 or dn <= 0:
        kappa = 1.0
        s = max(float(np.mean(np.abs(x - mu))) / np.log(4), 1e-6)
    else:
        kappa = float(np.sqrt(up / dn))
        s = float(up / (kappa * np.log(2)))
    kappa = float(np.clip(kappa, 1e-3, 1e3))
    s = max(s, 1e-6)
    # 同一套分位数逻辑放到 5/95 深度：描述"尾部"的不对称，与核心区的 kappa 对照
    up5, dn5 = q95 - mu, mu - q05
    if up5 > 0 and dn5 > 0:
        kappa05 = float(np.clip(np.sqrt(up5 / dn5), 1e-3, 1e3))
        theta_b05 = -np.log(kappa05)
    else:
        kappa05, theta_b05 = float("nan"), float("nan")
    fit = {"n": n, "mu": mu, "scale": s, "kappa": kappa,
           "theta_a": mu / s, "theta_b": -np.log(kappa),
           "kappa05": kappa05, "theta_b05": theta_b05,
           "core_tail_gap": theta_b05 - (-np.log(kappa))
           if np.isfinite(theta_b05) else float("nan")}
    # 尾部超额：落在 ALD 自身 0.5%/99.5% 分位之外的实际比例（理想为 1%）
    lo, hi = (float(v) for v in ald_ppf([0.005, 0.995], mu, s, kappa))
    fit["tail_out"] = float(np.mean((x < lo) | (x > hi)))
    fit["tail_excess"] = fit["tail_out"] - 0.01
    return fit


def ks_distance(x, mu, s, kappa):
    """模型 CDF 与经验 CDF 的最大偏差。"""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    F = ald_cdf(x, mu, s, kappa)
    i = np.arange(1, n + 1)
    d = np.maximum(np.abs(F - i / n), np.abs(F - (i - 1) / n))
    return float(d.max())


def describe(x, ref=None):
    fit = fit_ald(x)
    if fit is None:
        return None
    fit["A"] = ks_distance(x, fit["mu"], fit["scale"], fit["kappa"])
    fit["A_star"] = fit["A"] * np.sqrt(fit["n"])
    if ref is not None:
        xa = np.asarray(x, dtype=float)
        u = ald_cdf(xa, ref["mu"], ref["scale"], ref["kappa"])
        a = 2.0 * np.abs(u - 0.5)
        z = (xa - ref["mu"]) / ref["scale"]
        fit["u_mean"] = float(u.mean())
        fit["a_mean"] = float(a.mean())
        fit["a_p90"] = float(np.percentile(a, 90))
        fit["z_mean"] = float(z.mean())
        fit["z_std"] = float(z.std(ddof=1)) if z.size > 1 else 0.0
    return fit


def load(path: Path):
    rows = []
    with path.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            e = r["e_along"]
            rows.append({
                "video": r["video"], "frame": int(r["frame"]), "id": int(r["id"]),
                "cls": r["cls"], "h": float(r["h"]),
                "e": float(e) if e not in ("", "None") else None,
                "coasting": r["coasting"] == "True",
                "speed": float(r["speed"] or 0.0),
            })
    return rows


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "error_dist" / "rows.csv"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs" / "error_dist" / "分布模型"
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in load(src) if r["e"] is not None]
    print(f"输入 {src}（{len(rows)} 条含方向误差的记录）\n输出 {out}")

    # ---- 参照分布：类别 x 梯度
    ref: dict = {}
    for cls in ORDER:
        for h in HORIZONS:
            x = np.array([r["e"] for r in rows if r["cls"] == cls and r["h"] == h])
            f = fit_ald(x)
            if f:
                f["A"] = ks_distance(x, f["mu"], f["scale"], f["kappa"])
                f["A_star"] = f["A"] * np.sqrt(f["n"])
                ref[(cls, h)] = f

    # ---- 表1：类别 x 梯度
    t1 = [("类别", "梯度", "n", "mu(展示)", "scale", "theta_a", "theta_b(核心)",
           "theta_b(尾部)", "核心-尾部差", "尾部超额", "A", "A*")]
    for cls in ORDER:
        for h in HORIZONS:
            f = ref.get((cls, h))
            if f:
                t1.append((CLASS_ZH.get(cls, cls), h, f["n"], round(f["mu"], 2),
                           round(f["scale"], 2), round(f["theta_a"], 3),
                           round(f["theta_b"], 3),
                           round(f["theta_b05"], 3) if np.isfinite(f["theta_b05"]) else "",
                           round(f["core_tail_gap"], 3) if np.isfinite(f["core_tail_gap"]) else "",
                           f"{f['tail_out'] * 100:.2f}%",
                           round(f["A"], 4), round(f["A_star"], 2)))

    # ---- 表2：运动单位
    units: dict = defaultdict(list)
    for r in rows:
        units[(r["video"], r["id"], r["cls"], r["h"])].append(r["e"])
    t2 = [("视频", "轨迹", "类别", "梯度", "n", "mu(展示)", "scale", "theta_a",
           "theta_b(核心)", "theta_b(尾部)", "A", "A*", "z_mean(参照)",
           "a_mean(参照)", "a_p90(参照)", "自身拟合可用")]
    per_unit_rows = []
    for (video, tid, cls, h), xs in sorted(units.items()):
        xs = np.array(xs, dtype=float)
        rf = ref.get((cls, h))
        d = describe(xs, rf)
        if d is None:
            continue
        own = d["n"] >= MIN_OWN_FIT
        rec = {"video": video, "id": tid, "cls": cls, "h": h, "n": d["n"],
               "mu": d["mu"], "scale": d["scale"], "kappa": d["kappa"],
               "theta_a": d["theta_a"], "theta_b": d["theta_b"],
               "theta_b05": d.get("theta_b05", float("nan")),
               "A": d["A"], "A_star": d["A_star"],
               "z_mean": d.get("z_mean", float("nan")),
               "a_mean": d.get("a_mean", float("nan")),
               "a_p90": d.get("a_p90", float("nan")), "own": own}
        per_unit_rows.append(rec)
        t2.append((video, tid, CLASS_ZH.get(cls, cls), h, d["n"],
                   round(d["mu"], 2), round(d["scale"], 2), round(d["theta_a"], 3),
                   round(d["theta_b"], 3),
                   round(d["theta_b05"], 3) if np.isfinite(d.get("theta_b05", float("nan"))) else "",
                   round(d["A"], 4) if own else "",
                   round(d["A_star"], 2) if own else "",
                   round(d.get("z_mean", float("nan")), 3),
                   round(d.get("a_mean", float("nan")), 3),
                   round(d.get("a_p90", float("nan")), 3), "是" if own else "否"))

    # ---- 表3：帧（用参照分布，小样本也能算）
    frames: dict = defaultdict(list)
    for r in rows:
        frames[(r["video"], r["frame"], r["h"])].append(r)
    t3 = [("视频", "帧", "梯度", "n", "z_mean", "a_mean", "a_p90", "速度中位")]
    per_frame_rows = []
    for (video, fr, h), rs in sorted(frames.items()):
        zs, as_, sps = [], [], []
        for r in rs:
            rf = ref.get((r["cls"], h))
            if rf is None:
                continue
            u = float(ald_cdf([r["e"]], rf["mu"], rf["scale"], rf["kappa"])[0])
            zs.append((r["e"] - rf["mu"]) / rf["scale"])
            as_.append(2.0 * abs(u - 0.5))
            sps.append(r["speed"])
        if len(zs) < 2:
            continue
        rec = {"video": video, "frame": fr, "h": h, "n": len(zs),
               "z_mean": float(np.mean(zs)), "a_mean": float(np.mean(as_)),
               "a_p90": float(np.percentile(as_, 90)),
               "speed_med": float(np.median(sps))}
        per_frame_rows.append(rec)
        t3.append((video, fr, h, rec["n"], round(rec["z_mean"], 3),
                   round(rec["a_mean"], 3), round(rec["a_p90"], 3),
                   round(rec["speed_med"], 1)))

    for name, table in (("模型参数_类别梯度.csv", t1),
                        ("模型参数_运动单位.csv", t2),
                        ("模型参数_帧.csv", t3)):
        with (out / name).open("w", newline="", encoding="utf-8-sig") as fh:
            csv.writer(fh).writerows(table)

    # ------------------------------------------------------------------ 图
    # 图1：每类别一张独立的 ALD 拟合（1 s）
    for cls in ORDER:
        f = ref.get((cls, 1.0))
        if f is None or f["n"] < 50:
            continue
        x = np.array([r["e"] for r in rows if r["cls"] == cls and r["h"] == 1.0])
        lim = float(np.percentile(np.abs(x), 99.5)) or 1.0
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(x, bins=120, range=(-lim, lim), density=True, color="#BAB0AC",
                alpha=0.85, label="实测 e_along")
        xs = np.linspace(-lim, lim, 800)
        ax.plot(xs, ald_pdf(xs, f["mu"], f["scale"], f["kappa"]), color="#E45756",
                lw=2, label="ALD 拟合")
        ax.plot(xs, np.exp(-(xs - f["mu"]) ** 2 / (2 * f["scale"] ** 2))
                / (f["scale"] * np.sqrt(2 * np.pi)), color="#4C78A8", lw=1.6, ls="--",
                label="同尺度高斯（对照）")
        ax.axvline(0, color="0.35", lw=1)
        side = "（负侧更重）" if f["theta_b"] > 0 else "（正侧更重）"
        _note(ax, f"n = {f['n']}\n"
                  f"mu = {f['mu']:+.1f} px　（仅展示）\n"
                  f"s = {f['scale']:.1f} px\n"
                  f"kappa = {f['kappa']:.3f}\n"
                  f"theta_a = mu/s = {f['theta_a']:+.3f}\n"
                  f"theta_b(核心) = {f['theta_b']:+.3f}{side}\n"
                  f"theta_b(尾部) = {f['theta_b05']:+.3f}\n"
                  f"尾部超额 = {f['tail_out'] * 100:.2f}%（理想 1%）\n"
                  f"异常度 A = {f['A']:.3f}　A* = {f['A_star']:.1f}")
        ax.legend(fontsize=9)
        ax.set_title(f"{CLASS_ZH.get(cls, cls)}（{cls}）　误差分布与 ALD 拟合　梯度 1 s")
        ax.set_xlabel("e_along (px)　正=滞后　负=超前")
        ax.set_ylabel("概率密度")
        fig.tight_layout()
        fig.savefig(out / f"拟合_{CLASS_ZH.get(cls, cls)}_{cls}.png", dpi=170)
        plt.close(fig)

    # 图2：theta 平面（逐运动单位，1 s）
    u1 = [r for r in per_unit_rows if r["h"] == 1.0 and r["own"]]
    if u1:
        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        for cls in ORDER:
            sel = [r for r in u1 if r["cls"] == cls]
            if not sel:
                continue
            ax.scatter([r["theta_a"] for r in sel], [r["theta_b"] for r in sel],
                       s=26, alpha=0.75, color=COLORS[cls],
                       label=f"{CLASS_ZH.get(cls, cls)}（{len(sel)}）")
        ax.axhline(0, color="0.5", lw=1)
        ax.axvline(0, color="0.5", lw=1)
        ax.set_xlabel("theta_a = mu/s　（标准化偏移）")
        ax.set_ylabel("theta_b = -ln(kappa)　（>0 负侧更重＝偏超前）")
        ax.set_title("图2　两个偏差参数构成的平面（每个点=一条轨迹，1 s）")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "theta平面_逐单位.png", dpi=180)
        plt.close(fig)

    # 图3：异常度分布（单位级 vs 帧级）
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    au = np.array([r["a_mean"] for r in per_unit_rows if r["h"] == 1.0])
    af = np.array([r["a_mean"] for r in per_frame_rows if r["h"] == 1.0])
    for ax, a, name in ((axes[0], au, "运动单位级"), (axes[1], af, "帧级")):
        if a.size == 0:
            continue
        ax.hist(a, bins=50, color="#4C78A8", alpha=0.9)
        ax.axvline(float(np.median(a)), color="#E45756", lw=1.7, ls="--")
        _note(ax, f"n = {a.size}\n中位 {np.median(a):.3f}\n"
                  f"P90 {np.percentile(a, 90):.3f}\n最大 {a.max():.3f}")
        ax.set_title(f"{name}：a_mean（逐样本异常度均值，1 s）")
        ax.set_xlabel("a = 2*|F_ref(x) - 0.5|　0=典型　→1=极端")
        ax.set_ylabel("数量")
    fig.suptitle("图3　异常度分布", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "异常度_单位与帧.png", dpi=180)
    plt.close(fig)

    # 图4：核心区不对称 vs 尾部不对称（混合结构的证据）
    pts = [(cls, h, ref[(cls, h)]) for cls in ORDER for h in HORIZONS
           if (cls, h) in ref and np.isfinite(ref[(cls, h)]["theta_b05"])]
    if pts:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
        for cls in ORDER:
            sel = [(h, f) for c, h, f in pts if c == cls]
            if not sel:
                continue
            axes[0].plot([f["theta_b"] for _, f in sel], [f["theta_b05"] for _, f in sel],
                         "o-", color=COLORS[cls], lw=1.4, ms=5,
                         label=CLASS_ZH.get(cls, cls))
        lim = 0.9
        axes[0].plot([-lim, lim], [-lim, lim], color="0.5", lw=1.2, ls="--",
                     label="核心=尾部")
        axes[0].axhline(0, color="0.7", lw=1)
        axes[0].axvline(0, color="0.7", lw=1)
        axes[0].set_xlabel("theta_b 核心区（四分位）")
        axes[0].set_ylabel("theta_b 尾部（5/95 分位）")
        axes[0].set_title("离对角线越远＝分布不是单一机制")
        axes[0].legend(fontsize=8)
        for cls in ORDER:
            sel = [(h, f) for c, h, f in pts if c == cls]
            if not sel:
                continue
            axes[1].plot([h for h, _ in sel], [f["tail_out"] * 100 for _, f in sel],
                         "o-", color=COLORS[cls], lw=1.4, ms=5)
        axes[1].axhline(1.0, color="0.5", lw=1.2, ls="--")
        axes[1].text(0.15, 1.3, "理想 1%（ALD 自身 0.5/99.5 分位之外）",
                     fontsize=8, color="0.35")
        axes[1].set_xlabel("梯度 (s)")
        axes[1].set_ylabel("落在拟合分布极端分位之外的实测比例 (%)")
        axes[1].set_title("尾部超额：实测比拟合重多少")
        fig.suptitle("图4　分布是不是单一机制（核心 vs 尾部）", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out / "核心与尾部不对称.png", dpi=180)
        plt.close(fig)

    # 图5：每类别 PIT 直方图
    cols = [c for c in ORDER if (c, 1.0) in ref]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.0 * len(cols), 3.6), sharey=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, cls in zip(axes, cols):
        rf = ref[(cls, 1.0)]
        x = np.array([r["e"] for r in rows if r["cls"] == cls and r["h"] == 1.0])
        u = ald_cdf(x, rf["mu"], rf["scale"], rf["kappa"])
        ax.hist(u, bins=20, range=(0, 1), color=COLORS[cls], alpha=0.9)
        ax.axhline(len(u) / 20, color="#E45756", lw=1.6, ls="--")
        ax.set_title(f"{CLASS_ZH.get(cls, cls)}\nn={len(u)}", fontsize=10)
        ax.set_xlabel("F_ref(x)")
    fig.suptitle("图5　PIT 直方图（红虚线=完全拟合时应有的均匀高度）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / "PIT_按类别.png", dpi=180)
    plt.close(fig)

    # 图6：逐帧 z_mean / a_mean
    fr1 = [r for r in per_frame_rows if r["h"] == 1.0]
    videos = sorted({r["video"] for r in fr1})[:3]
    if videos:
        fig, axes = plt.subplots(len(videos), 2, figsize=(13, 2.6 * len(videos)),
                                 sharex="col")
        axes = np.atleast_2d(axes)
        for row, v in zip(axes, videos):
            s = sorted([r for r in fr1 if r["video"] == v], key=lambda r: r["frame"])
            row[0].plot([r["frame"] for r in s], [r["z_mean"] for r in s], lw=1.1,
                        color="#4C78A8")
            row[0].axhline(0, color="0.6", lw=1)
            row[0].set_title(f"{v[-6:]}　逐帧标准化残差 z_mean", fontsize=10)
            row[1].plot([r["frame"] for r in s], [r["a_mean"] for r in s], lw=1.1,
                        color="#E45756")
            row[1].set_title(f"{v[-6:]}　逐帧异常度 a_mean", fontsize=10)
            row[0].set_ylabel("z")
            row[1].set_ylabel("a")
        axes[-1][0].set_xlabel("帧号")
        axes[-1][1].set_xlabel("帧号")
        fig.suptitle("图6　逐帧的偏差与异常度（1 s）", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / "逐帧_偏差与异常度.png", dpi=180)
        plt.close(fig)

    # ------------------------------------------------------------------ 说明文档
    md = ["# 分布模型与两个判据量", "",
          f"数据源 `{src}`；产物目录 `{out}`。误差一律指沿运动方向的 e_along。", "",
          "## 分布函数", "",
          "```",
          "f(x; mu, s, kappa) = 1/(s(kappa+1/kappa)) * exp( -(x-mu)*kappa/s )    x >= mu",
          "                   = 1/(s(kappa+1/kappa)) * exp(  (x-mu)/(kappa*s) )   x <  mu",
          "",
          "F(x) = 1 - exp( -(x-mu)*kappa/s ) / (1+kappa^2)                 x >= mu",
          "     = kappa^2/(1+kappa^2) * exp( (x-mu)/(kappa*s) )          x <  mu",
          "```", "",
          "kappa=1 退化为对称拉普拉斯；kappa<1 负侧更重，kappa>1 正侧更重。", "",
          "## 参数分工", "",
          "| 量 | 定义 | 作用 |", "|---|---|---|",
          "| mu | 位置（中位数） | **仅展示**，不参与判据 |",
          "| s | 尺度 | 展宽 |",
          "| theta_a | mu / s | 标准化偏移（效应量） |",
          "| theta_b(核心) | -ln(kappa)，kappa 由 25/75 分位给出 | 核心区不对称，>0 负侧更重 |",
          "| theta_b(尾部) | 同上但由 5/95 分位给出 | 尾部不对称；与核心不一致＝分布非单一机制 |",
          "| 尾部超额 | 落在拟合分布 0.5/99.5 分位之外的实测比例 − 1% | 实测尾部比拟合重多少 |",
          "| A | KS(模型 CDF, 经验 CDF) | 自身拟合残差（异常度） |",
          "| A* | A*sqrt(n) | 去掉样本量影响的可比异常度（0.05 临界 1.36，0.01 临界 1.63） |",
          "| a_i | 2*abs(F_ref(x_i) - 0.5) | 逐样本异常度，0=典型，趋近 1=极端 |",
          "| z_i | (x_i - mu_ref)/s_ref | 以类别参照分布为基准的标准化残差 |",
          "", "## 估计方法", "",
          "mu 取中位数；kappa 由分位数闭式给出（(q75-mu)/(mu-q25) = kappa^2，由 ALD 上下",
          "两侧各为指数分布推出），s = (q75-mu)/(kappa*ln2)。全部闭式，无迭代、不受离群值牵动。",
          "", "## 类别 x 梯度（自身拟合）", "",
          "| " + " | ".join(t1[0]) + " |",
          "|" + "---|" * len(t1[0])]
    md += ["| " + " | ".join(str(v) for v in row) + " |" for row in t1[1:]]
    md += ["", "## 产物", "",
           "- `模型参数_类别梯度.csv`：类别 x 梯度 的 mu / s / kappa / theta_a / theta_b / A / A*",
           "- `模型参数_运动单位.csv`：逐轨迹的自身拟合参数 + 以类别为参照的残差",
           "- `模型参数_帧.csv`：逐帧的 z_mean / a_mean / a_p90（用参照分布，小样本可用）",
           "- `拟合_<类别>.png`：每类别一张独立的 ALD 拟合图（含同尺度高斯对照）",
           "- `theta平面_逐单位.png`：theta_a x theta_b 平面",
           "- `异常度_单位与帧.png`、`PIT_按类别.png`、`逐帧_偏差与异常度.png`", ""]
    (out / "分布模型与判据定义.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n参照分布（类别 x 梯度）已拟合 {len(ref)} 组；"
          f"运动单位 {len(per_unit_rows)} 条，帧 {len(per_frame_rows)} 个")
    print(f"产物写入 {out}")


if __name__ == "__main__":
    main()
