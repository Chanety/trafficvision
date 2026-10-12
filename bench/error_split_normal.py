"""用「不对称高斯（两片正态）」探查预判误差与目标模型的差异。

目的不是拟合，而是**度量偏离**：目标模型是零偏差的对称高斯；实际误差分布
允许一个不对称参数 a，a 取"让差异 theta 最小"的值。于是得到三个可直接用于判据的量：

  * **a\\***：最优不对称参数 —— 展示偏差的方向与大小
  * **theta\\*** = min_a theta(a)：把不对称放到最优之后**仍然剩下**的差异（不可约差异）
  * **theta0**：强制 a=0（即完全对称的目标模型）时的最小差异

## 分布函数

两片正态（split / two-piece normal）。给定分裂点 mu、尺度 sigma、不对称参数 a：

    sigma_L = sigma / (1 - a)        （左侧宽度）
    sigma_R = sigma / (1 + a)        （右侧宽度）

    f(x) = 2/(sigma_L + sigma_R) * phi((x - mu)/sigma_L)      x <  mu
         = 2/(sigma_L + sigma_R) * phi((x - mu)/sigma_R)      x >= mu

    F(x) = c * sigma_L * Phi((x-mu)/sigma_L)                          x <  mu
         = c * sigma_L/2 + c * sigma_R * (Phi((x-mu)/sigma_R) - 1/2)  x >= mu
           其中 c = 2/(sigma_L + sigma_R)

在 x=mu 处连续，全域积分为 1。**a>0 表示左侧更宽（负侧／超前侧铺得更开）**，
a<0 表示右侧更宽（正侧／滞后侧更开），a=0 退化为普通对称高斯。

## 差异 theta 与求解

theta(a, sigma) = sup_x |F_model(x) - F_emp(x)|        （KS 距离，越小越贴近）

  * mu 固定为中位数（**仅展示**，按用户口径不参与判据）
  * a\\* = argmin theta，sigma 同时在网格上搜索
  * theta\\* 即不可约差异；另外在左右两半各自取最大偏差，得到 theta左 / theta右，
    用来判断"差异主要出在哪一侧"

用法：python bench/error_split_normal.py [rows.csv] [输出目录]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from math import erf, sqrt
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
A_MIN, A_MAX, A_STEP = -0.92, 0.92, 0.01
MIN_OWN_FIT = 15

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
_SQRT2 = sqrt(2.0)


def _note(ax, txt, loc=0.02):
    ax.text(loc, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.92))


# --------------------------------------------------------------------------- 两片高斯

def _phi(z):
    return np.exp(-0.5 * np.asarray(z, dtype=float) ** 2) / sqrt(2.0 * np.pi)


def _Phi(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(z / _SQRT2))


def split_pdf(x, mu, s, a):
    x = np.asarray(x, dtype=float)
    sl, sr = s / (1.0 - a), s / (1.0 + a)
    c = 2.0 / (sl + sr)
    return np.where(x < mu, c * _phi((x - mu) / sl), c * _phi((x - mu) / sr))


def split_cdf(x, mu, s, a):
    x = np.asarray(x, dtype=float)
    sl, sr = s / (1.0 - a), s / (1.0 + a)
    c = 2.0 / (sl + sr)
    left = c * sl * _Phi((x - mu) / sl)
    right = c * sl * 0.5 + c * sr * (_Phi((x - mu) / sr) - 0.5)
    return np.clip(np.where(x < mu, left, right), 0.0, 1.0)


def ks_parts(x_sorted, mu, s, a):
    """返回 (总体 KS, 左侧 KS, 右侧 KS)。"""
    n = x_sorted.size
    F = split_cdf(x_sorted, mu, s, a)
    i = np.arange(1, n + 1)
    dev = np.maximum(np.abs(F - i / n), np.abs(F - (i - 1) / n))
    lo = x_sorted < mu
    hi = ~lo
    dl = float(dev[lo].max()) if lo.any() else 0.0
    dr = float(dev[hi].max()) if hi.any() else 0.0
    return float(dev.max()), dl, dr


def fit_split(x, mu=None, a_grid=None, s_grid=None):
    """在 (a, sigma) 网格上最小化 KS 差异。mu 固定为中位数（仅展示）。"""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n < MIN_OWN_FIT:
        return None
    if mu is None:
        mu = float(np.median(x))
    if s_grid is None:
        iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
        s0 = max(iqr / 1.349, 1e-6)
        s_grid = s0 * np.logspace(-0.7, 0.7, 29)
    if a_grid is None:
        a_grid = np.arange(A_MIN, A_MAX + 1e-9, A_STEP)

    # 网格搜索用抽样子序列加速，再在全体上做局部细化
    xs_sub = x[::max(1, n // 2500)] if n > 2500 else x

    def _search(seq, a_vals, s_vals):
        out = None
        for a in a_vals:
            for s in s_vals:
                d, dl, dr = ks_parts(seq, mu, s, float(a))
                if out is None or d < out[0]:
                    out = (d, dl, dr, float(a), float(s))
        return out

    best = _search(xs_sub, a_grid, s_grid)
    for a in np.linspace(max(A_MIN, best[3] - 0.03), min(A_MAX, best[3] + 0.03), 13):
        for s in np.linspace(best[4] * 0.92, best[4] * 1.08, 13):
            d, dl, dr = ks_parts(xs_sub, mu, s, float(a))
            if d < best[0]:
                best = (d, dl, dr, float(a), float(s))
    # 细化结果在全体上重算，保证 θ 是真实 KS 距离
    d, dl, dr = ks_parts(x, mu, best[4], best[3])
    best = (d, dl, dr, best[3], best[4])

    # 目标模型 A：对称高斯、位置自由（a=0）
    base = _search(xs_sub, (0.0,), s_grid)
    for s in np.linspace(base[4] * 0.9, base[4] * 1.1, 25):
        d, dl, dr = ks_parts(xs_sub, mu, s, 0.0)
        if d < base[0]:
            base = (d, dl, dr, 0.0, float(s))
    d0, dl0, dr0 = ks_parts(x, mu, base[4], 0.0)
    base = (d0, dl0, dr0, 0.0, base[4])

    # 目标模型 B：理想零误差（中心固定在 0 的对称高斯）
    zero = _search(xs_sub, (0.0,), s_grid)
    for s in np.linspace(zero[4] * 0.9, zero[4] * 1.1, 25):
        d, dl, dr = ks_parts(xs_sub, 0.0, s, 0.0)
        if d < zero[0]:
            zero = (d, dl, dr, 0.0, float(s))
    dz, _, _ = ks_parts(x, 0.0, zero[4], 0.0)
    return {"n": n, "mu": mu, "sigma": best[4], "a_star": best[3],
            "theta": best[0], "theta_left": best[1], "theta_right": best[2],
            "theta0": base[0], "sigma0": base[4], "theta_zero": dz,
            "sigma_zero": zero[4],
            "gain": base[0] - best[0],
            "gain_pct": (base[0] - best[0]) / base[0] * 100 if base[0] > 0 else 0.0}


def residual_against(x, ref):
    x = np.asarray(x, dtype=float)
    u = split_cdf(x, ref["mu"], ref["sigma"], ref["a_star"])
    z = (x - ref["mu"]) / ref["sigma"]
    a = 2 * np.abs(u - 0.5)
    return {"u_mean": float(u.mean()), "a_mean": float(a.mean()),
            "a_p90": float(np.percentile(a, 90)), "z_mean": float(z.mean())}


def load(path: Path):
    rows = []
    with path.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            e = r["e_along"]
            if e in ("", "None"):
                continue
            rows.append({"video": r["video"], "frame": int(r["frame"]),
                         "id": int(r["id"]), "cls": r["cls"], "h": float(r["h"]),
                         "e": float(e), "coasting": r["coasting"] == "True",
                         "speed": float(r["speed"] or 0.0)})
    return rows


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "error_dist" / "rows.csv"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs" / "error_dist" / "两片高斯"
    out.mkdir(parents=True, exist_ok=True)
    rows = load(src)
    print(f"输入 {src}（{len(rows)} 条）\n输出 {out}")

    ref: dict = {}
    for cls in ORDER:
        for h in HORIZONS:
            x = np.array([r["e"] for r in rows if r["cls"] == cls and r["h"] == h])
            f = fit_split(x)
            if f:
                ref[(cls, h)] = f
                print(f"  {cls} {h}s  n={f['n']}  a*={f['a_star']:+.2f}  "
                      f"sigma={f['sigma']:.1f}  theta*={f['theta']:.3f}  "
                      f"theta0={f['theta0']:.3f}")

    t1 = [("类别", "梯度", "n", "mu(展示)", "sigma", "a*", "theta*", "theta0(对称)",
           "theta零目标", "位置贡献", "不对称贡献", "剩余", "theta左", "theta右",
           "不对称改善%")]
    for cls in ORDER:
        for h in HORIZONS:
            f = ref.get((cls, h))
            if f:
                t1.append((CLASS_ZH.get(cls, cls), h, f["n"], round(f["mu"], 2),
                           round(f["sigma"], 2), round(f["a_star"], 3),
                           round(f["theta"], 4), round(f["theta0"], 4),
                           round(f["theta_zero"], 4),
                           round(f["theta_zero"] - f["theta0"], 4),
                           round(f["theta0"] - f["theta"], 4),
                           round(f["theta"], 4),
                           round(f["theta_left"], 4), round(f["theta_right"], 4),
                           round(f["gain_pct"], 1)))

    units: dict = defaultdict(list)
    for r in rows:
        units[(r["video"], r["id"], r["cls"], r["h"])].append(r["e"])
    t2 = [("视频", "轨迹", "类别", "梯度", "n", "mu(展示)", "sigma", "a*", "theta*",
           "theta0", "改善", "z_mean(参照)", "a_mean(参照)", "自身拟合")]
    unit_rows = []
    for (video, tid, cls, h), xs in sorted(units.items()):
        xs = np.array(xs, dtype=float)
        rf = ref.get((cls, h))
        f = fit_split(xs)
        res = residual_against(xs, rf) if rf else {}
        if f is None and not res:
            continue
        rec = {"video": video, "id": tid, "cls": cls, "h": h, "n": xs.size,
               "a_star": f["a_star"] if f else float("nan"),
               "theta": f["theta"] if f else float("nan"),
               "theta0": f["theta0"] if f else float("nan"),
               "gain": f["gain"] if f else float("nan"),
               "mu": f["mu"] if f else float("nan"),
               "sigma": f["sigma"] if f else float("nan"), **res}
        unit_rows.append(rec)
        t2.append((video, tid, CLASS_ZH.get(cls, cls), h, xs.size,
                   round(rec["mu"], 2) if f else "",
                   round(rec["sigma"], 2) if f else "",
                   round(rec["a_star"], 3) if f else "",
                   round(rec["theta"], 4) if f else "",
                   round(rec["theta0"], 4) if f else "",
                   round(rec["gain"], 4) if f else "",
                   round(res.get("z_mean", float("nan")), 3),
                   round(res.get("a_mean", float("nan")), 3),
                   "是" if f else "否"))

    frames: dict = defaultdict(list)
    for r in rows:
        frames[(r["video"], r["frame"], r["h"])].append(r)
    t3 = [("视频", "帧", "梯度", "n", "z_mean", "a_mean", "a_p90")]
    frame_rows = []
    for (video, fr, h), rs in sorted(frames.items()):
        zs, as_ = [], []
        for r in rs:
            rf = ref.get((r["cls"], h))
            if rf is None:
                continue
            u = float(split_cdf([r["e"]], rf["mu"], rf["sigma"], rf["a_star"])[0])
            zs.append((r["e"] - rf["mu"]) / rf["sigma"])
            as_.append(2.0 * abs(u - 0.5))
        if len(zs) < 2:
            continue
        rec = {"video": video, "frame": fr, "h": h, "n": len(zs),
               "z_mean": float(np.mean(zs)), "a_mean": float(np.mean(as_)),
               "a_p90": float(np.percentile(as_, 90))}
        frame_rows.append(rec)
        t3.append((video, fr, h, rec["n"], round(rec["z_mean"], 3),
                   round(rec["a_mean"], 3), round(rec["a_p90"], 3)))

    for name, table in (("两片高斯_类别梯度.csv", t1),
                        ("两片高斯_运动单位.csv", t2),
                        ("两片高斯_帧.csv", t3)):
        with (out / name).open("w", newline="", encoding="utf-8-sig") as fh:
            csv.writer(fh).writerows(table)

    for cls in ORDER:
        f = ref.get((cls, 1.0))
        if f is None:
            continue
        x = np.array([r["e"] for r in rows if r["cls"] == cls and r["h"] == 1.0])
        lim = float(np.percentile(np.abs(x), 99.5)) or 1.0
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        ax.hist(x, bins=120, range=(-lim, lim), density=True, color="#BAB0AC",
                alpha=0.85, label="实测 e_along")
        xs = np.linspace(-lim, lim, 900)
        ax.plot(xs, split_pdf(xs, f["mu"], f["sigma"], f["a_star"]), color="#E45756",
                lw=2.2, label=f"最优两片高斯  a*={f['a_star']:+.2f}")
        ax.plot(xs, split_pdf(xs, f["mu"], f["sigma0"], 0.0), color="#4C78A8",
                lw=1.7, ls="--", label="对称目标模型  a=0")
        ax.axvline(f["mu"], color="0.4", lw=1, ls=":")
        side = ("（左侧更宽＝偏超前侧）" if f["a_star"] > 0 else
                "（右侧更宽＝偏滞后侧）" if f["a_star"] < 0 else "（对称）")
        _note(ax, f"n = {f['n']}\n"
                  f"mu = {f['mu']:+.1f} px（仅展示）\n"
                  f"sigma = {f['sigma']:.1f} px\n"
                  f"a* = {f['a_star']:+.3f}{side}\n"
                  f"theta零目标 = {f['theta_zero']:.3f}\n"
                  f"theta0(对称) = {f['theta0']:.3f}\n"
                  f"theta*(不对称) = {f['theta']:.3f}\n"
                  f"拆解：位置 {f['theta_zero'] - f['theta0']:.3f}"
                  f" + 不对称 {f['theta0'] - f['theta']:.3f}"
                  f" + 剩余 {f['theta']:.3f}\n"
                  f"两侧差异：左 {f['theta_left']:.3f} / 右 {f['theta_right']:.3f}\n"
                  f"不对称只解释掉 {f['gain_pct']:.1f}% 的差异")
        ax.legend(fontsize=9)
        ax.set_title(f"{CLASS_ZH.get(cls, cls)}（{cls}）　与目标模型的差异　梯度 1 s")
        ax.set_xlabel("e_along (px)　正=滞后　负=超前")
        ax.set_ylabel("概率密度")
        fig.tight_layout()
        fig.savefig(out / f"差异_{CLASS_ZH.get(cls, cls)}_{cls}.png", dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    for cls in ORDER:
        pts = [(h, ref[(cls, h)]) for h in HORIZONS if (cls, h) in ref]
        if not pts:
            continue
        axes[0].plot([h for h, _ in pts], [f["a_star"] for _, f in pts], "o-",
                     color=COLORS[cls], lw=1.5, ms=5, label=CLASS_ZH.get(cls, cls))
    axes[0].axhline(0, color="0.5", lw=1.2)
    axes[0].set_xlabel("梯度 (s)")
    axes[0].set_ylabel("a*　（>0 左侧更宽＝偏超前侧）")
    axes[0].set_title("最优不对称参数随梯度的走向")
    axes[0].legend(fontsize=8)
    w = 0.27
    idx = np.arange(len(ORDER))
    tz = [ref[(c, 1.0)]["theta_zero"] if (c, 1.0) in ref else np.nan for c in ORDER]
    t0 = [ref[(c, 1.0)]["theta0"] if (c, 1.0) in ref else np.nan for c in ORDER]
    ts = [ref[(c, 1.0)]["theta"] if (c, 1.0) in ref else np.nan for c in ORDER]
    axes[1].bar(idx - w, tz, w, color="#BAB0AC", label="理想零误差目标")
    axes[1].bar(idx, t0, w, color="#4C78A8", label="对称高斯（位置自由）")
    axes[1].bar(idx + w, ts, w, color="#E45756", label="最优两片高斯")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels([CLASS_ZH.get(c, c) for c in ORDER], fontsize=9)
    axes[1].set_ylabel("KS 差异")
    axes[1].set_title("差异的三段拆解（1 s）")
    axes[1].legend(fontsize=9)
    fig.suptitle("图A　不对称参数与差异的削减（每个类别独立计算）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "不对称参数与差异削减.png", dpi=180)
    plt.close(fig)

    u1 = [r for r in unit_rows if r["h"] == 1.0 and np.isfinite(r["a_star"])]
    if u1:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        for cls in ORDER:
            sel = [r["a_star"] for r in u1 if r["cls"] == cls]
            if len(sel) < 3:
                continue
            axes[0].hist(sel, bins=25, histtype="step", lw=1.6, color=COLORS[cls],
                         label=f"{CLASS_ZH.get(cls, cls)}（{len(sel)}）")
        axes[0].axvline(0, color="0.4", lw=1.2)
        axes[0].set_xlabel("a*")
        axes[0].set_ylabel("轨迹数")
        axes[0].set_title("逐运动单位的最优不对称参数")
        axes[0].legend(fontsize=8)
        th = np.array([r["theta"] for r in u1])
        th0 = np.array([r["theta0"] for r in u1])
        axes[1].hist(th, bins=40, color="#E45756", alpha=0.85, label="theta*（允许不对称）")
        axes[1].hist(th0, bins=40, color="#4C78A8", alpha=0.5, label="theta0（强制对称）")
        axes[1].set_xlabel("KS 差异")
        axes[1].set_ylabel("轨迹数")
        axes[1].set_title("单位级不可约差异")
        axes[1].legend(fontsize=8)
        axes[2].scatter(th0, th, s=18, alpha=0.6, color="#F58518")
        lim = float(np.percentile(np.concatenate([th, th0]), 99))
        axes[2].plot([0, lim], [0, lim], color="0.4", lw=1.2, ls="--", label="无改善")
        axes[2].set_xlim(0, lim)
        axes[2].set_ylim(0, lim)
        axes[2].set_xlabel("theta0（对称目标）")
        axes[2].set_ylabel("theta*（最优不对称）")
        axes[2].set_title("对角线下方＝不对称确实解释了差异")
        axes[2].legend(fontsize=8)
        fig.suptitle("图B　运动单位级的 a* 与不可约差异（1 s）", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(out / "单位级_不对称与差异.png", dpi=180)
        plt.close(fig)

    md = ["# 两片高斯：与目标模型的差异", "",
          f"数据源 `{src}`；产物 `{out}`。误差指沿运动方向的 e_along。", "",
          "## 分布函数", "", "```",
          "sigma_L = sigma/(1-a)          sigma_R = sigma/(1+a)",
          "f(x) = 2/(sigma_L+sigma_R) * phi((x-mu)/sigma_L)   x <  mu",
          "     = 2/(sigma_L+sigma_R) * phi((x-mu)/sigma_R)   x >= mu",
          "",
          "F(x) = c*sigma_L*Phi((x-mu)/sigma_L)                        x <  mu",
          "     = c*sigma_L/2 + c*sigma_R*(Phi((x-mu)/sigma_R)-1/2)    x >= mu",
          "                              其中 c = 2/(sigma_L+sigma_R)",
          "```", "",
          "x=mu 处连续、全域积分为 1。**a>0 左侧更宽（偏超前侧），a<0 右侧更宽（偏滞后侧）**，",
          "a=0 退化为对称高斯（即目标模型）。", "",
          "## 判据量", "",
          "| 量 | 定义 | 含义 |", "|---|---|---|",
          "| a* | argmin_a theta(a) | 最优不对称参数，展示偏差方向与大小 |",
          "| theta* | min_a theta(a) = KS(模型, 经验) | 把不对称放到最优后仍剩下的差异 |",
          "| theta0 | a 强制为 0 时的最优 theta | 完全对称的目标模型下的差异 |",
          "| theta左/theta右 | 左半／右半各自的最大偏差 | 差异主要出在哪一侧 |",
          "| 改善% | (theta0 - theta*)/theta0 | 单靠一个不对称参数能解释掉多少差异 |",
          "",
          "mu 固定为中位数，**仅展示**，不参与判据；sigma 与 a 在网格上联合搜索。", "",
          "## 类别 x 梯度", "",
          "| " + " | ".join(t1[0]) + " |", "|" + "---|" * len(t1[0])]
    md += ["| " + " | ".join(str(v) for v in row) + " |" for row in t1[1:]]
    md += ["", "## 产物", "",
           "- `两片高斯_类别梯度.csv`、`两片高斯_运动单位.csv`、`两片高斯_帧.csv`",
           "- `差异_<类别>.png`：每类别一张独立的拟合与目标对照图",
           "- `不对称参数与差异削减.png`、`单位级_不对称与差异.png`", "",
           f"运动单位 {len(unit_rows)} 条（1 s 档可自身拟合 {len(u1)} 条）；"
           f"帧 {len(frame_rows)} 个。", ""]
    (out / "两片高斯与差异.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n运动单位 {len(unit_rows)}，帧 {len(frame_rows)}；产物写入 {out}")


if __name__ == "__main__":
    main()
