"""尺度混合分布（两成分高斯混合）的拟合与判据量提取。

形状诊断（`bench/error_dist_shape.py`）与分布族横评（`bench/error_dist_compare.py`）的
结论一致：误差分布不是带折点的单峰族，而是**主体 + 少数大尺度**的尺度混合。
本脚本把结构显式拟合出来，并给出可直接用于判据的量：

  * w（主体成分权重）、1−w（大尺度成分权重 ≈ "异常样本比例"）
  * sigma1（主体尺度）、sigma2（大尺度尺度）
  * **尺度比 sigma2/sigma1**、**对数比 ln(sigma2/sigma1)** —— 尾部比主体重多少倍
  * **方差占比**：大尺度成分贡献了总方差的百分之多少
  * **逐样本后验责任** r_i = P(样本属于大尺度成分 | x_i) —— 天然的逐帧/逐单位异常度

用法：python bench/error_mixture.py [rows.csv] [输出目录]
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
MIN_OWN_FIT = 30
MAX_RATIO = 50.0      # 尺度比超过此值判为退化解


def _valid(d):
    """排除退化拟合：权重贴边或尺度比爆炸。"""
    return (d is not None and 0.05 < d["w"] < 0.95
            and d["ratio"] < MAX_RATIO and np.isfinite(d["ratio"]))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _note(ax, txt, loc=0.02):
    ax.text(loc, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.92))


def _phi(z):
    return np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)


def _loglik(x, w, m, s):
    p = w * _phi((x - m[0]) / s[0]) / s[0] + (1 - w) * _phi((x - m[1]) / s[1]) / s[1]
    return float(np.log(np.maximum(p, 1e-300)).sum())


def fit_mix2(x, restarts=8, iters=1500, tol=1e-10):
    """一维两成分高斯混合的 EM。返回 (w, mu, sd)（主体在前）与对数似然。"""
    x = np.asarray(x, dtype=float)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sc = 1.4826 * mad if mad > 0 else float(np.std(x))
    z = (x - med) / max(sc, 1e-9)
    sd0 = float(np.std(z))
    best = (-np.inf, None)
    pairs = [(15, 70), (25, 85), (10, 50), (35, 90), (5, 30), (20, 60),
             (40, 95), (30, 75)]
    for r in range(restarts):
        q1, q2 = pairs[r % len(pairs)]
        m = np.array([np.percentile(z, q1), np.percentile(z, q2)], dtype=float)
        s = np.array([sd0 * 0.4, sd0 * 1.4], dtype=float)
        w = 0.7
        prev = -np.inf
        for it in range(iters):
            p1 = w * _phi((z - m[0]) / s[0]) / s[0]
            p2 = (1 - w) * _phi((z - m[1]) / s[1]) / s[1]
            tot = p1 + p2 + 1e-300
            r1 = p1 / tot
            w = float(r1.mean())
            n1, n2 = r1.sum(), (1 - r1).sum()
            if n1 < 1e-6 or n2 < 1e-6:
                break
            m[0] = float((r1 * z).sum() / n1)
            m[1] = float(((1 - r1) * z).sum() / n2)
            s[0] = float(np.sqrt(max((r1 * (z - m[0]) ** 2).sum() / n1, 1e-12)))
            s[1] = float(np.sqrt(max(((1 - r1) * (z - m[1]) ** 2).sum() / n2, 1e-12)))
            cur = _loglik(z, w, m, s)
            if abs(cur - prev) < tol * max(1.0, abs(cur)):
                break
            prev = cur
        ll = _loglik(z, w, m, s)
        if ll > best[0]:
            best = (ll, (w, m.copy(), s.copy()))
    if best[1] is None:
        return None
    w, m, s = best[1]
    # 主体 = 尺度较小的那个成分
    if s[0] > s[1]:
        w, m = 1.0 - w, m[::-1].copy()
        s = s[::-1].copy()
    w = float(np.clip(w, 1e-6, 1 - 1e-6))
    # 换算回像素
    mu_px = med + m * sc
    sd_px = s * sc
    k = 5
    ll_x = _loglik(x, w, mu_px, sd_px)
    n = x.size
    return {"n": n, "w": w, "mu": mu_px, "sd": sd_px,
            "logL": ll_x, "k": k,
            "BIC": k * np.log(n) - 2 * ll_x,
            "log_ratio": float(np.log(sd_px[1] / sd_px[0])),
            "ratio": float(sd_px[1] / sd_px[0]),
            "logit_w": float(np.log(w / (1 - w))),
            "var_share_heavy": float((1 - w) * sd_px[1] ** 2
                                     / (w * sd_px[0] ** 2 + (1 - w) * sd_px[1] ** 2)),
            "med": float(med), "scale": float(sc)}


def responsibilities(x, d):
    """逐样本的「属于大尺度成分」后验概率。"""
    x = np.asarray(x, dtype=float)
    m, s, w = d["mu"], d["sd"], d["w"]
    p1 = w * _phi((x - m[0]) / s[0]) / s[0]
    p2 = (1 - w) * _phi((x - m[1]) / s[1]) / s[1]
    return p2 / (p1 + p2 + 1e-300)


def mix_pdf(grid, d, which=None):
    grid = np.asarray(grid, dtype=float)
    m, s, w = d["mu"], d["sd"], d["w"]
    c1 = w * _phi((grid - m[0]) / s[0]) / s[0]
    c2 = (1 - w) * _phi((grid - m[1]) / s[1]) / s[1]
    if which == 1:
        return c1
    if which == 2:
        return c2
    return c1 + c2


def kde(z, grid, h, nbins=1024):
    lo, hi = float(grid[0]) - 6 * h, float(grid[-1]) + 6 * h
    dx = (hi - lo) / nbins
    cnt, _ = np.histogram(z, bins=nbins, range=(lo, hi))
    sig = max(h / dx, 1e-3)
    m = int(np.ceil(6 * sig))
    k = np.exp(-0.5 * (np.arange(-m, m + 1) / sig) ** 2)
    k /= k.sum()
    dens = np.convolve(cnt.astype(float), k, mode="same")
    centers = lo + (np.arange(nbins) + 0.5) * dx
    return np.interp(grid, centers, dens / (z.size * dx))


def load(path: Path):
    out = []
    with path.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            e = r["e_along"]
            if e in ("", "None"):
                continue
            out.append({"video": r["video"], "frame": int(r["frame"]),
                        "id": int(r["id"]), "cls": r["cls"], "h": float(r["h"]),
                        "e": float(e), "speed": float(r["speed"] or 0.0)})
    return out


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "error_dist" / "rows.csv"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs" / "error_dist" / "尺度混合"
    out.mkdir(parents=True, exist_ok=True)
    rows = load(src)
    print(f"输入 {src}（{len(rows)} 条）\n输出 {out}")

    ref: dict = {}
    t1 = [("类别", "梯度", "n", "w(主体)", "1-w(尾部)", "sigma1(px)", "sigma2(px)",
           "尺度比", "ln尺度比", "尾部方差占比", "mu1(px)", "mu2(px)", "BIC",
           "是否稳")]
    for cls in ORDER:
        for h in HORIZONS:
            x = np.array([r["e"] for r in rows if r["cls"] == cls and r["h"] == h])
            if x.size < 60:
                continue
            d = fit_mix2(x)
            if d is None:
                continue
            ref[(cls, h)] = d
            stable = "稳" if (0.2 <= d["w"] <= 0.9 and d["n"] >= 150) else "不稳"
            t1.append((CLASS_ZH.get(cls, cls), h, d["n"], round(d["w"], 3),
                       round(1 - d["w"], 3), round(d["sd"][0], 1),
                       round(d["sd"][1], 1), round(d["ratio"], 2),
                       round(d["log_ratio"], 3), round(d["var_share_heavy"], 3),
                       round(d["mu"][0], 1), round(d["mu"][1], 1), round(d["BIC"], 1),
                       stable))
            print(f"  {cls} {h}s n={d['n']:>6} w={d['w']:.3f} "
                  f"sigma1={d['sd'][0]:7.1f} sigma2={d['sd'][1]:8.1f} "
                  f"比={d['ratio']:6.2f} 尾部方差占比={d['var_share_heavy']:.2f}")

    with (out / "尺度混合_类别梯度.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows(t1)

    # ---- 运动单位级
    units: dict = defaultdict(list)
    for r in rows:
        units[(r["video"], r["id"], r["cls"], r["h"])].append(r["e"])
    t2 = [("视频", "轨迹", "类别", "梯度", "n", "w", "sigma1", "sigma2", "尺度比",
           "均值责任(尾部)", "P90责任")]
    unit_rows = []
    for (video, tid, cls, h), xs in sorted(units.items()):
        xs = np.array(xs, dtype=float)
        rf = ref.get((cls, h))
        resp = responsibilities(xs, rf) if rf else None
        d = fit_mix2(xs, restarts=5) if xs.size >= MIN_OWN_FIT else None
        if not _valid(d):
            d = None
        if d is None and resp is None:
            continue
        rec = {"video": video, "id": tid, "cls": cls, "h": h, "n": xs.size,
               "w": d["w"] if d else float("nan"),
               "s1": d["sd"][0] if d else float("nan"),
               "s2": d["sd"][1] if d else float("nan"),
               "ratio": d["ratio"] if d else float("nan"),
               "resp_mean": float(resp.mean()) if resp is not None else float("nan"),
               "resp_p90": float(np.percentile(resp, 90)) if resp is not None else float("nan")}
        unit_rows.append(rec)
        t2.append((video, tid, CLASS_ZH.get(cls, cls), h, xs.size,
                   round(rec["w"], 3) if d else "", round(rec["s1"], 1) if d else "",
                   round(rec["s2"], 1) if d else "", round(rec["ratio"], 2) if d else "",
                   round(rec["resp_mean"], 3), round(rec["resp_p90"], 3)))

    # ---- 帧级
    frames: dict = defaultdict(list)
    for r in rows:
        frames[(r["video"], r["frame"], r["h"])].append(r)
    t3 = [("视频", "帧", "梯度", "n", "责任均值(尾部)", "责任P90", "责任最大", "速度中位")]
    frame_rows = []
    for (video, fr, h), rs in sorted(frames.items()):
        vals, sps = [], []
        for r in rs:
            rf = ref.get((r["cls"], h))
            if rf is None:
                continue
            vals.append(float(responsibilities([r["e"]], rf)[0]))
            sps.append(r["speed"])
        if len(vals) < 2:
            continue
        rec = {"video": video, "frame": fr, "h": h, "n": len(vals),
               "resp_mean": float(np.mean(vals)), "resp_p90": float(np.percentile(vals, 90)),
               "resp_max": float(np.max(vals)), "speed_med": float(np.median(sps))}
        frame_rows.append(rec)
        t3.append((video, fr, h, rec["n"], round(rec["resp_mean"], 3),
                   round(rec["resp_p90"], 3), round(rec["resp_max"], 3),
                   round(rec["speed_med"], 1)))

    for name, table in (("尺度混合_运动单位.csv", t2), ("尺度混合_帧.csv", t3)):
        with (out / name).open("w", newline="", encoding="utf-8-sig") as fh:
            csv.writer(fh).writerows(table)

    # ---------------- 每类别独立一张页
    for cls in ORDER:
        d = ref.get((cls, 1.0))
        if d is None:
            continue
        x = np.array([r["e"] for r in rows if r["cls"] == cls and r["h"] == 1.0])
        lim = float(np.percentile(np.abs(x), 99.5)) or 1.0
        grid = np.linspace(-lim, lim, 900)
        fig, axes = plt.subplots(2, 3, figsize=(17, 8.8))

        ax = axes[0][0]
        ax.hist(x, bins=140, range=(-lim, lim), density=True, color="#D9D9D9",
                label="实测")
        ax.plot(grid, kde(x, grid, 0.02 * lim), color="0.25", lw=1.6, label="实测 KDE")
        ax.plot(grid, mix_pdf(grid, d), color="#000000", lw=2.0, label="混合 = 两者之和")
        ax.plot(grid, mix_pdf(grid, d, 1), color="#4C78A8", lw=1.6, ls="--",
                label=f"主体  w={d['w']:.2f}, σ={d['sd'][0]:.0f}px")
        ax.plot(grid, mix_pdf(grid, d, 2), color="#E45756", lw=1.6, ls="--",
                label=f"尾部  1−w={1 - d['w']:.2f}, σ={d['sd'][1]:.0f}px")
        ax.set_ylim(0, min(float(np.max(mix_pdf(grid, d))) * 1.35, 3 * float(np.max(kde(x, grid, 0.02 * lim)))))
        ax.set_title("两个成分的分解（1 s）")
        ax.set_xlabel("e_along (px)"); ax.set_ylabel("概率密度")
        ax.legend(fontsize=8)

        ax = axes[0][1]
        hs = [h for h in HORIZONS if (cls, h) in ref]
        ax.plot(hs, [ref[(cls, h)]["w"] for h in hs], "o-", color="#4C78A8")
        ax.set_ylim(0, 1)
        ax.set_xlabel("梯度 (s)"); ax.set_ylabel("主体权重 w")
        ax.set_title("主体权重随梯度（越接近 1 越单一机制）")
        _note(ax, "　".join(f"{h:g}s {ref[(cls, h)]['w']:.2f}" for h in hs))

        ax = axes[0][2]
        for i, (c, name) in enumerate((("#4C78A8", "主体 σ1"), ("#E45756", "尾部 σ2"))):
            ax.semilogy(hs, [ref[(cls, h)]["sd"][i] for h in hs], "o-", color=c,
                        label=name)
        ax.set_xlabel("梯度 (s)"); ax.set_ylabel("尺度 (px，对数轴)")
        ax.set_title("两个尺度随梯度的增长"); ax.legend(fontsize=8)

        ax = axes[1][0]
        ax.plot(hs, [ref[(cls, h)]["ratio"] for h in hs], "o-", color="#B279A2")
        ax.set_xlabel("梯度 (s)"); ax.set_ylabel("尺度比 σ2/σ1")
        ax.set_title("尾部比主体重多少倍")
        _note(ax, "　".join(f"{h:g}s {ref[(cls, h)]['ratio']:.1f}×" for h in hs))

        ax = axes[1][1]
        sel = [r for r in unit_rows if r["cls"] == cls and r["h"] == 1.0
               and np.isfinite(r["ratio"])]
        if sel:
            vals = np.array([r["ratio"] for r in sel])
            vals = vals[vals < MAX_RATIO]
            hi = float(np.percentile(vals, 99)) * 1.1 if vals.size else 1.0
            ax.hist(vals, bins=30, range=(0, hi), color=COLORS[cls], alpha=0.9)
            med = float(np.median(vals))
            ax.axvline(med, color="#E45756", lw=1.7, ls="--",
                       label=f"中位 {med:.1f}×")
            ax.axvline(ref[(cls, 1.0)]["ratio"], color="0.3", lw=1.4,
                       label=f"类别整体 {ref[(cls, 1.0)]['ratio']:.1f}×")
            ax.legend(fontsize=8)
        ax.set_xlabel("逐运动单位的尺度比 σ2/σ1")
        ax.set_ylabel("轨迹数")
        ax.set_title(f"单位级尺度比的分布（{len(sel)} 条轨迹）")

        ax = axes[1][2]
        # 只统计该类别自己的样本（之前误用了全体类别的帧聚合）
        by_frame: dict = defaultdict(list)
        for r in rows:
            if r["cls"] == cls and r["h"] == 1.0:
                by_frame[(r["video"], r["frame"])].append(r["e"])
        for video in sorted({v for v, _ in by_frame})[:3]:
            frs = sorted(f for v, f in by_frame if v == video)
            ys = []
            for f in frs:
                xs_f = np.array(by_frame[(video, f)], dtype=float)
                ys.append(float(responsibilities(xs_f, ref[(cls, 1.0)]).mean()))
            ax.plot(frs, ys, lw=1.0, label=video[-6:])
        ax.axhline(1 - ref[(cls, 1.0)]["w"], color="0.4", lw=1.2, ls="--",
                   label=f"整体尾部比例 {1 - ref[(cls, 1.0)]['w']:.2f}")
        ax.set_xlabel("帧号"); ax.set_ylabel("帧内平均责任")
        ax.set_title("逐帧的尾部责任（异常样本占比）"); ax.legend(fontsize=7, ncol=2)

        fig.suptitle(f"{CLASS_ZH.get(cls, cls)}（{cls}）　尺度混合分解　梯度 1 s",
                     fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out / f"混合_{CLASS_ZH.get(cls, cls)}_{cls}.png", dpi=165)
        plt.close(fig)

    # ---------------- 汇总
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for cls in ORDER:
        hs = [h for h in HORIZONS if (cls, h) in ref]
        if not hs:
            continue
        axes[0].plot(hs, [ref[(cls, h)]["w"] for h in hs], "o-", color=COLORS[cls],
                     lw=1.5, label=CLASS_ZH.get(cls, cls))
        axes[1].plot(hs, [ref[(cls, h)]["ratio"] for h in hs], "o-", color=COLORS[cls],
                     lw=1.5)
        axes[2].plot(hs, [ref[(cls, h)]["var_share_heavy"] for h in hs], "o-",
                     color=COLORS[cls], lw=1.5)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("梯度 (s)"); axes[0].set_ylabel("主体权重 w")
    axes[0].set_title("主体权重"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("梯度 (s)"); axes[1].set_ylabel("尺度比 σ2/σ1")
    axes[1].set_title("尾部/主体尺度比")
    axes[2].set_xlabel("梯度 (s)"); axes[2].set_ylabel("尾部成分的方差占比")
    axes[2].set_title("少数样本贡献了多少方差")
    fig.suptitle("图E　尺度混合的三个判据量（每类别独立拟合）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "尺度混合汇总.png", dpi=180)
    plt.close(fig)

    md = ["# 尺度混合：w / σ1 / σ2 与派生判据量", "",
          f"数据源 `{src}`；产物 `{out}`。模型：两成分高斯混合（EM，多次重启）。", "",
          "## 判据量", "",
          "| 量 | 含义 |", "|---|---|",
          "| w | 主体（小尺度）成分的权重 |",
          "| 1−w | 尾部成分权重 ≈ 异常样本比例 |",
          "| σ1, σ2 | 主体 / 尾部的尺度（px） |",
          "| σ2/σ1 | 尾部比主体重多少倍 |",
          "| ln(σ2/σ1) | 同上取对数，便于线性建模 |",
          "| 尾部方差占比 | (1−w)σ2² / (wσ1² + (1−w)σ2²) |",
          "| 责任 r_i | P(样本属于尾部成分 \\| x_i)，天然的逐样本异常度 |",
          "", "## 类别 × 梯度", "",
          "| " + " | ".join(t1[0]) + " |", "|" + "---|" * len(t1[0])]
    md += ["| " + " | ".join(str(v) for v in row) + " |" for row in t1[1:]]
    md += ["", "## 产物", "",
           "- `尺度混合_类别梯度.csv`、`尺度混合_运动单位.csv`、`尺度混合_帧.csv`",
           "- `混合_<类别>.png`：每类别一张独立的成分分解图",
           "- `尺度混合汇总.png`", "",
           f"运动单位 {len(unit_rows)} 条，帧 {len(frame_rows)} 个。", ""]
    (out / "尺度混合说明.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n运动单位 {len(unit_rows)}，帧 {len(frame_rows)}；产物写入 {out}")


if __name__ == "__main__":
    main()
