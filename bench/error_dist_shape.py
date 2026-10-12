"""先做非参数曲线拟合，再从曲线形状反推分布律。

不用先假定分布族。做法：

  1. **非参数密度估计**：FFT 卷积高斯核密度（KDE），带宽做扫描，确认形状不是
     带宽造出来的假象。
  2. **标准化**：z = (x − 中位) / (1.4826·MAD)，使各类别可比，方差约等于 1。
  3. **形状诊断**——这四条曲线是一眼看懂"是哪一族"的标准武器：

     | 曲线 | 形状含义 |
     |---|---|
     | 密度 log f(z) vs z | 抛物线=高斯；V 形（线性）=拉普拉斯；上凸变缓=t 类重尾 |
     | 对数生存 log S(z) vs z | 直线=指数尾；向下弯=比指数轻；向上弯=比指数重 |
     | 风险率 h(z) = f/S | 上升=比指数轻；水平=指数；下降=次指数（重尾） |
     | 峰形放大 | 圆顶=平滑密度；V 形尖点=折点密度 |

  4. **两段斜率**：对 log f(z) 在近尾（|z|∈[1,2.5]）与远尾（|z|∈[2.5,4]）各拟合
     一条直线。远尾比近尾更平 → 比指数还重；更陡 → 向轻尾收敛。

用法：python bench/error_dist_shape.py [rows.csv] [输出目录]
"""

from __future__ import annotations

import csv
import sys
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
BANDWIDTHS = (0.05, 0.10, 0.20, 0.40)   # 标准化单位下的核宽
Z_RANGE = 6.0

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _note(ax, txt, loc=0.02):
    ax.text(loc, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.92))


# --------------------------------------------------------------------------- 非参数密度

def kde(z, grid, h, nbins=1024):
    """FFT 思路的高斯核密度：直方图 + 高斯核卷积 + 插值。"""
    lo, hi = float(grid[0]) - 6 * h, float(grid[-1]) + 6 * h
    dx = (hi - lo) / nbins
    cnt, _ = np.histogram(z, bins=nbins, range=(lo, hi))
    sig = max(h / dx, 1e-3)
    m = int(np.ceil(6 * sig))
    k = np.exp(-0.5 * (np.arange(-m, m + 1) / sig) ** 2)
    k /= k.sum()
    dens = np.convolve(cnt.astype(float), k, mode="same")
    centers = lo + (np.arange(nbins) + 0.5) * dx
    dens = dens / (z.size * dx)
    return np.interp(grid, centers, dens)


def standardize(x):
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    s = 1.4826 * mad if mad > 0 else float(np.std(x))
    return (x - med) / max(s, 1e-9), med, s


def two_slopes(grid, logf, band, z=None, dens=None):
    """在给定 |z| 区间内对 log f 做直线拟合，返回 (左斜率, 右斜率)。

    只在**密度仍有数值意义**的区间内拟合：否则小样本类别会掉进 1e-300 的
    数值地板，log 值变成 −690，拟合出 −629 这种假斜率。
    """
    floor = float(np.max(dens)) * 1e-4 if dens is not None else 0.0
    out = []
    for sign in (-1, 1):
        lo, hi = sorted((sign * band[0], sign * band[1]))
        if z is not None:
            # 远端不超过实测分位，保证区间里真的有点
            edge = float(np.percentile(np.abs(z), 99.0))
            if sign > 0:
                hi = min(hi, edge)
            else:
                lo = max(lo, -edge)
        if hi - lo < 0.3:
            out.append(float("nan"))
            continue
        sel = (grid >= lo) & (grid <= hi) & np.isfinite(logf)
        if dens is not None:
            sel &= dens > floor
        out.append(float(np.polyfit(grid[sel], logf[sel], 1)[0]) if sel.sum() > 8
                   else float("nan"))
    return out[0], out[1]


def hazard(grid, dens):
    """风险率 h(z) = f(z)/S(z)，用 KDE 自身积分算生存函数。"""
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1])
                                           * np.diff(grid))])
    S = np.clip(1.0 - cdf, 1e-12, None)
    return dens / S


def fit_mix2(z, iters=200, seed=0):
    """一维两成分高斯混合的 EM（用于对照"尖峰是不是两机制叠加"）。"""
    rng = np.random.default_rng(seed)
    mu = np.array([np.percentile(z, 30), np.percentile(z, 70)])
    sd = np.array([np.std(z) * 0.6, np.std(z) * 1.2])
    w = 0.7
    for _ in range(iters):
        p1 = w * np.exp(-0.5 * ((z - mu[0]) / sd[0]) ** 2) / sd[0]
        p2 = (1 - w) * np.exp(-0.5 * ((z - mu[1]) / sd[1]) ** 2) / sd[1]
        tot = p1 + p2 + 1e-300
        r1 = p1 / tot
        w = float(r1.mean())
        mu[0] = float((r1 * z).sum() / max(r1.sum(), 1e-9))
        mu[1] = float(((1 - r1) * z).sum() / max((1 - r1).sum(), 1e-9))
        sd[0] = float(np.sqrt(max((r1 * (z - mu[0]) ** 2).sum() / max(r1.sum(), 1e-9), 1e-6)))
        sd[1] = float(np.sqrt(max(((1 - r1) * (z - mu[1]) ** 2).sum()
                                  / max((1 - r1).sum(), 1e-9), 1e-6)))
    return w, mu, sd


def mix2_pdf(grid, w, mu, sd):
    return (w * np.exp(-0.5 * ((grid - mu[0]) / sd[0]) ** 2) / (sd[0] * np.sqrt(2 * np.pi))
            + (1 - w) * np.exp(-0.5 * ((grid - mu[1]) / sd[1]) ** 2)
            / (sd[1] * np.sqrt(2 * np.pi)))


def references(grid):
    """方差归一化的参考曲线。"""
    ref = {}
    ref["标准高斯"] = np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi)
    b = 1.0 / np.sqrt(2.0)          # 拉普拉斯，方差=1
    ref["拉普拉斯(方差同)"] = np.exp(-np.abs(grid) / b) / (2 * b)
    for df in (3, 5):
        sc = np.sqrt((df - 2) / df)  # t 分布，方差=1
        from math import lgamma, pi, sqrt
        c = np.exp(lgamma((df + 1) / 2) - lgamma(df / 2)) / sqrt(df * pi)
        ref[f"学生t(df={df})"] = c / sc * (1 + ((grid / sc) ** 2) / df) ** (-(df + 1) / 2)
    return ref


def load(path: Path):
    out = []
    with path.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            e = r["e_along"]
            if e in ("", "None"):
                continue
            out.append({"cls": r["cls"], "h": float(r["h"]), "e": float(e)})
    return out


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "error_dist" / "rows.csv"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs" / "error_dist" / "形状诊断"
    out.mkdir(parents=True, exist_ok=True)
    rows = load(src)
    grid = np.linspace(-Z_RANGE, Z_RANGE, 1201)
    ref = references(grid)
    print(f"输入 {src}（{len(rows)} 条）\n输出 {out}")

    table = [("类别", "梯度", "n", "中位(px)", "稳健sigma(px)",
              "近尾斜率(左)", "近尾斜率(右)", "远尾斜率(右)", "远尾/近尾(右)",
              "风险率走势", "混合权重w", "混合两峰间距(z)")]
    summary: dict = {}

    for cls in ORDER:
        for h in HORIZONS:
            x = np.array([r["e"] for r in rows if r["cls"] == cls and r["h"] == h])
            if x.size < 60:
                continue
            z, med, sc = standardize(x)
            dens = kde(z, grid, 0.20)
            logf = np.log(np.clip(dens, 1e-300, None))
            l_near, r_near = two_slopes(grid, logf, (1.0, 2.5), z, dens)
            _, r_far = two_slopes(grid, logf, (2.5, 4.0), z, dens)
            hz = hazard(grid, dens)
            sel = np.abs(grid) <= 2.0
            hz_trend = float(np.polyfit(np.abs(grid[sel]), np.log(np.clip(hz[sel], 1e-12, None)), 1)[0])
            trend = ("上升（比指数轻）" if hz_trend > 0.05 else
                     "下降（重尾/次指数）" if hz_trend < -0.05 else "近似水平（指数尾）")
            w, mu, sd = fit_mix2(z)
            gap = abs(mu[1] - mu[0]) / np.sqrt(0.5 * (sd[0] ** 2 + sd[1] ** 2))
            summary[(cls, h)] = {"z": z, "dens": dens, "logf": logf, "hz": hz,
                                 "ref": ref, "w": w, "mu": mu, "sd": sd,
                                 "med": med, "scale": sc, "n": x.size,
                                 "r_near": r_near, "r_far": r_far, "trend": trend}
            table.append((CLASS_ZH.get(cls, cls), h, x.size, round(med, 2),
                          round(sc, 2), round(l_near, 3), round(r_near, 3),
                          round(r_far, 3),
                          round(r_far / r_near, 3) if r_near else "",
                          trend, round(w, 3), round(gap, 3)))

    with (out / "形状诊断.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows(table)

    # ---------------- 每类别独立一张形状诊断页（1 s）
    for cls in ORDER:
        d = summary.get((cls, 1.0))
        if not d:
            continue
        z, dens, logf, hz = d["z"], d["dens"], d["logf"], d["hz"]
        fig, axes = plt.subplots(2, 3, figsize=(17, 9))

        ax = axes[0][0]
        for bw, c in zip(BANDWIDTHS, ["#4C78A8", "#54A24B", "#F58518", "#E45756"]):
            ax.plot(grid, kde(z, grid, bw), color=c, lw=1.5, label=f"KDE h={bw}")
        ax.set_ylim(0, min(float(np.max(dens)) * 1.4, 3.0))
        ax.set_title("带宽扫描（形状是否随带宽改变）")
        ax.set_xlabel("标准化误差 z"); ax.set_ylabel("密度"); ax.legend(fontsize=8)
        _note(ax, f"n = {d['n']}\n中位 {d['med']:+.1f} px\n稳健 σ {d['scale']:.1f} px")

        ax = axes[0][1]
        ax.plot(grid, logf, color="0.2", lw=2.2, label="实测 KDE")
        for name, r in ref.items():
            ax.plot(grid, np.log(np.clip(r, 1e-300, None)), lw=1.4,
                    ls="--" if "高斯" in name else ":", label=name)
        ax.plot(grid, np.log(np.clip(mix2_pdf(grid, d["w"], d["mu"], d["sd"]), 1e-300, None)),
                color="#000000", lw=1.6, alpha=0.7, label="两成分混合")
        ax.set_ylim(-jnp_floor(), 1.0)
        ax.set_title("log f(z)：抛物线=高斯，V=拉普拉斯")
        ax.set_xlabel("z"); ax.set_ylabel("log 密度"); ax.legend(fontsize=7, ncol=2)
        _note(ax, f"近尾斜率 {d['r_near']:.2f}\n远尾斜率 {d['r_far']:.2f}\n"
                  f"远/近 {d['r_far'] / d['r_near']:.2f}（<1 = 比指数更重）"
                  if d["r_near"] else "")

        ax = axes[0][2]
        zs = np.sort(z)
        S = 1.0 - (np.arange(1, zs.size + 1)) / (zs.size + 1)
        ax.semilogy(zs, S, color="0.2", lw=2.2, label="实测生存函数")
        for name, r in ref.items():
            cdf = np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(grid))])
            ax.semilogy(grid, np.clip(1 - cdf, 1e-300, None), lw=1.3,
                        ls="--" if "高斯" in name else ":", label=name)
        ax.set_ylim(1e-4, 1.5)
        ax.set_title("log S(z)：直线=指数尾")
        ax.set_xlabel("z"); ax.set_ylabel("P(Z > z)"); ax.legend(fontsize=7, ncol=2)

        ax = axes[1][0]
        ax.plot(grid, hz, color="#E45756", lw=2.0)
        ax.set_ylim(0, min(float(np.percentile(hz[np.abs(grid) <= 3], 99)) * 1.3, 40))
        ax.set_title(f"风险率 h(z)=f/S —— {d['trend']}")
        ax.set_xlabel("z"); ax.set_ylabel("h(z)")

        ax = axes[1][1]
        fine = np.linspace(-0.8, 0.8, 601)
        ax.plot(fine, kde(z, fine, 0.04), color="0.2", lw=2.0, label="KDE h=0.04（放大）")
        ax.plot(fine, mix2_pdf(fine, d["w"], d["mu"], d["sd"]), color="#000000",
                lw=1.5, ls="-", alpha=0.7, label="两成分混合")
        b = 1.0 / np.sqrt(2.0)
        ax.plot(fine, np.exp(-np.abs(fine) / b) / (2 * b), color="#F58518", lw=1.5,
                ls="--", label="拉普拉斯（V 形）")
        ax.plot(fine, np.exp(-0.5 * fine ** 2) / np.sqrt(2 * np.pi), color="#4C78A8",
                lw=1.5, ls=":", label="高斯（圆顶）")
        ax.set_title("峰形放大：圆顶 还是 V 形尖点")
        ax.set_xlabel("z"); ax.set_ylabel("密度"); ax.legend(fontsize=8)

        ax = axes[1][2]
        q = stats_norm_ppf((np.arange(1, z.size + 1) - 0.5) / z.size)
        ax.plot(q, np.sort(z), ".", ms=2, color="#4C78A8", alpha=0.5)
        lim = 4.5
        ax.plot([-lim, lim], [-lim, lim], color="0.4", lw=1.3, ls="--", label="若为高斯")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title("正态 Q-Q：上弯=右尾重，下弯=左尾重")
        ax.set_xlabel("理论正态分位"); ax.set_ylabel("实测分位 z"); ax.legend(fontsize=8)

        fig.suptitle(f"{CLASS_ZH.get(cls, cls)}（{cls}）　形状诊断（先估曲线，再判分布律）"
                     f"　梯度 1 s", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out / f"形状_{CLASS_ZH.get(cls, cls)}_{cls}.png", dpi=165)
        plt.close(fig)

    # ---------------- 汇总：所有类别的 log f 叠加
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    for cls in ORDER:
        d = summary.get((cls, 1.0))
        if not d:
            continue
        axes[0].plot(grid, d["logf"], color=COLORS[cls], lw=1.8,
                     label=f"{CLASS_ZH.get(cls, cls)}")
    for name, r in ref.items():
        if "高斯" in name or "拉普拉斯" in name:
            axes[0].plot(grid, np.log(np.clip(r, 1e-300, None)), color="0.35", lw=1.4,
                         ls="--", label=name)
    axes[0].set_ylim(-jnp_floor(), 1.2)
    axes[0].set_title("标准化后各类型的 log f(z)")
    axes[0].set_xlabel("z"); axes[0].set_ylabel("log 密度"); axes[0].legend(fontsize=8)

    labels, near, far = [], [], []
    for cls in ORDER:
        d = summary.get((cls, 1.0))
        if not d:
            continue
        labels.append(CLASS_ZH.get(cls, cls))
        near.append(abs(d["r_near"])); far.append(abs(d["r_far"]))
    idx = np.arange(len(labels))
    axes[1].bar(idx - 0.2, near, 0.4, color="#4C78A8", label="近尾 |斜率|（|z| 1~2.5）")
    axes[1].bar(idx + 0.2, far, 0.4, color="#E45756", label="远尾 |斜率|（|z| 2.5~4）")
    axes[1].axhline(1.0, color="0.4", lw=1.2, ls=":")
    axes[1].text(0.05, 1.03, "斜率 1 = 指数尾（拉普拉斯）", fontsize=8, color="0.35")
    axes[1].set_xticks(idx); axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("log f 的衰减斜率")
    axes[1].set_title("尾巴衰减得比指数快还是慢")
    axes[1].legend(fontsize=9)
    fig.suptitle("图D　从曲线形状反推分布律（1 s，标准化后）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "形状汇总.png", dpi=180)
    plt.close(fig)

    md = ["# 先估曲线、再判分布律", "",
          f"数据源 `{src}`；产物 `{out}`。", "",
          "| " + " | ".join(table[0]) + " |", "|" + "---|" * len(table[0])]
    md += ["| " + " | ".join(str(v) for v in row) + " |" for row in table[1:]]
    md += ["", "## 读法", "",
           "- **log f(z)**：抛物线→高斯；V 形（线性）→拉普拉斯；两段线性+拐点→两机制叠加。",
           "- **log S(z)**：直线→指数尾；向下弯→比指数轻；向上弯→比指数重。",
           "- **风险率 h(z)**：上升→比指数轻；水平→指数；下降→重尾。",
           "- **两段斜率**：远尾比近尾更平（比值<1）→ 尾部比指数更重；更陡（>1）→ 向轻尾收敛。",
           "- **峰形放大**：圆顶=平滑密度；V 形=折点密度。",
           "",
           "带宽扫描一栏用于排除「形状是带宽造出来的」这一可能。", ""]
    (out / "形状诊断说明.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n产物写入 {out}")


def jnp_floor():
    return 9.0


def stats_norm_ppf(p):
    from scipy import stats
    return stats.norm.ppf(p)


if __name__ == "__main__":
    main()
