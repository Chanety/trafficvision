"""误差分布族的正经横评：平滑模型 vs 带折点的模型。

起因：先前选用 ALD（不对称拉普拉斯）的理由只有"比高斯对数似然高"，那是尾部证据，
没验证它在 μ 处的**尖点**。本脚本把这个问题当成模型选择来做：

候选（k = 自由参数个数，用于 AIC/BIC 惩罚）
  1. normal          高斯                     k=2   平滑
  2. laplace         拉普拉斯                  k=2   折点
  3. skewnorm        偏正态                    k=3   平滑
  4. student-t       学生 t                    k=3   平滑（重尾）
  5. split-normal    两片高斯                  k=3   折点
  6. asymmetric-laplace 不对称拉普拉斯          k=3   折点
  7. mix2            两成分高斯混合            k=5   平滑（可同时造出尖峰与重尾）

判据：
  * logL（越大越好）、AIC = 2k − 2logL、BIC = k·ln(n) − 2logL（越小越好）
  * KS 距离（模型 CDF 与经验 CDF 的最大偏差）
  * **近峰质量**：P(|x − mode| ≤ 0.1·scale) 的实测值与各模型预测值之比
    —— 这是直接检验"尖点/尖峰是不是真的存在"的量

用法：python bench/error_dist_compare.py [rows.csv] [输出目录]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize

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
MODELS = ["normal", "laplace", "skewnorm", "student-t", "split-normal",
          "asym-laplace", "mix2"]
MODEL_ZH = {"normal": "高斯", "laplace": "拉普拉斯", "skewnorm": "偏正态",
            "student-t": "学生t", "split-normal": "两片高斯",
            "asym-laplace": "不对称拉普拉斯", "mix2": "两成分混合"}
KINKED = {"laplace", "split-normal", "asym-laplace"}
NPARAM = {"normal": 2, "laplace": 2, "skewnorm": 3, "student-t": 3,
          "split-normal": 3, "asym-laplace": 3, "mix2": 5}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _note(ax, txt, loc=0.02):
    ax.text(loc, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.92))


# --------------------------------------------------------------------------- 各模型

def _nll(f):
    def g(theta, x):
        v = f(theta, x)
        if not np.all(np.isfinite(v)):
            return 1e12
        return -float(v.sum())
    return g


def pdf_normal(th, x):
    return stats.norm.pdf(x, th[0], np.exp(th[1]))


def pdf_laplace(th, x):
    return stats.laplace.pdf(x, th[0], np.exp(th[1]))


def pdf_skewnorm(th, x):
    return stats.skewnorm.pdf(x, th[0], th[1], np.exp(th[2]))


def pdf_t(th, x):
    return stats.t.pdf(x, 2.0 + np.exp(th[0]), th[1], np.exp(th[2]))


def pdf_split(th, x):
    mu, s, a = th[0], np.exp(th[1]), np.tanh(th[2]) * 0.999
    sl, sr = s / (1 - a), s / (1 + a)
    c = 2.0 / (sl + sr)
    z = np.where(x < mu, (x - mu) / sl, (x - mu) / sr)
    return c * np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)


def pdf_asymlaplace(th, x):
    kappa = np.exp(th[2])
    return stats.laplace_asymmetric.pdf(x, 1.0 / kappa, th[0], np.exp(th[1]))


def pdf_mix2(th, x):
    w = 1.0 / (1.0 + np.exp(-th[0]))
    m1, s1, m2, s2 = th[1], np.exp(th[2]), th[3], np.exp(th[4])
    return (w * stats.norm.pdf(x, m1, s1) + (1 - w) * stats.norm.pdf(x, m2, s2))


SPECS = {
    "normal": (pdf_normal, 2, lambda x: [np.median(x), np.log(max(x.std(), 1e-6))],
               [(-np.inf, np.inf), (-20, 20)]),
    "laplace": (pdf_laplace, 2, lambda x: [np.median(x), np.log(max(np.mean(np.abs(x - np.median(x))), 1e-6))],
                [(-np.inf, np.inf), (-20, 20)]),
    "skewnorm": (pdf_skewnorm, 3, lambda x: [0.0, np.median(x), np.log(max(x.std(), 1e-6))],
                 [(-20, 20), (-np.inf, np.inf), (-20, 20)]),
    "student-t": (pdf_t, 3, lambda x: [np.log(2.0), np.median(x), np.log(max(x.std(), 1e-6))],
                  [(-5, 8), (-np.inf, np.inf), (-20, 20)]),
    "split-normal": (pdf_split, 3, lambda x: [np.median(x), np.log(max(x.std(), 1e-6)), 0.0],
                     [(-np.inf, np.inf), (-20, 20), (-4, 4)]),
    "asym-laplace": (pdf_asymlaplace, 3, lambda x: [np.median(x), np.log(max(x.std(), 1e-6)), 0.0],
                     [(-np.inf, np.inf), (-20, 20), (-6, 6)]),
    "mix2": (pdf_mix2, 5,
             lambda x: [0.0, float(np.percentile(x, 25)), np.log(max(x.std(), 1e-6)),
                        float(np.percentile(x, 75)), np.log(max(x.std(), 1e-6))],
             [(-8, 8), (-np.inf, np.inf), (-20, 20), (-np.inf, np.inf), (-20, 20)]),
}


def _ll(pdf, th, x):
    """对数似然：注意是 log 密度之和，不是密度之和。"""
    v = pdf(th, x)
    if not np.all(np.isfinite(v)):
        return -np.inf
    # 极端处密度会下溢到 0，按数值地板处理，避免把"下溢"误判成"零密度"
    return float(np.log(np.maximum(v, 1e-300)).sum())


def _em_mix2(x, restarts=6):
    """两成分高斯混合的 EM（多次重启，取最优），比数值梯度更稳。"""
    best = (-np.inf, None)
    for r in range(restarts):
        rng = np.random.default_rng(r)
        q = [(20, 60), (10, 70), (30, 80), (5, 40), (40, 90), (25, 55)][r % 6]
        m = np.array([np.percentile(x, q[0]), np.percentile(x, q[1])], dtype=float)
        sd = np.array([np.std(x) * 0.5, np.std(x) * 1.3], dtype=float)
        w = 0.7
        for _ in range(400):
            p1 = w * np.exp(-0.5 * ((x - m[0]) / sd[0]) ** 2) / sd[0]
            p2 = (1 - w) * np.exp(-0.5 * ((x - m[1]) / sd[1]) ** 2) / sd[1]
            tot = p1 + p2 + 1e-300
            r1 = p1 / tot
            w = float(r1.mean())
            s1, s2 = r1.sum(), (1 - r1).sum()
            m[0] = float((r1 * x).sum() / max(s1, 1e-9))
            m[1] = float(((1 - r1) * x).sum() / max(s2, 1e-9))
            sd[0] = float(np.sqrt(max((r1 * (x - m[0]) ** 2).sum() / max(s1, 1e-9), 1e-9)))
            sd[1] = float(np.sqrt(max(((1 - r1) * (x - m[1]) ** 2).sum() / max(s2, 1e-9), 1e-9)))
        wf = 1.0 / (1.0 + np.exp(-np.log(max(w, 1e-9) / max(1 - w, 1e-9))))
        th = np.array([np.log(wf / (1 - wf)), m[0], np.log(sd[0]), m[1], np.log(sd[1])])
        v = _ll(pdf_mix2, th, x)
        if v > best[0]:
            best = (v, th)
    return best[1] if best[1] is not None else None


def _fit_split_mle(x):
    """两片正态的 MLE：分布函数在 μ 处不可导，梯度法会跑飞，改用网格搜索。"""
    med = float(np.median(x))
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    s0 = max(iqr / 1.349, 1e-6)
    best = (-np.inf, None)

    def ev(mu, s, a):
        th = np.array([mu, np.log(s), np.arctanh(a)])
        return _ll(pdf_split, th, x), th

    for mu in (med - 0.15 * s0, med, med + 0.15 * s0):
        for a in np.arange(-0.85, 0.851, 0.05):
            for s in s0 * np.logspace(-1.0, 1.0, 41):
                v, th = ev(mu, s, a)
                if v > best[0]:
                    best = (v, th)
    if best[1] is None:
        return None
    _, th0 = best
    mu0, s_best, a0 = th0[0], np.exp(th0[1]), np.tanh(th0[2])
    for mu in np.linspace(mu0 - 0.05 * s0, mu0 + 0.05 * s0, 5):
        for a in np.linspace(max(-0.85, a0 - 0.05), min(0.85, a0 + 0.05), 11):
            for s in np.linspace(s_best * 0.9, s_best * 1.1, 11):
                v, th = ev(mu, s, a)
                if v > best[0]:
                    best = (v, th)
    return best[1]


def fit_model(name, x):
    k = NPARAM[name]
    try:
        if name == "normal":
            p = stats.norm.fit(x)
            th = np.array([p[0], np.log(p[1])])
        elif name == "laplace":
            p = stats.laplace.fit(x)
            th = np.array([p[0], np.log(p[1])])
        elif name == "skewnorm":
            p = stats.skewnorm.fit(x)
            th = np.array([p[0], p[1], np.log(p[2])])
        elif name == "student-t":
            p = stats.t.fit(x)
            th = np.array([np.log(max(p[0] - 2.0, 1e-3)), p[1], np.log(p[2])])
        elif name == "asym-laplace":
            p = stats.laplace_asymmetric.fit(x)
            th = np.array([p[1], np.log(p[2]), -np.log(p[0])])
        elif name == "split-normal":
            th = _fit_split_mle(x)
        elif name == "mix2":
            th = _em_mix2(x)
        else:
            return None
    except Exception:
        return None
    if th is None or not np.all(np.isfinite(th)):
        return None
    ll = _ll(SPECS[name][0], th, x)
    if not np.isfinite(ll):
        return None
    n = x.size
    # KS 距离
    xs = np.sort(x)
    cdf = _cdf_for(name, th)
    F = np.clip(cdf(xs), 0.0, 1.0)
    i = np.arange(1, n + 1)
    ks = float(np.maximum(np.abs(F - i / n), np.abs(F - (i - 1) / n)).max())
    return {"model": name, "theta": th, "logL": ll, "k": k,
            "AIC": 2 * k - 2 * ll, "BIC": k * np.log(n) - 2 * ll, "KS": ks}


def _cdf_for(name, th):
    if name == "normal":
        return lambda v: stats.norm.cdf(v, th[0], np.exp(th[1]))
    if name == "laplace":
        return lambda v: stats.laplace.cdf(v, th[0], np.exp(th[1]))
    if name == "skewnorm":
        return lambda v: stats.skewnorm.cdf(v, th[0], th[1], np.exp(th[2]))
    if name == "student-t":
        return lambda v: stats.t.cdf(v, 2.0 + np.exp(th[0]), th[1], np.exp(th[2]))
    if name == "asym-laplace":
        return lambda v: stats.laplace_asymmetric.cdf(v, 1.0 / np.exp(th[2]),
                                                       th[0], np.exp(th[1]))
    if name == "split-normal":
        mu, s, a = th[0], np.exp(th[1]), np.tanh(th[2]) * 0.999
        sl, sr = s / (1 - a), s / (1 + a)
        c = 2.0 / (sl + sr)
        return lambda v: np.where(
            v < mu, c * sl * stats.norm.cdf((v - mu) / sl),
            c * sl * 0.5 + c * sr * (stats.norm.cdf((v - mu) / sr) - 0.5))
    if name == "mix2":
        w = 1.0 / (1.0 + np.exp(-th[0]))
        return lambda v: (w * stats.norm.cdf(v, th[1], np.exp(th[2]))
                          + (1 - w) * stats.norm.cdf(v, th[3], np.exp(th[4])))
    raise KeyError(name)


def peak_fraction(x, th, name, radius=0.1):
    """近峰质量：mode 附近 ±radius·scale 内的比例（实测 vs 模型预测）。"""
    pdf = SPECS[name][0]
    grid = np.linspace(x.min(), x.max(), 4000)
    p = pdf(th, grid)
    mode = float(grid[int(np.argmax(p))])
    scale = float(np.std(x))
    lo, hi = mode - radius * scale, mode + radius * scale
    emp = float(np.mean((x >= lo) & (x <= hi)))
    pred = float(np.trapezoid(pdf(th, np.linspace(lo, hi, 400)), np.linspace(lo, hi, 400)))
    return emp, pred, mode


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
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs" / "error_dist" / "模型横评"
    out.mkdir(parents=True, exist_ok=True)
    rows = load(src)
    print(f"输入 {src}（{len(rows)} 条）\n输出 {out}")

    table = [("类别", "梯度", "n", "最优(BIC)", "最优(KS)", "logL差额",
              "AIC(高斯)", "AIC(拉普拉斯)", "AIC(学生t)", "AIC(两片高斯)",
              "AIC(不对称拉普拉斯)", "AIC(两成分混合)",
              "近峰实测", "近峰(高斯)", "近峰(学生t)", "近峰(混合)", "近峰(两片)")]
    fits: dict = {}
    for cls in ORDER:
        for h in HORIZONS:
            x = np.array([r["e"] for r in rows if r["cls"] == cls and r["h"] == h])
            if x.size < 60:
                continue
            res = {}
            for name in MODELS:
                f = fit_model(name, x)
                if f:
                    res[name] = f
            if not res:
                continue
            fits[(cls, h)] = res
            best_bic = min(res, key=lambda m: res[m]["BIC"])
            best_ks = min(res, key=lambda m: res[m]["KS"])
            emp, preds, _ = peak_fraction(x, res[best_bic]["theta"], best_bic)
            pf = {}
            for m in ("normal", "student-t", "mix2", "split-normal"):
                if m in res:
                    _, pred, _ = peak_fraction(x, res[m]["theta"], m)
                    pf[m] = pred
                else:
                    pf[m] = float("nan")
            table.append((CLASS_ZH.get(cls, cls), h, x.size,
                          MODEL_ZH[best_bic], MODEL_ZH[best_ks],
                          round(res[best_bic]["BIC"] - res[best_ks]["BIC"], 1),
                          *[round(res[m]["AIC"], 1) if m in res else ""
                            for m in ("normal", "laplace", "student-t",
                                      "split-normal", "asym-laplace", "mix2")],
                          round(emp, 4), round(pf["normal"], 4),
                          round(pf["student-t"], 4), round(pf["mix2"], 4),
                          round(pf["split-normal"], 4)))
            print(f"  {cls} {h}s n={x.size}  BIC→{best_bic:<14} KS→{best_ks:<14} "
                  f"近峰实测={emp:.4f}")

    with (out / "模型横评.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows(table)

    # ---- 图1：每类别 1 s 的候选曲线叠加
    for cls in ORDER:
        res = fits.get((cls, 1.0))
        if not res:
            continue
        x = np.array([r["e"] for r in rows if r["cls"] == cls and r["h"] == 1.0])
        lim = float(np.percentile(np.abs(x), 99.5)) or 1.0
        grid = np.linspace(-lim, lim, 900)
        fig, ax = plt.subplots(figsize=(9.4, 5.4))
        ax.hist(x, bins=140, range=(-lim, lim), density=True, color="#D9D9D9",
                alpha=0.9, label="实测")
        kd = stats.gaussian_kde(x[::max(1, x.size // 4000)])
        ax.plot(grid, kd(grid), color="0.25", lw=1.6, label="实测 KDE")
        style = {"normal": ("#4C78A8", ":"), "student-t": ("#54A24B", "-."),
                 "skewnorm": ("#B279A2", (0, (3, 1, 1, 1))),
                 "asym-laplace": ("#F58518", "--"), "split-normal": ("#E45756", "--"),
                 "mix2": ("#000000", "-")}
        for m, (c, ls) in style.items():
            if m in res:
                ax.plot(grid, SPECS[m][0](res[m]["theta"], grid), color=c, ls=ls,
                        lw=1.9 if m == "mix2" else 1.5,
                        label=f"{MODEL_ZH[m]} ({'折点' if m in KINKED else '平滑'})")
        ax.set_ylim(0, min(ax.get_ylim()[1], 4 * max(kd(grid).max(), 1e-9)))
        best = min(res, key=lambda m: res[m]["BIC"])
        _note(ax, f"n = {x.size}\nBIC 最优：{MODEL_ZH[best]}\n"
                  f"KS 最优：{MODEL_ZH[min(res, key=lambda m: res[m]['KS'])]}")
        ax.legend(fontsize=8, ncol=2)
        ax.set_title(f"{CLASS_ZH.get(cls, cls)}（{cls}）　分布族横评　梯度 1 s")
        ax.set_xlabel("e_along (px)　正=滞后　负=超前")
        ax.set_ylabel("概率密度")
        fig.tight_layout()
        fig.savefig(out / f"横评_{CLASS_ZH.get(cls, cls)}_{cls}.png", dpi=170)
        plt.close(fig)

    # ---- 图2：AIC 差值汇总 + 近峰质量对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    base = "normal"
    labels, gaps = [], {m: [] for m in MODELS if m != base}
    for cls in ORDER:
        res = fits.get((cls, 1.0))
        if not res or base not in res:
            continue
        labels.append(CLASS_ZH.get(cls, cls))
        for m in gaps:
            gaps[m].append(res[m]["AIC"] - res[base]["AIC"] if m in res else np.nan)
    idx = np.arange(len(labels))
    w = 0.8 / max(1, len(gaps))
    for i, (m, vals) in enumerate(gaps.items()):
        axes[0].bar(idx + i * w - 0.4, vals, w,
                    color=COLORS_ALT.get(m, "#888888"),
                    label=f"{MODEL_ZH[m]}{'（折点）' if m in KINKED else '（平滑）'}")
    axes[0].axhline(0, color="0.4", lw=1.2)
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("AIC 相对高斯（负=更好）")
    axes[0].set_title("各模型相对高斯的 AIC 改进")
    axes[0].legend(fontsize=7, ncol=2)

    emp_all, pf_t = [], {m: [] for m in ("normal", "student-t", "mix2", "split-normal")}
    for cls in ORDER:
        res = fits.get((cls, 1.0))
        if not res:
            continue
        x = np.array([r["e"] for r in rows if r["cls"] == cls and r["h"] == 1.0])
        best = min(res, key=lambda m: res[m]["BIC"])
        emp, _, _ = peak_fraction(x, res[best]["theta"], best)
        emp_all.append(emp)
        for m in pf_t:
            if m in res:
                _, pred, _ = peak_fraction(x, res[m]["theta"], m)
                pf_t[m].append(pred)
            else:
                pf_t[m].append(np.nan)
    xpos = np.arange(len(emp_all))
    axes[1].bar(xpos - 2 * w, emp_all, w, color="#BAB0AC", label="实测近峰质量")
    for i, (m, vals) in enumerate(pf_t.items()):
        axes[1].bar(xpos + (i - 1) * w, vals, w, color=COLORS_ALT.get(m, "#888888"),
                    label=MODEL_ZH[m])
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([CLASS_ZH.get(c, c) for c in ORDER][:len(xpos)], fontsize=9)
    axes[1].set_ylabel("mode 附近 ±0.1σ 内的比例")
    axes[1].set_title("近峰质量：实测 vs 各模型预测")
    axes[1].legend(fontsize=8)
    fig.suptitle("图C　分布族比较（1 s）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "模型比较汇总.png", dpi=180)
    plt.close(fig)

    md = ["# 误差分布族横评", "",
          f"数据源 `{src}`；产物 `{out}`。误差指沿运动方向的 e_along。", "",
          "## 候选模型（k = 自由参数数）", "",
          "| 模型 | 平滑性 | k |", "|---|---|---|"]
    for m in MODELS:
        md.append(f"| {MODEL_ZH[m]} | {'折点（μ 处导数不连续）' if m in KINKED else '平滑'} | "
                  f"{NPARAM[m]} |")
    md += ["", "## 结果", "",
           "| " + " | ".join(table[0]) + " |", "|" + "---|" * len(table[0])]
    md += ["| " + " | ".join(str(v) for v in row) + " |" for row in table[1:]]
    md += ["", "## 读法", "",
           "- **BIC** 惩罚参数个数，是「该不该增加复杂度」的裁判；AIC 更宽松。",
           "- **近峰实测 > 所有平滑模型的预测** 说明峰比平滑分布能给的更尖——"
           "这才是「尖点」真正存在的证据；若两成分混合能同时命中近峰与尾部，"
           "说明尖峰其实是「两个机制叠加」的产物，而不是密度本身带折点。",
           "- 折点模型（拉普拉斯族、两片高斯）在 μ 处导数不连续，物理上对应"
           "「误差被某个极端项支配」的极限情形；检测噪声由大量独立小扰动叠加而成时"
           "更可能是平滑的。"]
    (out / "模型横评说明.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n产物写入 {out}")


COLORS_ALT = {"laplace": "#F58518", "skewnorm": "#B279A2", "student-t": "#54A24B",
              "split-normal": "#E45756", "asym-laplace": "#8C6D31", "mix2": "#000000"}


if __name__ == "__main__":
    main()
