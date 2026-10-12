"""按学术论文图形规范重做竞赛报告用图。

规范来源（三处一致）：
  * plotting-agent/references/chart-patterns.md —— 全局样式与低饱和学术配色
  * plotting-agent/references/aspect-ratios.md —— 12 种允许长宽比与像素目标
  * plot-from-data/references/bar_grouped_hatch.md、bar_paired_delta.md
    —— 分组柱与配对对比的具体参数（柱顶数值、斜线填充、开口坐标轴）

硬性要求：300 DPI；线图藏上/右边框；只用低饱和配色；每图配一句 caption。
数据来源：outputs/error_dist/（与实验日志逐条一致）。

输出：paper/competition/figures/*.png 与 captions.json
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "outputs" / "error_dist"
OUT = ROOT / "paper" / "competition" / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["SimSun", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "axes.linewidth": 0.7,
    "legend.fontsize": 7,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "grid.alpha": 0.16,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "axes.unicode_minus": False,
    "hatch.color": "white",
    "hatch.linewidth": 0.8,
    # 数学符号不要走 Unicode 字符（宋体缺下标字形，会渲染成方框）；
    # mathtext 自带字体，配合 STIX 与衬线正文更协调
    "mathtext.fontset": "stix",
})

# 学术配色（chart-patterns 规范）
BLUE, RED, GREEN, ORANGE, PURPLE, GOLD, GRAY = (
    "#2060cc", "#cc3030", "#208040", "#cc7020", "#8040cc", "#b08020", "#666666")

CLS_ORDER = ["汽车", "非机动车", "公交", "行人", "标志牌", "卡车"]


def size_for(ratio: str) -> tuple[float, float]:
    return {"16:9": (5.5, 3.09), "5:4": (4.5, 3.6), "3:2": (5.1, 3.4),
            "4:3": (4.0, 3.0)}[ratio]


def no_top_right(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_signed():
    xs = defaultdict(list)
    with (DIST / "rows.csv").open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            v = r["e_along"]
            if v in ("", "None"):
                continue
            xs[float(r["h"])].append(float(v))
    return xs


def load_mixture():
    out = defaultdict(dict)
    p = DIST / "尺度混合" / "尺度混合_类别梯度.csv"
    with p.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out[r["类别"]][float(r["梯度"])] = {
                "w": float(r["w(主体)"]), "ratio": float(r["尺度比"]),
                "share": float(r["尾部方差占比"])}
    return out


# --------------------------------------------------------------------------- 1

def fig_static_fix() -> None:
    """静态结构零位移预判：前后对比（对数纵轴，避免柱长误导）。"""
    hs = [0.1, 0.5, 1.0, 2.0]
    before = [4.3, 15.9, 31.9, 64.3]
    after = [2.9, 6.7, 9.5, 10.6]
    fig, ax = plt.subplots(figsize=size_for("5:4"))
    ax.plot(hs, before, "o-", color=RED, label="当前：按估计速度外推")
    ax.plot(hs, after, "s-", color=BLUE, label="改为零位移预判")
    for i, h in enumerate(hs):
        ax.annotate(f"-{100 * (before[i] - after[i]) / before[i]:.0f}%",
                    xy=(h, after[i]), xytext=(0, -13), textcoords="offset points",
                    ha="center", fontsize=7.5, fontweight="bold", color="#cc2200")
        ax.annotate(f"{before[i]:.1f}", xy=(h, before[i]), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=6.5, color=RED)
        ax.annotate(f"{after[i]:.1f}", xy=(h, after[i]), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=6.5, color=BLUE)
    ax.set_yscale("log")
    ax.set_ylim(1.8, 130)
    ax.set_xticks(hs)
    ax.set_xticklabels([f"{h:g}" for h in hs])
    ax.set_xlabel("预判梯度（秒）")
    ax.set_ylabel("误差中位（像素，对数刻度）")
    ax.set_title("标志牌类：零位移预判的局部收益")
    ax.grid(axis="y", linestyle="--")
    ax.legend(loc="upper left")
    no_top_right(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_static_fix.png")
    plt.close(fig)


# --------------------------------------------------------------------------- 2

def fig_error_attribution() -> None:
    """样本占比 vs 误差占比：分组柱 + 主系列斜线填充 + 柱顶数值。"""
    share_n = [65.9, 17.2, 7.7, 5.8, 2.0, 1.5]
    share_e = [68.5, 17.8, 5.5, 5.0, 1.6, 1.5]
    fig, ax = plt.subplots(figsize=size_for("5:4"))
    x = np.arange(len(CLS_ORDER))
    w = 0.36
    b1 = ax.bar(x - w / 2, share_n, w, color="#F5C5A3", edgecolor="white",
                linewidth=0.5, label="样本数占比")
    b2 = ax.bar(x + w / 2, share_e, w, color="#C0392B", edgecolor="white",
                linewidth=0.5, hatch="//", label="误差总量占比")
    for bars, best in ((b1, False), (b2, True)):
        for r in bars:
            ax.annotate(f"{r.get_height():.1f}", (r.get_x() + r.get_width() / 2,
                                                 r.get_height()),
                        xytext=(0, 2.5), textcoords="offset points", ha="center",
                        fontsize=6.3, color="#8B0000" if best else "#333333",
                        fontweight="bold" if best else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels(CLS_ORDER)
    ax.set_ylabel("占全部样本 / 全部误差的比例（%）")
    ax.set_ylim(0, 80)
    ax.set_title("像素误差质量的类别归属")
    ax.grid(axis="y", linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_error_attribution.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- 3

def fig_signed_error() -> None:
    xs = load_signed()
    fig, axes = plt.subplots(2, 2, figsize=size_for("16:9"))
    for ax, h in zip(axes.ravel(), (0.1, 0.5, 1.0, 2.0)):
        a = np.array(xs[h])
        lim = float(np.percentile(np.abs(a), 99))
        ax.hist(a, bins=70, range=(-lim, lim), color=BLUE, alpha=0.85,
                edgecolor="white", linewidth=0.2)
        ax.axvline(0, color=GRAY, linewidth=0.8)
        med = float(np.median(a))
        ax.axvline(med, color=RED, linewidth=1.2, linestyle="--")
        ax.set_title(f"{h:g} s", fontsize=8)
        ax.set_xlabel("沿运动方向误差（像素）", fontsize=7)
        ax.set_ylabel("样本数", fontsize=7)
        ax.annotate(f"n={a.size}\n中位 {med:+.1f}\n峰度 {float(((a - a.mean()) ** 4).mean() / a.std() ** 4):.1f}",
                    xy=(0.03, 0.96), xycoords="axes fraction", va="top", fontsize=6.2,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.5))
        no_top_right(ax)
    fig.suptitle("沿运动方向的有符号误差分布（正=滞后，负=超调）", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig_signed_error.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- 4

def fig_scale_mixture() -> None:
    mix = load_mixture()
    hs = [0.1, 0.5, 1.0, 2.0]
    colors = dict(zip(CLS_ORDER, [BLUE, ORANGE, GREEN, RED, PURPLE, GOLD]))
    # 卡车仅 619 个样本，各单元格不足以拟合混合分布（报告正文已标注为不稳定），
    # 故不纳入这张结论性图；仅保留 5 个可靠类别，避免一条噪声曲线引发对整图的怀疑。
    plot_order = [c for c in CLS_ORDER if c != "卡车"]
    fig, axes = plt.subplots(1, 3, figsize=size_for("16:9"))
    handles, labels = [], []
    for cls in plot_order:
        d = mix.get(cls, {})
        if not d:
            continue
        c = colors[cls]
        ln, = axes[0].plot(hs, [d[h]["w"] for h in hs], "o-", color=c, markersize=3)
        handles.append(ln)
        labels.append(cls)
        axes[1].plot(hs, [d[h]["ratio"] for h in hs], "o-", color=c, markersize=3)
        axes[2].plot(hs, [d[h]["share"] for h in hs], "o-", color=c, markersize=3)
    for ax, t, yl in ((axes[0], "主体权重 w", "权重"),
                      (axes[1], r"尺度比 $\sigma_2/\sigma_1$", "倍数"),
                      (axes[2], "尾部成分方差占比", "占比")):
        ax.set_xticks(hs)
        ax.set_xticklabels([f"{h:g}" for h in hs])
        ax.set_xlabel("预判梯度（秒）", fontsize=7)
        ax.set_ylabel(yl, fontsize=7)
        ax.set_title(t, fontsize=8)
        ax.grid(axis="y", linestyle="--")
        no_top_right(ax)
    axes[0].set_ylim(0, 1)
    fig.suptitle("误差的尺度混合结构：少数样本贡献多数方差", fontsize=9)
    # 三条曲线共用同一组系列：抽出一个共用图例放画布底部，避免压住数据与坐标轴
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, 0.005), handlelength=1.6,
               columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.075, 1, 0.94))
    fig.savefig(OUT / "fig_scale_mixture.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- 5

def fig_output_annotated() -> None:
    """真实输出画面 + 图例条，说明各类框的含义（不做可能标错的指引箭头）。"""
    from PIL import Image, ImageDraw, ImageFont
    src = ROOT / "outputs" / "demo" / "frame_40.png"
    frame = Image.open(src).convert("RGB")
    W = 1650
    fh = int(W * frame.height / frame.width)
    frame = frame.resize((W, fh), Image.LANCZOS)
    strip_h = 190
    canvas = Image.new("RGB", (W, fh + strip_h), "white")
    canvas.paste(frame, (0, 0))
    d = ImageDraw.Draw(canvas)

    def font(pt):
        for n in ("simsun.ttc", "msyh.ttc"):
            p = Path("C:/Windows/Fonts") / n
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), pt)
                except Exception:
                    pass
        return ImageFont.load_default()

    f_head, f_txt = font(26), font(22)
    d.text((14, fh + 10), "图例（画面为系统真实输出，未做修饰）：", font=f_head, fill="#111111")
    items = [
        ("#208040", "实线框：当前帧检测（含类型与跟踪 ID）"),
        ("#cc3030", "虚线框：预判位置（此处梯度 1 秒）"),
        ("#b08020", "黄色虚线 + “·记忆”：静态结构隐地图补回的固定结构"),
        ("#2060cc", "“遮挡续推”：被遮挡期间仍继续显示的轨迹"),
    ]
    x0, y0 = 24, fh + 52
    for i, (c, t) in enumerate(items):
        yy = y0 + (i % 2) * 52
        xx = x0 + (i // 2) * (W // 2 - 20)
        d.rectangle([xx, yy + 6, xx + 34, yy + 20], outline=c, width=4)
        d.text((xx + 46, yy), t, font=f_txt, fill="#222222")
    canvas.save(OUT / "fig_output_annotated.png")


# --------------------------------------------------------------------------- main

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_static_fix()
    fig_error_attribution()
    fig_signed_error()
    fig_scale_mixture()
    fig_output_annotated()
    caps = {
        "fig_architecture": "系统总体架构（system architecture）：主链路全程因果，静态结构支路负责固定场景元素的稳定显示，输出为视频与精度报告两件。",
        "fig_output_annotated": "系统实际输出画面（真实运行结果，未做修饰），下方为框线样式图例，用于说明检测框、预判框、静态记忆与遮挡续推四类元素的区别。",
        "fig_signed_error": "四个预判梯度上沿运动方向的有符号误差分布，中位偏置均为负且随梯度线性放大，说明以系统性超调为主。",
        "fig_scale_mixture": "误差的尺度混合结构：主体权重、尺度比与尾部成分方差占比随梯度的走向，约两成样本贡献七成以上方差。卡车类因样本量不足（619 个）未纳入本图。",
        "fig_error_attribution": "像素误差质量的类别归属，对比各类别的样本数占比与误差总量占比。",
        "fig_static_fix": "静态结构类改为零位移预判后，标志牌在四个梯度上的中位误差变化（对数纵轴）。",
        "fig_scale_control": "尺度归一化对照：五个特征变体（含逐个剔除尺度代理）的样本外曲线下面积，以及像素口径与归一化口径下误差的散点对照。",
        "fig_class_error": "误差幅值按类别与四个预判梯度分布的箱线图，用于说明各类别的单位样本误差量级相近。",
    }
    stale = OUT / "fig_output_sample.png"
    if stale.exists():
        stale.unlink()
    (OUT / "captions.json").write_text(
        json.dumps(caps, ensure_ascii=False, indent=2), encoding="utf-8")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name:<34} {p.stat().st_size // 1024:>5} KB")
    print(f"captions.json: {len(caps)} 条")


if __name__ == "__main__":
    main()
