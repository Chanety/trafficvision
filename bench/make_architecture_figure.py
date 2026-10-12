"""绘制竞赛技术报告用的系统架构图（无外部素材依赖，可复算）。

输出：paper/competition/figures/fig_architecture.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "competition" / "figures" / "fig_architecture.png"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["SimSun", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
})

# 学术论文用的低饱和配色（plotting-agent / chart-patterns 规范）
C_MAIN = "#2060cc"
C_AUX = "#b08020"
C_OUT = "#208040"
C_BG = "#fafbfc"


def box(ax, x, y, w, h, text, color, fs=10, fc="#FFFFFF", lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=lw, edgecolor=color, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#16222E", zorder=3, linespacing=1.55)


def arrow(ax, p, q, color="#4A5A6A", style="-|>", rad=0.0, lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13,
                                 color=color, linewidth=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def main() -> None:
    # 密集示意图保持大画布，以 300 DPI 输出；插入正文后等效分辨率 >600 DPI
    fig, ax = plt.subplots(figsize=(12.6, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.005, 0.005), 0.99, 0.99,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                linewidth=1.2, edgecolor="#B9C6D4",
                                facecolor=C_BG, zorder=0))

    # 主链路（上排）
    y1, h1, w1 = 0.720, 0.145, 0.115
    xs = [0.030, 0.165, 0.300, 0.435, 0.570, 0.705, 0.840]
    labels = [
        "输入\n2~30 秒单目视频\n（未标定、可能手持）",
        "抽帧与检测\nYOLO11n 链式微调（7 类）\n1280 分辨率",
        "多目标关联\nByteTrack\n+ 身份切断判据",
        "静动状态机\n净位移判据\n+ 框高缩放容差",
        "因果运动估计\n匀速卡尔曼\n+ 机动置信度",
        "四梯度外推\n0.1 / 0.5 / 1 / 2 秒\n只依赖当前帧及之前",
        "渲染输出\n类型 / 预判框 / 状态\n中文 HUD",
    ]
    for x, t in zip(xs, labels):
        box(ax, x, y1, w1, h1, t, C_MAIN, fs=8.4)
    for i in range(len(xs) - 1):
        arrow(ax, (xs[i] + w1, y1 + h1 / 2), (xs[i + 1], y1 + h1 / 2))

    # 静态结构支路（下排）
    y2, h2, w2 = 0.430, 0.135, 0.165
    xs2 = [0.140, 0.315, 0.490]
    box(ax, xs2[0], y2, w2, h2,
        "开放词汇静态扫描\n多提示词集成 · 低置信换召回", C_AUX, fs=8.8)
    box(ax, xs2[1], y2, w2, h2,
        "跨帧位置一致性投票\n≥80% 且命中 ≥4 帧", C_AUX, fs=8.8)
    box(ax, xs2[2], y2, w2, h2,
        "零样本语义裁定\n正反例词表 → 保留/待确认/否掉", C_AUX, fs=7.8)
    arrow(ax, (xs2[0] + w2, y2 + h2 / 2), (xs2[1], y2 + h2 / 2), color=C_AUX)
    arrow(ax, (xs2[1] + w2, y2 + h2 / 2), (xs2[2], y2 + h2 / 2), color=C_AUX)
    # 从检测下探到静态扫描
    arrow(ax, (0.2225, y1), (0.2225, y2 + h2), color=C_AUX)
    # 场景先验缓存（在裁定框正下方）
    box(ax, xs2[2], 0.285, w2, 0.105,
        "场景先验缓存\n每个机位只算一次", C_AUX, fs=7.8, fc="#FCF7EC")
    arrow(ax, (xs2[2] + w2 / 2, y2), (xs2[2] + w2 / 2, 0.390), color=C_AUX)
    # 裁定结果注入渲染
    arrow(ax, (xs2[2] + w2, y2 + h2), (0.8975, y1), color=C_AUX, rad=-0.22, ls="--")

    # 交付物（右侧竖排一个块）
    box(ax, 0.700, 0.262, 0.270, 0.303,
        "交付物\n\n① 带预判的连续视频\nMP4 · 逐帧 HUD · 中文类型标签\n\n"
        "② 机器可读精度报告\n每梯度中位 / 均值 / P90\n可预判比例与轨迹稳定性",
        C_OUT, fs=9, fc="#F1F8F4")
    arrow(ax, (0.8975, y1), (0.860, 0.565), color=C_OUT, rad=0.15)

    ax.text(0.030, 0.215,
            "静态结构通道的设计要点：位置稳定 ≠ 语义正确。"
            "路缘石、花坛边缘、店面招牌同样常年不动，因此跨帧投票之后仍要语义裁定；"
            "已确认位置按机位缓存复用。",
            fontsize=8.6, color="#5A4A22", ha="left")

    ax.text(0.030, 0.965, "图 1　系统总体架构", fontsize=13, color="#16222E",
            ha="left", va="top")
    ax.text(0.030, 0.915,
            "深蓝 = 主链路（因果，只用当前帧及之前）；棕色 = 静态结构支路；绿色 = 交付物",
            fontsize=9, color="#44525F", ha="left", va="top")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, facecolor="white")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
