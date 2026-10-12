"""验收口径 = 像素误差：误差质量归属 + 保覆盖改动的量化。

前提：验收看像素误差 |Δ|（不是相对误差）。因此只考虑**不丢目标**的改动，
"把大目标筛掉让指标好看"这类做法不列入。

本脚本给出：
  1. 像素误差质量的归属：按类别、按梯度、按机制标记（遮挡续推/机动/尺度档）
  2. **静态结构零位移预判**的确切收益（sign / bollard / 可能的 roadlamp、trafficlight）
  3. 全局位移增益校正的收益
  4. 门控（事后筛样本）的上界，仅作参照

零位移预判的算法（可精确计算，不需重跑推理）：
    预判框 = 当前框（位移为 0）
    新误差向量 Δ' = 实测 − 当前
    沿运动方向分量 = a_true；垂直分量与原来相同 = e_cross（因为 û ⟂ n̂）
    故 |Δ'| = sqrt(a_true² + e_cross²)

用法：python bench/px_metric_plan.py [rows.csv] [输出目录]
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
STATIC_CLASSES = ("sign", "bollard")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load(path: Path):
    out = []
    with path.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["e_along"] in ("", "None"):
                continue
            g = lambda k, d=0.0: (float(r[k]) if r[k] not in ("", "None") else d)
            out.append({"cls": r["cls"], "h": float(r["h"]), "video": r["video"],
                        "mag": float(r["mag"]), "e_along": g("e_along"),
                        "e_cross": g("e_cross"), "a_true": g("a_true"),
                        "a_pred": g("a_pred"), "box_h": g("box_h"),
                        "coasting": r["coasting"] == "True",
                        "maneuver": r["maneuver"] == "True",
                        "had_dir": r["e_along"] not in ("", "None")})
    return out


def zero_disp_mag(rows):
    """零位移预判下的 |Δ|。方向不可用的样本退回原值（因为算不了）。"""
    m = np.empty(len(rows))
    for i, r in enumerate(rows):
        if r["had_dir"]:
            m[i] = np.hypot(r["a_true"], r["e_cross"])
        else:
            m[i] = r["mag"]
    return m


def summarize(mag, tag):
    if mag.size == 0:
        return None
    return {"tag": tag, "n": int(mag.size), "median": float(np.median(mag)),
            "p90": float(np.percentile(mag, 90)), "mean": float(mag.mean())}


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "error_dist" / "rows.csv"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs" / "error_dist" / "像素口径"
    out.mkdir(parents=True, exist_ok=True)
    rows = load(src)
    mag = np.array([r["mag"] for r in rows])
    print(f"样本 {len(rows)}，总像素误差（Σ|Δ|）= {mag.sum() / 1e6:.2f} M px，"
          f"中位 {np.median(mag):.1f}px，P90 {np.percentile(mag, 90):.1f}px")

    # ---- 1. 误差质量归属
    t1 = [("类别", "梯度", "n", "中位|Δ|", "P90|Δ|", "均值|Δ|", "误差总量占比")]
    for cls in ORDER:
        for h in HORIZONS:
            sel = [r for r in rows if r["cls"] == cls and r["h"] == h]
            if len(sel) < 20:
                continue
            m = np.array([r["mag"] for r in sel])
            t1.append((CLASS_ZH[cls], h, len(sel), round(np.median(m), 1),
                       round(np.percentile(m, 90), 1), round(m.mean(), 1),
                       f"{m.sum() / mag.sum() * 100:.2f}%"))
    share_cls = []
    for cls in ORDER:
        m = np.array([r["mag"] for r in rows if r["cls"] == cls])
        if m.size:
            share_cls.append((CLASS_ZH[cls], m.size, m.sum() / mag.sum() * 100,
                              float(np.median(m))))
    for c, n, s, md in share_cls:
        print(f"  {c:<6} n={n:>6}  误差总量占比 {s:5.2f}%  中位 {md:6.1f}px")

    # ---- 2. 静态结构零位移
    t2 = [("类别", "梯度", "n", "改前中位", "改后中位", "改前P90", "改后P90",
           "误差总量变化")]
    tot_before = tot_after = 0.0
    for cls in STATIC_CLASSES:
        for h in HORIZONS:
            sel = [r for r in rows if r["cls"] == cls and r["h"] == h]
            if len(sel) < 20:
                continue
            b = np.array([r["mag"] for r in sel])
            a = zero_disp_mag(sel)
            tot_before += b.sum()
            tot_after += a.sum()
            t2.append((CLASS_ZH[cls], h, len(sel), round(np.median(b), 1),
                       round(np.median(a), 1), round(np.percentile(b, 90), 1),
                       round(np.percentile(a, 90), 1),
                       f"{(a.sum() - b.sum()) / 1e3:+.1f} k px"))
            print(f"  {CLASS_ZH[cls]} {h}s: 中位 {np.median(b):.1f} -> "
                  f"{np.median(a):.1f}px，P90 {np.percentile(b, 90):.1f} -> "
                  f"{np.percentile(a, 90):.1f}px")
    # 全体影响
    cls_arr = np.array([r["cls"] for r in rows])
    static_mask = np.isin(cls_arr, STATIC_CLASSES)
    mag_new = mag.copy()
    mag_new[static_mask] = zero_disp_mag([r for r in rows if r["cls"] in STATIC_CLASSES])
    print(f"\n静态结构零位移（全体影响）：总量 {mag.sum() / 1e6:.3f} -> "
          f"{mag_new.sum() / 1e6:.3f} M px（{(mag_new.sum() - mag.sum()) / mag.sum() * 100:+.2f}%）；"
          f"中位 {np.median(mag):.1f} -> {np.median(mag_new):.1f}px；"
          f"P90 {np.percentile(mag, 90):.1f} -> {np.percentile(mag_new, 90):.1f}px")

    # ---- 3. 全局增益校正
    dir_mask = np.array([r["had_dir"] for r in rows])
    at = np.array([r["a_true"] for r in rows])[dir_mask]
    ap = np.array([r["a_pred"] for r in rows])[dir_mask]
    g = float((at * ap).sum() / (ap * ap).sum())
    ec = np.array([r["e_cross"] for r in rows])[dir_mask]
    gain_mag = np.hypot(at - g * ap, ec)
    base_mag = mag[dir_mask]
    print(f"\n全局位移增益 g*={g:.4f}：中位 {np.median(base_mag):.1f} -> "
          f"{np.median(gain_mag):.1f}px，均值 {base_mag.mean():.1f} -> "
          f"{gain_mag.mean():.1f}px（{gain_mag.mean() / base_mag.mean() * 100 - 100:+.2f}%）")

    # ---- 4. 机制标记的误差占比
    t4 = [("机制标记", "n", "误差总量占比", "中位|Δ|")]
    marks = [("遮挡续推样本", np.array([r["coasting"] for r in rows])),
             ("机动样本（转角>阈值）", np.array([r["maneuver"] for r in rows])),
             ("静态结构类别", static_mask)]
    small = np.array([r["box_h"] for r in rows])
    med_box = float(np.median(small))
    marks.append((f"小目标（框高<{med_box:.0f}px）", small < med_box))
    marks.append((f"大目标（框高≥{med_box:.0f}px）", small >= med_box))
    for name, m in marks:
        if m.sum() == 0:
            continue
        t4.append((name, int(m.sum()), round(mag[m].sum() / mag.sum() * 100, 2),
                   round(float(np.median(mag[m])), 1)))
        print(f"  {name}: n={m.sum()} 占比 {mag[m].sum() / mag.sum() * 100:.2f}% "
              f"中位 {np.median(mag[m]):.1f}px")

    for name, table in (("像素口径_误差归属.csv", t1),
                        ("像素口径_静态零位移.csv", t2),
                        ("像素口径_机制占比.csv", t4)):
        with (out / name).open("w", newline="", encoding="utf-8-sig") as fh:
            csv.writer(fh).writerows(table)

    # ---------------------------------------------------------------- 图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar([c for c, _, _, _ in share_cls], [s for _, _, s, _ in share_cls],
                color="#4C78A8")
    axes[0].set_ylabel("占全部像素误差的百分比 (%)")
    axes[0].set_title("像素误差质量按类别归属")
    for i, (c, n, s, md) in enumerate(share_cls):
        axes[0].text(i, s + 0.6, f"{s:.1f}%", ha="center", fontsize=8)

    ls, bs, as_ = [], [], []
    for cls in STATIC_CLASSES:
        for h in HORIZONS:
            sel = [r for r in rows if r["cls"] == cls and r["h"] == h]
            if len(sel) < 20:
                continue
            ls.append(f"{CLASS_ZH[cls]}\n{h:g}s")
            bs.append(np.median([r["mag"] for r in sel]))
            as_.append(np.median(zero_disp_mag(sel)))
    xp = np.arange(len(ls))
    axes[1].bar(xp - 0.2, bs, 0.4, color="#E45756", label="现在（速度外推）")
    axes[1].bar(xp + 0.2, as_, 0.4, color="#54A24B", label="零位移预判")
    axes[1].set_xticks(xp)
    axes[1].set_xticklabels(ls, fontsize=8)
    axes[1].set_ylabel("中位 |Δ| (px)")
    axes[1].set_title("静态结构：零位移预判的收益")
    axes[1].legend(fontsize=9)

    names = [t[0] for t in t4[1:]]
    vals = [t[2] for t in t4[1:]]
    axes[2].barh(names[::-1], vals[::-1], color="#F58518")
    axes[2].set_xlabel("占全部像素误差的百分比 (%)")
    axes[2].set_title("误差质量按机制归属")
    fig.suptitle("图N　像素口径下的误差质量归属与可控项（1 s 与各梯度）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "像素口径_归属与收益.png", dpi=180)
    plt.close(fig)

    md = ["# 像素口径：误差质量归属与保覆盖改动", "",
          f"数据源 `{src}`；产物 `{out}`。口径 = 像素误差 |Δ|。",
          "原则：只列**不丢目标**的改动。", "",
          "## 关键数字", "",
          f"- 全部样本 Σ|Δ| = **{mag.sum() / 1e6:.3f} M px**，中位 "
          f"{np.median(mag):.1f}px，P90 {np.percentile(mag, 90):.1f}px",
          f"- 静态结构零位移：总量 {(mag_new.sum() - mag.sum()) / mag.sum() * 100:+.2f}%，"
          f"中位 {np.median(mag):.1f}→{np.median(mag_new):.1f}px，"
          f"P90 {np.percentile(mag, 90):.1f}→{np.percentile(mag_new, 90):.1f}px",
          f"- 全局位移增益 g*={g:.4f}：均值 "
          f"{gain_mag.mean() / base_mag.mean() * 100 - 100:+.2f}%", "",
          "## 误差质量归属（类别）", "",
          "| 类别 | n | 占误差总量 | 中位 \\|Δ\\| |", "|---|---|---|---|"]
    for c, n, s, md_ in share_cls:
        md.append(f"| {c} | {n} | {s:.2f}% | {md_:.1f} |")
    md += ["", "| 机制标记 | n | 占误差总量 | 中位 \\|Δ\\| |", "|---|---|---|---|"]
    for t in t4[1:]:
        md.append(f"| {t[0]} | {t[1]} | {t[2]}% | {t[3]} |")
    md += ["", "## 类别 × 梯度明细", "",
           "| " + " | ".join(t1[0]) + " |", "|" + "---|" * len(t1[0])]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in t1[1:]]
    md += ["", "## 静态结构零位移明细", "",
           "| " + " | ".join(t2[0]) + " |", "|" + "---|" * len(t2[0])]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in t2[1:]]
    (out / "像素口径说明.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n产物写入 {out}")


if __name__ == "__main__":
    main()
