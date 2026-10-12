"""剥离标志牌（并顺带对比再去掉路桩）后的对照统计。

动因：归一化口径下标志牌的样本外 AUC 只有 0.266——低于随机，说明它那部分的
混合拟合本身不稳（样本少、结构异常）。本脚本量化"把它从统计里剥出去"的影响：

  1. 门控五个变体的样本外 AUC：全体 / 去标志牌 / 再去路桩
  2. 工作点（查准-召回）与「保留率—误差」曲线：含 vs 不含
  3. 尺度混合结构：主体权重、尺度比、尾部方差占比
  4. 分类型 AUC（剥离后）

用法：python bench/error_gate_nosign.py [rows.csv] [输出目录]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_gate_norm import (IDX_NO_BOX, IDX_NO_SCALE, ORDER,  # noqa: E402
                             CLASS_ZH, build)

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import (average_precision_score,  # noqa: E402
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _note(ax, txt, loc=0.02):
    ax.text(loc, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.92))


def oof_prob(X, y, groups, cols=None):
    Xs = X if cols is None else X[:, cols]
    p = np.full(Xs.shape[0], np.nan)
    for tr, te in LeaveOneGroupOut().split(Xs, y, groups):
        m = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
            random_state=0).fit(Xs[tr], y[tr])
        p[te] = m.predict_proba(Xs[te])[:, 1]
    return p


def benefit_curve(mag, p, keep_list=(0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5)):
    order = np.argsort(p)
    out = [(1.0, float(np.median(mag)), float(np.percentile(mag, 90)))]
    for keep in keep_list:
        k = int(round(keep * order.size))
        sel = order[:k]
        out.append((keep, float(np.median(mag[sel])), float(np.percentile(mag[sel], 90))))
    return out


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "error_dist" / "rows.csv"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs" / "error_dist" / "剥离标志牌"
    out.mkdir(parents=True, exist_ok=True)
    X, meta, mix_px, mix_rel, e_px, e_rel, box_h = build(src)
    cls_arr = np.array([m["cls"] for m in meta])
    h_arr = np.array([m["h"] for m in meta])
    groups_all = np.array([m["video"] for m in meta])
    mag = np.array([m["mag"] for m in meta])
    y_px = np.array([1 if m["r_px"] > 0.5 else 0 for m in meta])
    y_rel = np.array([1 if m["r_rel"] > 0.5 else 0 for m in meta])

    subsets = [
        ("全体", np.ones(len(meta), dtype=bool)),
        ("去标志牌", cls_arr != "sign"),
        ("再去路桩", (cls_arr != "sign") & (cls_arr != "bollard")),
    ]
    variants = [("A 像素・全部", y_px, None), ("B 像素・去框高", y_px, IDX_NO_BOX),
                ("C 归一化・全部", y_rel, None), ("D 归一化・去框高", y_rel, IDX_NO_BOX),
                ("E 归一化・去尺度代理", y_rel, IDX_NO_SCALE)]

    print("样本量：", {n: int(m.sum()) for n, m in subsets})
    table = [("子集", "变体", "n", "正类占比", "样本外AUC", "平均精度")]
    auc_map = {}
    for sname, mask in subsets:
        for vname, y, cols in variants:
            p = oof_prob(X[mask], y[mask], groups_all[mask], cols)
            auc = roc_auc_score(y[mask], p)
            ap = average_precision_score(y[mask], p)
            auc_map[(sname, vname)] = auc
            table.append((sname, vname, int(mask.sum()),
                          round(float(y[mask].mean()), 3), round(auc, 3), round(ap, 3)))
            print(f"  {sname:<8} {vname:<20} n={mask.sum():>6} "
                  f"正类 {y[mask].mean() * 100:5.1f}%  AUC {auc:.3f}  AP {ap:.3f}")

    # 工作点（用 D 变体，去标志牌）
    mask_ns = dict(subsets)["去标志牌"]
    p_ns = oof_prob(X[mask_ns], y_rel[mask_ns], groups_all[mask_ns], IDX_NO_BOX)
    prec, rec, thr = precision_recall_curve(y_rel[mask_ns], p_ns)
    rows_tbl = [("查准率门槛", "召回", "被标记比例", "保留率")]
    for target in (0.4, 0.5, 0.6):
        ok = np.where(prec[:-1] >= target)[0]
        if ok.size == 0:
            rows_tbl.append((target, "—", "—", "—"))
            continue
        i = ok[np.argmax(rec[:-1][ok])]
        flagged = float((p_ns >= thr[i]).mean())
        rows_tbl.append((target, round(float(rec[i]), 3), round(flagged, 3),
                         round(1 - flagged, 3)))
        print(f"  去标志牌·查准≥{target}: 召回 {rec[i]:.3f}，保留 {1 - flagged:.3f}")

    # 保留率—误差曲线（像素口径的 |Δ|，便于与其他结果对齐）
    curves = {}
    for sname, mask in subsets:
        p = oof_prob(X[mask], y_rel[mask], groups_all[mask], IDX_NO_BOX)
        curves[sname] = benefit_curve(mag[mask], p)

    # 混合结构汇总（去掉标志牌后按类别重算）
    t2 = [("类别", "梯度", "n", "w", "sigma1", "sigma2", "尺度比", "尾部方差占比")]
    for cls in ORDER:
        if cls == "sign":
            continue
        for h in (0.1, 0.5, 1.0, 2.0):
            d = mix_rel.get((cls, h))
            if not d:
                continue
            t2.append((CLASS_ZH[cls], h, d["n"], round(d["w"], 3),
                       round(d["sd"][0], 4), round(d["sd"][1], 4),
                       round(d["ratio"], 2), round(d["var_share_heavy"], 3)))

    # 分类型 AUC（去标志牌、归一化、去框高）
    per_cls = []
    for cls in ORDER:
        sel = mask_ns & (cls_arr == cls)
        if sel.sum() < 80:
            continue
        yy = y_rel[sel]
        if yy.sum() == 0 or yy.sum() == yy.size:
            continue
        pp = oof_prob(X[sel], yy, groups_all[sel], IDX_NO_BOX)
        per_cls.append((CLASS_ZH[cls], int(sel.sum()), float(yy.mean()),
                        float(roc_auc_score(yy, pp))))
        print(f"  {CLASS_ZH[cls]}: n={sel.sum()} 尾部 {yy.mean() * 100:.0f}% "
              f"AUC {per_cls[-1][3]:.3f}")

    with (out / "剥离_AUC对照.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows(table)
    with (out / "剥离_工作点.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows(rows_tbl)
    with (out / "剥离_保留率效果.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["子集", "保留率", "中位|Δ|", "P90|Δ|"])
        for sname, curve in curves.items():
            for keep, med, p90 in curve:
                w.writerow([sname, keep, round(med, 1), round(p90, 1)])
    with (out / "剥离_混合结构.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows(t2)
    with (out / "剥离_分类型AUC.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["类别", "n", "尾部占比", "AUC(归一化·去框高)"])
        for r in per_cls:
            w.writerow([r[0], r[1], round(r[2], 3), round(r[3], 3)])

    # ---------------------------------------------------------------- 图
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    vnames = [v[0] for v in variants]
    xpos = np.arange(len(vnames))
    w = 0.26
    for i, (sname, _) in enumerate(subsets):
        axes[0].bar(xpos + (i - 1) * w, [auc_map[(sname, v)] for v in vnames], w,
                    label=sname)
    axes[0].axhline(0.5, color="0.5", lw=1.2, ls=":")
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels(vnames, rotation=15, fontsize=8)
    axes[0].set_ylim(0.4, 0.9)
    axes[0].set_ylabel("样本外 AUC")
    axes[0].set_title("门控可分性：剥离静态结构类别前后")
    axes[0].legend(fontsize=9)
    for i, (sname, mask) in enumerate(subsets):
        for j, v in enumerate(vnames):
            axes[0].text(j + (i - 1) * w, auc_map[(sname, v)] + 0.008,
                         f"{auc_map[(sname, v)]:.3f}", ha="center", fontsize=7)

    for sname, curve in curves.items():
        keeps = [c[0] for c in curve]
        axes[1].plot(keeps, [c[2] for c in curve], "o-", label=f"{sname} P90")
    axes[1].set_xlabel("保留率"); axes[1].set_ylabel("P90 |Δ| (px)")
    axes[1].set_title("门控后的 P90（归一化标签・去框高特征）")
    axes[1].legend(fontsize=9)
    fig.suptitle("图L　剥离标志牌的前后对照", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "剥离_对照总览.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for sname, curve in curves.items():
        keeps = [c[0] for c in curve]
        axes[0].plot(keeps, [c[1] for c in curve], "o-", label=sname)
    axes[0].set_xlabel("保留率"); axes[0].set_ylabel("中位 |Δ| (px)")
    axes[0].set_title("中位误差")
    axes[0].legend(fontsize=9)
    axes[1].bar([r[0] for r in per_cls], [r[3] for r in per_cls], color="#4C78A8")
    axes[1].axhline(0.5, color="0.5", lw=1.2, ls=":")
    axes[1].set_ylim(0.4, 0.9)
    axes[1].set_ylabel("样本外 AUC")
    axes[1].set_title("剥离标志牌后的分类型 AUC")
    for i, r in enumerate(per_cls):
        axes[1].text(i, r[3] + 0.008, f"{r[3]:.3f}", ha="center", fontsize=8)
    fig.suptitle("图M　收益曲线与分类型可分性", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "剥离_收益与分类型.png", dpi=180)
    plt.close(fig)

    md = ["# 剥离标志牌的对照统计", "",
          f"数据源 `{src}`；产物 `{out}`。", "",
          "## 门控 AUC 对照", "",
          "| " + " | ".join(table[0]) + " |", "|" + "---|" * len(table[0])]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in table[1:]]
    md += ["", "## 工作点（去标志牌，归一化标签・去框高特征）", "",
           "| " + " | ".join(rows_tbl[0]) + " |", "|---|---|---|---|"]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows_tbl[1:]]
    md += ["", "## 保留率—误差（归一化标签，|Δ| 仍按像素报告）", "",
           "| 子集 | 保留率 | 中位 \\|Δ\\| | P90 \\|Δ\\| |", "|---|---|---|---|"]
    for sname, curve in curves.items():
        for keep, med, p90 in curve:
            md.append(f"| {sname} | {keep:.0%} | {med:.1f} | {p90:.1f} |")
    md += ["", "## 分类型 AUC（剥离标志牌后）", "",
           "| 类别 | n | 尾部占比 | AUC |", "|---|---|---|---|"]
    for r in per_cls:
        md.append(f"| {r[0]} | {r[1]} | {r[2] * 100:.0f}% | {r[3]:.3f} |")
    md += ["", "## 说明", "",
           "- 标志牌样本量小、混合拟合不稳，其归一化 AUC 低于随机，属于污染源；",
           "- 路桩在多数类别×梯度单元格里样本不足（<60），本身不参与拟合，",
           "  单独列出「再去路桩」一列只是确认它不影响结论。", ""]
    (out / "剥离说明.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n产物写入 {out}")


if __name__ == "__main__":
    main()
