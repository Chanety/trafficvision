"""尾部样本门控实验：能不能只靠因果特征，提前捞出"要出问题"的样本。

背景（见 `bench/error_mixture.py`）：误差是两个尺度的混合——约 22%~36% 的样本
属于一个大尺度成分（σ2 是 σ1 的 3~8 倍），而这些少数样本贡献了 75%~98% 的总方差。
逐样本责任 r_i = P(属于尾部成分 | x_i) 是最合适的标签，但**它需要等到未来帧才知道**，
所以不能直接当在线判据，只能当**训练标签**。

本脚本回答三个问题：
  1. 只用预测时刻已知的特征，能否把尾部样本捞出来（AUC、固定查准率下的召回）？
  2. 按门控降级那批样本后，误差的中位/P90 降到多少（含随机门控对照与理想门控上界）？
  3. 门控在不同类别、不同梯度上是否通用？

评估用**留一视频交叉验证**（LeaveOneGroupOut by video），避免同段视频内样本泄漏。

用法：python bench/error_gate.py [rows.csv] [输出目录]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_mixture import fit_mix2, responsibilities  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                            roc_auc_score, roc_curve)  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (0.1, 0.5, 1.0, 2.0)
CLASS_ZH = {
    "car": "汽车", "truck": "卡车", "bus": "公交", "nonmotor": "非机动车",
    "person": "行人", "sign": "标志牌", "bollard": "路桩",
}
ORDER = ["car", "bus", "truck", "nonmotor", "person", "sign", "bollard"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _note(ax, txt, loc=0.02):
    ax.text(loc, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.92))


FEATS = ["速度", "转角速度", "机动标记", "框高", "检测置信度", "遮挡续推",
         "预判位移", "预判速度", "轨迹年龄", "同帧目标数", "梯度"] + \
        [f"类:{CLASS_ZH[c]}" for c in ORDER]


def build(path: Path):
    rows = []
    with path.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["e_along"] in ("", "None"):
                continue
            rows.append(r)
    # 轨迹年龄与同帧目标数（都只用当前帧及之前的信息）
    first_seen: dict = {}
    crowd: dict = defaultdict(int)
    for r in rows:
        key = (r["video"], r["id"])
        f = int(r["frame"])
        if key not in first_seen or f < first_seen[key]:
            first_seen[key] = f
        crowd[(r["video"], int(r["frame"]), float(r["h"]))] += 1

    # 参照混合（类别 × 梯度）→ 标签
    ref = {}
    by_cls_h: dict = defaultdict(list)
    for r in rows:
        by_cls_h[(r["cls"], float(r["h"]))].append(float(r["e_along"]))
    for k, v in by_cls_h.items():
        if len(v) >= 60:
            ref[k] = fit_mix2(np.array(v))

    X, y, meta = [], [], []
    for r in rows:
        h = float(r["h"])
        rf = ref.get((r["cls"], h))
        if rf is None:
            continue
        e = float(r["e_along"])
        r_i = float(responsibilities([e], rf)[0])
        speed = float(r["speed"] or 0.0)
        turn = r["turn"]
        turn = float(turn) if turn not in ("", "None") else 0.0
        a_pred = float(r["a_pred"]) if r["a_pred"] not in ("", "None") else 0.0
        age = int(r["frame"]) - first_seen[(r["video"], r["id"])]
        box_h = float(r["box_h"])
        vec = [speed, turn, 1.0 if r["maneuver"] == "True" else 0.0, box_h,
               float(r["conf"] or 0.0), 1.0 if r["coasting"] == "True" else 0.0,
               a_pred, a_pred / h, float(age),
               float(crowd[(r["video"], int(r["frame"]), h)]), h]
        vec += [1.0 if r["cls"] == c else 0.0 for c in ORDER]
        X.append(vec)
        y.append(1 if r_i > 0.5 else 0)
        meta.append({"video": r["video"], "cls": r["cls"], "h": h,
                     "mag": float(r["mag"]), "e": e, "r": r_i})
    return np.array(X, dtype=float), np.array(y, dtype=int), meta, ref


def oof_predict(X, y, groups, make_model):
    """留一视频的样本外预测概率。"""
    p = np.full(X.shape[0], np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = make_model()
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def gate_curve(mag, keep_mask, name):
    m = mag[keep_mask]
    if m.size == 0:
        return None
    return {"name": name, "keep": float(keep_mask.mean()),
            "median": float(np.median(m)), "p90": float(np.percentile(m, 90)),
            "mean": float(m.mean())}


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "error_dist" / "rows.csv"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs" / "error_dist" / "门控"
    out.mkdir(parents=True, exist_ok=True)
    X, y, meta, ref = build(src)
    mag = np.array([m["mag"] for m in meta])
    groups = np.array([m["video"] for m in meta])
    print(f"样本 {len(y)}，尾部标签占比 {y.mean() * 100:.1f}%，"
          f"视频 {len(set(groups))} 段\n输出 {out}")

    p_gbm = oof_predict(X, y, groups,
                        lambda: HistGradientBoostingClassifier(
                            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                            min_samples_leaf=40, l2_regularization=1.0,
                            early_stopping=False, random_state=0))
    p_lr = oof_predict(X, y, groups,
                       lambda: make_pipeline(
                           StandardScaler(),
                           LogisticRegression(max_iter=5000, C=1.0)))
    auc_gbm = roc_auc_score(y, p_gbm)
    auc_lr = roc_auc_score(y, p_lr)
    ap_gbm = average_precision_score(y, p_gbm)
    print(f"样本外 AUC：梯度提升 {auc_gbm:.3f}｜逻辑回归 {auc_lr:.3f}"
          f"｜随机基线 0.500")
    print(f"PR-AUC（平均精度）：{ap_gbm:.3f}（随机基线 = 正类占比 {y.mean():.3f}）")

    # 固定查准率下的召回
    prec, rec, thr = precision_recall_curve(y, p_gbm)
    rows_tbl = [("查准率门槛", "对应召回", "被标记比例", "保留率")]
    for target in (0.4, 0.5, 0.6, 0.7):
        ok = np.where(prec[:-1] >= target)[0]
        if ok.size == 0:
            rows_tbl.append((target, "—", "—", "—"))
            continue
        i = ok[np.argmax(rec[:-1][ok])]
        flagged = float((p_gbm >= thr[i]).mean())
        rows_tbl.append((target, round(float(rec[i]), 3), round(flagged, 3),
                         round(1 - flagged, 3)))
        print(f"  查准率≥{target}: 召回 {rec[i]:.3f}，标记 {flagged:.3f}，"
              f"保留 {1 - flagged:.3f}")

    # 门控效果：按预测概率从低到高保留
    order = np.argsort(p_gbm)
    curves = []
    base = gate_curve(mag, np.ones_like(mag, dtype=bool), "全部保留")
    curves.append(base)
    rng = np.random.default_rng(0)
    for keep in (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5):
        k = int(round(keep * len(order)))
        sel = np.zeros(len(order), dtype=bool)
        sel[order[:k]] = True
        curves.append(gate_curve(mag, sel, f"门控保留 {keep:.0%}"))
    # 随机门控对照（同样保留率，重复 30 次取中位）
    rand_curve = {}
    for keep in (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5):
        meds = []
        for _ in range(30):
            sel = rng.random(mag.size) < keep
            meds.append(np.median(mag[sel]))
        rand_curve[keep] = float(np.median(meds))
    # 理想门控：直接按真实责任 r_i（需要未来，只能当上界）
    oracle = {"keeps": [], "med": [], "p90": []}
    r_arr = np.array([m["r"] for m in meta])
    o = np.argsort(r_arr)
    for keep in (0.9, 0.8, 0.7, 0.6, 0.5):
        k = int(round(keep * len(o)))
        sel = np.zeros(len(o), dtype=bool)
        sel[o[:k]] = True
        oracle["keeps"].append(keep)
        oracle["med"].append(float(np.median(mag[sel])))
        oracle["p90"].append(float(np.percentile(mag[sel], 90)))

    with (out / "门控_查准召回.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows(rows_tbl)
    with (out / "门控_保留率效果.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["策略", "保留率", "中位|Δ|", "P90|Δ|", "均值|Δ|"])
        for c in curves:
            w.writerow([c["name"], round(c["keep"], 3), round(c["median"], 1),
                        round(c["p90"], 1), round(c["mean"], 1)])

    # 分组 AUC
    per_cls = {}
    for cls in ORDER:
        sel = np.array([m["cls"] == cls for m in meta])
        if sel.sum() > 50 and 0 < y[sel].sum() < sel.sum():
            per_cls[cls] = (int(sel.sum()), float(roc_auc_score(y[sel], p_gbm[sel])),
                            float(y[sel].mean()))
    per_h = {}
    for h in HORIZONS:
        sel = np.array([m["h"] == h for m in meta])
        if sel.sum() > 50 and 0 < y[sel].sum() < sel.sum():
            per_h[h] = (int(sel.sum()), float(roc_auc_score(y[sel], p_gbm[sel])),
                        float(y[sel].mean()))
    print("按类别 AUC:", {CLASS_ZH[k]: round(v[1], 3) for k, v in per_cls.items()})
    print("按梯度 AUC:", {k: round(v[1], 3) for k, v in per_h.items()})

    # ---------------------------------------------------------------- 图
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    fpr, tpr, _ = roc_curve(y, p_gbm)
    axes[0].plot(fpr, tpr, color="#4C78A8", lw=2,
                 label=f"梯度提升 AUC={auc_gbm:.3f}")
    fpr2, tpr2, _ = roc_curve(y, p_lr)
    axes[0].plot(fpr2, tpr2, color="#F58518", lw=1.6, ls="--",
                 label=f"逻辑回归 AUC={auc_lr:.3f}")
    axes[0].plot([0, 1], [0, 1], color="0.5", lw=1.2, ls=":", label="随机")
    axes[0].set_xlabel("误报率"); axes[0].set_ylabel("命中率")
    axes[0].set_title("ROC（留一视频样本外）"); axes[0].legend(fontsize=9)
    axes[1].plot(rec, prec, color="#4C78A8", lw=2)
    axes[1].axhline(y.mean(), color="0.5", lw=1.2, ls=":", label="随机基线")
    axes[1].set_xlabel("召回"); axes[1].set_ylabel("查准率")
    axes[1].set_title(f"查准-召回（平均精度 {ap_gbm:.3f}）")
    axes[1].legend(fontsize=9)
    fig.suptitle("图F　尾部样本门控的可分性", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "门控_ROC与PR.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    keeps = [c["keep"] for c in curves[1:]]
    axes[0].plot(keeps, [c["median"] for c in curves[1:]], "o-", color="#4C78A8",
                 label="门控后中位")
    axes[0].plot(list(rand_curve), list(rand_curve.values()), "s--", color="0.5",
                 label="随机门控（同保留率）")
    axes[0].plot(oracle["keeps"], oracle["med"], "^-", color="#E45756",
                 label="理想门控（用真实责任，上界）")
    axes[0].axhline(base["median"], color="#54A24B", lw=1.3, ls=":",
                    label=f"不门控 {base['median']:.1f}px")
    axes[0].set_xlabel("保留率"); axes[0].set_ylabel("中位 |Δ| (px)")
    axes[0].set_title("降级尾部样本后的中位误差"); axes[0].legend(fontsize=9)
    axes[1].plot(keeps, [c["p90"] for c in curves[1:]], "o-", color="#F58518",
                 label="门控后 P90")
    axes[1].plot(oracle["keeps"], oracle["p90"], "^-", color="#E45756",
                 label="理想门控 P90")
    axes[1].axhline(base["p90"], color="#54A24B", lw=1.3, ls=":",
                    label=f"不门控 {base['p90']:.1f}px")
    axes[1].set_xlabel("保留率"); axes[1].set_ylabel("P90 |Δ| (px)")
    axes[1].set_title("P90 误差"); axes[1].legend(fontsize=9)
    fig.suptitle("图G　门控的收益与代价（横轴=还留多少样本）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "门控_收益代价.png", dpi=180)
    plt.close(fig)

    # 特征重要性
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                       max_leaf_nodes=31, min_samples_leaf=40,
                                       l2_regularization=1.0, early_stopping=False,
                                       random_state=0).fit(X, y)
    sub = np.random.default_rng(0).choice(X.shape[0], size=min(6000, X.shape[0]),
                                          replace=False)
    imp = permutation_importance(m, X[sub], y[sub], n_repeats=5, random_state=0,
                                 scoring="roc_auc")
    idx = np.argsort(imp.importances_mean)[::-1][:12]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.barh([FEATS[i] for i in idx][::-1], imp.importances_mean[idx][::-1],
            xerr=imp.importances_std[idx][::-1], color="#4C78A8")
    ax.set_xlabel("置换后 AUC 的下降（越大越重要）")
    ax.set_title("图H　门控特征重要性")
    fig.tight_layout()
    fig.savefig(out / "门控_特征重要性.png", dpi=180)
    plt.close(fig)

    # 分组 AUC 条形
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    if per_cls:
        ks = list(per_cls)
        axes[0].bar([CLASS_ZH[k] for k in ks], [per_cls[k][1] for k in ks],
                    color="#4C78A8")
        axes[0].axhline(0.5, color="0.5", lw=1.2, ls=":")
        axes[0].set_ylim(0.4, 1.0)
        axes[0].set_ylabel("样本外 AUC")
        axes[0].set_title("按类别")
        for i, k in enumerate(ks):
            axes[0].text(i, per_cls[k][1] + 0.01, f"{per_cls[k][1]:.2f}\nn={per_cls[k][0]}",
                         ha="center", fontsize=8)
    if per_h:
        ks = list(per_h)
        axes[1].bar([f"{k:g}s" for k in ks], [per_h[k][1] for k in ks], color="#F58518")
        axes[1].axhline(0.5, color="0.5", lw=1.2, ls=":")
        axes[1].set_ylim(0.4, 1.0)
        axes[1].set_ylabel("样本外 AUC")
        axes[1].set_title("按梯度")
        for i, k in enumerate(ks):
            axes[1].text(i, per_h[k][1] + 0.01,
                         f"{per_h[k][1]:.2f}\n尾部{y[k] if False else per_h[k][2] * 100:.0f}%",
                         ha="center", fontsize=8)
    fig.suptitle("图I　门控在不同类别与梯度上的可迁移性", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "门控_分组AUC.png", dpi=180)
    plt.close(fig)

    md = ["# 尾部样本门控实验", "",
          f"数据源 `{src}`；产物 `{out}`。标签：责任 r_i > 0.5（属于大尺度成分）。",
          f"评估：留一视频交叉验证。样本 {len(y)}，尾部标签占比 {y.mean() * 100:.1f}%。",
          "", "## 可分性", "",
          f"- 梯度提升样本外 AUC **{auc_gbm:.3f}**，逻辑回归 {auc_lr:.3f}（随机 0.500）",
          f"- 平均精度 {ap_gbm:.3f}（随机基线 {y.mean():.3f}）", "",
          "| " + " | ".join(rows_tbl[0]) + " |", "|---|---|---|---|"]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows_tbl[1:]]
    md += ["", "## 门控效果（降级被标记的样本）", "",
           "| 策略 | 保留率 | 中位 \\|Δ\\| | P90 \\|Δ\\| | 均值 \\|Δ\\| |",
           "|---|---|---|---|---|"]
    for c in curves:
        md.append(f"| {c['name']} | {c['keep']:.0%} | {c['median']:.1f} | "
                  f"{c['p90']:.1f} | {c['mean']:.1f} |")
    md += ["", "## 分组 AUC", "", "| 类别 | n | 尾部占比 | AUC |", "|---|---|---|---|"]
    for k, v in per_cls.items():
        md.append(f"| {CLASS_ZH[k]} | {v[0]} | {v[2] * 100:.0f}% | {v[1]:.3f} |")
    md += ["", "| 梯度 | n | 尾部占比 | AUC |", "|---|---|---|---|"]
    for k, v in per_h.items():
        md.append(f"| {k:g}s | {v[0]} | {v[2] * 100:.0f}% | {v[1]:.3f} |")
    md += ["", "## 必须注意", "",
           "- 标签 r_i 本身是用误差算出来的，**只有未来帧到了才知道**；",
           "  所以 AUC 衡量的是「这个标签可不可预测」，不是「能不能在线判」。",
           "- 门控降级样本会损失覆盖率：保留率每降一档，画面里少画一批框。",
           "- 同一轨迹的样本高度相关，留一视频可以避免同段泄漏，但只有 6 段，",
           "  AUC 的置信区间较宽。", ""]
    (out / "门控说明.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n产物写入 {out}")


if __name__ == "__main__":
    main()
