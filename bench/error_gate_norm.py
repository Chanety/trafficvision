"""对照实验：把误差换成尺度归一化口径后，尾部结构与门控可分性还剩多少。

动机（见 `bench/error_gate.py`）：像素口径下的门控特征重要性第一名是**框高**，
也就是"离相机多近"。这说明"尾部样本"可能有一部分只是**透视尺度的产物**——
近处目标像素误差天然大，而不是预判真的坏了。

做法：
  1. 归一化误差 **e_rel = e_along / 框高**（单位：目标自身高度的倍数）。
     透视下"像素位移 ∝ 框高"，所以这个归一化把像素换算成了近似与距离无关的量。
  2. 在归一化口径下重做尺度混合拟合，看尾部结构是否还在。
  3. 门控做四组对照，回答"像素口径的门控有多少是尺度效应"：

     | 编号 | 标签 | 特征 |
     |---|---|---|
     | A | 像素口径尾部 | 全部特征（上一轮的基线） |
     | B | 像素口径尾部 | 去掉框高 |
     | C | 归一化口径尾部 | 全部特征 |
     | D | 归一化口径尾部 | 去掉框高 |
     | E | 归一化口径尾部 | 去掉全部尺度代理（框高/速度/预判位移/预判速度） |

  4. 标签翻转率：像素口径与归一化口径判为尾部的样本，有多少是同一批。

用法：python bench/error_gate_norm.py [rows.csv] [输出目录]
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
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (0.1, 0.5, 1.0, 2.0)
CLASS_ZH = {
    "car": "汽车", "truck": "卡车", "bus": "公交", "nonmotor": "非机动车",
    "person": "行人", "sign": "标志牌", "bollard": "路桩",
}
ORDER = ["car", "bus", "truck", "nonmotor", "person", "sign", "bollard"]

# 特征名（顺序与 build 中一致）
F_ALL = ["速度", "转角速度", "机动标记", "框高", "检测置信度", "遮挡续推",
         "预判位移", "预判速度", "轨迹年龄", "同帧目标数", "梯度"] + \
        [f"类:{CLASS_ZH[c]}" for c in ORDER]
SCALE_PROXIES = {"框高", "速度", "预判位移", "预判速度"}
IDX_NO_BOX = [i for i, n in enumerate(F_ALL) if n != "框高"]
IDX_NO_SCALE = [i for i, n in enumerate(F_ALL) if n not in SCALE_PROXIES]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _note(ax, txt, loc=0.02):
    ax.text(loc, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.92))


def build(path: Path):
    rows = []
    with path.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["e_along"] in ("", "None"):
                continue
            rows.append(r)
    first_seen: dict = {}
    crowd: dict = defaultdict(int)
    for r in rows:
        key = (r["video"], r["id"])
        f = int(r["frame"])
        first_seen[key] = min(first_seen.get(key, f), f)
        crowd[(r["video"], f, float(r["h"]))] += 1

    def mixture_map(values, key):
        buckets: dict = defaultdict(list)
        for r, v in zip(rows, values):
            buckets[(r["cls"], float(r["h"]))].append(v)
        out = {}
        for k, v in buckets.items():
            if len(v) >= 60:
                d = fit_mix2(np.array(v))
                if d is not None:
                    out[k] = d
        return out

    e_px = np.array([float(r["e_along"]) for r in rows])
    box_h = np.array([max(float(r["box_h"]), 1e-6) for r in rows])
    e_rel = e_px / box_h

    mix_px = mixture_map(e_px, "px")
    mix_rel = mixture_map(e_rel, "rel")

    X, meta = [], []
    for i, r in enumerate(rows):
        h = float(r["h"])
        key = (r["cls"], h)
        if key not in mix_px or key not in mix_rel:
            continue
        turn = r["turn"]
        turn = float(turn) if turn not in ("", "None") else 0.0
        a_pred = float(r["a_pred"]) if r["a_pred"] not in ("", "None") else 0.0
        speed = float(r["speed"] or 0.0)
        age = int(r["frame"]) - first_seen[(r["video"], r["id"])]
        vec = [speed, turn, 1.0 if r["maneuver"] == "True" else 0.0, box_h[i],
               float(r["conf"] or 0.0), 1.0 if r["coasting"] == "True" else 0.0,
               a_pred, a_pred / h, float(age),
               float(crowd[(r["video"], int(r["frame"]), h)]), h]
        vec += [1.0 if r["cls"] == c else 0.0 for c in ORDER]
        r_px = float(responsibilities([e_px[i]], mix_px[key])[0])
        r_rel = float(responsibilities([e_rel[i]], mix_rel[key])[0])
        X.append(vec)
        meta.append({"video": r["video"], "cls": r["cls"], "h": h,
                     "mag": float(r["mag"]), "box_h": box_h[i],
                     "e_px": e_px[i], "e_rel": e_rel[i],
                     "r_px": r_px, "r_rel": r_rel})
    return np.array(X, dtype=float), meta, mix_px, mix_rel, e_px, e_rel, box_h


def oof_auc(X, y, groups, cols=None):
    Xs = X if cols is None else X[:, cols]
    p = np.full(Xs.shape[0], np.nan)
    for tr, te in LeaveOneGroupOut().split(Xs, y, groups):
        m = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
            random_state=0).fit(Xs[tr], y[tr])
        p[te] = m.predict_proba(Xs[te])[:, 1]
    return roc_auc_score(y, p), p


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "error_dist" / "rows.csv"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs" / "error_dist" / "归一化对照"
    out.mkdir(parents=True, exist_ok=True)
    X, meta, mix_px, mix_rel, e_px, e_rel, box_h = build(src)
    print(f"样本 {len(meta)}")

    y_px = np.array([1 if m["r_px"] > 0.5 else 0 for m in meta])
    y_rel = np.array([1 if m["r_rel"] > 0.5 else 0 for m in meta])
    groups = np.array([m["video"] for m in meta])
    flip = float(np.mean(y_px != y_rel))

    # 尺度本身能解释多少像素误差
    ok = e_px != 0
    lg = np.polyfit(np.log(box_h[ok]), np.log(np.abs(e_px[ok])), 1)
    corr = float(np.corrcoef(np.log(box_h[ok]), np.log(np.abs(e_px[ok])))[0, 1])
    print(f"log|e_px| ~ log(框高) 斜率 {lg[0]:.3f}，相关系数 {corr:.3f}"
          f"（R²={corr ** 2:.3f}）")
    print(f"像素口径尾部占比 {y_px.mean() * 100:.1f}%，"
          f"归一化口径 {y_rel.mean() * 100:.1f}%，标签翻转率 {flip * 100:.1f}%")

    variants = [
        ("A 像素口径・全部特征", y_px, None),
        ("B 像素口径・去框高", y_px, IDX_NO_BOX),
        ("C 归一化・全部特征", y_rel, None),
        ("D 归一化・去框高", y_rel, IDX_NO_BOX),
        ("E 归一化・去全部尺度代理", y_rel, IDX_NO_SCALE),
    ]
    res = []
    for name, y, cols in variants:
        auc, _ = oof_auc(X, y, groups, cols)
        res.append({"name": name, "auc": auc, "pos": float(y.mean())})
        print(f"  {name}: 样本外 AUC {auc:.3f}（正类 {y.mean() * 100:.1f}%）")

    # 归一化口径下按类别看
    _, p_rel = oof_auc(X, y_rel, groups, IDX_NO_SCALE)
    per_cls = {}
    for cls in ORDER:
        sel = np.array([m["cls"] == cls for m in meta])
        if sel.sum() > 80 and 0 < y_rel[sel].sum() < sel.sum():
            per_cls[cls] = (int(sel.sum()),
                            float(roc_auc_score(y_rel[sel], p_rel[sel])),
                            float(y_rel[sel].mean()))

    # 混合结构对照
    t = [("口径", "类别", "梯度", "n", "w", "sigma1", "sigma2", "尺度比", "尾部方差占比")]
    for tag, mix in (("像素", mix_px), ("归一化", mix_rel)):
        for cls in ORDER:
            for h in HORIZONS:
                d = mix.get((cls, h))
                if not d:
                    continue
                t.append((tag, CLASS_ZH[cls], h, d["n"], round(d["w"], 3),
                          round(d["sd"][0], 4 if tag == "归一化" else 1),
                          round(d["sd"][1], 4 if tag == "归一化" else 1),
                          round(d["ratio"], 2), round(d["var_share_heavy"], 3)))
    with (out / "归一化对照_混合结构.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows(t)
    with (out / "归一化对照_门控AUC.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["变体", "样本外AUC", "正类占比"])
        for r in res:
            w.writerow([r["name"], round(r["auc"], 3), round(r["pos"], 3)])
        w.writerow([])
        w.writerow(["类别", "n", "归一化尾部占比", "AUC(去尺度代理)"])
        for k, v in per_cls.items():
            w.writerow([CLASS_ZH[k], v[0], round(v[2], 3), round(v[1], 3)])

    # ---------------------------------------------------------------- 图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].bar(range(len(res)), [r["auc"] for r in res],
                color=["#4C78A8", "#4C78A8", "#E45756", "#E45756", "#B279A2"])
    axes[0].axhline(0.5, color="0.5", lw=1.3, ls=":")
    axes[0].set_xticks(range(len(res)))
    axes[0].set_xticklabels([r["name"] for r in res], rotation=18, fontsize=8)
    axes[0].set_ylim(0.4, 0.9)
    axes[0].set_ylabel("样本外 AUC")
    axes[0].set_title("门控可分性：像素口径 vs 归一化口径")
    for i, r in enumerate(res):
        axes[0].text(i, r["auc"] + 0.008, f"{r['auc']:.3f}", ha="center", fontsize=8)
    _note(axes[0], f"像素口径尾部 {y_px.mean() * 100:.1f}%\n"
                   f"归一化口径尾部 {y_rel.mean() * 100:.1f}%\n"
                   f"标签翻转率 {flip * 100:.1f}%")

    axes[1].scatter(e_px, e_rel, s=3, alpha=0.12, color="#4C78A8")
    axes[1].axhline(0, color="0.5", lw=1)
    axes[1].axvline(0, color="0.5", lw=1)
    lim = float(np.percentile(np.abs(e_px), 99))
    axes[1].set_xlim(-lim, lim)
    ylim = float(np.percentile(np.abs(e_rel), 99))
    axes[1].set_ylim(-ylim, ylim)
    axes[1].set_xlabel("像素口径 e_along (px)")
    axes[1].set_ylabel("归一化 e_along / 框高（目标高度倍数）")
    axes[1].set_title("两个口径的样本对照（散点，1 s 全类别）")
    fig.suptitle("图J　归一化口径对照", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "归一化_对照总览.png", dpi=180)
    plt.close(fig)

    # 混合结构对照曲线
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    for cls in ORDER:
        hs = [h for h in HORIZONS if (cls, h) in mix_px and (cls, h) in mix_rel]
        if not hs:
            continue
        axes[0].plot(hs, [mix_px[(cls, h)]["ratio"] for h in hs], "o-",
                     label=f"{CLASS_ZH[cls]} 像素", lw=1.5)
        axes[0].plot(hs, [mix_rel[(cls, h)]["ratio"] for h in hs], "s--",
                     color=axes[0].lines[-1].get_color(), lw=1.3, alpha=0.8)
        axes[1].plot(hs, [mix_px[(cls, h)]["w"] for h in hs], "o-", lw=1.5)
        axes[1].plot(hs, [mix_rel[(cls, h)]["w"] for h in hs], "s--",
                     color=axes[1].lines[-1].get_color(), lw=1.3, alpha=0.8)
    axes[0].set_xlabel("梯度 (s)"); axes[0].set_ylabel("尺度比 σ2/σ1")
    axes[0].set_title("尺度比：实线=像素口径，虚线=归一化口径")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_xlabel("梯度 (s)"); axes[1].set_ylabel("主体权重 w")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("主体权重：实线=像素口径，虚线=归一化口径")
    fig.suptitle("图K　归一化后，尺度混合结构是否还在", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "归一化_混合结构.png", dpi=180)
    plt.close(fig)

    md = ["# 归一化口径对照实验", "",
          f"数据源 `{src}`；产物 `{out}`。归一化尺度 = 目标框高（预测时刻已知）。", "",
          "## 尺度本身解释了多少像素误差", "",
          f"- log|e_px| 对 log(框高) 回归：斜率 **{lg[0]:.3f}**，相关系数 "
          f"**{corr:.3f}**（R²={corr ** 2:.3f}）",
          f"- 像素口径尾部占比 {y_px.mean() * 100:.1f}%，归一化口径 "
          f"{y_rel.mean() * 100:.1f}%，**标签翻转率 {flip * 100:.1f}%**", "",
          "## 门控可分性对照", "",
          "| 变体 | 样本外 AUC | 正类占比 |", "|---|---|---|"]
    for r in res:
        md.append(f"| {r['name']} | {r['auc']:.3f} | {r['pos'] * 100:.1f}% |")
    md += ["", "## 归一化口径下按类别（去掉全部尺度代理特征）", "",
           "| 类别 | n | 尾部占比 | AUC |", "|---|---|---|---|"]
    for k, v in per_cls.items():
        md.append(f"| {CLASS_ZH[k]} | {v[0]} | {v[2] * 100:.0f}% | {v[1]:.3f} |")
    md += ["", "## 读法", "",
           "- **A→B 的落差** = 像素口径门控里有多少是靠框高（尺度）撑起来的。",
           "- **C 与 D 的落差**同理；**D 的绝对值**才是「不靠尺度信息，"
           "能不能预判真正出错的样本」。",
           "- **标签翻转率**高，说明两个口径认定的尾部不是同一批样本，"
           "即像素口径的尾部有相当部分是尺度产物。", ""]
    (out / "归一化对照说明.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n产物写入 {out}")


if __name__ == "__main__":
    main()
