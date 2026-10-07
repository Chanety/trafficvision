"""预判误差归因：到底是"运动模型算不准"还是"跟踪断了/ID 跳了"。

背景：新素材上 1 秒预判误差中位 69.6 px、均值 177.6 px，远大于旧素材的 5~15 px。
误差要么来自运动模型，要么来自跟踪——如果目标中途被遮挡、ID 断掉又换了新 ID，
用同 ID 的"未来位置"当真值就会和预测对不上，误差会瞬间涨到几百像素。

判定方法：对每个误差样本（帧 t、目标 id、梯度 h）回看 t → t+h 这段窗口：
  · 断档：该 id 在窗口内是否消失过（消失=被遮挡或跟丢，之后同 id 可能已经是别的目标）
  · 贴边：目标框在预测时刻是否贴着画面边缘（即将出画，真值天然不可靠）
  · 跳变：窗口内的真实位移换算成速度，是否远大于预测时刻的估计速度（ID 换人的典型特征）
  · 尺度突变：框高在窗口内的变化倍数（换人往往伴随尺度突变）

用法：python bench/track_diagnostics.py <json目录> [梯度=1.0] [标签]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def analyze(payload: dict, horizon: float) -> list[dict]:
    frames = payload["frames"]
    info = payload["video_info"]
    width, height = float(info["width"]), float(info["height"])
    times = np.array([o["t"] for o in frames[0]] and
                     [f[0]["t"] if f else 0.0 for f in frames], dtype=float)
    # 每帧时间以该帧第一个目标的时间为准；空帧用上一帧时间
    last = 0.0
    for i, f in enumerate(frames):
        if f:
            last = float(f[0]["t"])
        times[i] = last

    # 真值与"是否断档"都只能看真实观测：遮挡续推出来的位置是推算值，不能当真值
    index_of: list[dict[int, dict]] = [
        {o["id"]: o for o in f if not o.get("coasting")} for f in frames
    ]
    present: dict[int, list[int]] = {}
    for i, f in enumerate(frames):
        for o in f:
            if not o.get("coasting"):
                present.setdefault(o["id"], []).append(i)

    samples = []
    for t, frame in enumerate(frames):
        target = times[t] + horizon
        i1 = int(np.searchsorted(times, target))
        if i1 >= len(frames):
            continue
        i0 = max(0, i1 - 1)
        span = times[i1] - times[i0]
        frac = (target - times[i0]) / span if span > 1e-9 else 0.0
        for obj in frame:
            pred = obj["preds"].get(f"{horizon:g}") or obj["preds"].get(str(horizon))
            if pred is None or not obj["predictable"]:
                continue
            tid = obj["id"]
            a = index_of[i0].get(tid)
            b = index_of[i1].get(tid)
            if a is not None and b is not None:
                truth = (a["centroid"][0] + (b["centroid"][0] - a["centroid"][0]) * frac,
                         a["centroid"][1] + (b["centroid"][1] - a["centroid"][1]) * frac)
                scale_end = b["bbox"][3] - b["bbox"][1]
            elif a is not None and frac < 0.5:
                truth = tuple(a["centroid"]); scale_end = a["bbox"][3] - a["bbox"][1]
            elif b is not None and frac >= 0.5:
                truth = tuple(b["centroid"]); scale_end = b["bbox"][3] - b["bbox"][1]
            else:
                continue

            pc = ((pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0)
            error = float(np.hypot(pc[0] - truth[0], pc[1] - truth[1]))

            c = obj["centroid"]
            x1, y1, x2, y2 = obj["bbox"]
            span_frames = present.get(tid, [])
            in_window = [f for f in span_frames if t <= f <= i1]
            # 窗口 [t, i1] 共 i1-t+1 帧；缺口 = 应有帧数 - 实际出现帧数
            gap = (i1 - t + 1) - len(in_window)
            border = min(x1, y1, width - x2, height - y2)
            start_h = max(1.0, y2 - y1)
            actual_disp = float(np.hypot(truth[0] - c[0], truth[1] - c[1]))
            speed = float(obj.get("speed_px_s", 0.0))
            implied_speed = actual_disp / max(1e-3, horizon)
            samples.append({
                "video": Path(payload["video"]).name,
                "cls": obj["cls"], "id": tid, "t": float(times[t]),
                "frame_idx": t, "bbox": list(obj["bbox"]),
                "track_len": len(present.get(tid, [])),
                "error": error, "gap_frames": gap, "border_px": float(border),
                "out_of_frame": border < 4.0,
                "implied_ratio": implied_speed / max(1.0, speed),
                "scale_ratio": scale_end / start_h,
                "speed": speed, "actual_disp": actual_disp,
                "coasting": bool(obj.get("coasting")),
            })
    return samples


def summarize(rows: list[dict]) -> None:
    errs = np.array([r["error"] for r in rows])
    total = errs.sum()

    gaps = np.array([r["gap_frames"] for r in rows])
    print("窗口内缺失帧数分布：")
    buckets = [("0 帧（全程连续）", gaps == 0), ("1-3 帧", (gaps >= 1) & (gaps <= 3)),
               ("4-9 帧", (gaps >= 4) & (gaps <= 9)), ("≥10 帧", gaps >= 10)]
    for name, mask in buckets:
        if mask.sum():
            sub = errs[mask]
            print(f"  {name:<18}{mask.sum():>6} 个（{mask.sum() / len(rows) * 100:>4.1f}%）"
                  f"  中位误差 {np.median(sub):>6.1f} px   占误差总量 {sub.sum() / total * 100:>4.0f}%")
    neg = sum(1 for r in rows if r["id"] < 0)
    if neg:
        print(f"  ⚠ 其中 {neg} 个样本来自跟踪器未分配 ID 的检测（负 ID），不计入连续性分析")
    lengths = np.array([r["track_len"] for r in rows])
    print(f"  贡献样本的轨迹长度：中位 {np.median(lengths):.0f} 帧，"
          f"P10 {np.percentile(lengths, 10):.0f}，P90 {np.percentile(lengths, 90):.0f}\n")

    def share(mask) -> tuple[int, float, float, float]:
        sub = errs[mask]
        if sub.size == 0:
            return 0, 0.0, 0.0, 0.0
        return int(sub.size), sub.sum() / total * 100, float(np.median(sub)), float(sub.mean())

    conditions = {
        "全程连续": [r["gap_frames"] == 0 for r in rows],
        "窗口内断档≥1帧": [r["gap_frames"] >= 1 for r in rows],
        "其中断档≥5帧": [r["gap_frames"] >= 5 for r in rows],
        "贴边/出画": [r["out_of_frame"] for r in rows],
        "速度异常(>2倍)": [r["implied_ratio"] > 2.0 for r in rows],
        "尺度突变(>1.5或<0.67)": [r["scale_ratio"] > 1.5 or r["scale_ratio"] < 0.67 for r in rows],
    }
    print(f"{'条件':<24}{'样本':>7}{'误差占比':>10}{'中位':>9}{'均值':>9}")
    for name, mask in conditions.items():
        n, share_pct, med, mean = share(np.array(mask))
        print(f"{name:<24}{n:>7}{share_pct:>9.0f}%{med:>9.1f}{mean:>9.1f}")

    clean = np.array([r["gap_frames"] == 0 and not r["out_of_frame"] for r in rows])
    print(f"\n只看'全程连续且未出画'的样本：{clean.sum()} / {len(rows)} 个"
          f"（{clean.sum() / len(rows) * 100:.0f}%）")
    if clean.sum():
        sub = errs[clean]
        print(f"  误差 中位 {np.median(sub):.1f} px  均值 {sub.mean():.1f} px  "
              f"P90 {np.percentile(sub, 90):.1f} px")
    dirty = ~clean
    if dirty.sum():
        sub = errs[dirty]
        print(f"  被排除的样本（断档或出画）：{dirty.sum()} 个，占误差总量 "
              f"{sub.sum() / total * 100:.0f}%，中位 {np.median(sub):.1f} px")

    worst = sorted(rows, key=lambda r: -r["error"])[:8]
    print("\n最严重的 8 个样本：")
    print(f"{'视频':<26}{'类别':<10}{'id':>7}{'帧':>6}{'误差':>9}{'缺帧':>6}{'轨迹长':>8}"
          f"{'速度异常':>9}{'尺度比':>8}")
    for r in worst:
        print(f"{r['video']:<26}{r['cls']:<10}{r['id']:>7}{r['t']:>6.1f}{r['error']:>9.0f}"
              f"{r['gap_frames']:>6}{r['track_len']:>8}{r['implied_ratio']:>9.1f}"
              f"{r['scale_ratio']:>8.2f}")

    print("\n连续性 × 速度一致性（误差总量占比）：")
    cells = [
        ("连续 + 速度正常", lambda r: r["gap_frames"] == 0 and r["implied_ratio"] <= 2.0),
        ("连续 + 速度异常", lambda r: r["gap_frames"] == 0 and r["implied_ratio"] > 2.0),
        ("断档 + 速度正常", lambda r: r["gap_frames"] > 0 and r["implied_ratio"] <= 2.0),
        ("断档 + 速度异常", lambda r: r["gap_frames"] > 0 and r["implied_ratio"] > 2.0),
    ]
    for name, fn in cells:
        sub = [r["error"] for r in rows if fn(r)]
        if sub:
            arr = np.array(sub)
            print(f"  {name:<18}{len(sub):>6} 个（{len(sub) / len(rows) * 100:>4.1f}%）"
                  f"  中位 {np.median(arr):>6.1f} px   占误差总量 {arr.sum() / total * 100:>4.0f}%")

    coast = np.array([r.get("coasting", False) for r in rows])
    if coast.any():
        print(f"\n遮挡续推样本：{coast.sum()} 个（占 {coast.mean() * 100:.1f}%）"
              f"  误差中位 {np.median(errs[coast]):.1f} px  均值 {errs[coast].mean():.1f} px"
              f"  占误差总量 {errs[coast].sum() / total * 100:.0f}%")
        rest = ~coast
        print(f"实测样本：    {rest.sum()} 个  误差中位 {np.median(errs[rest]):.1f} px"
              f"  均值 {errs[rest].mean():.1f} px")


def main() -> None:
    out_dir = Path(sys.argv[1])
    horizon = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    label = sys.argv[3] if len(sys.argv) > 3 else out_dir.name
    rows: list[dict] = []
    files = sorted(out_dir.glob("*.json"))
    for path in files:
        if path.name == "summary.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(analyze(payload, horizon))
    if not rows:
        print("没有样本")
        return
    errs = np.array([r["error"] for r in rows])
    print(f"=== {label} · {len(files)} 段视频 · {len(rows)} 个 {horizon:g}s 误差样本 ===")
    print(f"总体：中位 {np.median(errs):.1f} px，均值 {errs.mean():.1f} px，"
          f"P90 {np.percentile(errs, 90):.1f} px\n")
    summarize(rows)

    print("\n--- 按类别 ---")
    for cls in sorted({r["cls"] for r in rows}):
        sub = np.array([r["error"] for r in rows if r["cls"] == cls])
        clean = np.array([r["error"] for r in rows if r["cls"] == cls
                          and r["gap_frames"] == 0 and not r["out_of_frame"]])
        if clean.size:
            print(f"{cls:<12}{sub.size:>6} 个   中位 {np.median(sub):>7.1f} px"
                  f"   连续样本 {clean.size:>4} 个 中位 {np.median(clean):>7.1f} px")
        else:
            print(f"{cls:<12}{sub.size:>6} 个   中位 {np.median(sub):>7.1f} px   无连续样本")


if __name__ == "__main__":
    main()
