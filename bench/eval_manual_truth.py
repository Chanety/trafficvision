"""用**独立人工真值**评测预判精度——打破"自证式指标"。

为什么需要（仓库审计 P0-02、用户 2026-10-09 决定"前置"）：
  现在预判误差的真值取自**同一个跟踪器的未来检测帧**。跟踪漂移时，预测和"真值"
  会一起漂——指标自洽，但不一定真实。这是比赛 25 分项"可核验效果数据"最怕的漏洞。

人工真值格式：MOT 风格，每行 `frame,id,x,y,w,h,cls[,conf]`（像素坐标，左上角 + 宽高）。
**必须有稳定的 id**：否则无法判断"1 秒后的那个框"是不是同一个目标。

评测逻辑（刻意避免循环论证）：
  1. 在**当前帧 t**，用 IoU≥0.5 把模型检测框与人工框配对 → 建立对应关系（两边都是
     同一时刻的观测，不是预测，因此这一步不引入循环）；
  2. 沿人工 id 找到 t+h 时刻的人工框 → 这就是"真实位置"；
  3. 比较模型在 t 时刻给出的 t+h 预判框 与 该人工框：报中心误差与 IoU。

用法：
    python bench/eval_manual_truth.py \\
        --video 片段.mp4 --truth truth/segment01.txt \\
        --weights outputs/train/v7/weights/best.pt \\
        --horizon 1 --out outputs/manual_eval_seg01.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trafficvision.pipeline import probe_video, run_clip  # noqa: E402


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.0


def load_truth(path: Path):
    """读 MOT 风格人工真值 → {frame: [(id, cls, box)]} 与 {id: {frame: (cls, box)}}。"""
    by_frame = defaultdict(list)
    by_id = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.replace(",", " ").split()]
        if len(parts) < 7:
            continue
        try:
            f = int(float(parts[0])); tid = int(float(parts[1]))
            x, y, w, h = (float(v) for v in parts[2:6])
            cls = parts[6]
        except ValueError:
            continue
        box = [x, y, x + w, y + h]
        by_frame[f].append((tid, cls, box))
        by_id[tid][f] = (cls, box)
    return by_frame, by_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--truth", required=True, help="MOT 风格人工真值文件")
    ap.add_argument("--weights", default=str(ROOT / "outputs" / "train" / "v7" / "weights" / "best.pt"))
    ap.add_argument("--horizon", type=float, default=1.0)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    video = Path(args.video)
    truth_path = Path(args.truth)
    info = probe_video(video)
    fps = info["fps"]
    by_frame, by_id = load_truth(truth_path)
    if not by_frame:
        raise SystemExit(f"真值文件里没读到有效行：{truth_path}")

    tmp_json = Path(args.out).with_suffix(".pred.json") if args.out else ROOT / "outputs" / "_manual_pred.json"
    tmp_json.parent.mkdir(parents=True, exist_ok=True)
    print(f"跑预判：{video.name}  真值 {len(by_id)} 条轨迹 / {len(by_frame)} 帧")
    run_clip(
        video=video, horizon=args.horizon,
        out_video=tmp_json.with_suffix(".mp4"), weights=args.weights,
        imgsz=args.imgsz, device="0", render=False, out_json=tmp_json,
        static_map=False,          # 人工真值对照只看动目标，静态先验只会引入额外框
    )
    data = json.loads(tmp_json.read_text(encoding="utf-8"))
    frames = data["frames"]
    times = [i / fps for i in range(len(frames))]
    step = max(1, int(round(args.horizon * fps)))

    samples = []
    matched_tracks = set()
    for t, objs in enumerate(frames):
        t2 = t + step
        if t2 >= len(frames):
            break
        for o in objs:
            if not o.get("predictable"):
                continue
            pred = (o["preds"] or {}).get(str(args.horizon)) or (o["preds"] or {}).get(str(float(args.horizon)))
            if pred is None:
                continue
            # ① 在当前帧与人工框配对（同一时刻的观测对观测，不构成循环）
            best, best_iou = None, 0.0
            for tid, cls, box in by_frame.get(t, []):
                if cls != o["cls"]:
                    continue
                v = iou(o["bbox"], box)
                if v > best_iou:
                    best, best_iou = tid, v
            if best is None or best_iou < args.match_iou:
                continue
            # ② 找该 id 在 t+h 的人工框 = 独立真值
            fut = by_id.get(best, {}).get(t2)
            if fut is None:
                continue
            matched_tracks.add(best)
            tb = fut[1]
            pc = ((pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2)
            tc = ((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2)
            samples.append({
                "frame": t, "track": best, "cls": o["cls"],
                "center_err_px": float(np.hypot(pc[0] - tc[0], pc[1] - tc[1])),
                "box_iou": iou(pred, tb),
            })

    if not samples:
        raise SystemExit(
            "没有匹配到任何样本。请检查：\n"
            "  · 真值的 cls 名是否与模型类别名一致（如 nonmotor / person）；\n"
            "  · 帧号是否从 0 开始、与视频对齐；\n"
            "  · 人工框与模型检测的 IoU 是否普遍低于匹配阈值。")

    err = np.array([s["center_err_px"] for s in samples])
    ious = np.array([s["box_iou"] for s in samples])
    result = {
        "video": video.name, "truth": truth_path.name,
        "horizon_s": args.horizon, "fps": round(fps, 3),
        "samples": len(samples), "matched_tracks": len(matched_tracks),
        "center_err_px": {"mean": round(float(err.mean()), 2),
                          "median": round(float(np.median(err)), 2),
                          "p90": round(float(np.percentile(err, 90)), 2)},
        "box_iou": {"mean": round(float(ious.mean()), 3),
                    "median": round(float(np.median(ious)), 3)},
        "per_class": {},
    }
    per = defaultdict(list)
    for s, e, i_ in zip(samples, err, ious):
        per[s["cls"]].append((e, i_))
    for cls, vals in sorted(per.items()):
        e = np.array([v[0] for v in vals]); i_ = np.array([v[1] for v in vals])
        result["per_class"][cls] = {
            "samples": len(vals),
            "center_median": round(float(np.median(e)), 2),
            "iou_median": round(float(np.median(i_)), 3),
        }

    print(f"\n=== 独立人工真值评测（{args.horizon:g} 秒梯度）===")
    print(f"样本 {result['samples']} 个，覆盖 {result['matched_tracks']} 条人工轨迹")
    print(f"中心误差 中位 {result['center_err_px']['median']} px"
          f"  均值 {result['center_err_px']['mean']}"
          f"  P90 {result['center_err_px']['p90']}")
    print(f"预测框 IoU 中位 {result['box_iou']['median']}"
          f"（此前从未测量过——所有历史指标都只评中心点）")
    for cls, v in result["per_class"].items():
        print(f"   {cls:<12} n={v['samples']:<5} 中心中位 {v['center_median']:>6.2f}px"
              f"  IoU 中位 {v['iou_median']:.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果：{args.out}")


if __name__ == "__main__":
    main()
