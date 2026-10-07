"""预判端对比：换用微调后的检测器，看预判指标是否真的改善。

对比对象：
  A. 原开放词汇模型（yolov8s-worldv2，8 类自定义）
  B. 微调模型（v3 / v4 的 best.pt）

对比指标（这是整条链路最终要看的）：
  · 轨迹中心抖动占每帧位移的比例（原来 26%）
  · 1 秒预判误差（原来中位 24.4 像素）
  · 可预判覆盖率
  · 目标密度与类别构成

用法：python bench/predict_compare.py [每段帧数]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trafficvision.pipeline import run_clip  # noqa: E402

VIDEO_DIR = Path(r"<用户目录>\Desktop\acdamic writing skill\videoset_01")
VIDEOS = ["VID_20261003_135104.mp4", "VID_20261003_135739.mp4", "VID_20261003_140156.mp4"]
MODELS = {
    "A_开放词汇": (ROOT / "trafficlab-3d" / "repo" / "yolov8s-worldv2.pt", True),
    "B_微调v3": (ROOT / "outputs" / "train" / "v3" / "weights" / "best.pt", False),
}


def jitter_stats(records: list[list[dict]], window: int = 15):
    """轨迹中心抖动 RMS 与每帧位移。"""
    series: dict[int, list[tuple[float, float]]] = {}
    for frame in records:
        for obj in frame:
            if obj["id"] > 0:
                series.setdefault(obj["id"], []).append(obj["centroid"])
    jitters, steps = [], []
    for pts in series.values():
        if len(pts) < window:
            continue
        arr = np.array(pts, dtype=float)
        steps.extend(np.linalg.norm(np.diff(arr, axis=0), axis=1).tolist())
        half = window // 2
        for c in range(half, len(arr) - half):
            seg = arr[c - half:c + half + 1]
            t = np.arange(len(seg), dtype=float)
            for dim in (0, 1):
                resid = seg[:, dim] - np.polyval(np.polyfit(t, seg[:, dim], 2), t)
                jitters.append(float(np.sqrt(np.mean(resid ** 2))))
    if not jitters:
        return None, None, None
    j, s = float(np.mean(jitters)), float(np.mean(steps))
    return j, s, (j / s * 100 if s else None)


def main() -> None:
    max_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    out_dir = ROOT / "outputs" / "predict_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = {}
    for label, (weights, world) in MODELS.items():
        agg = {"err1": [], "coverage": [], "density": [], "jitter": [], "step": [], "classes": {}}
        for video in VIDEOS:
            json_path = out_dir / f"{label}_{video.replace('.mp4','')}.json"
            result = run_clip(
                video=VIDEO_DIR / video, horizon=1.0, out_video=out_dir / "unused.mp4",
                weights=str(weights), imgsz=960, conf=0.25, device="0", world=world,
                out_json=json_path, render=False, max_frames_override=max_frames,
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            frames = payload["frames"]
            objects = [o for f in frames for o in f]
            agg["err1"].append(payload["metrics"]["1s"].get("mean_px"))
            agg["coverage"].append(
                sum(1 for o in objects if o.get("predictable")) / max(1, len(objects)))
            agg["density"].append(len(objects) / max(1, len(frames)))
            j, s, ratio = jitter_stats(frames)
            if j:
                agg["jitter"].append(j); agg["step"].append(s)
            for o in objects:
                agg["classes"][o["cls"]] = agg["classes"].get(o["cls"], 0) + 1
            print(f"  {label} {video}: 1秒误差 {payload['metrics']['1s'].get('mean_px')} "
                  f"覆盖率 {agg['coverage'][-1]:.3f} 密度 {agg['density'][-1]:.2f}")
        table[label] = agg

    print("\n=== 预判端对比（3 段验证视频平均）===")
    print(f"{'模型':<14}{'1秒误差(px)':>12}{'可预判覆盖率':>14}{'目标/帧':>10}"
          f"{'抖动RMS(px)':>13}{'每帧位移':>10}{'抖动占比':>10}")
    for label, agg in table.items():
        err = np.mean([v for v in agg["err1"] if v is not None])
        cov = np.mean(agg["coverage"])
        den = np.mean(agg["density"])
        j = np.mean(agg["jitter"]) if agg["jitter"] else float("nan")
        s = np.mean(agg["step"]) if agg["step"] else float("nan")
        print(f"{label:<14}{err:>12.1f}{cov:>14.3f}{den:>10.2f}{j:>13.2f}{s:>10.2f}"
              f"{j / s * 100:>9.1f}%")

    print("\n=== 类别构成对比 ===")
    for label, agg in table.items():
        total = sum(agg["classes"].values())
        top = sorted(agg["classes"].items(), key=lambda kv: -kv[1])[:8]
        print(f"{label}: 共 {total} 个目标 → " + "  ".join(f"{k}={v}" for k, v in top))


if __name__ == "__main__":
    main()
