"""预判端对比（第二轮）：微调 v3 vs 微调 v5（加入三轮车数据后）。

与 predict_compare.py 的区别：
  · 对比的是 v3 与 v5，两版都在同一批 3 段验证视频、同样的 300 帧上跑；
  · 输出写到 outputs/predict_compare_v5/，不覆盖第一轮（A 开放词汇 vs B v3）的结果；
  · 跑完后用 bench/jitter_moving_only.py 与 bench/jitter_matched.py 做抖动复盘。

用法：python bench/predict_compare_v5.py [每段帧数]
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
    "B_微调v3": ROOT / "outputs" / "train" / "v3" / "weights" / "best.pt",
    "C_微调v5": ROOT / "outputs" / "train" / "v5" / "weights" / "best.pt",
}


def main() -> None:
    max_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    out_dir = ROOT / "outputs" / "predict_compare_v5"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = {}
    for label, weights in MODELS.items():
        if not weights.exists():
            print(f"跳过 {label}（找不到 {weights}，训练还没完成？）")
            continue
        agg = {"err1": [], "coverage": [], "density": [], "classes": {}}
        for video in VIDEOS:
            json_path = out_dir / f"{label}_{video.replace('.mp4', '')}.json"
            run_clip(
                video=VIDEO_DIR / video, horizon=1.0, out_video=out_dir / "unused.mp4",
                weights=str(weights), imgsz=960, conf=0.25, device="0", world=False,
                out_json=json_path, render=False, max_frames_override=max_frames,
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            frames = payload["frames"]
            objects = [o for f in frames for o in f]
            err = payload["metrics"]["1s"].get("mean_px")
            coverage = sum(1 for o in objects if o.get("predictable")) / max(1, len(objects))
            agg["err1"].append(err)
            agg["coverage"].append(coverage)
            agg["density"].append(len(objects) / max(1, len(frames)))
            for o in objects:
                agg["classes"][o["cls"]] = agg["classes"].get(o["cls"], 0) + 1
            print(f"  {label} {video}: 1秒误差 {err}  覆盖率 {coverage:.3f}"
                  f"  密度 {agg['density'][-1]:.2f}")
        table[label] = agg

    if not table:
        return
    print("\n=== 预判端对比（3 段 × 300 帧）===")
    print(f"{'模型':<12}{'1秒误差(px)':>12}{'可预判覆盖率':>14}{'目标/帧':>10}")
    for label, agg in table.items():
        err = np.mean([v for v in agg["err1"] if v is not None])
        print(f"{label:<12}{err:>12.1f}{np.mean(agg['coverage']):>14.3f}"
              f"{np.mean(agg['density']):>10.2f}")
    print("\n=== 类别构成 ===")
    for label, agg in table.items():
        total = sum(agg["classes"].values())
        top = sorted(agg["classes"].items(), key=lambda kv: -kv[1])[:8]
        print(f"{label}: 共 {total} → " + "  ".join(f"{k}={v}" for k, v in top))


if __name__ == "__main__":
    main()
