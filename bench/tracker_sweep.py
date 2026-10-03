"""跟踪器横评：在同一批密集视频上对比各跟踪器的碎片化程度与预判精度。

评判口径：
  · 轨迹条数越少、每条越长 → 碎片化越低（ID 跳变少）
  · 可预判比例越高 → 覆盖率越好
  · 误差越低 → 精度越好

用法：python bench/tracker_sweep.py <数据集目录> <输出目录> [每段帧数]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.batch_predict import summarize  # noqa: E402
from trafficvision.pipeline import run_clip  # noqa: E402

TRACKERS = ["bytetrack.yaml", "botsort.yaml", "ocsort.yaml", "fasttrack.yaml", "tracktrack.yaml"]
VIDEOS = ["VID_20261003_135739.mp4", "VID_20261003_135823.mp4", "VID_20261003_140539.mp4"]


def main() -> None:
    root = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = Path(__file__).resolve().parents[1] / "trafficlab-3d" / "repo" / "yolov8s-worldv2.pt"

    rows = []
    for video in VIDEOS:
        for tracker in TRACKERS:
            tag = f"{video.split('_')[-1].replace('.mp4', '')}_{tracker.replace('.yaml', '')}"
            payload_path = out_dir / f"{tag}.json"
            try:
                run_clip(
                    video=root / video, horizon=1.0, out_video=out_dir / f"{tag}.mp4",
                    weights=str(weights), imgsz=960, conf=0.25, device="0", world=True,
                    out_json=payload_path, render=False, max_frames_override=max_frames,
                    tracker=tracker,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"{tag:<34} 失败：{type(exc).__name__} {exc}")
                continue
            row = summarize(json.loads(payload_path.read_text(encoding="utf-8")))
            row["tracker"] = tracker.replace(".yaml", "")
            rows.append(row)
            print(f"{tag:<34} 轨迹{row['tracks']:>4} "
                  f"长轨迹{row['tracks_ge_5frames']:>4} "
                  f"可预判{row['predictable_ratio']:>6} "
                  f"1s误差中位{str(row['err_1s_mean']):>8}")

    (out_dir / "tracker_sweep.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n汇总 -> {out_dir / 'tracker_sweep.json'}")


if __name__ == "__main__":
    main()
