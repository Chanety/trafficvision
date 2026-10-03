"""批处理：对整个数据集逐段跑预判流程，输出每段报告与汇总表。

用法：
    python bench/batch_predict.py <数据集目录> <输出目录> [每段最大帧数]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trafficvision.config import HORIZONS  # noqa: E402
from trafficvision.pipeline import run_clip  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def summarize(payload: dict) -> dict:
    frames = payload["frames"]
    objects = [o for f in frames for o in f]
    track_frames: dict[int, int] = {}
    for frame in frames:
        for obj in frame:
            if obj["id"] > 0:
                track_frames[obj["id"]] = track_frames.get(obj["id"], 0) + 1

    lengths = np.array(list(track_frames.values())) if track_frames else np.array([0])
    predictable = sum(1 for o in objects if o.get("predictable"))
    row = {
        "video": Path(payload["video"]).name,
        "duration_s": payload["video_info"]["duration"],
        "fps": payload["video_info"]["fps"],
        "resolution": f"{payload['video_info']['width']}x{payload['video_info']['height']}",
        "frames_processed": len(frames),
        "objects_per_frame": round(len(objects) / max(1, len(frames)), 2),
        "tracks": len(track_frames),
        "track_len_median": float(np.median(lengths)),
        "track_len_p90": float(np.percentile(lengths, 90)),
        "tracks_ge_5frames": int((lengths >= 5).sum()),
        "predictable_ratio": round(predictable / max(1, len(objects)), 3),
    }
    for h in HORIZONS:
        m = payload["metrics"].get(f"{h:g}s", {})
        row[f"err_{h:g}s_mean"] = m.get("mean_px")
        row[f"err_{h:g}s_p90"] = m.get("p90_px")
        row[f"err_{h:g}s_n"] = m.get("samples", 0)
    return row


def main() -> None:
    root = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = Path(__file__).resolve().parents[1] / "trafficlab-3d" / "repo" / "yolov8s-worldv2.pt"
    videos = sorted(p for p in root.iterdir() if p.suffix.lower() in VIDEO_EXT)

    rows = []
    for path in videos:
        per_video = out_dir / f"{path.stem}.json"
        try:
            run_clip(
                video=path,
                horizon=1.0,
                out_video=out_dir / f"{path.stem}.mp4",
                weights=str(weights),
                imgsz=960,
                conf=0.25,
                device="0",
                world=True,
                out_json=per_video,
                render=False,
                max_frames_override=max_frames,
            )
        except TypeError:
            # 兼容未提供 max_frames 参数的版本
            run_clip(video=path, horizon=1.0, out_video=out_dir / f"{path.stem}.mp4",
                     weights=str(weights), imgsz=960, conf=0.25, device="0",
                     world=True, out_json=per_video, render=False,
                     max_frames_override=max_frames)
        except SystemExit as exc:
            print(f"{path.name:<30} 跳过：{exc}")
            continue
        payload = json.loads(per_video.read_text(encoding="utf-8"))
        row = summarize(payload)
        rows.append(row)
        print(f"{row['video']:<30} {row['fps']:>6}fps {row['objects_per_frame']:>6}目标/帧 "
              f"轨迹{row['tracks']:>4} 可预判{row['predictable_ratio']:>6} "
              f"1s误差{str(row['err_1s_mean']):>8}")

    summary = out_dir / "summary.json"
    summary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总 -> {summary}")


if __name__ == "__main__":
    main()
