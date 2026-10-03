"""数据集检测能力体检：逐视频采样帧，统计目标密度、类别分布、目标尺寸、检出率。

用途：判断当前检测器在这批素材上"够不够用"，以及误差主要来自哪里。

用法：python bench/dataset_report.py <数据集目录> [每段采样帧数] [输出json]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def timestamp_of(name: str) -> str:
    """从文件名里抽出 8 位日期 + 6 位时间，用于场景图与视频配对。"""
    m = re.search(r"(\d{8})_(\d{6})", name)
    return f"{m.group(1)}_{m.group(2)}" if m else ""


def sample_frames(path: Path, want: int):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    stride = max(1, total // want)
    frames = []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if ok:
                frames.append(frame)
        idx += 1
    cap.release()
    return frames


def main() -> None:
    root = Path(sys.argv[1])
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    from ultralytics import YOLO

    weights = Path(__file__).resolve().parents[1] / "trafficlab-3d" / "repo" / "yolov8s-worldv2.pt"
    classes = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
    model = YOLO(str(weights))
    model.set_classes(classes)

    videos = sorted(p for p in root.iterdir()
                    if p.suffix.lower() in VIDEO_EXT)
    pictures = sorted((p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".png"}),
                      key=lambda p: timestamp_of(p.name))

    rows = []
    for path in videos:
        frames = sample_frames(path, want)
        if not frames:
            continue
        height, width = frames[0].shape[:2]
        area = float(height * width)
        per_frame_counts, per_frame_conf, class_counts = [], [], {}
        small, medium, large = 0, 0, 0
        empty = 0

        results = model.predict(frames, imgsz=960, conf=0.20, device=0, verbose=False)
        for result in results:
            n = 0 if result.boxes is None else len(result.boxes)
            per_frame_counts.append(n)
            if n == 0:
                empty += 1
                continue
            for i in range(n):
                name = result.names[int(result.boxes.cls[i])]
                class_counts[name] = class_counts.get(name, 0) + 1
                per_frame_conf.append(float(result.boxes.conf[i]))
                x1, y1, x2, y2 = (float(v) for v in result.boxes.xyxy[i])
                ratio = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / area
                if ratio < 0.001:
                    small += 1
                elif ratio < 0.02:
                    medium += 1
                else:
                    large += 1

        ts = timestamp_of(path.name)
        nearest = None
        if ts and pictures:
            def gap(pic):
                pt = timestamp_of(pic.name)
                if not pt:
                    return 10 ** 9
                d = abs(int(pt[:8]) - int(ts[:8])) * 86400
                h1, m1, s1 = int(pt[9:11]), int(pt[11:13]), int(pt[13:15])
                h2, m2, s2 = int(ts[9:11]), int(ts[11:13]), int(ts[13:15])
                return abs(d + (h1 * 3600 + m1 * 60 + s1) - (h2 * 3600 + m2 * 60 + s2))
            best = min(pictures, key=gap)
            if gap(best) <= 300:
                nearest = {"picture": best.name, "gap_s": gap(best)}

        rows.append({
            "file": path.name,
            "resolution": f"{width}x{height}",
            "sampled_frames": len(frames),
            "objects_per_frame_mean": round(float(np.mean(per_frame_counts)), 2),
            "objects_per_frame_max": int(np.max(per_frame_counts)),
            "empty_frame_ratio": round(empty / len(frames), 3),
            "mean_confidence": round(float(np.mean(per_frame_conf)), 3) if per_frame_conf else None,
            "class_counts": dict(sorted(class_counts.items(), key=lambda kv: -kv[1])),
            "box_size": {"small_lt_0.1pct": small, "medium": medium, "large_gt_2pct": large},
            "scene_picture": nearest,
        })
        print(f"{path.name:<28} 平均 {rows[-1]['objects_per_frame_mean']:>6} 个/帧  "
              f"峰值 {rows[-1]['objects_per_frame_max']:>3}  "
              f"空帧率 {rows[-1]['empty_frame_ratio']:.2f}  "
              f"{rows[-1]['class_counts']}")

    if out_path:
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告 -> {out_path}")


if __name__ == "__main__":
    main()
