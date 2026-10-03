"""端到端检测吞吐基线与对比测试。

用途：在改动前后给出可比的性能数字（FPS / 单帧延迟 / 检出数）。
不依赖网络：权重使用本地文件路径。
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2


def read_frames(video: Path, count: int, stride: int = 1) -> list:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video}")
    frames = []
    idx = 0
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def bench(model_path: str, frames: list, imgsz: int, device: str, half: bool,
          conf: float, warmup: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path)
    kwargs = dict(imgsz=imgsz, device=device, half=half, conf=conf, verbose=False)

    for frame in frames[:warmup]:
        model.predict(frame, **kwargs)

    latencies = []
    detections = 0
    classes = {}
    run_frames = frames[warmup:]
    start = time.perf_counter()
    for frame in run_frames:
        t0 = time.perf_counter()
        result = model.predict(frame, **kwargs)[0]
        latencies.append(time.perf_counter() - t0)
        boxes = result.boxes
        detections += len(boxes)
        for cls in boxes.cls.tolist() if boxes is not None else []:
            name = result.names[int(cls)]
            classes[name] = classes.get(name, 0) + 1
    wall = time.perf_counter() - start

    ordered = sorted(latencies)

    def pct(p: float) -> float:
        if not ordered:
            return 0.0
        k = min(len(ordered) - 1, int(round(p / 100 * len(ordered) - 1)))
        return ordered[k]

    return {
        "model": Path(model_path).name,
        "device": device,
        "imgsz": imgsz,
        "half": half,
        "frames": len(run_frames),
        "wall_s": round(wall, 3),
        "fps": round(len(run_frames) / wall, 2) if wall else 0.0,
        "latency_mean_ms": round(statistics.mean(latencies) * 1000, 2) if latencies else 0.0,
        "latency_p50_ms": round(pct(50) * 1000, 2),
        "latency_p95_ms": round(pct(95) * 1000, 2),
        "detections": detections,
        "det_per_frame": round(detections / len(run_frames), 1) if run_frames else 0.0,
        "class_hist": dict(sorted(classes.items(), key=lambda kv: -kv[1])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    frames = read_frames(Path(args.video), args.frames + args.warmup, args.stride)
    if len(frames) <= args.warmup:
        raise SystemExit("视频帧数不足")

    import json

    print(json.dumps(bench(args.model, frames, args.imgsz, args.device, args.half,
                           args.conf, args.warmup), ensure_ascii=False))


if __name__ == "__main__":
    main()
