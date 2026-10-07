"""把扫描结果里的三轮车候选帧标出来并放大，供人工判断真假。

输入是 scan_tricycle.py 产出的 scan.json：挑出含三轮车命中的帧，
重新推理一次（同参数，结果确定），画出两个模型的框并裁切放大，
拼成"原图 + 放大裁片"的复核图。

用法：python bench/mark_candidates.py <扫描输出目录> [最多几张] [每帧最多裁片数]
例：  python bench/mark_candidates.py outputs/scan_videoset_02 12 4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORLD_WEIGHTS = ROOT / "trafficlab-3d" / "repo" / "yolov8s-worldv2.pt"
FINETUNE_WEIGHTS = ROOT / "outputs" / "train" / "v5" / "weights" / "best.pt"
PROMPTS = ["tricycle", "motorcycle", "bicycle", "car", "person", "truck", "bus"]
COLORS = {"world": (0, 200, 255), "v5": (255, 80, 255)}   # BGR：开放词汇橙、v5 紫


def tricycle_boxes(result) -> list[tuple[float, tuple[int, int, int, int]]]:
    out = []
    if result.boxes is None:
        return out
    for i in range(len(result.boxes)):
        if result.names[int(result.boxes.cls[i])] != "tricycle":
            continue
        x1, y1, x2, y2 = (int(v) for v in result.boxes.xyxy[i])
        out.append((float(result.boxes.conf[i]), (x1, y1, x2, y2)))
    return out


def crop_patch(frame: np.ndarray, box, pad: float = 0.25, target_h: int = 320):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad), int(h * pad)
    H, W = frame.shape[:2]
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(W, x2 + px), min(H, y2 + py)
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    scale = max(1.0, target_h / patch.shape[0])
    return cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def main() -> None:
    scan_dir = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    per_frame = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    video_root = scan_dir.parent.parent / "datasets" / "videoset_02_origin"
    if not video_root.exists():
        video_root = Path(r"datasets\videoset_02_origin")
    out_dir = scan_dir / "review"
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    world = YOLO(str(WORLD_WEIGHTS))
    world.set_classes(PROMPTS)
    fine = YOLO(str(FINETUNE_WEIGHTS))

    payload = json.loads((scan_dir / "scan.json").read_text(encoding="utf-8"))
    hits = [f for f in payload["frames"] if f["tri_world_h"] or f["tri_v5_h"]]
    hits.sort(key=lambda f: -max(f["world"].get("tricycle", [0]) + f["v5"].get("tricycle", [0])))
    hits = hits[:limit]

    written = 0
    for rec in hits:
        video = video_root / rec["video"]
        if not video.exists():
            print(f"跳过 {video}（找不到原视频）")
            continue
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(rec["t"] * fps)))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue

        boxes_world = tricycle_boxes(world.predict(frame, imgsz=1280, conf=0.15,
                                                   device=0, verbose=False)[0])
        boxes_v5 = tricycle_boxes(fine.predict(frame, imgsz=960, conf=0.25,
                                               device=0, verbose=False)[0])

        marked = frame.copy()
        for tag, boxes in (("world", boxes_world), ("v5", boxes_v5)):
            for conf, (x1, y1, x2, y2) in boxes:
                cv2.rectangle(marked, (x1, y1), (x2, y2), COLORS[tag], 3)
                cv2.putText(marked, f"{tag} {conf:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[tag], 2)

        patches = []
        for tag, boxes in (("world", boxes_world), ("v5", boxes_v5)):
            for conf, box in sorted(boxes, key=lambda b: -b[0])[:per_frame]:
                patch = crop_patch(frame, box)
                if patch is None:
                    continue
                bar = np.zeros((36, patch.shape[1], 3), dtype=np.uint8)
                cv2.putText(bar, f"{tag} {conf:.2f}", (8, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[tag], 2)
                patches.append(np.vstack([bar, patch]))
        if patches:
            H = max(p.shape[0] for p in patches)
            row = np.hstack([
                np.vstack([p, np.zeros((H - p.shape[0], p.shape[1], 3), dtype=np.uint8)])
                for p in patches
            ])
            scale = min(1.0, 1800 / row.shape[1])
            row = cv2.resize(row, None, fx=scale, fy=scale)
            marked = cv2.resize(marked, (row.shape[1],
                                         int(marked.shape[0] * row.shape[1] / marked.shape[1])))
            marked = np.vstack([marked, np.zeros((6, row.shape[1], 3), dtype=np.uint8), row])

        name = f"{Path(rec['video']).stem}_t{rec['t']:07.2f}.jpg"
        cv2.imwrite(str(out_dir / name), marked, [cv2.IMWRITE_JPEG_QUALITY, 88])
        written += 1

    print(f"复核图 {written} 张 -> {out_dir}")


if __name__ == "__main__":
    main()
