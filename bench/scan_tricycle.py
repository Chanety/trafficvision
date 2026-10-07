"""扫原始素材里的三轮车：两个检测器交叉确认，并保存候选帧供复核。

为什么要两个检测器：
  · 开放词汇模型（yolov8s-worldv2 + 提示词 tricycle）没见过我们的三轮车，
    但只要被提示就能"找相似物"，召回偏高、置信度偏低；
  · 微调 v5 有三轮车这一类，但它只在 VisDrone 俯拍数据上学过，
    街面表现未知，可能漏检也可能误检。
  两者都说"有"的帧，是可信度最高的候选；只有一个说的，也保留，交人工判断。

用法：
    python bench/scan_tricycle.py <视频目录> [抽帧间隔秒] [输出目录]
例：
    python bench/scan_tricycle.py datasets/videoset_02_origin 1.0 outputs/scan_videoset_02
输出：
    <输出目录>/candidates/      含三轮车候选的帧（jpg，文件名带时间戳与置信度）
    <输出目录>/scan.json        逐帧统计
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORLD_WEIGHTS = ROOT / "trafficlab-3d" / "repo" / "yolov8s-worldv2.pt"
FINETUNE_WEIGHTS = ROOT / "outputs" / "train" / "v5" / "weights" / "best.pt"
PROMPTS = ["tricycle", "motorcycle", "bicycle", "car", "person", "truck", "bus"]
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def detections(result, names_of_interest: set[str]):
    """把一次预测拆成 {类别: [(conf, 框高像素)]}。"""
    found: dict[str, list[tuple[float, float]]] = {}
    if result.boxes is None:
        return found
    for i in range(len(result.boxes)):
        name = result.names[int(result.boxes.cls[i])]
        if name not in names_of_interest:
            continue
        x1, y1, x2, y2 = (float(v) for v in result.boxes.xyxy[i])
        found.setdefault(name, []).append((float(result.boxes.conf[i]), y2 - y1))
    return found


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    root = Path(sys.argv[1])
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "outputs" / "scan"
    cand_dir = out / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    world = YOLO(str(WORLD_WEIGHTS))
    world.set_classes(PROMPTS)
    fine = YOLO(str(FINETUNE_WEIGHTS))
    print(f"开放词汇 {WORLD_WEIGHTS.name}  +  微调 v5 {FINETUNE_WEIGHTS.name}")

    videos = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    records = []
    per_video: dict[str, dict] = {}
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(1, int(round(interval * fps)))
        stats = {"frames_scanned": 0, "world": Counter(), "v5": Counter(),
                 "both_tricycle": 0, "any_tricycle": 0}
        idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    t = idx / fps
                    hits = {}
                    for tag, model, size, conf in (("world", world, 1280, 0.15),
                                                   ("v5", fine, 960, 0.25)):
                        found = detections(
                            model.predict(frame, imgsz=size, conf=conf, device=0,
                                          verbose=False)[0],
                            set(PROMPTS),
                        )
                        hits[tag] = found
                        for name, boxes in found.items():
                            stats[tag][name] += len(boxes)
                    stats["frames_scanned"] += 1
                    tri_world = hits["world"].get("tricycle", [])
                    tri_v5 = hits["v5"].get("tricycle", [])
                    if tri_world or tri_v5:
                        stats["any_tricycle"] += 1
                    if tri_world and tri_v5:
                        stats["both_tricycle"] += 1
                    if tri_world or tri_v5:
                        best = max([c for c, _ in tri_world + tri_v5], default=0.0)
                        name = f"{video.stem}_t{t:07.2f}_conf{best:.2f}.jpg"
                        cv2.imwrite(str(cand_dir / name), frame)
                    records.append({
                        "video": video.name, "t": round(t, 3),
                        "world": {k: [round(c, 3) for c, _ in v] for k, v in hits["world"].items()},
                        "v5": {k: [round(c, 3) for c, _ in v] for k, v in hits["v5"].items()},
                        "tri_world_h": [round(h, 1) for _, h in tri_world],
                        "tri_v5_h": [round(h, 1) for _, h in tri_v5],
                    })
            idx += 1
            if total and idx >= total:
                break
        cap.release()
        per_video[video.name] = {
            "frames_scanned": stats["frames_scanned"],
            "world_counts": dict(stats["world"]),
            "v5_counts": dict(stats["v5"]),
            "frames_with_tricycle_any": stats["any_tricycle"],
            "frames_with_tricycle_both": stats["both_tricycle"],
        }
        print(f"  {video.name}: 扫 {stats['frames_scanned']} 帧　"
              f"三轮(开放词汇) {stats['world']['tricycle']}　"
              f"三轮(v5) {stats['v5']['tricycle']}　"
              f"两者都有的帧 {stats['both_tricycle']}")

    (out / "scan.json").write_text(
        json.dumps({"interval_s": interval, "per_video": per_video, "frames": records},
                   ensure_ascii=False), encoding="utf-8")

    total_world = sum(v["world_counts"].get("tricycle", 0) for v in per_video.values())
    total_v5 = sum(v["v5_counts"].get("tricycle", 0) for v in per_video.values())
    both = sum(v["frames_with_tricycle_both"] for v in per_video.values())
    print(f"\n合计：三轮车框 开放词汇 {total_world} 个 / v5 {total_v5} 个；"
          f"两者同时命中的扫描帧 {both} 帧")
    print(f"候选帧 -> {cand_dir}（{len(list(cand_dir.glob('*.jpg')))} 张）")
    print(f"逐帧记录 -> {out / 'scan.json'}")


if __name__ == "__main__":
    main()
