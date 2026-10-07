"""扩样本：针对稀有类（自行车、摩托）从自采视频中定向挖掘更多帧。

两个用途：
  1. 扩验证集——从验证视频里挖，优先包含自行车的帧，让单车类的指标更稳
  2. 扩训练集——从训练视频里挖（严格排除验证视频，避免数据泄漏）

用法：
    python bench/expand_samples.py <视频目录> <输出目录> <视频名单> <目标帧数> [优先级类别]
    视频名单：逗号分隔的视频名前缀，或 all
    优先级类别：默认 bicycle,motorcycle（含这些类的帧优先保存）
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trafficvision.config import DEFAULT_CLASSES  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def main() -> None:
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    only = sys.argv[3] if len(sys.argv) > 3 else "all"
    target = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    priority = (sys.argv[5].split(",") if len(sys.argv) > 5
                else ["bicycle", "motorcycle"])
    want = set() if only == "all" else {v.strip() for v in only.split(",") if v.strip()}

    images_dir = out / "images"
    labels_dir = out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    weights = Path(__file__).resolve().parents[1] / "trafficlab-3d" / "repo" / "yolov8s-worldv2.pt"
    model = YOLO(str(weights))
    model.set_classes(list(DEFAULT_CLASSES))
    names = list(DEFAULT_CLASSES)
    priority_ids = {names.index(p) for p in priority if p in names}

    videos = sorted(p for p in root.iterdir() if p.suffix.lower() in VIDEO_EXT)
    if want:
        videos = [p for p in videos if any(w in p.stem for w in want)]
    print(f"待处理视频 {len(videos)} 段，目标 {target} 帧，优先级类别 {sorted(priority)}")

    saved = 0
    stats: dict[int, int] = {}
    for path in videos:
        if saved >= target:
            break
        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        step = max(1, int(round(fps * 1.0)))       # 每秒一帧
        index = 0
        while saved < target:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                result = model.predict(frame, imgsz=1280, conf=0.15, iou=0.5,
                                       device=0, verbose=False)[0]
                boxes = []
                if result.boxes is not None and len(result.boxes):
                    h, w = frame.shape[:2]
                    for i in range(len(result.boxes)):
                        cls_name = result.names[int(result.boxes.cls[i])]
                        if cls_name not in names:
                            continue
                        x1, y1, x2, y2 = (float(v) for v in result.boxes.xyxy[i])
                        boxes.append([float(names.index(cls_name)),
                                      ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h,
                                      (x2 - x1) / w, (y2 - y1) / h])
                if any(int(b[0]) in priority_ids for b in boxes):
                    stem = f"{path.stem}_f{index:06d}"
                    if (images_dir / f"{stem}.jpg").exists():
                        index += 1
                        continue
                    cv2.imwrite(str(images_dir / f"{stem}.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    (labels_dir / f"{stem}.txt").write_text(
                        "\n".join(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
                                  for b in boxes) + "\n", encoding="utf-8")
                    saved += 1
                    for b in boxes:
                        stats[int(b[0])] = stats.get(int(b[0]), 0) + 1
            index += 1
        cap.release()
        print(f"  {path.stem:<28} 累计 {saved} 帧")

    print(f"\n共保存 {saved} 帧 -> {out}")
    for cid, n in sorted(stats.items()):
        print(f"    {names[cid]:<11} {n}")


if __name__ == "__main__":
    main()
