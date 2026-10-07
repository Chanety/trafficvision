"""从原始视频中挖掘稀有类样本（自行车、摩托），作为补充训练数据。

背景：外网数据集不可用（COCO 站点实测约 14 KB/s，2000 张要 8 小时以上），
而自采视频里本身就含大量自行车与摩托，且**域完全匹配**（同机位、同视角），
比街景数据集更适合补充这两类。

做法：按固定间隔抽帧，用开放词汇模型以较高分辨率全类检测，
保留含自行车或摩托的帧，写出完整标注（不是只写这两类，避免教模型忽略其他类）。

用法：python bench/mine_supplement.py <视频目录> <输出目录> [目标帧数] [抽帧间隔秒]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trafficvision.config import CLASS_COLORS, DEFAULT_CLASSES  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
RARE = {"bicycle", "motorcycle"}


def main() -> None:
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    target = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    interval_s = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5

    images_dir = out / "images"
    labels_dir = out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    weights = Path(__file__).resolve().parents[1] / "trafficlab-3d" / "repo" / "yolov8s-worldv2.pt"
    model = YOLO(str(weights))
    model.set_classes(list(DEFAULT_CLASSES))
    names = list(DEFAULT_CLASSES)

    videos = sorted(p for p in root.iterdir() if p.suffix.lower() in VIDEO_EXT)
    saved = 0
    stats = {name: 0 for name in RARE}

    for path in videos:
        if saved >= target:
            break
        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        step = max(1, int(round(fps * interval_s)))
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
                # 只保留含稀有类的帧
                if any(int(b[0]) in (3, 4) for b in boxes):
                    stem = f"{path.stem}_f{index:06d}"
                    cv2.imwrite(str(images_dir / f"{stem}.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    (labels_dir / f"{stem}.txt").write_text(
                        "\n".join(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
                                  for b in boxes) + "\n",
                        encoding="utf-8")
                    saved += 1
                    for b in boxes:
                        if int(b[0]) == 3:
                            stats["motorcycle"] += 1
                        elif int(b[0]) == 4:
                            stats["bicycle"] += 1
            index += 1
        cap.release()
        print(f"{path.name:<30} 累计已保存 {saved} 帧")

    (out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"\n共保存 {saved} 帧含稀有类的样本")
    print(f"其中 摩托框 {stats['motorcycle']} 个，自行车框 {stats['bicycle']} 个")
    print(f"输出目录：{out}")


if __name__ == "__main__":
    main()
