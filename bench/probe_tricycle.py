"""探测三轮车识别的可行性：现有素材里"三轮车"这类目标能否被单独检出。

方法：用开放词汇模型以 tricycle / 三轮车 为提示类别，在验证集帧上检测，
统计检出数量与置信度分布，并看它与 motorcycle 框的重叠情况，
判断"把三轮从摩托里拆出来"是否可行、大概有多少样本。

用法：python bench/probe_tricycle.py [帧数上限]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "datasets" / "train_v4" / "images" / "val"
WEIGHTS = ROOT / "trafficlab-3d" / "repo" / "yolov8s-worldv2.pt"


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    model.set_classes(["tricycle", "motorcycle", "bicycle"])

    images = sorted(FRAMES.glob("*.jpg"))[:limit]
    counts = {"tricycle": 0, "motorcycle": 0, "bicycle": 0}
    frames_with_tricycle = 0
    confs = []
    for image in images:
        frame = cv2.imread(str(image))
        if frame is None:
            continue
        result = model.predict(frame, imgsz=1280, conf=0.15, iou=0.5,
                               device=0, verbose=False)[0]
        hit = 0
        if result.boxes is not None:
            for i in range(len(result.boxes)):
                name = result.names[int(result.boxes.cls[i])]
                counts[name] = counts.get(name, 0) + 1
                if name == "tricycle":
                    confs.append(float(result.boxes.conf[i]))
                    hit += 1
        if hit:
            frames_with_tricycle += 1

    print(f"扫描验证集帧 {len(images)} 张（imgsz=1280, conf=0.15）")
    for name, n in counts.items():
        print(f"  {name:<11} 检出 {n} 个")
    print(f"含三轮车的帧：{frames_with_tricycle} 张")
    if confs:
        confs.sort()
        print(f"三轮车置信度：最低 {confs[0]:.2f}  中位 {confs[len(confs)//2]:.2f}  最高 {confs[-1]:.2f}")
    print("\n说明：检出数量代表『三轮车』这类目标在当前素材里的可见规模，"
          "可直接用来判断拆类的样本量是否够。")


if __name__ == "__main__":
    main()
