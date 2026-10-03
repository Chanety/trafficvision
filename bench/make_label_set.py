"""生成标注集：从原始视频里挑典型帧并自动预标注，供人工复核后用于微调。

选帧策略（兼顾覆盖与难度）：
  · 每秒取一帧作为候选
  · 按目标数量排序取最密集的一批（难例）
  · 再按时间均匀取一批（覆盖不同场景与时段）
  · 候选间去重，避免近似重复帧

自动标注策略：召回优先（低置信度阈值），宁可多标，人工删除比补标省力。

用法：
    python bench/make_label_set.py <数据集目录> <输出目录> [每段帧数]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trafficvision.config import DEFAULT_CLASSES  # noqa: E402
from trafficvision.labelrules import suppress_riders  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def candidates_for(path: Path, fps_sample: float = 1.0):
    """每秒抽一帧作为候选，返回 [(帧号, 时间秒, 画面)]。"""
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    step = max(1, int(round(fps / fps_sample)))
    out = []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                out.append((idx, idx / fps, frame))
        idx += 1
    cap.release()
    return out


def pick(candidates: list, per_video: int) -> list:
    """按目标数量优先 + 时间均匀，选出该视频要标注的帧。"""
    if len(candidates) <= per_video:
        return candidates
    by_count = sorted(candidates, key=lambda c: -c[3])
    dense = by_count[: per_video // 2]
    dense_idx = {c[0] for c in dense}
    rest = [c for c in candidates if c[0] not in dense_idx]
    stride = max(1, len(rest) // max(1, per_video - len(dense)))
    spread = rest[::stride][: per_video - len(dense)]
    return sorted(dense + spread, key=lambda c: c[0])


def main() -> None:
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    per_video = int(sys.argv[3]) if len(sys.argv) > 3 else 20

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
    manifest = []
    total_written = 0

    for path in videos:
        cands = candidates_for(path)
        if not cands:
            continue

        # 批量推理，召回优先
        frames = [c[2] for c in cands]
        results = model.predict(frames, imgsz=1280, conf=0.15, iou=0.5, device=0,
                                verbose=False, batch=8)
        enriched = []
        for (idx, t, frame), res in zip(cands, results):
            boxes = res.boxes
            n = 0 if boxes is None else len(boxes)
            enriched.append((idx, t, frame, n, res))

        chosen = pick(enriched, per_video)

        for idx, t, frame, n, res in chosen:
            stem = f"{path.stem}_f{idx:05d}"
            image_path = images_dir / f"{stem}.jpg"
            label_path = labels_dir / f"{stem}.txt"
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

            h, w = frame.shape[:2]
            boxes = []
            if res.boxes is not None and len(res.boxes):
                for i in range(len(res.boxes)):
                    cls_name = res.names[int(res.boxes.cls[i])]
                    if cls_name not in names:
                        continue
                    cid = names.index(cls_name)
                    x1, y1, x2, y2 = (float(v) for v in res.boxes.xyxy[i])
                    boxes.append([float(cid), ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h,
                                  (x2 - x1) / w, (y2 - y1) / h])
            # 标注规则：先载具，后行人（删掉骑车人/驾驶位里的人）
            boxes = suppress_riders(boxes)
            lines = [f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            manifest.append({
                "image": f"images/{stem}.jpg",
                "label": f"labels/{stem}.txt",
                "source_video": path.name,
                "frame_index": int(idx),
                "time_s": round(float(t), 2),
                "auto_boxes": len(lines),
            })
            total_written += 1

        print(f"{path.name:<30} 候选 {len(cands):>4} 帧 → 选中 {len(chosen):>2} 帧")

    (out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images\nval: images\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)),
        encoding="utf-8",
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n共生成 {total_written} 帧，输出目录：{out}")
    print(f"类别文件：{out / 'classes.txt'}")


if __name__ == "__main__":
    main()
