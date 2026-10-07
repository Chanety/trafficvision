"""路桩漏检诊断：定位静态小目标为什么检不到 / 时有时无。

背景：用户反馈 videoset_02 里近处并排的几根路桩，只有最右侧那根被标注出来，
左侧两根完全没有；而且同一根路桩"这一帧有、人走过去之后就没有了"。

本脚本做三件事：
  1. 把指定区域裁剪放大存图，便于人眼确认到底有几根、长什么样；
  2. 用多套权重（开放词汇 / 微调 v5 / 微调 v7）+ 多套提示词，
     在极低置信度（默认 0.01）下把所有候选框打出来——看是"完全没响应"
     还是"有响应但分数低于阈值"；
  3. 逐帧统计某类目标的出现/消失，量化"时有时无"。

用法：
    # 单帧诊断 + 裁剪
    python bench/bollard_probe.py --video datasets/videoset_02_origin/VID_20261004_112222.mp4 \
        --frames 0,30,60 --roi 0.0,0.60,0.50,0.95 --crops outputs/probe_bollard

    # 逐帧时序统计
    python bench/bollard_probe.py --video ... --timeline --cls bollard
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# 开放词汇模型用的类别提示词。第一组 = 产品当前用的英文单词（与 DEFAULT_CLASSES 一致），
# 后面几组只替换 bollard 一项，用来验证"是不是提示词选得不好导致不响应"。
PROMPT_VARIANTS: dict[str, list[str]] = {
    "baseline":     ["car", "truck", "bus", "motorcycle", "bicycle", "person", "sign", "bollard", "tricycle"],
    "bollard短柱":   ["car", "truck", "bus", "motorcycle", "bicycle", "person", "sign", "traffic bollard", "tricycle"],
    "concrete":     ["car", "truck", "bus", "motorcycle", "bicycle", "person", "sign", "concrete bollard", "tricycle"],
    "post":         ["car", "truck", "bus", "motorcycle", "bicycle", "person", "sign", "post", "tricycle"],
    "pillar":       ["car", "truck", "bus", "motorcycle", "bicycle", "person", "sign", "pillar", "tricycle"],
    "barrier":      ["car", "truck", "bus", "motorcycle", "bicycle", "person", "sign", "barrier post", "tricycle"],
    "shortpost":    ["car", "truck", "bus", "motorcycle", "bicycle", "person", "sign", "short post", "tricycle"],
    "中文路桩":      ["car", "truck", "bus", "motorcycle", "bicycle", "person", "sign", "路桩", "tricycle"],
}


def grab_frames(video: Path, indices: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    want = set(indices)
    out: dict[int, np.ndarray] = {}
    idx = 0
    while want:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            out[idx] = frame
            want.discard(idx)
        idx += 1
    cap.release()
    return out


def crop_roi(frame: np.ndarray, roi: tuple[float, float, float, float], scale: float = 2.0):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    a = frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
    return cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def in_roi(box, roi, w, h) -> bool:
    cx = (box[0] + box[2]) / 2 / w
    cy = (box[1] + box[3]) / 2 / h
    x1, y1, x2, y2 = roi
    return x1 <= cx <= x2 and y1 <= cy <= y2


def run_model(model, frames: list[np.ndarray], imgsz: int, conf: float):
    return model.predict(frames, imgsz=imgsz, conf=conf, iou=0.5, device=0, verbose=False, batch=8)


def probe_prompts(video: Path, frames_idx: list[int], roi, conf: float, imgsz: int) -> list[dict]:
    from ultralytics import YOLO

    frames = grab_frames(video, frames_idx)
    order = sorted(frames)
    imgs = [frames[i] for i in order]
    h, w = imgs[0].shape[:2]

    world = YOLO(str(ROOT / "trafficlab-3d" / "repo" / "yolov8s-worldv2.pt"))
    rows = []
    for name, prompts in PROMPT_VARIANTS.items():
        world.set_classes(prompts)
        results = run_model(world, imgs, imgsz, conf)
        for i, res in zip(order, results):
            for b in res.boxes:
                nm = res.names[int(b.cls)]
                if nm not in ("bollard", "post", "pillar", "traffic bollard", "concrete bollard",
                              "barrier post", "short post", "路桩"):
                    continue
                xyxy = [float(v) for v in b.xyxy[0]]
                if not in_roi(xyxy, roi, w, h):
                    continue
                rows.append({
                    "prompt_set": name, "frame": i, "class": nm,
                    "conf": round(float(b.conf), 4),
                    "cx": round((xyxy[0] + xyxy[2]) / 2 / w, 4),
                    "cy": round((xyxy[1] + xyxy[3]) / 2 / h, 4),
                    "w": round((xyxy[2] - xyxy[0]) / w, 4),
                    "h": round((xyxy[3] - xyxy[1]) / h, 4),
                })
    return rows


def probe_finetuned(video: Path, frames_idx: list[int], roi, conf: float, imgsz: int,
                    weights: list[tuple[str, Path]]) -> list[dict]:
    from ultralytics import YOLO

    frames = grab_frames(video, frames_idx)
    order = sorted(frames)
    imgs = [frames[i] for i in order]
    h, w = imgs[0].shape[:2]
    rows = []
    for tag, path in weights:
        if not path.exists():
            print(f"[跳过] 权重不存在：{path}")
            continue
        model = YOLO(str(path))
        names = model.names
        results = run_model(model, imgs, imgsz, conf)
        for i, res in zip(order, results):
            for b in res.boxes:
                nm = names[int(b.cls)]
                xyxy = [float(v) for v in b.xyxy[0]]
                if nm not in ("bollard", "sign") or not in_roi(xyxy, roi, w, h):
                    continue
                rows.append({
                    "model": tag, "frame": i, "class": nm,
                    "conf": round(float(b.conf), 4),
                    "cx": round((xyxy[0] + xyxy[2]) / 2 / w, 4),
                    "cy": round((xyxy[1] + xyxy[3]) / 2 / h, 4),
                    "w": round((xyxy[2] - xyxy[0]) / w, 4),
                    "h": round((xyxy[3] - xyxy[1]) / h, 4),
                })
    return rows


def timeline(video: Path, cls_name: str, weights: Path, imgsz: int, conf: float, step: float):
    """逐帧统计某类目标的出现/消失，量化"时有时无"。"""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model.names
    ids = [i for i, n in names.items() if n == cls_name] if isinstance(names, dict) else \
          [i for i, n in enumerate(names) if n == cls_name]
    cid = ids[0]

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    stride = max(1, int(round(fps * step)))
    rows = []
    idx = 0
    batch, batch_idx = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            batch.append(frame)
            batch_idx.append(idx)
        idx += 1
    cap.release()

    results = model.predict(batch, imgsz=imgsz, conf=conf, iou=0.5, device=0, verbose=False, batch=8)
    for i, res in zip(batch_idx, results):
        hits = []
        for b in res.boxes:
            if int(b.cls) == cid:
                xyxy = [float(v) for v in b.xyxy[0]]
                hits.append((round(float(b.conf), 3), round((xyxy[0] + xyxy[2]) / 2 / res.orig_shape[1], 3)))
        rows.append({"frame": i, "t": round(i / fps, 2), "n": len(hits), "hits": hits})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", default="0", help="逗号分隔的帧号")
    ap.add_argument("--roi", default="0.0,0.60,0.50,0.95", help="x1,y1,x2,y2（归一化）")
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--crops", default=None, help="裁剪放大图的输出目录")
    ap.add_argument("--crop-scale", type=float, default=2.0)
    ap.add_argument("--timeline", action="store_true", help="额外做逐帧时序统计")
    ap.add_argument("--cls", default="bollard")
    ap.add_argument("--step", type=float, default=0.1, help="时序统计的抽样间隔（秒）")
    ap.add_argument("--json", dest="out_json", default=None)
    args = ap.parse_args()

    video = Path(args.video)
    roi = tuple(float(v) for v in args.roi.split(","))
    frames_idx = [int(v) for v in args.frames.split(",")]

    print(f"视频：{video.name}")
    print(f"ROI ：x {roi[0]}~{roi[2]}  y {roi[1]}~{roi[3]}（归一化）")
    print(f"低阈值扫描：conf={args.conf}  imgsz={args.imgsz}\n")

    payload: dict = {"video": video.name, "roi": roi, "conf": args.conf}

    if args.crops:
        out = Path(args.crops)
        out.mkdir(parents=True, exist_ok=True)
        frames = grab_frames(video, frames_idx)
        for i, fr in sorted(frames.items()):
            cv2.imwrite(str(out / f"{video.stem}_f{i:05d}_roi.jpg"),
                        crop_roi(fr, roi, args.crop_scale),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"裁剪图已保存：{out}\n")

    print("=== 一、开放词汇模型：换提示词看响应 ===")
    rows = probe_prompts(video, frames_idx, roi, args.conf, args.imgsz)
    payload["prompt_scores"] = rows
    if rows:
        by = {}
        for r in rows:
            key = (r["prompt_set"], r["frame"])
            by.setdefault(key, []).append(r)
        for (ps, fr), items in sorted(by.items()):
            s = "  ".join(f"{it['class']}@{it['conf']:.3f}(x={it['cx']:.2f},y={it['cy']:.2f},h={it['h']:.3f})"
                          for it in sorted(items, key=lambda z: -z["conf"]))
            print(f"  {ps:<12} 帧{fr:<5} {s}")
    else:
        print("  （ROI 内没有任何 bollard 类响应，连 0.01 都没有）")

    print("\n=== 二、微调模型：同一区域 ===")
    w7 = ROOT / "outputs" / "train" / "v7" / "weights" / "best.pt"
    w5 = ROOT / "outputs" / "train" / "v5" / "weights" / "best.pt"
    rows2 = probe_finetuned(video, frames_idx, roi, args.conf, args.imgsz,
                            [("v7(7类)", w7), ("v5(9类)", w5)])
    payload["finetuned"] = rows2
    if rows2:
        for r in sorted(rows2, key=lambda z: (z["frame"], -z["conf"])):
            print(f"  {r['model']:<9} 帧{r['frame']:<5} {r['class']:<8} conf={r['conf']:.3f} "
                  f"x={r['cx']:.2f} y={r['cy']:.2f} w={r['w']:.3f} h={r['h']:.3f}")
    else:
        print("  （微调模型在 ROI 内连 0.01 都没有 bollard）")

    if args.timeline:
        print(f"\n=== 三、逐帧时序：{args.cls} 每 {args.step}s 采样（v7 权重）===")
        tl = timeline(video, args.cls, w7, args.imgsz, 0.05, args.step)
        payload["timeline"] = tl
        seen = sum(1 for r in tl if r["n"] > 0)
        print(f"  采样 {len(tl)} 帧，出现 {seen} 帧（{seen / max(1, len(tl)) * 100:.0f}%）")
        for r in tl:
            mark = "".join("●" if r["n"] else "·") 
            hits = "  ".join(f"conf={c}(x={x})" for c, x in r["hits"])
            print(f"  f{r['frame']:<5} t={r['t']:<6} n={r['n']} {mark} {hits}")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入：{args.out_json}")


if __name__ == "__main__":
    main()
