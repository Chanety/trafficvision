"""视频预判主流程：检测跟踪 → 因果运动估计 → 画面渲染 → 精度评测。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .config import (
    DEFAULT_CLASSES,
    HORIZONS,
    MAX_CLIP_SECONDS,
    MIN_CLIP_SECONDS,
    MOTION_MODEL,
    STILL_THRESHOLD_PX,
    STILL_WINDOW,
    TRACKER,
)
from .motion import TrackMotion
from .render import draw_object, hud
from .state import StateTracker


def probe_video(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频：{path}")
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 25.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    info["duration"] = info["frames"] / info["fps"] if info["fps"] else 0.0
    return info


def _plan_clip(info: dict) -> int:
    """校验时长并返回实际处理的帧数。超出上限时截取，不足下限时报错。"""
    d = info["duration"]
    if d < MIN_CLIP_SECONDS - 0.05:
        raise SystemExit(
            f"视频时长 {d:.1f} 秒，短于可处理下限 {MIN_CLIP_SECONDS:g} 秒"
        )
    limit = int(round(MAX_CLIP_SECONDS * info["fps"]))
    if info["frames"] > limit:
        print(
            f"[提示] 视频 {d:.1f} 秒，超过 {MAX_CLIP_SECONDS:g} 秒上限，"
            f"自动截取前 {MAX_CLIP_SECONDS:g} 秒处理"
        )
        return limit
    return info["frames"]


def _iter_tracked(video: str | Path, model, info: dict, classes: tuple[str, ...],
                  imgsz: int, conf: float, device: str, max_frames: int,
                  motion_model: str = "cv", accel_std: float = 400.0,
                  accel_prior: float = 0.0, use_timestamps: bool = True,
                  tracker: str = TRACKER):
    """逐帧检测+跟踪，产出逐帧目标列表与全部轨迹对象。"""
    cap = cv2.VideoCapture(str(video))
    fps = info["fps"]
    size = (info["width"], info["height"])
    motions: dict[int, TrackMotion] = {}
    state_tracker = StateTracker(window=STILL_WINDOW, threshold_px=STILL_THRESHOLD_PX)
    index = 0
    records: list[list[dict]] = []
    times: list[float] = []

    while index < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        # 用容器里的真实时间戳，兼容手机常见的可变帧率视频
        stamp_s = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0 if use_timestamps else 0.0
        now = stamp_s if stamp_s > 0 else index / fps
        times.append(now)
        result = model.track(
            frame,
            persist=True,
            tracker=tracker,
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )[0]

        objects = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            ids = boxes.id
            for i in range(len(boxes)):
                cls_name = result.names[int(boxes.cls[i])]
                if cls_name not in classes:
                    continue
                # 跟踪器尚未分配 ID 时（例如画面中目标刚出现），
                # 必须给每帧内的目标各自独立的临时 ID，否则会把不同目标
                # 错并成一条长轨迹，导致速度估计完全失真。
                track_id = int(ids[i]) if ids is not None else -(index * 1000 + i + 1)
                bbox = [float(v) for v in boxes.xyxy[i]]
                motion = motions.get(track_id)
                if motion is None:
                    motion = TrackMotion(track_id, cls_name, fps, model=motion_model,
                                         accel_std=accel_std, accel_prior=accel_prior,
                                         cx_img=info["width"] / 2.0,
                                         frame_w=info["width"], frame_h=info["height"])
                    motions[track_id] = motion
                cx, cy = motion.update(bbox, now)
                # 运动状态：连续多帧未移动判为 constant（不显示、不预判）
                state, hidden = state_tracker.update(track_id, cls_name, index, cx, cy)

                preds = {h: motion.predict_box(h) for h in HORIZONS}
                objects.append(
                    {
                        "id": track_id,
                        "cls": cls_name,
                        "conf": float(boxes.conf[i]),
                        "bbox": bbox,
                        "centroid": (cx, cy),
                        "speed_px_s": motion.speed,
                        "predictable": motion.is_predictable() and not hidden,
                        "state": state,
                        "hidden": hidden,
                        "preds": {str(h): preds[h] for h in HORIZONS},
                        "trail": list(motion.history),
                        "t": now,
                    }
                )
        records.append(objects)
        index += 1

    cap.release()
    return records, motions, times


def _estimate_horizon(motions: dict, height: int) -> float | None:
    """跨轨迹联合估计地平线（相机属性），单条轨迹的估计只作为一个样本。"""
    estimates = []
    for motion in motions.values():
        value = motion.horizon_estimate()
        if value is not None and 0.0 < value < height:
            estimates.append(value)
    if not estimates:
        return None
    return float(np.median(estimates))


def _apply_scale_predictions(records: list[list[dict]], motions: dict, fps: float,
                             y_horizon: float | None, accel_prior: float,
                             min_horizon: float = 0.0) -> int:
    """用维度判定后的归一化匀速模型重算预判；不可用时保留原匀速结果。"""
    if y_horizon is None:
        return 0
    used = 0
    for t, frame in enumerate(records):
        for obj in frame:
            now = float(obj.get("t", t / fps))
            motion = motions.get(obj["id"])
            if motion is None:
                continue
            preds = {}
            for h in HORIZONS:
                if h < min_horizon:
                    preds[str(h)] = obj["preds"].get(str(h))
                    continue
                box = motion.predict_scale(now, h, y_horizon, accel_prior)
                if box is not None:
                    preds[str(h)] = box
                    used += 1
                else:
                    preds[str(h)] = obj["preds"].get(str(h))
            obj["preds"] = preds
    return used


def _evaluate(records: list[list[dict]], times: list[float], diagonal: float) -> dict:
    """用视频自身的未来帧作为真值，评估各时间梯度的预判误差。"""
    by_frame: list[dict[int, tuple[float, float]]] = [
        {o["id"]: o["centroid"] for o in frame} for frame in records
    ]
    t_arr = np.array(times, dtype=float)
    report = {}
    for h in HORIZONS:
        errors: list[float] = []
        for t, frame in enumerate(records):
            target = times[t] + h
            i1 = int(np.searchsorted(t_arr, target))
            if i1 >= len(records):
                continue
            i0 = max(0, i1 - 1)
            span = times[i1] - times[i0]
            frac = (target - times[i0]) / span if span > 1e-9 else 0.0
            for obj in frame:
                pred = obj["preds"].get(str(h))
                if pred is None or not obj["predictable"]:
                    continue
                tid = obj["id"]
                a = by_frame[i0].get(tid)
                b = by_frame[i1].get(tid)
                if a is not None and b is not None:
                    truth = (a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac)
                elif a is not None and frac < 0.5:
                    truth = a
                elif b is not None and frac >= 0.5:
                    truth = b
                else:
                    continue
                pc = ((pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0)
                errors.append(math.hypot(pc[0] - truth[0], pc[1] - truth[1]))
        if errors:
            arr = np.array(errors)
            report[f"{h:g}s"] = {
                "samples": int(arr.size),
                "mean_px": round(float(arr.mean()), 2),
                "median_px": round(float(np.median(arr)), 2),
                "p90_px": round(float(np.percentile(arr, 90)), 2),
                "mean_pct_of_diagonal": round(float(arr.mean() / diagonal * 100), 2),
            }
        else:
            report[f"{h:g}s"] = {"samples": 0}
    return report


def run_clip(
    video: str | Path,
    horizon: float,
    out_video: str | Path,
    weights: str,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    imgsz: int = 960,
    conf: float = 0.25,
    device: str = "0",
    world: bool = True,
    out_json: str | Path | None = None,
    gif: str | Path | None = None,
    gif_stride: int = 3,
    motion_model: str = MOTION_MODEL,
    accel_std: float = 400.0,
    accel_prior: float = 0.0,
    render: bool = True,
    max_frames_override: int | None = None,
    use_timestamps: bool = True,
    tracker: str = TRACKER,
) -> dict:
    from ultralytics import YOLO

    if horizon not in HORIZONS:
        raise SystemExit(f"预判梯度必须是 {HORIZONS} 之一，收到 {horizon}")

    info = probe_video(video)
    max_frames = _plan_clip(info)
    if max_frames_override is not None:
        max_frames = min(max_frames, int(max_frames_override))
    diagonal = math.hypot(info["width"], info["height"])

    model = YOLO(weights)
    if world:
        model.set_classes(list(classes))

    # 第一遍：检测跟踪 + 运动估计
    records, motions, times = _iter_tracked(video, model, info, classes, imgsz, conf, device,
                                            max_frames, motion_model, accel_std, accel_prior,
                                            use_timestamps, tracker)

    y_horizon = _estimate_horizon(motions, info["height"])
    scale_used = 0
    if motion_model == "scale":
        scale_used = _apply_scale_predictions(records, motions, info["fps"], y_horizon, accel_prior)
    elif motion_model == "auto":
        # 实测结论：0.1 秒档普通匀速更好（归一化反而引入方差），
        # 0.5 秒及以上维度判定明显更准 —— 因此按梯度自动分工。
        scale_used = _apply_scale_predictions(records, motions, info["fps"], y_horizon,
                                              accel_prior, min_horizon=0.5)

    metrics = _evaluate(records, times, diagonal)

    if not render:
        if out_json:
            Path(out_json).write_text(
                json.dumps(
                    {
                        "video": str(video),
                        "horizon_s": horizon,
                        "video_info": info,
                        "metrics": metrics,
                        "frames": [
                            [{k: v for k, v in o.items() if k != "trail"} for o in frame]
                            for frame in records
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return {
            "info": info,
            "metrics": metrics,
            "y_horizon": y_horizon,
            "scale_predictions": scale_used,
        }

    # 第二遍：渲染输出视频
    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info["fps"],
        (info["width"], info["height"]),
    )
    gif_frames = []
    idx = 0
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        for obj in records[idx] if idx < len(records) else []:
            if obj.get("hidden"):
                continue          # constant 状态的目标暂时不显示
            obj = dict(obj)
            obj["pred_box"] = obj["preds"].get(str(horizon))
            draw_object(frame, obj, horizon)
        hud(
            frame,
            [
                f"预判梯度: {horizon:g} s",
                f"帧: {idx + 1}/{max_frames}",
                f"跟踪目标: {len(records[idx]) if idx < len(records) else 0}",
            ],
        )
        writer.write(frame)
        if gif and idx % gif_stride == 0:
            small = cv2.resize(frame, (480, int(480 * info["height"] / info["width"])))
            gif_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    writer.release()

    if gif and gif_frames:
        from PIL import Image

        Image.fromarray(gif_frames[0]).save(
            str(gif),
            save_all=True,
            append_images=[Image.fromarray(f) for f in gif_frames[1:]],
            duration=int(1000 * gif_stride / info["fps"]),
            loop=0,
        )

    if out_json:
        payload = {
            "video": str(video),
            "horizon_s": horizon,
            "video_info": info,
            "first_frame_timestamp": times[0] if times else None,
            "metrics": metrics,
            "frames": [
                [
                    {k: v for k, v in o.items() if k != "trail"}
                    for o in frame
                ]
                for frame in records
            ],
        }
        Path(out_json).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    return {
        "info": info,
        "metrics": metrics,
        "y_horizon": y_horizon,
        "scale_predictions": scale_used,
    }
