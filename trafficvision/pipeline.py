"""视频预判主流程：检测跟踪 → 因果运动估计 → 画面渲染 → 精度评测。"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .config import (
    COAST_FRAMES,
    DEFAULT_CLASSES,
    HORIZONS,
    JUMP_IOU_MAX,
    JUMP_MIN_PX,
    JUMP_STEP_FACTOR,
    MAX_CLIP_SECONDS,
    MIN_SPEED_PX_S,
    MIN_TRACK_AGE,
    MIN_CLIP_SECONDS,
    NMS_IOU,
    MERGE_NAME_MAP,
    MOTION_MODEL,
    STILL_THRESHOLD_PX,
    STILL_BIAS_PX,
    STILL_WINDOW,
    STATIC_CLASSES,
    STATIC_CONF,
    STATIC_IMGSZ,
    STATIC_PROMPTS,
    STATIC_STRIDE,
    MIN_TARGET_AREA_PX,
    MIN_TARGET_SIDE_PX,
    TRACKER,
    static_thin_exempt,
    target_too_small,
)
from .motion import TrackMotion
from .render import draw_object, hud
from .state import StateTracker
from .staticmap import StaticObjectMap


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


class _TrackGuard:
    """轨迹跳变防护：把"跳到了另一个目标"的串轨切断，避免污染预判。

    实测（路口素材，6 段 300 帧）：7.8% 的误差样本是速度异常样本，
    却贡献了 25% 的误差总量，典型特征就是轨迹中心一帧内跳几百像素。

    判据必须同时满足两条，避免把真正的急加速/急转弯误杀：
      · 位移 > max(JUMP_MIN_PX, JUMP_STEP_FACTOR × 该轨迹步长中位数)
      · 与上一帧框的 IoU < JUMP_IOU_MAX（框几乎不重叠 = 换了个目标）
    触发后给该目标分配一个新的合成 ID（≥ SYNTHETIC_BASE），
    于下游看来这是"一条新轨迹"：运动模型重新初始化，不会从错误历史外推。
    """

    SYNTHETIC_BASE = 10_000_000

    def __init__(self, enabled: bool = True, min_px: float = JUMP_MIN_PX,
                 factor: float = JUMP_STEP_FACTOR, iou_max: float = JUMP_IOU_MAX) -> None:
        self.enabled = enabled
        self.min_px = float(min_px)
        self.factor = float(factor)
        self.iou_max = float(iou_max)
        self.active: dict[int, int] = {}
        self.last_centroid: dict[int, tuple[float, float]] = {}
        self.last_bbox: dict[int, list[float]] = {}
        self.steps: dict[int, deque[float]] = {}
        self._synthetic = 0
        self.breaks = 0

    @staticmethod
    def _iou(a: list[float], b: list[float]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def resolve(self, raw_id: int, bbox: list[float]) -> int:
        """返回该检测实际使用的轨迹 ID（可能是切断后的新 ID）。"""
        cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        current = self.active.get(raw_id, raw_id)
        prev = self.last_centroid.get(current)
        history = self.steps.get(current)

        if self.enabled and prev is not None and history and len(history) >= 3:
            dist = math.hypot(cx - prev[0], cy - prev[1])
            typical = float(np.median(history))
            if (dist > max(self.min_px, self.factor * typical)
                    and self._iou(bbox, self.last_bbox[current]) < self.iou_max):
                self._synthetic += 1
                current = self.SYNTHETIC_BASE + self._synthetic
                self.breaks += 1
                prev, history = None, None
                self.active[raw_id] = current

        self.last_centroid[current] = (cx, cy)
        self.last_bbox[current] = list(bbox)
        bucket = self.steps.setdefault(current, deque(maxlen=30))
        if prev is not None:
            bucket.append(math.hypot(cx - prev[0], cy - prev[1]))
        return current


def _iter_tracked(video: str | Path, model, info: dict, classes: tuple[str, ...],
                  imgsz: int, conf: float, device: str, max_frames: int,
                  motion_model: str = "cv", accel_std: float = 400.0,
                  accel_prior: float = 0.0, use_timestamps: bool = True,
                  tracker: str = TRACKER, guard: bool = True,
                  coast_frames: int = COAST_FRAMES, horizon: float = 1.0,
                  gate_maneuver: bool = False, nms_iou: float = NMS_IOU,
                  merge_names: bool = True,
                  min_side: float | None = None,
                  min_area: float = MIN_TARGET_AREA_PX):
    """逐帧检测+跟踪，产出逐帧目标列表与全部轨迹对象。"""
    cap = cv2.VideoCapture(str(video))
    fps = info["fps"]
    size = (info["width"], info["height"])
    motions: dict[int, TrackMotion] = {}
    track_guard = _TrackGuard(enabled=guard)
    too_small = 0
    coast_offscene = 0
    state_tracker = StateTracker(window=STILL_WINDOW, threshold_px=STILL_THRESHOLD_PX,
                                 bias_px=STILL_BIAS_PX)
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
            iou=nms_iou,
            device=device,
            verbose=False,
        )[0]

        objects = []
        seen_ids: set[int] = set()
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            ids = boxes.id
            for i in range(len(boxes)):
                cls_name = result.names[int(boxes.cls[i])]
                # 类别合并：开放词汇模型输出的是 motorcycle/bicycle/tricycle，
                # 统一映射成 nonmotor，保证产品输出口径与用户要求一致。
                if merge_names:
                    cls_name = MERGE_NAME_MAP.get(cls_name, cls_name)
                if cls_name not in classes:
                    continue
                # 跟踪器尚未分配 ID 时（例如画面中目标刚出现），
                # 必须给每帧内的目标各自独立的临时 ID，否则会把不同目标
                # 错并成一条长轨迹，导致速度估计完全失真。
                bbox = [float(v) for v in boxes.xyxy[i]]
                # 过小目标一律不识别（用户要求：动态静态一视同仁），
                # 这类框既不可靠、画出来也只是噪声。
                if target_too_small(bbox, cls_name, min_side, min_area):
                    too_small += 1
                    continue
                raw_id = int(ids[i]) if ids is not None else -(index * 1000 + i + 1)
                # 串轨防护：位移离谱且框不重叠时，切断并换新 ID
                track_id = track_guard.resolve(raw_id, bbox) if raw_id > 0 else raw_id
                seen_ids.add(track_id)
                motion = motions.get(track_id)
                if motion is None:
                    motion = TrackMotion(track_id, cls_name, fps, model=motion_model,
                                         accel_std=accel_std, accel_prior=accel_prior,
                                         cx_img=info["width"] / 2.0,
                                         frame_w=info["width"], frame_h=info["height"])
                    motions[track_id] = motion
                cx, cy = motion.update(bbox, now)
                # 运动状态：连续多帧未移动判为 constant（不显示、不预判）
                state, hidden, in_bias, spread = state_tracker.update(
                    track_id, cls_name, index, cx, cy, abs(bbox[3] - bbox[1]))

                preds = {h: motion.predict_box(h) for h in HORIZONS}
                # 机动判定：转弯/变道中的目标，匀速外推本来就不可靠，
                # 标出低置信（--gate-maneuver 时直接不给该梯度的预判）
                is_maneuver, turn, deviation = motion.maneuver(horizon)
                if gate_maneuver and is_maneuver:
                    preds = {h: None for h in HORIZONS}
                objects.append(
                    {
                        "id": track_id,
                        "cls": cls_name,
                        "conf": float(boxes.conf[i]),
                        "bbox": bbox,
                        "centroid": (cx, cy),
                        "speed_px_s": motion.speed,
                        "predictable": (motion.is_predictable() and not hidden
                                        and not (gate_maneuver and is_maneuver)),
                        "maneuver": is_maneuver,
                        "turn_deg_s": round(turn, 1) if turn is not None else None,
                        "maneuver_dev_px": round(deviation, 1),
                        "confidence": "low" if is_maneuver else "high",
                        "state": state,
                        "hidden": hidden,
                        "state_bias": in_bias,
                        "spread_px": round(spread, 2),
                        "preds": {str(h): preds[h] for h in HORIZONS},
                        "trail": list(motion.history),
                        "t": now,
                    }
                )

        # 遮挡续推：本帧没被跟踪器输出的成熟轨迹，短时间内继续外推显示，
        # 让画面上的框"穿过去"，而不是消失后再冒出来。
        if coast_frames > 0:
            for tid, motion in motions.items():
                if tid in seen_ids or motion.size is None or motion.last_t is None:
                    continue
                gap = float(now) - motion.last_t
                if gap <= 1e-6 or gap * fps > coast_frames:
                    continue
                if motion.frames < MIN_TRACK_AGE or motion.speed < MIN_SPEED_PX_S:
                    continue
                if not motion.quality_ok():
                    continue
                center = motion.kf.predict(gap)
                if center is None:
                    continue
                w, h = motion.size
                box = [float(center[0] - w / 2.0), float(center[1] - h / 2.0),
                       float(center[0] + w / 2.0), float(center[1] + h / 2.0)]
                # 幽灵续推防护：实测 64% 的续推目标此后再没被检出过，
                # 其中相当一部分是"目标已经驶出画面"，继续画只会留下一个
                # 跟着走但没人/没车的空框。中心点一旦离开画面就停止续推。
                if _left_scene(box, info["width"], info["height"]):
                    coast_offscene += 1
                    continue
                preds = {hh: motion.predict_box(gap + hh) for hh in HORIZONS}
                is_maneuver, turn, deviation = motion.maneuver(horizon)
                if gate_maneuver and is_maneuver:
                    preds = {hh: None for hh in HORIZONS}
                objects.append({
                    "id": tid,
                    "cls": motion.cls_name,
                    "conf": 0.0,
                    "bbox": box,
                    "centroid": (float(center[0]), float(center[1])),
                    "speed_px_s": motion.speed,
                    "predictable": not (gate_maneuver and is_maneuver),
                    "maneuver": is_maneuver,
                    "turn_deg_s": round(turn, 1) if turn is not None else None,
                    "maneuver_dev_px": round(deviation, 1),
                    "confidence": "low" if is_maneuver else "high",
                    "state": "variable",
                    "hidden": False,
                    "state_bias": False,
                    "spread_px": 0.0,
                    "coasting": True,
                    "preds": {str(hh): preds[hh] for hh in HORIZONS},
                    "trail": list(motion.history),
                    "t": now,
                })

        records.append(objects)
        index += 1

    cap.release()
    coasting = sum(1 for frame in records for o in frame if o.get("coasting"))
    stats = {
        "track_breaks": track_guard.breaks,
        "coasting_objects": coasting,
        "guard_enabled": bool(guard),
        "coast_frames": int(coast_frames),
        "too_small_skipped": too_small,
        "coast_offscene_skipped": coast_offscene,
    }
    return records, motions, times, stats


def _nms(items: list[tuple[float, list[float]]], iou_thr: float = 0.5):
    """同一概念下多提示词的互相去重（按置信度从高到低贪心保留）。"""
    kept: list[tuple[float, list[float]]] = []
    for conf, box in sorted(items, key=lambda z: -z[0]):
        drop = False
        for _, kb in kept:
            ix1, iy1 = max(box[0], kb[0]), max(box[1], kb[1])
            ix2, iy2 = min(box[2], kb[2]), min(box[3], kb[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0:
                continue
            aa = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            ab = max(0.0, kb[2] - kb[0]) * max(0.0, kb[3] - kb[1])
            union = aa + ab - inter
            if union > 0 and inter / union >= iou_thr:
                drop = True
                break
        if not drop:
            kept.append((conf, box))
    return kept


def scan_static_channel(video: str | Path, info: dict, weights: str | Path,
                        max_frames: int, device: str, conf: float = STATIC_CONF,
                        imgsz: int = STATIC_IMGSZ, stride: int = STATIC_STRIDE,
                        classes: tuple[str, ...] = STATIC_CLASSES,
                        min_side: float | None = None,
                        min_area: float = MIN_TARGET_AREA_PX) -> dict:
    """独立扫一遍静态目标（多提示词 + 低阈值），供 StaticObjectMap 投票。

    为什么要单独一条通道，而不是直接用主检测器：
      · 开放词汇模型对"路桩/bollard"这种单一提示词响应很弱（实测同一根混凝土
        路桩只有 0.02~0.16 分），换成 "concrete bollard" 立刻有 0.55~0.91 分。
        多提示词集成 + 低阈值 = 先把召回拉满；
      · 抓回来的假阳性交给跨帧投票去伪：真正的静态目标会在几十帧里反复出现，
        随机误检不会。
    """
    from ultralytics import YOLO

    prompts: list[str] = []
    mapping: dict[str, str] = {}
    for cls in classes:
        for word in STATIC_PROMPTS.get(cls, (cls,)):
            prompts.append(word)
            mapping[word] = cls
    if not prompts:
        return {}

    model = YOLO(str(weights))
    model.set_classes(prompts)

    cap = cv2.VideoCapture(str(video))
    batch, batch_idx = [], []
    idx = 0
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if stride <= 1 or idx % stride == 0:
            batch.append(frame)
            batch_idx.append(idx)
        idx += 1
    cap.release()

    out: dict[int, list[tuple[str, list[float], float]]] = {}
    if not batch:
        return out
    results = model.predict(batch, imgsz=imgsz, conf=conf, iou=0.6, device=device,
                            verbose=False, batch=4)
    for i, res in zip(batch_idx, results):
        per: dict[str, list[tuple[float, list[float]]]] = {}
        for b in res.boxes:
            name = res.names[int(b.cls)]
            cls = mapping.get(name)
            if cls is None:
                continue
            box = [float(v) for v in b.xyxy[0]]
            # 静态目标同样适用"过小不识别"，但灯杆/信号灯杆这类细长结构豁免：
            # 它们短边经常不到 10px，一刀切会把整根杆子丢掉。
            if target_too_small(box, cls, min_side, min_area) and not static_thin_exempt(box, cls):
                continue
            per.setdefault(cls, []).append((float(b.conf), box))
        dets: list[tuple[str, list[float], float]] = []
        for cls, items in per.items():
            for c, box in _nms(items):
                dets.append((cls, box, c))
        out[i] = dets
    return out


def _grab_frames(video: str | Path, indices) -> dict:
    """顺序扫一遍视频，抓出需要的帧（用于给静态目标做上下文裁剪）。"""
    want = sorted(set(int(i) for i in indices))
    if not want:
        return {}
    cap = cv2.VideoCapture(str(video))
    out: dict[int, np.ndarray] = {}
    target = 0
    idx = 0
    while target < len(want):
        ok = cap.grab()
        if not ok:
            break
        if idx == want[target]:
            ok, frame = cap.retrieve()
            if ok:
                out[idx] = frame
            target += 1
        idx += 1
    cap.release()
    return out


def _left_scene(box, width: int, height: int) -> bool:
    """目标是否已经离开画面（用于终止遮挡续推）。

    判据取"中心点出画"而不是"框完全出画"：部分出画的目标（车头刚压线）
    仍然是真实存在的，继续续推合理；中心点都出画了，就没有继续画的必要。
    """
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return not (0.0 <= cx < width and 0.0 <= cy < height)


def _resolve_prior_path(scene_prior, camera: str | None, video) -> Path | None:
    """解析「场景先验」文件路径。

    off   → None（每次现场重算，最慢）
    auto  → outputs/scene_prior/<机位名>.json，机位名默认取视频文件名
    其他   → 当作显式路径
    """
    if not scene_prior or str(scene_prior).lower() == "off":
        return None
    if str(scene_prior).lower() == "auto":
        key = camera or Path(video).stem
        return (Path(__file__).resolve().parents[1] / "outputs" / "scene_prior"
                / f"{key}.json")
    return Path(scene_prior)


def _verify_static(clusters: list[dict], video: str | Path, frame_w: int, frame_h: int,
                   device: str) -> dict:
    """对已确认的静态位置做一次上下文校验，返回 {idx: (decision, reason, top)}。"""
    from .verify import StaticVerifier, crop_with_context, decide

    needed = [f for c in clusters for f in c.get("sample_frames", [])]
    frames = _grab_frames(video, needed)
    verifier = StaticVerifier(device=device)
    out: dict[int, tuple[str, str, str]] = {}
    for c in clusters:
        crops = []
        for f in c.get("sample_frames", []):
            img = frames.get(f)
            if img is None:
                continue
            crop = crop_with_context(img, c["box"])
            if crop is not None:
                crops.append(crop)
        verdict = verifier.score_many(crops, c["cls"])
        decision, reason = decide(c["cls"], c["box"], frame_w, frame_h, verdict)
        out[c["idx"]] = (decision, reason, verdict.get("top", ""))
    return out


def _memory_object(item: dict, tid: int, t: float) -> dict:
    box = item["box"]
    return {
        "id": tid,
        "cls": item["cls"],
        "conf": 0.0,
        "bbox": box,
        "centroid": ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
        "speed_px_s": 0.0,
        "predictable": False,
        "maneuver": False,
        "turn_deg_s": None,
        "maneuver_dev_px": 0.0,
        "confidence": "high",
        "state": "constant",
        "hidden": False,
        "state_bias": False,
        "spread_px": 0.0,
        "coasting": False,
        "memory": True,
        "static_ratio": item.get("ratio"),
        "static_seen": item.get("seen"),
        "preds": {str(h): None for h in HORIZONS},
        "trail": [],
        "t": t,
    }


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
        # 遮挡续推出来的位置是"推算"，不是观测，绝不能当真值用
        {o["id"]: o["centroid"] for o in frame if not o.get("coasting")}
        for frame in records
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
    # 推理分辨率：实测 1280 明显优于 960（新素材验证集 mAP50 0.585→0.637，
    # 精确率 0.459→0.604；旧街面 0.680→0.701），1600 反而回落。
    imgsz: int = 1280,
    conf: float = 0.25,
    device: str = "0",
    # None = 按权重文件名自动判断（含 "world" 即开放词汇模型）。
    # 早前默认 True，传微调权重时会在 set_classes 上直接报错。
    world: bool | None = None,
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
    guard: bool = True,
    coast_frames: int = COAST_FRAMES,
    gate_maneuver: bool = False,
    static_map: bool = True,
    static_weights: str | None = None,
    static_conf: float = STATIC_CONF,
    static_imgsz: int = STATIC_IMGSZ,
    static_stride: int = STATIC_STRIDE,
    static_classes: tuple[str, ...] | None = None,
    static_verify: bool = True,
    scene_prior: str | None = "auto",
    camera: str | None = None,
    merge_names: bool = True,
    min_side: float | None = None,
    min_area: float = MIN_TARGET_AREA_PX,
    nms_iou: float = NMS_IOU,
) -> dict:
    from ultralytics import YOLO

    if horizon not in HORIZONS:
        raise SystemExit(f"预判梯度必须是 {HORIZONS} 之一，收到 {horizon}")

    info = probe_video(video)
    max_frames = _plan_clip(info)
    if max_frames_override is not None:
        max_frames = min(max_frames, int(max_frames_override))
    diagonal = math.hypot(info["width"], info["height"])

    weight_path = Path(weights)
    if not weight_path.exists():
        # 默认权重是微调模型，不随仓库分发；直接把 Ultralytics 的
        # "文件不存在" 抛给用户很容易被误判成程序坏了。
        raise SystemExit(
            f"找不到权重文件：{weight_path}\n"
            f"  默认权重是微调合并模型 v7（体积较大，不随仓库分发）。\n"
            f"  获取方式与 SHA-256 见 models/README.md 与 models/manifest.json；\n"
            f"  也可以用 --weights 指定其他权重，例如开放词汇模型\n"
            f"  trafficlab-3d/repo/yolov8s-worldv2.pt（会走开放词汇模式）。"
        )
    model = YOLO(str(weight_path))
    if world is None:
        world = "world" in Path(weights).name.lower()
    if world:
        model.set_classes(list(classes))
        # 合并口径：提示词仍是英文原词（模型只认这些），但过滤表换成合并名
        if merge_names:
            classes = tuple(dict.fromkeys(MERGE_NAME_MAP.get(c, c) for c in classes))
    else:
        # 微调模型自带类别表：以它为准，避免默认类别表把模型认得的类过滤掉
        # （例如 7 类合并模型里的 nonmotor 不在默认 9 类里）
        classes = tuple(model.names[i] for i in sorted(model.names))

    # 第一遍：检测跟踪 + 运动估计
    records, motions, times, track_stats = _iter_tracked(
        video, model, info, classes, imgsz, conf, device,
        max_frames, motion_model, accel_std, accel_prior,
        use_timestamps, tracker, guard, coast_frames, horizon, gate_maneuver,
        nms_iou=nms_iou,
        merge_names=merge_names, min_side=min_side, min_area=min_area)

    # ------------------------------------------------------------ 静态目标地图
    # 用户方案：非阻挡状态下 ≥80% 的帧被识别 → 该位置写入隐标记；
    # 之后检测器漏掉的帧按隐标记补回。实测正是"几根并排路桩只标出一根"、
    # "人走过去就没了"这两类问题的直接解法。
    static_stats: dict = {"enabled": False}
    if static_map:
        sweep_classes = tuple(static_classes) if static_classes else STATIC_CLASSES
        # 场景先验缓存：摄像机长期不动，同一机位"哪些位置是固定结构"只需算一次。
        # 现场重算要跑「开放词汇扫描 + CLIP 语义校验」，实测把端到端从约
        # 0.09 s/帧 抬到 0.53 s/帧；命中缓存后这两步全跳过。
        prior_path = _resolve_prior_path(scene_prior, camera, video)
        smap = None
        if prior_path is not None and prior_path.exists():
            try:
                smap = StaticObjectMap.load_prior(prior_path)
                print(f"[场景先验] 命中缓存 {prior_path}（{len(smap.clusters)} 个位置），"
                      f"跳过扫描与语义校验")
            except Exception as exc:
                print(f"[场景先验] 缓存不可用（{type(exc).__name__}: {exc}），改为现场重算")
                smap = None

        default_world = (Path(__file__).resolve().parents[1]
                         / "trafficlab-3d" / "repo" / "yolov8s-worldv2.pt")
        sweep_weights = Path(static_weights) if static_weights else default_world
        if smap is not None:
            confirmed = smap.confirmed()
            verify_stats: dict = {"enabled": False, "source": "cache",
                                  "prior": str(prior_path)}
            source = "cache"
        elif sweep_weights.exists():
            try:
                scan = scan_static_channel(video, info, sweep_weights, max_frames, device,
                                           static_conf, static_imgsz, static_stride,
                                           classes=sweep_classes,
                                           min_side=min_side, min_area=min_area)
                smap = StaticObjectMap(classes=sweep_classes)
                # 只对"真正扫过"的帧做登记；每 N 帧扫一次时，一次结果覆盖 N 帧窗口
                for i in sorted(scan):
                    if i >= len(records):
                        break
                    dynamics = [o for o in records[i] if o["cls"] not in STATIC_CLASSES]
                    smap.observe(i, scan[i], dynamics, span=static_stride)
                confirmed = smap.confirmed()
                # 语义校验：跨帧投票只证明"位置稳定"，证明不了"这就是那类东西"。
                # 花坛边缘、路缘石、广告牌同样常年不动，一样能通过 80% 检验。
                # 这里把每个位置连上下文裁出来做一次 CLIP 判断：明确不像的不画，
                # 拿不准的照旧画出来——宁可让人看见一个误检，也不要漏掉真目标。
                verify_stats: dict = {"enabled": False}
                if static_verify and confirmed:
                    from .verify import MissingClipWeights as _MissingClipWeights
                    try:
                        decisions = _verify_static(confirmed, video, info["width"],
                                                   info["height"], device)
                        kept, unsure, dropped = [], [], []
                        for c in confirmed:
                            d, why, top = decisions.get(c["idx"], ("keep", "", ""))
                            c["verify"] = d
                            c["verify_reason"] = why
                            c["verify_top"] = top
                            # 写回簇对象，落盘场景先验时才会带上校验结论
                            smap.clusters[c["idx"]].verify = d
                            if d == "keep":
                                kept.append(c)
                            elif d == "review":
                                unsure.append(c)
                            else:
                                dropped.append(c)
                        verify_stats = {
                            "enabled": True,
                            "keep": len(kept), "review": len(unsure), "reject": len(dropped),
                            "rejected_items": [
                                {"cls": c["cls"], "ratio": c["ratio"],
                                 "reason": c["verify_reason"], "top": c["verify_top"]}
                                for c in dropped],
                        }
                        confirmed = kept + unsure
                    except _MissingClipWeights as exc:
                        verify_stats = {"enabled": True, "skipped": str(exc)}
                    except Exception as exc:
                        verify_stats = {"enabled": True,
                                        "error": f"{type(exc).__name__}: {exc}"}
                source = "scan"
            except Exception as exc:  # 静态通道失败不应阻断主流程
                static_stats = {"enabled": True, "error": f"{type(exc).__name__}: {exc}"}
        else:
            static_stats = {"enabled": False, "reason": f"找不到开放词汇权重 {sweep_weights}"}

        if smap is not None:
            injected = 0
            allowed = {c["idx"] for c in confirmed}
            for i, frame in enumerate(records):
                dynamics = [o for o in frame if o["cls"] not in STATIC_CLASSES]
                for item in smap.missing(i, dynamics, present=frame):
                    if item["idx"] not in allowed:
                        continue
                    frame.append(_memory_object(item, 20_000_000 + item["idx"], times[i]))
                    injected += 1

            # 现场重算的结果落盘，供同一机位的其他视频复用
            saved = None
            if source == "scan" and prior_path is not None and static_verify:
                try:
                    smap.save_prior(prior_path, camera=camera or Path(video).stem,
                                    frame_size=(info["width"], info["height"]),
                                    extra={"source_video": Path(video).name})
                    saved = str(prior_path)
                except Exception as exc:
                    print(f"[场景先验] 保存失败：{type(exc).__name__}: {exc}")

            summary = smap.summary()
            static_stats = {
                "enabled": True,
                "source": source,
                "prior_path": str(prior_path) if prior_path else None,
                "prior_saved": saved,
                "weights": sweep_weights.name,
                "conf": static_conf,
                "imgsz": static_imgsz,
                "stride": static_stride,
                "classes": list(sweep_classes),
                "clusters": summary["clusters"],
                "confirmed": summary["confirmed"],
                "rejected": summary["rejected"],
                "dropped_small": summary.get("dropped_small", 0),
                "by_class": summary["by_class"],
                "injected_boxes": injected,
                "verify": verify_stats,
                "confirmed_after_verify": len(confirmed),
                "items": [
                    {"cls": d["cls"], "ratio": d["ratio"], "seen": d["seen"],
                     "visible": d["visible"], "occluded": d["occluded"],
                     "box": [round(v, 1) for v in d["box"]],
                     "verify": d.get("verify", "keep")}
                    for d in confirmed
                ],
            }

    y_horizon = _estimate_horizon(motions, info["height"])
    scale_used = 0
    if motion_model == "scale":
        scale_used = _apply_scale_predictions(records, motions, info["fps"], y_horizon, accel_prior)
    elif motion_model == "auto":
        # 实测结论（videoset_02 6 段 × 300 帧，7 类模型，同一批目标配对比较）：
        #   0.5 秒：匀速 中位16.4/均值26.8  vs  维度判定 17.8/31.9
        #   1 秒  ：匀速 34.2/56.9            vs  34.0/62.8
        #   2 秒  ：匀速 78.2/133.6           vs  75.2/131.4（维度判定略优）
        # 也就是说"维度判定"只在 2 秒档有意义，0.5~1 秒用它是负收益
        # （早期合成实验显示斜视机位更好，但真实素材上不成立）。
        # 因此 auto 只在 2 秒档启用维度判定。
        scale_used = _apply_scale_predictions(records, motions, info["fps"], y_horizon,
                                              accel_prior, min_horizon=2.0)

    metrics = _evaluate(records, times, diagonal)

    if not render:
        if out_json:
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text(
                json.dumps(
                    {
                        "video": str(video),
                        "horizon_s": horizon,
                        "video_info": info,
                        "track_stats": track_stats,
                        "static_map": static_stats,
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
            "static_map": static_stats,
            "track_stats": track_stats,
        }

    # 第二遍：渲染输出视频
    # 输出目录统一先建好：cv2.VideoWriter 在父目录不存在时会**静默失败**
    # （不抛异常、不写文件），JSON/GIF 也会在最后一步才报错。
    out_video_path = Path(out_video)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info["fps"],
        (info["width"], info["height"]),
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(
            f"无法创建输出视频：{out_video_path}\n"
            f"  常见原因：路径不可写、磁盘已满、或 mp4v 编解码器不可用。"
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
        frame_objs = records[idx] if idx < len(records) else []
        live = sum(1 for o in frame_objs if not o.get("memory"))
        memo = len(frame_objs) - live
        hud(
            frame,
            [
                f"预判梯度: {horizon:g} s",
                f"帧: {idx + 1}/{max_frames}",
                f"跟踪目标: {live}" + (f"  （静态记忆 {memo}）" if memo else ""),
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

        Path(gif).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(gif_frames[0]).save(
            str(gif),
            save_all=True,
            append_images=[Image.fromarray(f) for f in gif_frames[1:]],
            duration=int(1000 * gif_stride / info["fps"]),
            loop=0,
        )

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "video": str(video),
            "horizon_s": horizon,
            "video_info": info,
            "first_frame_timestamp": times[0] if times else None,
            "track_stats": track_stats,
            "static_map": static_stats,
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
        "static_map": static_stats,
    }
