"""逐帧分析会话：把「模型只加载一次」变成可能。

为什么需要它
------------
`trafficvision.cli` 是无状态批处理器：每个视频从零加载模型、重扫静态结构、结束即退出。
在线分析要求把这两项固定成本从「每段视频付一次」变成「每个进程/机位付一次」。
本模块直接驱动 `trafficvision.pipeline` 里同一批算法类（TrackMotion / _TrackGuard /
StateTracker），逐帧推进，不复制任何算法逻辑。

用法（服务端内部）::

    sess = StreamSession(SessionConfig(camera="cam01"))
    for frame, t in frames:
        result = sess.push(frame, t)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from trafficvision.config import (
    COAST_FRAMES, HORIZONS, MERGE_NAME_MAP, MIN_SPEED_PX_S, MIN_TARGET_AREA_PX,
    MIN_TRACK_AGE, NMS_IOU, STILL_BIAS_PX, STILL_THRESHOLD_PX, STILL_WINDOW,
    TRACKER,
)
from trafficvision.pipeline import (
    TrackMotion, _TrackGuard, _left_scene, target_too_small,
)
from trafficvision.state import StateTracker

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "outputs" / "train" / "v7" / "weights" / "best.pt"


@dataclass
class SessionConfig:
    """一个分析会话的参数。等价于 CLI 的一组开关。"""

    weights: str = str(DEFAULT_WEIGHTS)
    horizon: float = 1.0
    imgsz: int = 1280
    conf: float = 0.25
    device: str = "0"
    tracker: str = TRACKER
    motion_model: str = "auto"
    accel_std: float = 400.0
    coast: int = COAST_FRAMES
    guard: bool = True
    min_side: float | None = None
    min_area: float = MIN_TARGET_AREA_PX
    fps: float = 30.0
    frame_w: int = 1920
    frame_h: int = 1080
    camera: str | None = None
    render: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d.pop("extras", None)
        d["weights"] = Path(self.weights).name
        return d


class StreamSession:
    """常驻会话：模型只加载一次，之后逐帧推入。"""

    def __init__(self, cfg: SessionConfig):
        from ultralytics import YOLO  # 延迟导入，便于无 GPU 环境做单元测试

        self.cfg = cfg
        self.model = YOLO(cfg.weights)
        self.motions: dict[int, TrackMotion] = {}
        self.guard = _TrackGuard(enabled=cfg.guard)
        self.state = StateTracker(window=STILL_WINDOW, threshold_px=STILL_THRESHOLD_PX,
                                  bias_px=STILL_BIAS_PX)
        self.index = 0
        self.frames_seen = 0
        self.frames_analyzed = 0
        self.latency_sum = 0.0
        self._seen_ids: set[int] = set()
        self.created = time.time()

    # ------------------------------------------------------------------ 逐帧
    def push(self, frame: np.ndarray, now: float | None = None,
             analyze: bool = True) -> dict[str, Any]:
        """推入一帧。analyze=False 表示该帧跳过检测（跳帧模式的中间帧）。

        返回该帧的结构化结果；不返回画好的图，渲染交给前端。
        """
        t0 = time.perf_counter()
        cfg = self.cfg
        t = float(now) if now is not None else self.frames_seen / max(cfg.fps, 1e-6)
        self.frames_seen += 1

        objects: list[dict[str, Any]] = []
        self._seen_ids = set()
        if analyze:
            self.frames_analyzed += 1
            objects = self._detect_and_track(frame, t)
        objects += self._coast(t)

        latency = (time.perf_counter() - t0) * 1000.0
        self.latency_sum += latency
        self.index += 1
        return {
            "frame_index": self.frames_seen - 1,
            "t": round(t, 4),
            "analyzed": analyze,
            "objects": objects,
            "stats": {
                "tracked": sum(1 for o in objects if not o.get("coasting")),
                "predictable": sum(1 for o in objects if o.get("predictable")),
                "coasting": sum(1 for o in objects if o.get("coasting")),
                "memory": sum(1 for o in objects if o.get("memory")),
            },
            "latency_ms": round(latency, 2),
        }

    # ------------------------------------------------------------------ 内部
    def _detect_and_track(self, frame: np.ndarray, t: float) -> list[dict[str, Any]]:
        cfg = self.cfg
        res = self.model.track(frame, persist=True, tracker=cfg.tracker,
                               imgsz=cfg.imgsz, conf=cfg.conf, iou=NMS_IOU,
                               device=cfg.device, verbose=False)[0]
        out: list[dict[str, Any]] = []
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return out
        ids = boxes.id
        for i in range(len(boxes)):
            cls_name = res.names[int(boxes.cls[i])]
            cls_name = MERGE_NAME_MAP.get(cls_name, cls_name)
            bbox = [float(v) for v in boxes.xyxy[i]]
            if target_too_small(bbox, cls_name, cfg.min_side, cfg.min_area):
                continue
            raw_id = int(ids[i]) if ids is not None else -(self.index * 1000 + i + 1)
            tid = self.guard.resolve(raw_id, bbox) if raw_id > 0 else raw_id
            self._seen_ids.add(tid)
            motion = self.motions.get(tid)
            if motion is None:
                motion = TrackMotion(tid, cls_name, cfg.fps, model=cfg.motion_model,
                                     accel_std=cfg.accel_std,
                                     cx_img=cfg.frame_w / 2.0,
                                     frame_w=cfg.frame_w, frame_h=cfg.frame_h)
                self.motions[tid] = motion
            cx, cy = motion.update(bbox, t)
            state, hidden, in_bias, spread = self.state.update(
                tid, cls_name, self.index, cx, cy, abs(bbox[3] - bbox[1]))
            out.append(self._object(motion, tid, bbox, cx, cy, state, hidden,
                                    in_bias, spread, float(boxes.conf[i]), t))
        return out

    def _object(self, motion, tid, bbox, cx, cy, state, hidden, in_bias, spread,
                conf, t, coasting=False) -> dict[str, Any]:
        cfg = self.cfg
        preds = {str(h): motion.predict_box(h) for h in HORIZONS}
        is_maneuver, turn, deviation = motion.maneuver(cfg.horizon)
        return {
            "id": tid,
            "cls": motion.cls_name,
            "conf": round(conf, 3),
            "bbox": [round(v, 1) for v in bbox],
            "centroid": [round(cx, 1), round(cy, 1)],
            "speed_px_s": round(motion.speed, 1),
            "predictable": bool(motion.is_predictable() and not hidden),
            "maneuver": bool(is_maneuver),
            "turn_deg_s": round(turn, 1) if turn is not None else None,
            "state": state,
            "hidden": bool(hidden),
            "state_bias": bool(in_bias),
            "spread_px": round(spread, 2),
            "coasting": coasting,
            "preds": {k: ([round(x, 1) for x in v] if v else None)
                      for k, v in preds.items()},
            "t": round(t, 4),
        }

    def _coast(self, t: float) -> list[dict[str, Any]]:
        """遮挡续推：本帧没被检测到的成熟轨迹继续外推显示（带出画即停）。"""
        cfg = self.cfg
        if cfg.coast <= 0:
            return []
        out: list[dict[str, Any]] = []
        for tid, motion in self.motions.items():
            if tid in self._seen_ids or motion.size is None or motion.last_t is None:
                continue
            gap = float(t) - float(motion.last_t)
            if gap <= 1e-6 or gap * cfg.fps > cfg.coast:
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
            if _left_scene(box, cfg.frame_w, cfg.frame_h):
                continue  # 幽灵续推防护：外推中心一旦出画即停止
            out.append(self._object(motion, tid, box, float(center[0]),
                                    float(center[1]), "variable",
                                    False, False, 0.0, 0.0, t, coasting=True))
        return out

    # ------------------------------------------------------------------ 报告
    def report(self) -> dict[str, Any]:
        return {
            "config": self.cfg.as_dict(),
            "frames_seen": self.frames_seen,
            "frames_analyzed": self.frames_analyzed,
            "tracks": len(self.motions),
            "mean_latency_ms": round(self.latency_sum / max(self.frames_analyzed, 1), 2),
            "uptime_s": round(time.time() - self.created, 1),
        }
