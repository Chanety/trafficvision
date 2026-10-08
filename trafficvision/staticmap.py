"""静态目标地图：把整段视频里反复识别到的静态目标固化成"隐标记"。

对应用户提的方案：
  "在一张图内有的路桩无法被识别，有的此次被识别但人走过之后就没有识别了。
   建议采用一种回档查看的方式——在非阻挡状态下百分之八十多的情况下
   该目标被识别为路障，那么就在训练完成后把这个位置隐标记为路障所在位置。"

这里把它做成一个与训练解耦的**推理期机制**，因此不重训也能生效：

  1. `observe()`  逐帧登记静态目标的检测结果，并统计该位置在"未被遮挡"的帧里
                  被检出的比例；
  2. 比例 ≥ `min_ratio`（默认 0.80）且命中帧数 ≥ `min_frames` → 确认为静态目标；
  3. `missing()`  之后每一帧，只要该位置没被遮挡、而检测器这一帧又没输出，
                  就把隐标记补回画面。

遮挡判据：任一动态目标的框覆盖该静态框面积 ≥ `occlude_cover`（默认 50%），
或盖住了它的中心点，就算这一帧"看不见"。看不见的帧不参与比例统计，
也不会被补回——否则会在一辆驶过的公交车上画出一根柱子。
"""

from __future__ import annotations

import json
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import (
    STATIC_CLASSES,
    STATIC_MAP_MATCH_IOU,
    STATIC_MAP_MERGE_DIST,
    STATIC_MAP_MERGE_IOU,
    STATIC_MAP_MIN_FRAMES,
    STATIC_MAP_MIN_RATIO,
    STATIC_MAP_MIN_AREA,
    STATIC_MAP_OCCLUDE_COVER,
)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _covered_ratio(target, other) -> float:
    """other 覆盖了 target 多少比例（按 target 自身面积算）。"""
    ix1, iy1 = max(target[0], other[0]), max(target[1], other[1])
    ix2, iy2 = min(target[2], other[2]), min(target[3], other[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a = _area(target)
    return inter / a if a > 0 else 0.0


def _contains_point(box, px: float, py: float) -> bool:
    return box[0] <= px <= box[2] and box[1] <= py <= box[3]


def _center(box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _long_side(box) -> float:
    return max(box[2] - box[0], box[3] - box[1])


@dataclass
class _Cluster:
    """同一个静态目标在多帧里的观测集合。"""

    cls: str
    boxes: list[list[float]] = field(default_factory=list)
    seen: set[int] = field(default_factory=set)
    visible: set[int] = field(default_factory=set)
    occluded: set[int] = field(default_factory=set)
    first_frame: int = 0
    last_seen: int = 0
    best_conf: float = 0.0
    # 从磁盘上的「场景先验」文件恢复出来的簇：不必重新投票，直接视为已确认，
    # 但保留原始 ratio/seen/visible 供报告与排查引用。
    forced: bool = False
    meta: dict = field(default_factory=dict)
    # 语义校验结论（keep / review / reject），落盘场景先验时需要带上
    verify: str = "keep"

    @property
    def box(self) -> list[float]:
        """代表框：各次观测的中位数，抗单帧框抖动。"""
        arr = np.asarray(self.boxes, dtype=float)
        return [float(v) for v in np.median(arr, axis=0)]

    @property
    def ratio(self) -> float:
        """"看得见的时候"被检出的比例。

        分母 = 可见帧 ∪ 已检出帧：
          · 被遮挡且没检出 → 不计入（那不算漏检）；
          · 被遮挡但依然检出了 → 计入（这是支持确认的证据，不能反过来惩罚它）。
        早前只用 len(visible) 作分母，会把"隔着车流仍然检出"的帧算成矛盾，
        出现 ratio = 8.5 这种大于 1 的荒谬值。
        """
        if self.forced:
            return float(self.meta.get("ratio", 1.0))
        denom = len(self.visible | self.seen)
        return len(self.seen) / denom if denom else 0.0

    @property
    def denom(self) -> int:
        if self.forced:
            return int(self.meta.get("visible", 0))
        return len(self.visible | self.seen)


class StaticObjectMap:
    """跨帧累积的静态目标隐标记。"""

    def __init__(
        self,
        classes: tuple[str, ...] = STATIC_CLASSES,
        min_ratio: float = STATIC_MAP_MIN_RATIO,
        min_frames: int = STATIC_MAP_MIN_FRAMES,
        match_iou: float = STATIC_MAP_MATCH_IOU,
        occlude_cover: float = STATIC_MAP_OCCLUDE_COVER,
        merge_iou: float = STATIC_MAP_MERGE_IOU,
        merge_dist: float = STATIC_MAP_MERGE_DIST,
        min_area: float = STATIC_MAP_MIN_AREA,
    ) -> None:
        self.classes = set(classes)
        self.min_ratio = float(min_ratio)
        self.min_frames = int(min_frames)
        self.match_iou = float(match_iou)
        self.occlude_cover = float(occlude_cover)
        self.merge_iou = float(merge_iou)
        self.merge_dist = float(merge_dist)
        self.min_area = float(min_area)
        self.clusters: list[_Cluster] = []
        self.frames_observed = 0
        self.dropped_small = 0
        self.prior_meta: dict = {}

    # ---------------------------------------------------------------- 遮挡
    def occluded(self, box, dynamics) -> bool:
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        for obj in dynamics:
            other = obj["bbox"]
            if _contains_point(other, cx, cy):
                return True
            if _covered_ratio(box, other) >= self.occlude_cover:
                return True
        return False

    def _match(self, cls: str, box):
        best, best_iou = None, 0.0
        for c in self.clusters:
            if c.cls != cls:
                continue
            value = iou(box, c.box)
            if value > best_iou:
                best, best_iou = c, value
        return best if best_iou >= self.match_iou else None

    # ---------------------------------------------------------------- 合并
    def _should_merge(self, a, b) -> bool:
        """同一个物理目标的碎片（灯头/灯杆/灯箱各自成簇）应该并回一个。"""
        if iou(a, b) >= self.merge_iou:
            return True
        (ax, ay), (bx, by) = _center(a), _center(b)
        gap = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        return gap <= self.merge_dist * max(_long_side(a), _long_side(b))

    def _merge_pass(self) -> None:
        changed = True
        while changed:
            changed = False
            n = len(self.clusters)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = self.clusters[i], self.clusters[j]
                    if a.cls != b.cls or not self._should_merge(a.box, b.box):
                        continue
                    a.boxes.extend(b.boxes)
                    a.seen |= b.seen
                    a.visible |= b.visible
                    a.occluded |= b.occluded
                    a.first_frame = min(a.first_frame, b.first_frame)
                    a.last_seen = max(a.last_seen, b.last_seen)
                    a.best_conf = max(a.best_conf, b.best_conf)
                    del self.clusters[j]
                    changed = True
                    break
                if changed:
                    break

    # ---------------------------------------------------------------- 观测
    def observe(self, frame_idx: int, detections, dynamics, span: int = 1) -> None:
        """登记一次扫描结果。

        detections = [(cls, box, conf)]；dynamics = 该次扫描所在帧的其它目标。
        span = 这次扫描代表多少帧（静态通道每 N 帧扫一次时 N>1）：
        静态目标在相邻帧之间不会变，所以一次扫描的结果可以覆盖一个窗口；
        这样窗口内的每一帧都算"看见过"，不会因为抽样而在中间帧冒出记忆框。
        """
        self.frames_observed += 1
        window = list(range(frame_idx, frame_idx + max(1, span)))
        for det in detections:
            cls, box = det[0], [float(v) for v in det[1]]
            conf = float(det[2]) if len(det) > 2 else 0.0
            if cls not in self.classes:
                continue
            if _area(box) < self.min_area:
                self.dropped_small += 1
                continue
            cluster = self._match(cls, box)
            if cluster is None:
                cluster = _Cluster(cls=cls, first_frame=frame_idx)
                self.clusters.append(cluster)
            cluster.boxes.append(box)
            cluster.seen.update(window)
            cluster.last_seen = frame_idx
            cluster.best_conf = max(cluster.best_conf, conf)

        # 可见性统计：每一帧都要对所有已出现的簇判一次（含本帧新建的）
        for cluster in self.clusters:
            if cluster.first_frame > frame_idx:
                continue
            for f in window:
                if self.occluded(cluster.box, dynamics):
                    cluster.occluded.add(f)
                else:
                    cluster.visible.add(f)

        self._merge_pass()

    # ---------------------------------------------------------------- 汇报
    def is_confirmed(self, c: _Cluster) -> bool:
        if c.forced:
            return True          # 来自场景先验文件，已在上次运行中确认过
        return (len(c.seen) >= self.min_frames
                and c.denom > 0
                and c.ratio >= self.min_ratio)

    def confirmed(self) -> list[dict]:
        out = []
        for idx, c in enumerate(self.clusters):
            if not self.is_confirmed(c):
                continue
            seen = sorted(c.seen)
            sample = sorted({seen[0], seen[len(seen) // 2], seen[-1]}) if seen else []
            meta = c.meta if c.forced else {}
            out.append({
                "idx": idx,
                "cls": c.cls,
                "box": c.box,
                "sample_frames": sample,
                "ratio": round(c.ratio, 3),
                "seen": meta.get("seen", len(c.seen)),
                "visible": meta.get("visible", len(c.visible)),
                "occluded": meta.get("occluded", len(c.occluded)),
                "first_frame": meta.get("first_frame", c.first_frame),
                "last_seen": meta.get("last_seen", c.last_seen),
                "best_conf": meta.get("best_conf", round(c.best_conf, 3)),
                "verify": meta.get("verify", c.verify),
            })
        out.sort(key=lambda d: (-d["ratio"], -d["seen"]))
        return out

    def rejected(self) -> list[dict]:
        out = []
        for c in self.clusters:
            if self.is_confirmed(c):
                continue
            out.append({
                "cls": c.cls, "box": c.box, "ratio": round(c.ratio, 3),
                "seen": len(c.seen), "visible": len(c.visible),
            })
        out.sort(key=lambda d: -d["seen"])
        return out

    # ---------------------------------------------------------------- 回补
    def missing(self, frame_idx: int, dynamics, present=None,
                present_iou: float = 0.30) -> list[dict]:
        """本帧需要按隐标记补回的已确认静态目标。

        关键：判断"这一帧要不要补"看的是**主检测器**有没有输出它，
        而不是静态通道自己有没有检出——静态通道的检出情况已经用在
        `observe()` 的投票里了。实测场景正是：
          静态通道在 300/300 帧都认得出左边那根路桩（投票通过），
          但微调主模型一帧都认不出（它的训练集里根本没有这根），
          所以必须由隐标记把它补回画面。

        present：本帧主检测器输出的目标列表；其中已有同类框与隐标记重叠
        （IoU ≥ present_iou）时不再补，用实测框更好。
        """
        present = present or []
        out = []
        for idx, c in enumerate(self.clusters):
            if not self.is_confirmed(c):
                continue
            box = c.box
            if any(o.get("cls") == c.cls and iou(box, o["bbox"]) >= present_iou
                   for o in present):
                continue
            if self.occluded(box, dynamics):
                continue
            out.append({
                "idx": idx,
                "cls": c.cls, "box": box,
                "ratio": round(c.ratio, 3),
                "seen": len(c.seen), "visible": len(c.visible),
                "best_conf": round(c.best_conf, 3),
            })
        return out

    def summary(self) -> dict:
        conf = self.confirmed()
        return {
            "frames_observed": self.frames_observed,
            "clusters": len(self.clusters),
            "confirmed": len(conf),
            "rejected": len(self.clusters) - len(conf),
            "dropped_small": self.dropped_small,
            "by_class": {c: sum(1 for d in conf if d["cls"] == c) for c in sorted(self.classes)},
            "items": conf,
        }

    # ---------------------------------------------------------------- 场景先验
    # 摄像机是长期不动的，同一个机位的"哪些位置是固定结构"只需要算一次。
    # 把结果落盘成场景先验文件，之后每段视频直接加载，省掉
    # 「开放词汇扫描 + CLIP 校验」这两步（实测占端到端耗时的绝大部分）。
    PRIOR_FORMAT = 1

    def save_prior(self, path: str | Path, camera: str | None = None,
                   frame_size: tuple[int, int] | None = None,
                   extra: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": self.PRIOR_FORMAT,
            "camera": camera,
            "frame_size": list(frame_size) if frame_size else None,
            "created": datetime.now().isoformat(timespec="seconds"),
            "classes": sorted(self.classes),
            "min_ratio": self.min_ratio,
            "min_frames": self.min_frames,
            "items": [
                {k: v for k, v in d.items() if k != "sample_frames"}
                for d in self.confirmed()
                if d.get("verify") != "reject"      # 语义校验已否掉的不写进先验
            ],
            "extra": extra or {},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_prior(cls, path: str | Path) -> "StaticObjectMap":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        fmt = data.get("format")
        if fmt != cls.PRIOR_FORMAT:
            raise ValueError(f"场景先验文件格式不兼容：{path}（format={fmt}）")
        smap = cls(classes=tuple(data.get("classes") or STATIC_CLASSES),
                   min_ratio=float(data.get("min_ratio", STATIC_MAP_MIN_RATIO)),
                   min_frames=int(data.get("min_frames", STATIC_MAP_MIN_FRAMES)))
        for item in data.get("items", []):
            c = _Cluster(cls=item["cls"],
                         boxes=[[float(v) for v in item["box"]]],
                         forced=True)
            c.meta = {k: item.get(k) for k in
                      ("ratio", "seen", "visible", "occluded", "first_frame",
                       "last_seen", "best_conf", "verify") if k in item}
            smap.clusters.append(c)
        smap.prior_meta = {k: data.get(k) for k in ("camera", "created", "frame_size", "extra")}
        return smap
