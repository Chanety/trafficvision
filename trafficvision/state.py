"""目标运动状态判定：constant（静止）/ variable（运动）。

规则（按客户要求）：
  · 某目标在**连续 6 帧**内位置未发生变化 → 状态为 constant
  · constant 状态的目标**暂时不显示**（不画框、不做预判）
  · 一旦重新移动，恢复显示并重新框选，状态回到 variable
  · sign（警示标志）、bollard（路桩）两类**不受此状态约束**，始终显示

状态是独立属性，与类别无关。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

STILL_WINDOW = 6          # 连续静止多少帧判定为 constant
STILL_THRESHOLD_PX = 3.0  # 位置波动不超过多少像素算"未移动"
EXEMPT_CLASSES = {"sign", "bollard"}


@dataclass
class TrackState:
    samples: deque[tuple[int, float, float]] = field(default_factory=lambda: deque(maxlen=STILL_WINDOW))
    state: str = "variable"

    def update(self, frame_index: int, x: float, y: float,
               window: int, threshold: float) -> str:
        # 只统计"连续帧"：中间断帧就重新计数
        if self.samples and frame_index != self.samples[-1][0] + 1:
            self.samples.clear()
        self.samples.append((frame_index, x, y))
        if len(self.samples) < window:
            self.state = "variable"
            return self.state
        xs = [s[1] for s in self.samples]
        ys = [s[2] for s in self.samples]
        spread = max(max(xs) - min(xs), max(ys) - min(ys))
        self.state = "constant" if spread <= threshold else "variable"
        return self.state


class StateTracker:
    """管理所有轨迹的运动状态。"""

    def __init__(self, window: int = STILL_WINDOW, threshold_px: float = STILL_THRESHOLD_PX,
                 exempt: set[str] | None = None) -> None:
        self.window = int(window)
        self.threshold = float(threshold_px)
        self.exempt = set(EXEMPT_CLASSES if exempt is None else exempt)
        self.tracks: dict[int, TrackState] = {}

    def update(self, track_id: int, cls_name: str, frame_index: int,
               x: float, y: float) -> tuple[str, bool]:
        """返回（状态, 是否隐藏）。sign / bollard 类始终为 variable 且不隐藏。"""
        if cls_name in self.exempt:
            return "variable", False
        state = self.tracks.setdefault(track_id, TrackState()).update(
            frame_index, x, y, self.window, self.threshold
        )
        return state, state == "constant"
