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
STILL_THRESHOLD_PX = 3.0  # 进入 constant 的位置波动阈值（基础阈值）
STILL_BIAS_PX = 2.0       # bias 允许值：迟滞带宽，吸收检测噪声与视角误差
# 静态结构类不受"连续 6 帧不动就不显示"的约束：它们本来就是不动的，
# 隐藏它们等于把画面里的固定参考物全删掉。2026-10-06 增补路灯与信号灯。
EXEMPT_CLASSES = {"sign", "bollard", "roadlamp", "trafficlight"}
REFERENCE_BOX_HEIGHT = 60.0   # 参考框高（像素），用于按目标大小缩放容差


@dataclass
class TrackState:
    samples: deque[tuple[int, float, float]] = field(default_factory=lambda: deque(maxlen=STILL_WINDOW))
    state: str = "variable"
    spread: float = 0.0

    def update(self, frame_index: int, x: float, y: float,
               window: int, threshold: float, bias: float,
               size_h: float = 0.0) -> tuple[str, bool]:
        """返回（状态, 是否落在 bias 容差带内）。

        迟滞判定（关键设计）：
          · 当前为 variable：波动 ≤ 基础阈值 才允许进入 constant
          · 当前为 constant：波动 ≤ 基础阈值 + bias 就继续认定为 constant
        这样小抖动（检测噪声、透视造成的框形变化）不会让静止目标在
        constant/variable 之间反复跳变；只有真正移动（超过 基础阈值+bias）
        才会退出 constant。中间那段即为 bias 标记区。
        """
        # 只统计"连续帧"：中间断帧就重新计数
        if self.samples and frame_index != self.samples[-1][0] + 1:
            self.samples.clear()
        self.samples.append((frame_index, x, y))
        if len(self.samples) < window:
            self.state = "variable"
            return self.state, False
        xs = [s[1] for s in self.samples]
        ys = [s[2] for s in self.samples]
        self.spread = max(max(xs) - min(xs), max(ys) - min(ys))

        # 容差随目标尺寸缩放：框越大（离相机越近），同样的世界位移在画面上的
        # 像素抖动也越大，固定阈值会造成"近处误判为运动"。
        scale = 1.0
        if size_h > 0:
            scale = min(3.0, max(0.5, size_h / REFERENCE_BOX_HEIGHT))
        enter = threshold * scale
        keep = (threshold + bias) * scale

        # 三区间判定：
        #   波动 ≤ 基础阈值          → 明确静止（constant，无 bias 标记）
        #   基础阈值 < 波动 ≤ +bias  → 判为静止但打 bias 标记（容差带）
        #   波动 > 基础阈值 + bias   → 判为运动（variable）
        if self.spread <= enter:
            self.state, in_bias = "constant", False
        elif self.spread <= keep:
            self.state, in_bias = "constant", True
        else:
            self.state, in_bias = "variable", False
        return self.state, in_bias


class StateTracker:
    """管理所有轨迹的运动状态。"""

    def __init__(self, window: int = STILL_WINDOW, threshold_px: float = STILL_THRESHOLD_PX,
                 bias_px: float = STILL_BIAS_PX, exempt: set[str] | None = None) -> None:
        self.window = int(window)
        self.threshold = float(threshold_px)
        self.bias = float(bias_px)
        self.exempt = set(EXEMPT_CLASSES if exempt is None else exempt)
        self.tracks: dict[int, TrackState] = {}

    def update(self, track_id: int, cls_name: str, frame_index: int,
               x: float, y: float, size_h: float = 0.0) -> tuple[str, bool, bool, float]:
        """返回（状态, 是否隐藏, 是否在 bias 带内, 当前波动像素）。

        sign / bollard 类始终为 variable、不隐藏，也不带 bias 标记。
        """
        if cls_name in self.exempt:
            return "variable", False, False, 0.0
        track = self.tracks.setdefault(track_id, TrackState())
        state, in_bias = track.update(
            frame_index, x, y, self.window, self.threshold, self.bias, size_h
        )
        return state, state == "constant", in_bias, track.spread
