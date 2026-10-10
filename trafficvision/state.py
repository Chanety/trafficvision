"""目标运动状态判定：constant（静止）/ variable（运动）。

规则（按客户要求）：
  · 某目标在**连续 6 帧**内位置未发生变化 → 状态为 constant
  · constant 状态的目标**暂时不显示**（不画框、不做预判）
  · 一旦重新移动，恢复显示并重新框选，状态回到 variable
  · sign（警示标志）、bollard（路桩）两类**不受此状态约束**，始终显示

状态是独立属性，与类别无关。

2026-10-10 补充（用户抽检反馈"行人近点目标未识别"）：
  只看"窗口内波动范围"有个漏洞——**检测框抖动也会把范围撑大**，于是阈值只能放宽，
  放宽后走得慢的目标又被当成静止。实测 112254 第 24 帧 18 条行人里 14 条被藏掉，
  其中几条速度是 25~40 px/s。
  这里加第二条判据：**净位移**（窗口前 2 帧均值 → 后 2 帧均值）。
  抖动在净位移上互相抵消，真移动不会。实测净位移 ≥5px 时，
  行人能救回 13.9%，而停放车辆只有 0.2% 被误判为运动。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import STILL_NET_PX

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
    net: float = 0.0          # 窗口内净位移（前 2 帧均值 → 后 2 帧均值）

    def update(self, frame_index: int, x: float, y: float,
               window: int, threshold: float, bias: float,
               size_h: float = 0.0, net_px: float = STILL_NET_PX) -> tuple[str, bool]:
        """返回（状态, 是否落在 bias 容差带内）。

        迟滞判定（关键设计）：
          · 当前为 variable：波动 ≤ 基础阈值 才允许进入 constant
          · 当前为 constant：波动 ≤ 基础阈值 + bias 就继续认定为 constant
        这样小抖动（检测噪声、透视造成的框形变化）不会让静止目标在
        constant/variable 之间反复跳变；只有真正移动（超过 基础阈值+bias）
        才会退出 constant。中间那段即为 bias 标记区。

        净位移线索优先于上面的迟滞判断：只要窗口内净位移够大，
        无论波动范围多小，都判定为在移动（慢速目标不会被藏掉）。
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
        # 净位移：前 2 帧与后 2 帧的均值之差。抖动会互相抵消，真移动不会。
        pts = list(self.samples)
        ax = (pts[0][1] + pts[1][1]) / 2.0
        ay = (pts[0][2] + pts[1][2]) / 2.0
        bx = (pts[-1][1] + pts[-2][1]) / 2.0
        by = (pts[-1][2] + pts[-2][2]) / 2.0
        self.net = float(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)

        # 容差随目标尺寸缩放：框越大（离相机越近），同样的世界位移在画面上的
        # 像素抖动也越大，固定阈值会造成"近处误判为运动"。
        scale = 1.0
        if size_h > 0:
            scale = min(3.0, max(0.5, size_h / REFERENCE_BOX_HEIGHT))
        enter = threshold * scale
        keep = (threshold + bias) * scale

        # 净位移够大 → 明确在移动（慢速目标的主要救济通道）
        if self.net >= net_px:
            self.state, in_bias = "variable", False
            return self.state, in_bias

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
                 bias_px: float = STILL_BIAS_PX, exempt: set[str] | None = None,
                 net_px: float = STILL_NET_PX) -> None:
        self.window = int(window)
        self.threshold = float(threshold_px)
        self.bias = float(bias_px)
        self.net_px = float(net_px)
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
            frame_index, x, y, self.window, self.threshold, self.bias, size_h,
            self.net_px,
        )
        return state, state == "constant", in_bias, track.spread
