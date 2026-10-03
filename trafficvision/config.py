"""全局配置：类别、颜色、预判梯度与有效性阈值。"""

from __future__ import annotations

# 支持的预判时间梯度（秒）
HORIZONS: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0)

# 默认目标类型
DEFAULT_CLASSES: tuple[str, ...] = (
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "person",
)

# 每个类别的绘制颜色（BGR）
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (80, 220, 80),
    "truck": (0, 150, 255),
    "bus": (255, 130, 40),
    "motorcycle": (220, 60, 220),
    "bicycle": (220, 220, 60),
    "person": (60, 220, 255),
}
DEFAULT_COLOR: tuple[int, int, int] = (180, 180, 180)

# 预判有效性门槛
MIN_TRACK_AGE = 5          # 连续观测帧数不足则不给预判
MIN_SPEED_PX_S = 8.0       # 近乎静止的目标不给预判

# 片段时长约束（秒）
MIN_CLIP_SECONDS = 2.0
MAX_CLIP_SECONDS = 30.0

# 尾迹长度（秒）
TRAIL_SECONDS = 1.0

# 运动模型：cv = 匀速；ca = 匀加速（对透视引起的画面内加减速更稳）
MOTION_MODEL = "cv"

# 跟踪器：在 videoset_01 全集（14 段，排除手持段）实测后确定
# 结论：tracktrack 虽把轨迹条数降低 79%，但"每帧可预判目标数"下降 44.6%、
# 误差反而变差（≈58.9 → ≈81.5 px），因此不用它做默认。
# fasttrack 在 3 段横评中与 bytetrack 召回相当、误差略优，列为待全集验证的候选。
TRACKER = "bytetrack.yaml"
