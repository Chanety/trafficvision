"""因果运动估计与短时轨迹外推。

只使用当前帧及之前的数据，保证画面上的"预判"是真正的预判，
而不是利用视频后续帧的事后拟合。
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .config import MIN_SPEED_PX_S, MIN_TRACK_AGE, TRAIL_SECONDS


class ConstantVelocityKF:
    """匀速模型卡尔曼滤波。

    状态向量 [x, y, vx, vy]：位置（像素）与速度（像素/秒）。
    观测向量 [x, y]：检测框中心点。
    """

    def __init__(self, dt: float, accel_std: float = 400.0, meas_std: float = 4.0) -> None:
        self.dt = float(dt)
        self.q = float(accel_std) ** 2
        self.r = float(meas_std) ** 2
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None

    def _F(self, dt: float) -> np.ndarray:
        return np.array(
            [[1.0, 0.0, dt, 0.0],
             [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        )

    def _Q(self, dt: float) -> np.ndarray:
        dt2, dt3, dt4 = dt ** 2, dt ** 3, dt ** 4
        return self.q * np.array(
            [[dt4 / 4, 0.0, dt3 / 2, 0.0],
             [0.0, dt4 / 4, 0.0, dt3 / 2],
             [dt3 / 2, 0.0, dt2, 0.0],
             [0.0, dt3 / 2, 0.0, dt2]]
        )

    def update(self, measurement: tuple[float, float], dt: float | None = None) -> None:
        """dt 为该帧与上一帧的真实时间间隔（秒）。

        手机视频常见可变帧率，必须用真实时间戳，否则 1 秒梯度的位移会算错。
        """
        step = self.dt if dt is None or dt <= 0 else float(dt)
        z = np.array([[float(measurement[0])], [float(measurement[1])]])
        if self.x is None:
            self.x = np.array([[z[0, 0]], [z[1, 0]], [0.0], [0.0]])
            # 初始速度不确定度给大一些，让滤波器自己收敛
            self.P = np.diag([self.r, self.r, 200.0 ** 2, 200.0 ** 2])
            return

        F = self._F(step)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(step)

        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        S = H @ self.P @ H.T + np.eye(2) * self.r
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P

    @property
    def velocity(self) -> np.ndarray:
        if self.x is None:
            return np.zeros(2)
        return self.x[2:, 0]

    @property
    def position(self) -> np.ndarray | None:
        if self.x is None:
            return None
        return self.x[:2, 0]

    def predict(self, horizon: float) -> np.ndarray | None:
        if self.x is None:
            return None
        return self.x[:2, 0] + self.x[2:, 0] * float(horizon)


class ConstantAccelerationKF:
    """匀加速模型卡尔曼滤波。

    状态向量 [x, y, vx, vy, ax, ay]。相比匀速模型，它能跟上
    透视效应造成的画面内加减速（车开远时像素位移越来越小）。
    """

    def __init__(self, dt: float, jerk_std: float = 600.0, meas_std: float = 4.0) -> None:
        self.dt = float(dt)
        self.q = float(jerk_std) ** 2
        self.r = float(meas_std) ** 2
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None

    def _F(self, dt: float) -> np.ndarray:
        h2 = dt ** 2 / 2.0
        return np.array(
            [[1.0, 0.0, dt, 0.0, h2, 0.0],
             [0.0, 1.0, 0.0, dt, 0.0, h2],
             [0.0, 0.0, 1.0, 0.0, dt, 0.0],
             [0.0, 0.0, 0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        )

    def _Q(self, dt: float) -> np.ndarray:
        d2, d3, d4, d5 = dt ** 2, dt ** 3, dt ** 4, dt ** 5
        return self.q * np.array(
            [[d5 / 20, 0.0, d4 / 8, 0.0, d3 / 6, 0.0],
             [0.0, d5 / 20, 0.0, d4 / 8, 0.0, d3 / 6],
             [d4 / 8, 0.0, d3 / 3, 0.0, d2 / 2, 0.0],
             [0.0, d4 / 8, 0.0, d3 / 3, 0.0, d2 / 2],
             [d3 / 6, 0.0, d2 / 2, 0.0, dt, 0.0],
             [0.0, d3 / 6, 0.0, d2 / 2, 0.0, dt]]
        )

    def update(self, measurement: tuple[float, float], dt: float | None = None) -> None:
        step = self.dt if dt is None or dt <= 0 else float(dt)
        z = np.array([[float(measurement[0])], [float(measurement[1])]])
        if self.x is None:
            self.x = np.array([[z[0, 0]], [z[1, 0]], [0.0], [0.0], [0.0], [0.0]])
            self.P = np.diag([self.r, self.r, 200.0 ** 2, 200.0 ** 2, 400.0 ** 2, 400.0 ** 2])
            return

        F = self._F(step)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(step)

        H = np.zeros((2, 6))
        H[0, 0] = H[1, 1] = 1.0
        S = H @ self.P @ H.T + np.eye(2) * self.r
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(6) - K @ H) @ self.P

    @property
    def velocity(self) -> np.ndarray:
        if self.x is None:
            return np.zeros(2)
        return self.x[2:4, 0]

    def predict(self, horizon: float) -> np.ndarray | None:
        if self.x is None:
            return None
        p = self.x[:2, 0]
        v = self.x[2:4, 0]
        a = self.x[4:6, 0]
        return p + v * float(horizon) + 0.5 * a * float(horizon) ** 2


class TrackMotion:
    """单条轨迹的运动状态。"""

    def __init__(self, track_id: int, cls_name: str, fps: float, model: str = "cv",
                 accel_std: float = 400.0, accel_prior: float = 0.0,
                 cx_img: float = 0.0, frame_w: float = 0.0, frame_h: float = 0.0) -> None:
        self.id = track_id
        self.cls_name = cls_name
        self.fps = float(fps)
        self.model = model
        self.accel_prior = float(accel_prior)
        self.cx_img = float(cx_img)
        self.frame_w = float(frame_w)
        self.frame_h = float(frame_h)
        self.kf = (
            ConstantAccelerationKF(dt=1.0 / float(fps))
            if model == "ca"
            else ConstantVelocityKF(dt=1.0 / float(fps), accel_std=accel_std)
        )
        self.size: tuple[float, float] | None = None
        # (时间秒, 中心x, 中心y, 宽, 高, 未贴边标记)：维度判定用
        self.samples: deque[tuple[float, float, float, float, float, bool]] = deque(maxlen=90)
        self.history: deque[tuple[float, float]] = deque(
            maxlen=max(2, int(round(TRAIL_SECONDS * float(fps))))
        )
        self.frames = 0
        self.last_t: float | None = None

    def update(self, bbox: list[float], now: float | None = None) -> tuple[float, float]:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)

        if now is None:
            now = self.frames / self.fps
        step = None if self.last_t is None else max(1e-3, float(now) - self.last_t)
        self.kf.update((cx, cy), step)
        # 框贴到画面边缘时会被裁切，尺寸不再反映真实尺度，必须标记出来
        margin = 1.0
        clean = not (
            x1 <= margin or y1 <= margin
            or (self.frame_w and x2 >= self.frame_w - margin)
            or (self.frame_h and y2 >= self.frame_h - margin)
        )
        self.samples.append((float(now), cx, cy, w, h, clean))
        self.last_t = float(now)
        if self.size is None:
            self.size = (w, h)
        else:
            # 尺寸做指数平滑，避免框大小抖动
            self.size = (0.7 * self.size[0] + 0.3 * w, 0.7 * self.size[1] + 0.3 * h)
        self.history.append((cx, cy))
        self.frames += 1
        return cx, cy

    @property
    def speed(self) -> float:
        v = self.kf.velocity
        return float(np.hypot(v[0], v[1]))

    def is_predictable(self) -> bool:
        return (
            self.frames >= MIN_TRACK_AGE
            and self.speed >= MIN_SPEED_PX_S
            and self.quality_ok()
        )

    def quality_ok(self) -> bool:
        """轨迹可信度门控。

        退化检测框（例如把多个目标并成一个大框）会让框中心剧烈摆动，
        进而污染速度与航向估计。这类轨迹不能用于预判，也不应计入统计。

        判据：
          1. 框的高宽比在合理范围内（排除细长/异常框）
          2. 单帧框尺寸变化不超过 5%（退化框会剧烈伸缩）
          3. 框面积不超过画面的 30%
          4. 速度不超过物理上限（超过说明发生了 ID 跳变）
        """
        if len(self.samples) < MIN_TRACK_AGE:
            return False
        w = np.array([s[3] for s in self.samples], dtype=float)
        h = np.array([s[4] for s in self.samples], dtype=float)
        if h.mean() <= 1.0 or w.mean() <= 1.0:
            return False

        aspect = float(w.mean() / h.mean())
        if not (0.2 <= aspect <= 6.0):
            return False

        dw = np.abs(np.diff(w)) / np.maximum(w[:-1], 1.0)
        dh = np.abs(np.diff(h)) / np.maximum(h[:-1], 1.0)
        if dw.size and float(np.median(dw)) > 0.05:
            return False
        if dh.size and float(np.median(dh)) > 0.05:
            return False

        if self.frame_w and self.frame_h:
            max_ratio = 0.30
            if float((w.mean() * h.mean()) / (self.frame_w * self.frame_h)) > max_ratio:
                return False

        if self.speed > 2500.0:
            return False
        return True

    def predict_box(self, horizon: float) -> tuple[float, float, float, float] | None:
        """返回预判时刻的包围框（沿用当前框大小）。"""
        if not self.is_predictable() or self.size is None:
            return None
        center = self.kf.predict(horizon)
        if center is None:
            return None
        w, h = self.size
        return (
            float(center[0] - w / 2.0),
            float(center[1] - h / 2.0),
            float(center[0] + w / 2.0),
            float(center[1] + h / 2.0),
        )

    # ------------------------------------------------------------------
    # 维度判定 + 归一化匀速模型
    # ------------------------------------------------------------------
    def scale_variation(self) -> float:
        """轨迹内目标框高度的相对变化幅度，用于判断深度变化是否显著。"""
        if len(self.samples) < MIN_TRACK_AGE:
            return 0.0
        h = np.array([s[4] for s in self.samples if s[5]], dtype=float)
        if h.size < MIN_TRACK_AGE:
            return 0.0
        mean = float(h.mean())
        return 0.0 if mean <= 1e-6 else float((h.max() - h.min()) / mean)

    def horizon_estimate(self) -> float | None:
        """单条轨迹给出的地平线估计（y 对框高线性回归的截距）。

        单独一条轨迹的估计方差很大，只作为全局估计的一个样本使用。
        """
        clean = [s for s in self.samples if s[5]]
        if len(clean) < MIN_TRACK_AGE or self.scale_variation() < 0.10:
            return None
        h = np.array([s[4] for s in clean], dtype=float)
        y = np.array([s[2] for s in clean], dtype=float)
        slope, intercept = np.polyfit(h, y, 1)
        if slope <= 0.2:      # 斜率应为正（相机高度/目标高度）
            return None
        return float(intercept)

    def predict_scale(self, now: float, horizon: float, y_horizon: float,
                      accel_prior: float = 0.0,
                      max_samples: int = 30) -> tuple[float, float, float, float] | None:
        """在地平线已知的前提下，用尺度归一化坐标做匀速外推。

        关键点：地平线是相机属性，跨轨迹联合估计；单条轨迹只拟合自己的一维参数，
        避免小样本下二维回归被噪声放大。
        """
        hist = [s for s in self.samples if s[0] <= now + 1e-6 and s[5]][-max_samples:]
        if len(hist) < MIN_TRACK_AGE:
            return None

        t = np.array([s[0] for s in hist], dtype=float)
        cx = np.array([s[1] for s in hist], dtype=float)
        cy = np.array([s[2] for s in hist], dtype=float)
        bw = np.array([s[3] for s in hist], dtype=float)
        bh = np.array([s[4] for s in hist], dtype=float)

        if (bh.max() - bh.min()) / max(1e-6, bh.mean()) < 0.10:
            return None                      # 深度变化不显著，交给匀速模型兜底
        if np.any(cy - y_horizon <= 1e-6):
            return None

        # 单参数拟合：c = (y - 地平线)/框高，理论上为常数
        c = float(np.median((cy - y_horizon) / bh))
        if c <= 0.1:
            return None

        t_pred = float(now + horizon)

        # 深度方向：1/h 对时间线性
        z_coef = np.polyfit(t, 1.0 / bh, 1)
        depth_pred = float(np.polyval(z_coef, t_pred))
        if depth_pred <= 1e-6:
            return None
        h_pred = 1.0 / depth_pred

        # 把观测到的速度变化按经验权重部分采纳（收缩估计，抑制噪声）
        if accel_prior > 0.0 and len(t) >= 8:
            z_quad = float(np.polyval(np.polyfit(t, 1.0 / bh, 2), t_pred))
            weight = max(0.0, min(1.0, accel_prior))
            depth_pred = depth_pred + weight * (z_quad - depth_pred)
            if depth_pred <= 1e-6:
                return None
            h_pred = 1.0 / depth_pred

        if not (0.25 * bh[-1] <= h_pred <= 4.0 * bh[-1]):
            return None

        # 横向：归一化横坐标对时间线性
        xhat = (cx - self.cx_img) / bh
        x_pred = self.cx_img + float(np.polyval(np.polyfit(t, xhat, 1), t_pred)) * h_pred
        y_pred = y_horizon + c * h_pred

        aspect = float(bw[-1] / max(1e-6, bh[-1]))
        w_pred = max(2.0, aspect * h_pred)
        return (
            float(x_pred - w_pred / 2.0),
            float(y_pred - h_pred / 2.0),
            float(x_pred + w_pred / 2.0),
            float(y_pred + h_pred / 2.0),
        )
