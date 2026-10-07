"""最小卡尔曼滤波器：只为驱动 filterpy 的 IMMEstimator 而写（本项目自研，非 filterpy 代码）。

IMMEstimator 只要求每个子滤波器提供这套接口：
    x, P, F, Q, R, H, predict(), update(z), likelihood, x_prior/x_post, P_prior/P_post
因此这里实现一个标准线性卡尔曼，避免为了一个滤波器引入 filterpy 的整棵依赖树。

状态 [x, y, vx, vy]，观测 [x, y]，全部按像素与秒。
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv


class MinimalKalmanFilter:
    def __init__(self, dim_x: int = 4, dim_z: int = 2) -> None:
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x)
        self.F = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)
        self.H = np.zeros((dim_z, dim_x))
        self.likelihood = 1.0
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        self._I = np.eye(dim_x)

    def predict(self, u=None) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z) -> None:
        z = np.atleast_2d(z)
        if z.shape[0] != self.dim_z:
            z = z.T
        y = z - self.H @ self.x                     # 新息
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ inv(S)
        self.x = self.x + K @ y
        self.P = (self._I - K @ self.H) @ self.P
        # 似然：IMM 用它给各模型加权
        denom = np.sqrt(((2 * np.pi) ** self.dim_z) * np.linalg.det(S))
        mahalanobis = float((y.T @ inv(S) @ y).item())
        self.likelihood = float(np.exp(-0.5 * mahalanobis) / denom)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    @property
    def position(self) -> np.ndarray:
        return self.x[:2, 0]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:, 0]
