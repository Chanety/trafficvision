"""合成受控实验：验证"维度判定 + 归一化匀速"是否真的能消除透视偏差。

构造一个标准针孔相机下的地面运动场景，物体做**严格的匀速直线运动**（物理速度恒定），
然后比较两种预判方法的误差：

  A. 图像匀速模型（直接对像素坐标做匀速外推）
  B. 维度判定模型（按目标框高度归一化后外推，再换算回图像坐标）

如果 B 明显优于 A，说明"拉伸"（透视）确实是误差来源，且该方案有效。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trafficvision.motion import TrackMotion  # noqa: E402

FPS = 25.0
SECONDS = 6.0
HORIZONS = (0.1, 0.5, 1.0, 2.0)

# 相机与场景参数
F_PX = 1000.0          # 焦距（像素）
CX = 960.0             # 主点
Y_HORIZON = 320.0      # 地平线（斜视角：地平线在画面内）
CAM_HEIGHT = 8.0       # 相机高度（米）
OBJ_HEIGHT = 1.5       # 目标高度（米）
OBJ_WIDTH = 1.8        # 目标宽度（米）
MEAS_NOISE_PX = 1.5
SIZE_NOISE = 0.03


def project(x_m: float, z_m: float) -> tuple[float, float, float, float]:
    """把地面坐标 (x, z) 上的长方体投影成图像包围框。"""
    y_ground = Y_HORIZON + F_PX * CAM_HEIGHT / z_m
    h_px = F_PX * OBJ_HEIGHT / z_m
    w_px = F_PX * OBJ_WIDTH / z_m
    x_px = CX + F_PX * x_m / z_m
    return (x_px - w_px / 2, y_ground - h_px, x_px + w_px / 2, y_ground)


def center(box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def make_tracks() -> list[dict]:
    """三条典型轨迹：驶近、驶离、横向等深。"""
    return [
        dict(name="驶近", x0=1.0, z0=70.0, vx=0.0, vz=-14.0),
        dict(name="驶离", x0=-2.0, z0=35.0, vx=0.0, vz=11.0),
        dict(name="横向", x0=-8.0, z0=50.0, vx=6.0, vz=0.0),
    ]


def run() -> None:
    rng = np.random.default_rng(7)
    n_frames = int(FPS * SECONDS)
    results: dict[str, dict[str, list[float]]] = {
        "A_图像匀速": {h: [] for h in HORIZONS},
        "B_维度判定": {h: [] for h in HORIZONS},
    }

    for spec in make_tracks():
        motion_cv = TrackMotion(1, "car", FPS, model="cv")
        motion_scale = TrackMotion(2, "car", FPS, model="scale")
        truths: dict[int, tuple[float, float]] = {}

        boxes = []
        for t in range(n_frames):
            time_s = t / FPS
            x_m = spec["x0"] + spec["vx"] * time_s
            z_m = spec["z0"] + spec["vz"] * time_s
            if z_m < 6.0:      # 已驶过相机，退出该轨迹
                break
            box = list(project(x_m, z_m))
            box[0] += rng.normal(0, MEAS_NOISE_PX)
            box[2] += rng.normal(0, MEAS_NOISE_PX)
            box[1] += rng.normal(0, MEAS_NOISE_PX)
            box[3] += rng.normal(0, MEAS_NOISE_PX)
            boxes.append(box)
            truths[t] = center(box)

        n_frames_track = len(boxes)
        for t, box in enumerate(boxes):
            motion_cv.update(box)
            motion_scale.update(box)
            for horizon in HORIZONS:
                t_target = t + horizon * FPS
                i0, frac = int(np.floor(t_target)), t_target - int(np.floor(t_target))
                i1 = i0 + 1
                if i1 >= n_frames_track:
                    continue
                truth = (
                    truths[i0][0] + (truths[i1][0] - truths[i0][0]) * frac,
                    truths[i0][1] + (truths[i1][1] - truths[i0][1]) * frac,
                )
                for label, motion, scale in (("A_图像匀速", motion_cv, False),
                                             ("B_维度判定", motion_scale, True)):
                    if not motion.is_predictable():
                        continue
                    if scale:
                        pred = motion.predict_scale(t / FPS, horizon, Y_HORIZON)
                    else:
                        pred = motion.predict_box(horizon)
                    if pred is None:
                        continue
                    pc = center(pred)
                    results[label][horizon].append(
                        float(np.hypot(pc[0] - truth[0], pc[1] - truth[1]))
                    )

    print(f"合成场景：{len(make_tracks())} 条轨迹，{SECONDS:g} 秒 @ {FPS:g} fps，"
          f"地平线 y={Y_HORIZON:g}px（斜视机位）")
    print(f"{'时间梯度':<10}{'图像匀速 平均':>14}{'图像匀速 P90':>14}"
          f"{'维度判定 平均':>14}{'维度判定 P90':>14}{'改进':>10}")
    for horizon in HORIZONS:
        row = []
        for label in ("A_图像匀速", "B_维度判定"):
            errs = np.array(results[label][horizon])
            row.append((errs.mean(), np.percentile(errs, 90)) if errs.size else (float("nan"),) * 2)
        gain = (1 - row[1][0] / row[0][0]) * 100 if row[0][0] else 0.0
        print(f"{horizon:<10g}{row[0][0]:>14.2f}{row[0][1]:>14.2f}"
              f"{row[1][0]:>14.2f}{row[1][1]:>14.2f}{gain:>9.1f}%")


if __name__ == "__main__":
    run()
