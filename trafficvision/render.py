"""画面绘制：检测框、类型标签、预判框、箭头与尾迹。"""

from __future__ import annotations

import cv2
import numpy as np

from .config import CLASS_COLORS, DEFAULT_COLOR, display_name


def class_color(cls_name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(cls_name, DEFAULT_COLOR)


def _label(img, text: str, org: tuple[int, int], color, scale: float = 0.5) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv2.LINE_AA)


def solid_box(img, box, color, thickness: int = 2) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def dashed_box(img, box, color, thickness: int = 2, dash: int = 9, gap: int = 6) -> None:
    """虚线框：沿四条边按 dash/gap 交替画线段。"""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    step = dash + gap

    for x in range(x1, x2, step):
        x_end = min(x + dash, x2)
        cv2.line(img, (x, y1), (x_end, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y2), (x_end, y2), color, thickness, cv2.LINE_AA)
    for y in range(y1, y2, step):
        y_end = min(y + dash, y2)
        cv2.line(img, (x1, y), (x1, y_end), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y), (x2, y_end), color, thickness, cv2.LINE_AA)


def arrow(img, p0, p1, color, thickness: int = 2) -> None:
    a = (int(round(p0[0])), int(round(p0[1])))
    b = (int(round(p1[0])), int(round(p1[1])))
    if a == b:
        return
    cv2.arrowedLine(img, a, b, color, thickness, cv2.LINE_AA, tipLength=0.25)


def trail(img, points, color) -> None:
    if len(points) < 2:
        return
    pts = np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32)
    cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)


def hud(img, lines: list[str]) -> None:
    scale, pad = 0.55, 8
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0] for t in lines]
    width = max(w for w, _ in sizes) + pad * 2
    height = sum(h for _, h in sizes) + pad * 2 + 6 * (len(lines) - 1)
    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    y = 8 + pad
    for text, (_, h) in zip(lines, sizes):
        y += h
        cv2.putText(img, text, (8 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 1, cv2.LINE_AA)
        y += 6


def draw_object(img, obj: dict, horizon: float) -> None:
    color = class_color(obj["cls"])

    # 静态目标记忆（隐标记回补）：这一帧检测器并没有输出它，是整段视频的
    # 跨帧投票把它补回来的。用"密虚线 + ·记忆"标注，跟实测框区分开，
    # 也不做预判——它本来就是静止的。
    if obj.get("memory"):
        dashed_box(img, obj["bbox"], color, 2, dash=4, gap=4)
        _label(img, f'{display_name(obj["cls"])}·记忆',
               (int(obj["bbox"][0]), int(obj["bbox"][1])), color, 0.45)
        return

    trail(img, obj["trail"], color)
    if obj.get("coasting"):
        # 遮挡续推：目标当前没有被检测到，框是推算出来的，用虚线区分，避免误认为是实测
        dashed_box(img, obj["bbox"], color, 2)
        _label(img, f'{display_name(obj["cls"])} #{obj["id"]} 遮挡续推',
               (int(obj["bbox"][0]), int(obj["bbox"][1])), color, 0.45)
    else:
        solid_box(img, obj["bbox"], color, 2)
        _label(img, f'{display_name(obj["cls"])} #{obj["id"]}',
               (int(obj["bbox"][0]), int(obj["bbox"][1])), color)

    pred = obj.get("pred_box")
    if pred is None:
        return
    # 机动中的目标：匀速外推本来就不准，用灰色细虚线 + 文字标注，提醒别太当真
    low = obj.get("confidence") == "low"
    if low:
        dashed_box(img, pred, (140, 140, 140), 1)
    else:
        dashed_box(img, pred, color, 2)
    center_now = obj["centroid"]
    center_pred = ((pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0)
    arrow(img, center_now, center_pred, (140, 140, 140) if low else color, 1 if low else 2)
    _label(
        img,
        f'预判 {horizon:g}s' + ("（机动·低置信）" if low else ""),
        (int(pred[0]), int(max(12, pred[1] - 4))),
        (140, 140, 140) if low else color,
        0.45,
    )
