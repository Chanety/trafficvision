"""标注规则：先识别载具，再识别不属于载具的行人。

骑车人、驾驶位里的人，其身份已由载具表达。若再单独标一个 person 框，
会与载具框重叠，给模型自相矛盾的监督信号。

实测：本数据集原有 2025 个行人框中有 561 个与载具重叠（中心落在载具框内、
中位覆盖率 0.79），属于需要清除的情况。
"""

from __future__ import annotations

CLASS_NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
VEHICLE_IDS = {0, 1, 2, 3, 4}
PERSON_ID = 5


def _xyxy(box):
    _, cx, cy, w, h = box
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _coverage(person, vehicle) -> tuple[float, bool]:
    px1, py1, px2, py2 = _xyxy(person)
    vx1, vy1, vx2, vy2 = _xyxy(vehicle)
    ix1, iy1 = max(px1, vx1), max(py1, vy1)
    ix2, iy2 = min(px2, vx2), min(py2, vy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0, False
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = max(1e-9, (px2 - px1) * (py2 - py1))
    inside = vx1 <= person[1] <= vx2 and vy1 <= person[2] <= vy2
    return inter / area, inside


def suppress_riders(boxes: list[list[float]], ratio: float = 0.5,
                    min_scale: float = 0.8) -> list[list[float]]:
    """删除与载具框重叠的行人框（骑车人、驾驶位里的人）。

    判定需同时满足：行人框中心在载具框内、被覆盖面积比例 ≥ ratio、
    载具框面积 ≥ 行人框面积 × min_scale（避免小车吞掉大行人）。
    """
    vehicles = [b for b in boxes if int(b[0]) in VEHICLE_IDS]
    if not vehicles:
        return list(boxes)

    kept: list[list[float]] = []
    for box in boxes:
        if int(box[0]) != PERSON_ID:
            kept.append(box)
            continue
        drop = False
        for vehicle in vehicles:
            cover, inside = _coverage(box, vehicle)
            if (inside and cover >= ratio
                    and (vehicle[3] * vehicle[4]) >= (box[3] * box[4]) * min_scale):
                drop = True
                break
        if not drop:
            kept.append(box)
    return kept
