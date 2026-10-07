"""标注规则：先识别载具，再识别不属于载具的行人。

历史规则（v1）：骑车人、驾驶位里的人其身份已由载具表达，因此把这类 person
框直接删除。实测本数据集 2025 个行人框里有 561 个与载具重叠，属于该情况。

2026-10-06 按用户要求升级（v2）：不再是"删掉骑车人"，而是把它**单独标成一类**
`rider`（非机动车驾驶员），与纯步行的 `person`（行人）区分开。
于是同一帧里：两三轮车 = 载具框，骑它的人 = rider 框，旁边走路的人 = person 框。

两种模式都保留：
  · `suppress_riders()` —— 旧行为，删除骑乘中的 person（仅用于复现旧数据集）
  · `classify_riders()` —— 新行为，把骑乘中的 person 改判为 rider
"""

from __future__ import annotations

CLASS_NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person",
               "sign", "bollard", "tricycle", "roadlamp", "trafficlight", "rider"]
# 三轮车（8）同样是载具：三轮车驾驶人身上若再叠一个 person 框，
# 等于给模型"人和车是两回事"的错误监督，因此一并纳入清理范围。
VEHICLE_IDS = {0, 1, 2, 3, 4, 8}
PERSON_ID = 5
RIDER_ID = 11
# 只有"两三轮"才算骑乘：骑在汽车上（天窗探头）不算
RIDER_VEHICLE_IDS = {3, 4, 8}


def _h_overlap(a, b) -> float:
    """两个框的水平重叠量 / 较窄者的宽度。"""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    narrow = min(a[2] - a[0], b[2] - b[0])
    return ix / narrow if narrow > 1e-9 else 0.0


def classify_riders(boxes: list[list[float]], min_h_overlap: float = 0.40,
                    foot_margin: float = 0.20) -> tuple[list[list[float]], int]:
    """把"骑在两三轮车上的人"由 person 改判为 rider。

    判据（全部用框的几何关系，不需要额外模型）：
      1. 水平重叠 ≥ min_h_overlap × 较窄者宽度（人横向落在车上）；
      2. 人的脚底（框下边）落在载具框的垂直范围内（允许上缘外扩 foot_margin）；
      3. 人的框顶明显高于载具框底（人确实"坐在上面"，而不是站在车后的行人）。

    返回（新框列表, 改判数量）。
    """
    vehicles = [b for b in boxes if int(b[0]) in RIDER_VEHICLE_IDS]
    if not vehicles:
        return list(boxes), 0

    out: list[list[float]] = []
    changed = 0
    for box in boxes:
        if int(box[0]) != PERSON_ID:
            out.append(box)
            continue
        px1, py1, px2, py2 = _xyxy(box)
        hit = False
        for v in vehicles:
            vx1, vy1, vx2, vy2 = _xyxy(v)
            vh = vy2 - vy1
            if _h_overlap((px1, py1, px2, py2), (vx1, vy1, vx2, vy2)) < min_h_overlap:
                continue
            if not (vy1 - foot_margin * vh <= py2 <= vy2 + 0.05 * vh):
                continue
            if py1 >= vy2:
                continue
            hit = True
            break
        if hit:
            out.append([float(RIDER_ID)] + [float(v) for v in box[1:]])
            changed += 1
        else:
            out.append(box)
    return out, changed


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
