"""中文标注复核工具（专为本项目定制）。

相比 labelImg 的优势：
  · 全中文界面与中文按键提示
  · 滚轮直接缩放（以光标为中心），按住中键或空格拖动平移
  · 预加载已有自动标注，鼠标拖动即可改框
  · 类别用数字键一键切换，删除一键完成

操作方式：
  滚轮            缩放（以光标位置为中心）
  中键拖动 / 空格+左键拖动    平移画面
  左键在空白处拖动    画新框
  左键点框 / 拖动框    选中 / 移动框
  1~6             切换选中框（或新框）的类别
  R               应用"先载具后行人"规则（清掉与载具重叠的行人框）
  Delete / Backspace  删除选中框
  D / 右方向键      下一张（自动保存）
  A / 左方向键      上一张（自动保存）
  Ctrl+S          立即保存
  F               适应窗口
  Ctrl+Z          撤销上一步
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def resolve_dataset() -> Path:
    """数据集目录：命令行第一个参数优先，默认仍是 videoset_01_label。

    用法：python tools/annotate_zh.py datasets/videoset_02_label
    """
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        given = Path(sys.argv[1])
        return given if given.is_absolute() else (ROOT / given)
    return ROOT / "datasets" / "videoset_01_label"


DATASET = resolve_dataset()
IMAGES = DATASET / "images"
LABELS = DATASET / "labels"
REVIEW_ORDER = DATASET / "review_order.txt"   # 可选：人工复核的优先顺序

sys.path.insert(0, str(ROOT))
from trafficvision.labelrules import suppress_riders  # noqa: E402

from PyQt5.QtCore import QPoint, Qt  # noqa: E402
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QFont  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# 类别（顺序必须与 classes.txt 一致，决定了训练时的类别编号）
CLASS_NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person",
               "sign", "bollard", "tricycle"]
CLASS_CN = ["汽车", "卡车", "公交", "摩托", "自行车", "行人", "标志牌", "路桩", "三轮车"]
CLASS_COLORS = [
    QColor(80, 220, 80),
    QColor(255, 150, 0),
    QColor(40, 130, 255),
    QColor(220, 60, 220),
    QColor(220, 220, 60),
    QColor(60, 220, 255),
    QColor(255, 255, 255),
    QColor(150, 150, 255),
    QColor(0, 215, 255),
]


class Canvas(QWidget):
    """图像画布：负责缩放、平移、画框、选中与移动。"""

    def __init__(self, window: "MainWindow") -> None:
        super().__init__()
        self.window = window
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(640, 480)
        self.pixmap: QPixmap | None = None
        self.scale = 1.0
        self.offset = QPoint(0, 0)
        self.boxes: list[list[float]] = []       # [cls, cx, cy, w, h] 归一化
        self.selected = -1
        self.mode: str | None = None
        self.start_img = (0.0, 0.0)
        self.start_screen = QPoint(0, 0)
        self.drag_origin_box: list[float] | None = None
        self.pan_origin = QPoint(0, 0)
        self.pan_offset_start = QPoint(0, 0)
        self.space_held = False
        self.history: list[list[list[float]]] = []
        self.user_zoomed = False          # 用户手动缩放后，窗口改变大小时不再自动适应
        self.active_handle: int | None = None   # 正在拖动的手柄编号（0~7）

    def push_history(self) -> None:
        """在修改前记录快照，供 Ctrl+Z 撤销。"""
        self.history.append([list(b) for b in self.boxes])
        if len(self.history) > 50:
            self.history.pop(0)

    # ---------- 坐标换算 ----------
    def to_image(self, point: QPoint) -> tuple[float, float]:
        return ((point.x() - self.offset.x()) / self.scale,
                (point.y() - self.offset.y()) / self.scale)

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        return (x * self.scale + self.offset.x(), y * self.scale + self.offset.y())

    def image_size(self) -> tuple[int, int]:
        if self.pixmap is None:
            return (0, 0)
        return (self.pixmap.width(), self.pixmap.height())

    # ---------- 显示 ----------
    def set_image(self, path: Path | None, boxes: list[list[float]]) -> None:
        self.pixmap = QPixmap(str(path)) if path else None
        self.boxes = boxes
        self.selected = -1
        self.user_zoomed = False
        self.fit_window()
        self.update()
        self.window.refresh_status()

    def resizeEvent(self, event) -> None:  # noqa: N802
        """窗口尺寸变化时自动重新适应（除非用户已手动缩放）。"""
        if not self.user_zoomed:
            self.fit_window()
        super().resizeEvent(event)

    def fit_window(self) -> None:
        w, h = self.image_size()
        if not w or not h:
            return
        self.scale = min(self.width() / w, self.height() / h) * 0.98
        self.offset = QPoint(int((self.width() - w * self.scale) / 2),
                             int((self.height() - h * self.scale) / 2))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(45, 45, 45))
        if self.pixmap is None:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignCenter, "没有图片")
            return

        w, h = self.image_size()
        target_w, target_h = int(w * self.scale), int(h * self.scale)
        painter.drawPixmap(self.offset.x(), self.offset.y(), target_w, target_h, self.pixmap)

        font = QFont()
        font.setPointSize(max(9, int(12 * min(1.6, self.scale + 0.4))))
        painter.setFont(font)
        for index, (cls, cx, cy, bw, bh) in enumerate(self.boxes):
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            sx1, sy1 = self.to_screen(x1, y1)
            sx2, sy2 = self.to_screen(x2, y2)
            color = CLASS_COLORS[int(cls) % len(CLASS_COLORS)]
            pen = QPen(color, 3 if index == self.selected else 2)
            painter.setPen(pen)
            painter.drawRect(int(sx1), int(sy1), int(sx2 - sx1), int(sy2 - sy1))
            label = CLASS_CN[int(cls) % len(CLASS_CN)]
            painter.fillRect(int(sx1), int(sy1) - 20, 18 + 16 * len(label), 20, color)
            painter.setPen(QColor(20, 20, 20))
            painter.drawText(int(sx1) + 4, int(sy1) - 5, label)

            # 选中的框画 8 个缩放手柄
            if index == self.selected:
                painter.setPen(QPen(QColor(20, 20, 20), 1))
                for hx, hy in self.handle_points(self.boxes[index]):
                    px, py = self.to_screen(hx, hy)
                    painter.setBrush(QColor(255, 255, 255))
                    painter.drawRect(int(px) - 4, int(py) - 4, 9, 9)
                painter.setBrush(Qt.NoBrush)   # 复位，避免影响后续标签底色

        # 正在拖出的新框
        if self.mode == "draw":
            painter.setPen(QPen(QColor(255, 255, 255), 2, Qt.DashLine))
            x0, y0 = self.start_screen.x(), self.start_screen.y()
            cur = self.mapFromGlobal(self.cursor().pos())
            painter.drawRect(int(min(x0, cur.x())), int(min(y0, cur.y())),
                             abs(cur.x() - x0), abs(cur.y() - y0))

    # ---------- 交互 ----------
    def wheelEvent(self, event) -> None:  # noqa: N802
        if self.pixmap is None:
            return
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        cursor = event.pos()
        img_x, img_y = self.to_image(cursor)
        self.user_zoomed = True
        self.scale *= 1.15 ** steps
        self.scale = max(0.05, min(20.0, self.scale))
        self.offset = QPoint(int(cursor.x() - img_x * self.scale),
                             int(cursor.y() - img_y * self.scale))
        self.window.refresh_status()
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.pixmap is None:
            return
        self.setFocus()
        pos = event.pos()
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self.space_held):
            self.mode = "pan"
            self.pan_origin = pos
            self.pan_offset_start = QPoint(self.offset)
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton:
            return

        # 先判断是否抓住了缩放手柄（只对已选中的框生效）
        handle = self.handle_at(pos)
        if handle is not None:
            self.push_history()
            self.mode = "resize"
            self.active_handle = handle
            return

        hit = self.hit_test(pos)
        if hit >= 0:
            self.selected = hit
            self.mode = "move"
            self.drag_origin_box = list(self.boxes[hit])
            self.start_img = self.to_image(pos)
            self.window.refresh_status()
            self.update()
            return
        self.selected = -1
        self.mode = "draw"
        self.start_screen = pos
        self.start_img = self.to_image(pos)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.pos()
        if self.mode == "pan":
            delta = pos - self.pan_origin
            self.offset = self.pan_offset_start + delta
            self.update()
            return
        if self.mode == "draw":
            self.update()
            return
        if self.mode == "resize" and self.active_handle is not None:
            img_x, img_y = self.to_image(pos)
            self.apply_resize(self.active_handle, img_x, img_y)
            self.update()
            return
        if self.mode == "move" and self.selected >= 0 and self.drag_origin_box:
            img_x, img_y = self.to_image(pos)
            dx = img_x - self.start_img[0]
            dy = img_y - self.start_img[1]
            w, h = self.image_size()
            cls, cx, cy, bw, bh = self.drag_origin_box
            self.boxes[self.selected] = [cls,
                                         min(1.0, max(0.0, cx + dx / w)),
                                         min(1.0, max(0.0, cy + dy / h)),
                                         bw, bh]
            self.update()
            return
        # 光标反馈：手柄上显示缩放光标，框内显示移动光标，空白处显示抓手
        handle = self.handle_at(pos)
        if handle in (0, 4):
            self.setCursor(Qt.SizeFDiagCursor)
        elif handle in (2, 6):
            self.setCursor(Qt.SizeBDiagCursor)
        elif handle in (1, 5):
            self.setCursor(Qt.SizeVerCursor)
        elif handle in (3, 7):
            self.setCursor(Qt.SizeHorCursor)
        elif self.hit_test(pos) >= 0:
            self.setCursor(Qt.SizeAllCursor)
        else:
            self.setCursor(Qt.OpenHandCursor)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self.mode == "pan":
            self.mode = None
            self.setCursor(Qt.ArrowCursor)
            return
        if self.mode == "draw":
            end_img_x, end_img_y = self.to_image(event.pos())
            w, h = self.image_size()
            x1, x2 = sorted((self.start_img[0], end_img_x))
            y1, y2 = sorted((self.start_img[1], end_img_y))
            if (x2 - x1) > 4 and (y2 - y1) > 4:
                self.push_history()
                x1 = max(0.0, min(w, x1)); x2 = max(0.0, min(w, x2))
                y1 = max(0.0, min(h, y1)); y2 = max(0.0, min(h, y2))
                self.boxes.append([float(self.window.current_class),
                                   ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h,
                                   (x2 - x1) / w, (y2 - y1) / h])
                self.selected = len(self.boxes) - 1
                # 标了载具就顺手清掉被它盖住的行人框（骑车人/驾驶位里的人）
                if int(self.boxes[self.selected][0]) in {0, 1, 2, 3, 4}:
                    before = len(self.boxes)
                    self.boxes = suppress_riders(self.boxes)
                    if len(self.boxes) != before:
                        self.selected = -1
                        self.window.mark_dirty()
            self.mode = None
            self.window.refresh_status()
            self.update()
            return
        if self.mode == "move":
            self.mode = None
            self.window.mark_dirty()
            self.update()
            return
        if self.mode == "resize":
            self.mode = None
            self.active_handle = None
            self.window.mark_dirty()
            self.setCursor(Qt.ArrowCursor)
            self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key_Space:
            self.space_held = True
            self.setCursor(Qt.OpenHandCursor)
            return
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            if 0 <= self.selected < len(self.boxes):
                self.push_history()
                self.boxes.pop(self.selected)
                self.selected = -1
                self.window.mark_dirty()
                self.window.refresh_status()
                self.update()
            return
        if Qt.Key_1 <= key <= Qt.Key_9:
            idx = key - Qt.Key_1
            if 0 <= self.selected < len(self.boxes):
                self.push_history()
                self.boxes[self.selected][0] = float(idx)
                self.window.mark_dirty()
            else:
                self.window.current_class = idx
            self.window.refresh_status()
            self.update()
            return
        if key == Qt.Key_F:
            self.fit_window()
            return
        if key == Qt.Key_R:
            self.push_history()
            before = len(self.boxes)
            self.boxes = suppress_riders(self.boxes)
            if len(self.boxes) != before:
                self.selected = -1
                self.window.mark_dirty()
                self.window.refresh_status(extra=f"已清理 {before - len(self.boxes)} 个重叠行人框")
            else:
                self.window.refresh_status(extra="没有重叠的行人框需要清理")
            self.update()
            return
        if key == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
            if self.history:
                self.boxes = self.history.pop()
                self.selected = -1
                self.window.mark_dirty()
                self.window.refresh_status(extra="已撤销上一步")
                self.update()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Space:
            self.space_held = False
            self.setCursor(Qt.ArrowCursor)

    def hit_test(self, pos: QPoint) -> int:
        """返回被点中的框下标，从后往前找（后画的在上层）。"""
        w, h = self.image_size()
        if not w or not h:
            return -1
        x, y = self.to_image(pos)
        for index in range(len(self.boxes) - 1, -1, -1):
            _, cx, cy, bw, bh = self.boxes[index]
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            if x1 <= x <= x2 and y1 <= y <= y2:
                return index
        return -1

    # ---------- 拖角缩放 ----------
    HANDLE_IDS = ("左上", "上", "右上", "右", "右下", "下", "左下", "左")

    def box_rect(self, box: list[float]) -> tuple[float, float, float, float]:
        """归一化框 → 图像坐标的 x1, y1, x2, y2。"""
        w, h = self.image_size()
        _, cx, cy, bw, bh = box
        return ((cx - bw / 2) * w, (cy - bh / 2) * h,
                (cx + bw / 2) * w, (cy + bh / 2) * h)

    def handle_points(self, box: list[float]) -> list[tuple[float, float]]:
        x1, y1, x2, y2 = self.box_rect(box)
        return [(x1, y1), ((x1 + x2) / 2, y1), (x2, y1), (x2, (y1 + y2) / 2),
                (x2, y2), ((x1 + x2) / 2, y2), (x1, y2), (x1, (y1 + y2) / 2)]

    def handle_at(self, pos: QPoint) -> int | None:
        """返回命中的手柄编号；只对当前选中的框生效。"""
        if not (0 <= self.selected < len(self.boxes)):
            return None
        tolerance = 9.0
        px, py = pos.x(), pos.y()
        for index, (hx, hy) in enumerate(self.handle_points(self.boxes[self.selected])):
            sx, sy = self.to_screen(hx, hy)
            if abs(px - sx) <= tolerance and abs(py - sy) <= tolerance:
                return index
        return None

    def apply_resize(self, handle: int, image_x: float, image_y: float) -> None:
        """按拖动的手柄调整框的边，保持最小尺寸并限制在画面内。"""
        if not (0 <= self.selected < len(self.boxes)):
            return
        w, h = self.image_size()
        if not w or not h:
            return
        box = self.boxes[self.selected]
        x1, y1, x2, y2 = self.box_rect(box)
        image_x = min(max(image_x, 0.0), float(w))
        image_y = min(max(image_y, 0.0), float(h))
        if handle in (0, 6, 7):
            x1 = image_x
        if handle in (2, 3, 4):
            x2 = image_x
        if handle in (0, 1, 2):
            y1 = image_y
        if handle in (4, 5, 6):
            y2 = image_y
        if x2 - x1 < 5 or y2 - y1 < 5:
            return
        self.boxes[self.selected] = [box[0],
                                     ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h,
                                     (x2 - x1) / w, (y2 - y1) / h]
        self.window.mark_dirty()


class MainWindow(QMainWindow):
    @staticmethod
    def _ordered_images() -> list[Path]:
        """按 review_order.txt 排优先顺序（值钱的帧放前面，可随时收工）；无该文件则按文件名。"""
        every = sorted(IMAGES.glob("*.jpg"))
        if not REVIEW_ORDER.exists():
            return every
        wanted = [line.strip() for line in
                  REVIEW_ORDER.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_name = {p.name: p for p in every}
        ordered = [by_name[name] for name in wanted if name in by_name]
        taken = {p.name for p in ordered}
        return ordered + [p for p in every if p.name not in taken]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("交通标注工具 · 中文版 v2")
        self.images = self._ordered_images()
        if not self.images:
            raise SystemExit(f"没有找到图片，请先运行 bench/make_label_set.py：{IMAGES}")
        self.index = 0
        self.current_class = 0
        self.dirty = False
        self.undo_stack: list[list[list[float]]] = []

        self.canvas = Canvas(self)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # ---- 顶部：类别选择 ----
        top = QHBoxLayout()
        top.setSpacing(6)
        top_label = QLabel("新框类别")
        top_label.setStyleSheet("font-weight: bold;")
        top.addWidget(top_label)
        self.class_buttons = []
        for i, name in enumerate(CLASS_CN):
            color = CLASS_COLORS[i].name()
            btn = QPushButton(f"{i + 1}  {name}")
            btn.setCheckable(True)
            btn.setMinimumHeight(34)
            btn.setMinimumWidth(96)
            btn.setStyleSheet(
                f"QPushButton{{border:2px solid {color}; border-radius:6px;"
                f" padding:4px 10px; font-size:13px; color:#eaeaea;}}"
                f"QPushButton:checked{{background:{color}; color:#101010; font-weight:bold;}}"
            )
            btn.clicked.connect(lambda _, idx=i: self.set_class(idx))
            top.addWidget(btn)
            self.class_buttons.append(btn)
        top.addStretch(1)
        for text, func in (("放大", lambda: self.zoom(1.25)),
                           ("缩小", lambda: self.zoom(0.8)),
                           ("适应窗口", lambda: self.canvas.fit_window()),
                           ("原始大小", self.zoom_reset)):
            btn = QPushButton(text)
            btn.setMinimumHeight(34)
            btn.clicked.connect(func)
            top.addWidget(btn)
        layout.addLayout(top)

        # ---- 中部：画布 + 右侧信息栏 ----
        middle = QHBoxLayout()
        middle.setSpacing(8)
        middle.addWidget(self.canvas, 1)
        panel = QWidget()
        panel.setFixedWidth(250)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(10)
        self.panel_title = QLabel("当前图片")
        self.panel_title.setStyleSheet("font-weight:bold; font-size:14px;")
        self.panel_info = QLabel()
        self.panel_info.setWordWrap(True)
        self.panel_info.setStyleSheet("font-size:13px; line-height:150%;")
        self.panel_counts = QLabel()
        self.panel_counts.setWordWrap(True)
        self.panel_counts.setStyleSheet("font-size:13px;")
        hint = QLabel(
            "操作方式\n"
            "滚轮　　以光标为中心缩放\n"
            "中键 / 空格+左键　平移\n"
            "左键空白处拖动　画新框\n"
            "点框 / 拖框　　选中 / 移动\n"
            "数字 1~6　　切换类别\n"
            "R　　清理重叠行人框\n"
            "Delete　　删除选中框\n"
            "Ctrl+Z　　撤销\n"
            "A / D　　上一张 / 下一张\n"
            "Ctrl+S　　保存\n"
            "F　　适应窗口"
        )
        hint.setStyleSheet("font-size:12px; color:#9fd0ff; line-height:160%;")
        panel_layout.addWidget(self.panel_title)
        panel_layout.addWidget(self.panel_info)
        panel_layout.addWidget(self.panel_counts)
        panel_layout.addStretch(1)
        panel_layout.addWidget(hint)
        middle.addWidget(panel)
        layout.addLayout(middle, 1)

        # ---- 底部：导航与状态 ----
        bottom = QHBoxLayout()
        self.info = QLabel()
        self.info.setStyleSheet("font-size:13px;")
        bottom.addWidget(self.info, 1)
        for text, func in (("上一张 (A)", self.prev_image),
                           ("下一张 (D)", self.next_image),
                           ("保存 (Ctrl+S)", self.save)):
            btn = QPushButton(text)
            btn.setMinimumHeight(34)
            btn.setMinimumWidth(120)
            btn.clicked.connect(func)
            bottom.addWidget(btn)
        layout.addLayout(bottom)
        self.setCentralWidget(central)
        self.set_class(0)
        self.load_image()

    def zoom(self, factor: float) -> None:
        """按钮缩放：以画布中心为基准。"""
        self.canvas.user_zoomed = True
        center = QPoint(self.canvas.width() // 2, self.canvas.height() // 2)
        img_x, img_y = self.canvas.to_image(center)
        self.canvas.scale = max(0.05, min(20.0, self.canvas.scale * factor))
        self.canvas.offset = QPoint(int(center.x() - img_x * self.canvas.scale),
                                    int(center.y() - img_y * self.canvas.scale))
        self.canvas.update()
        self.refresh_status()

    def zoom_reset(self) -> None:
        self.canvas.user_zoomed = True
        self.canvas.scale = 1.0
        self.canvas.offset = QPoint(0, 0)
        self.canvas.update()
        self.refresh_status()

    # ---------- 数据读写 ----------
    def label_path(self) -> Path:
        return LABELS / (self.images[self.index].stem + ".txt")

    def load_image(self) -> None:
        boxes: list[list[float]] = []
        path = self.label_path()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 5:
                    boxes.append([float(parts[0]), float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4])])
        self.canvas.set_image(self.images[self.index], boxes)
        self.dirty = False
        self.undo_stack = []
        self.refresh_status()

    def save(self) -> None:
        path = self.label_path()
        lines = []
        for cls, cx, cy, w, h in self.canvas.boxes:
            lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.dirty = False
        self.refresh_status(extra="已保存")

    def mark_dirty(self) -> None:
        self.dirty = True
        self.refresh_status()

    # ---------- 导航 ----------
    def prev_image(self) -> None:
        if self.dirty:
            self.save()
        if self.index > 0:
            self.index -= 1
            self.load_image()

    def next_image(self) -> None:
        if self.dirty:
            self.save()
        if self.index < len(self.images) - 1:
            self.index += 1
            self.load_image()

    def set_class(self, idx: int) -> None:
        self.current_class = idx
        if 0 <= self.canvas.selected < len(self.canvas.boxes):
            self.canvas.boxes[self.canvas.selected][0] = float(idx)
            self.mark_dirty()
        for i, btn in enumerate(self.class_buttons):
            btn.setChecked(i == idx)
        self.refresh_status()
        self.canvas.update()

    def refresh_status(self, extra: str = "") -> None:
        name = self.images[self.index].name
        n = len(self.canvas.boxes)
        cls_counts: dict[int, int] = {}
        for box in self.canvas.boxes:
            cls_counts[int(box[0])] = cls_counts.get(int(box[0]), 0) + 1
        detail = "  ".join(f"{CLASS_CN[k]}{v}" for k, v in sorted(cls_counts.items()))
        state = "未保存" if self.dirty else "已保存"
        text = (f"第 {self.index + 1}/{len(self.images)} 张 · {name} · "
                f"框 {n} 个（{detail}）· 缩放 {self.canvas.scale * 100:.0f}% · {state}")
        if extra:
            text += f" · {extra}"
        self.info.setText(text)
        state_color = "#ff9f43" if self.dirty else "#4cd137"
        self.info.setStyleSheet(f"font-size:13px; color:{state_color};")

        # 右侧面板
        stem = self.images[self.index].stem
        source = stem.split("_f")[0].replace("VID_", "")
        frame_no = stem.split("_f")[-1]
        self.panel_info.setText(
            f"图像：{name}\n来源视频：{source}\n帧号：{frame_no}\n"
            f"尺寸：{self.canvas.image_size()[0]} × {self.canvas.image_size()[1]}\n"
            f"缩放：{self.canvas.scale * 100:.0f}%\n"
            f"选中：{'第 ' + str(self.canvas.selected + 1) + ' 个框' if self.canvas.selected >= 0 else '无'}"
        )
        rows = [f'<span style="color:{CLASS_COLORS[k].name()}">■</span> '
                f"{CLASS_CN[k]}：{v}" for k, v in sorted(cls_counts.items())]
        self.panel_counts.setText(
            f'<b>共 {n} 个框</b><br>' + ("<br>".join(rows) if rows else "（本图暂无标注）")
        )

    # ---------- 快捷键 ----------
    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        mods = event.modifiers()
        if mods & Qt.ControlModifier and key == Qt.Key_S:
            self.save()
            return
        if key in (Qt.Key_D, Qt.Key_Right):
            self.next_image()
            return
        if key in (Qt.Key_A, Qt.Key_Left):
            self.prev_image()
            return
        super().keyPressEvent(event)


def main() -> None:
    # 只把程序名交给 Qt：数据集目录参数是我们自己的，别让 Qt 去解析
    app = QApplication([sys.argv[0]])
    # 内置深色样式：不依赖第三方主题包，避免缓存目录权限问题
    app.setStyleSheet(
        "QMainWindow, QWidget{background:#232629; color:#e8e8e8;}"
        "QPushButton{background:#34383c; border:1px solid #4d5157; border-radius:6px;"
        " padding:6px 14px; font-size:13px;}"
        "QPushButton:hover{background:#41464b;}"
        "QPushButton:pressed{background:#2b2f33;}"
        "QLabel{color:#e8e8e8;}"
    )
    window = MainWindow()
    window.resize(1440, 900)
    window.show()
    window.canvas.fit_window()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
