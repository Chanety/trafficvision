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
DATASET = ROOT / "datasets" / "videoset_01_label"
IMAGES = DATASET / "images"
LABELS = DATASET / "labels"

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
CLASS_NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
CLASS_CN = ["汽车", "卡车", "公交", "摩托", "自行车", "行人"]
CLASS_COLORS = [
    QColor(80, 220, 80),
    QColor(255, 150, 0),
    QColor(40, 130, 255),
    QColor(220, 60, 220),
    QColor(220, 220, 60),
    QColor(60, 220, 255),
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
        self.fit_window()
        self.update()
        self.window.refresh_status()

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
        self.setCursor(Qt.OpenHandCursor if self.hit_test(pos) < 0 else Qt.SizeAllCursor)

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
                x1 = max(0.0, min(w, x1)); x2 = max(0.0, min(w, x2))
                y1 = max(0.0, min(h, y1)); y2 = max(0.0, min(h, y2))
                self.boxes.append([float(self.window.current_class),
                                   ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h,
                                   (x2 - x1) / w, (y2 - y1) / h])
                self.selected = len(self.boxes) - 1
            self.mode = None
            self.window.refresh_status()
            self.update()
            return
        if self.mode == "move":
            self.mode = None
            self.window.mark_dirty()
            self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key_Space:
            self.space_held = True
            self.setCursor(Qt.OpenHandCursor)
            return
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            if 0 <= self.selected < len(self.boxes):
                self.boxes.pop(self.selected)
                self.selected = -1
                self.window.mark_dirty()
                self.window.refresh_status()
                self.update()
            return
        if Qt.Key_1 <= key <= Qt.Key_6:
            idx = key - Qt.Key_1
            if 0 <= self.selected < len(self.boxes):
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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("交通视频标注复核工具")
        self.images = sorted(IMAGES.glob("*.jpg"))
        if not self.images:
            raise SystemExit(f"没有找到图片，请先运行 bench/make_label_set.py：{IMAGES}")
        self.index = 0
        self.current_class = 0
        self.dirty = False
        self.undo_stack: list[list[list[float]]] = []

        self.canvas = Canvas(self)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        top = QHBoxLayout()
        top.addWidget(QLabel("新框类别："))
        self.class_buttons = []
        for i, name in enumerate(CLASS_CN):
            btn = QPushButton(f"{i + 1} {name}")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, idx=i: self.set_class(idx))
            top.addWidget(btn)
            self.class_buttons.append(btn)
        top.addStretch(1)
        layout.addLayout(top)
        layout.addWidget(self.canvas, 1)

        bottom = QHBoxLayout()
        self.info = QLabel()
        bottom.addWidget(self.info)
        bottom.addStretch(1)
        for text, func in (("上一张 (A)", self.prev_image),
                           ("下一张 (D)", self.next_image),
                           ("保存 (Ctrl+S)", self.save),
                           ("适应窗口 (F)", lambda: self.canvas.fit_window())):
            btn = QPushButton(text)
            btn.clicked.connect(func)
            bottom.addWidget(btn)
        layout.addLayout(bottom)
        self.setCentralWidget(central)
        self.set_class(0)
        self.load_image()

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
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1280, 860)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
