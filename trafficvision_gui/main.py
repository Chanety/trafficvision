"""图形界面入口：python -m trafficvision_gui.main

布局（三栏 + 底部）：
  左栏 输入与参数 · 中栏 视频预览与播放 · 右栏 指标与四梯度对照 · 底部 运行与日志
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QSplitter, QStatusBar, QVBoxLayout, QWidget,
)

from . import theme
from .worker import PipelineJob

ROOT = Path(__file__).resolve().parents[1]
VIVID = ["#2060cc", "#208040", "#cc7020", "#8040cc"]


class MetricCard(QFrame):
    """单个指标卡：上面数值，下面标签。"""

    def __init__(self, label: str, color: str = theme.BLUE):
        super().__init__()
        self.setStyleSheet(
            f"QFrame{{background:{theme.BG_INPUT};border:1px solid {theme.BORDER};"
            f"border-radius:8px;}}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(2)
        self.value = QLabel("—")
        self.value.setObjectName("metricValue")
        self.value.setStyleSheet(f"color:{color};font-size:20px;font-weight:bold;")
        self.label = QLabel(label)
        self.label.setObjectName("metricLabel")
        lay.addWidget(self.value)
        lay.addWidget(self.label)

    def set(self, text: str) -> None:
        self.value.setText(text)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("交通视频短时预判软件 · v0.3")
        self.resize(1500, 900)
        self.job = PipelineJob(self)
        self.job.log.connect(self.append_log)
        self.job.finished.connect(self.on_finished)
        self.job.metrics.connect(self.on_metrics)
        self._build()
        self.append_log("就绪。左侧选择视频与参数，点「开始预判」。")

    # ------------------------------------------------------------------ 构建
    def _build(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 12, 14, 10)
        outer.setSpacing(10)
        outer.addWidget(self._header())

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._left_panel())
        split.addWidget(self._center_panel())
        split.addWidget(self._right_panel())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([340, 760, 360])
        outer.addWidget(split, 1)
        outer.addWidget(self._bottom_bar())

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("未加载视频")

    def _header(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(2)
        t = QLabel("交通视频短时预判软件")
        t.setObjectName("title")
        s = QLabel("输入 2~30 秒单目视频，选 0.1/0.5/1/2 秒预判梯度，"
                   "输出带类型与预判位置的视频与逐梯度精度报告")
        s.setObjectName("subtitle")
        lay.addWidget(t)
        lay.addWidget(s)
        return w

    # ------------------------------------------------------------------ 左栏
    def _left_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(10)

        g1 = QGroupBox("输入")
        v1 = QVBoxLayout(g1)
        self.ed_video = QLineEdit()
        self.ed_video.setPlaceholderText("选择视频文件…")
        b1 = QPushButton("浏览…")
        b1.clicked.connect(self.pick_video)
        row = QHBoxLayout()
        row.addWidget(self.ed_video, 1)
        row.addWidget(b1)
        v1.addLayout(row)
        self.ed_camera = QLineEdit()
        self.ed_camera.setPlaceholderText("机位名（可留空；同机位复用场景先验）")
        v1.addWidget(self.ed_camera)
        lay.addWidget(g1)

        g2 = QGroupBox("预判参数")
        v2 = QVBoxLayout(g2)
        self.radios: dict[str, QRadioButton] = {}
        for i, h in enumerate(["0.1", "0.5", "1.0", "2.0"]):
            r = QRadioButton(f"{h} 秒")
            r.setStyleSheet(f"QRadioButton{{color:{VIVID[i]};font-weight:bold;}}")
            if h == "1.0":
                r.setChecked(True)
            self.radios[h] = r
            v2.addWidget(r)
        row2 = QHBoxLayout()
        self.sp_imgsz = QSpinBox()
        self.sp_imgsz.setRange(320, 1920)
        self.sp_imgsz.setSingleStep(32)
        self.sp_imgsz.setValue(1280)
        row2.addWidget(QLabel("分辨率"))
        row2.addWidget(self.sp_imgsz)
        self.sp_conf = QDoubleSpinBox()
        self.sp_conf.setRange(0.05, 0.9)
        self.sp_conf.setSingleStep(0.05)
        self.sp_conf.setValue(0.25)
        row2.addWidget(QLabel("置信度"))
        row2.addWidget(self.sp_conf)
        v2.addLayout(row2)
        self.chk_static = QCheckBox("静态结构通道（路桩/路灯/信号灯稳定显示）")
        self.chk_static.setChecked(True)
        self.chk_coast = QCheckBox("遮挡续推（被遮挡时继续显示）")
        self.chk_coast.setChecked(True)
        v2.addWidget(self.chk_static)
        v2.addWidget(self.chk_coast)
        row3 = QHBoxLayout()
        self.sp_stride = QSpinBox()
        self.sp_stride.setRange(1, 4)
        self.sp_stride.setValue(1)
        row3.addWidget(QLabel("跳帧"))
        row3.addWidget(self.sp_stride)
        self.sp_maxframes = QSpinBox()
        self.sp_maxframes.setRange(0, 100000)
        self.sp_maxframes.setValue(0)
        self.sp_maxframes.setToolTip("0 表示处理整段视频")
        row3.addWidget(QLabel("帧数(0=全部)"))
        row3.addWidget(self.sp_maxframes)
        v2.addLayout(row3)
        lay.addWidget(g2)

        g3 = QGroupBox("输出")
        v3 = QVBoxLayout(g3)
        self.ed_out = QLineEdit(str(ROOT / "outputs" / "gui"))
        b3 = QPushButton("浏览…")
        b3.clicked.connect(self.pick_out)
        row4 = QHBoxLayout()
        row4.addWidget(self.ed_out, 1)
        row4.addWidget(b3)
        v3.addLayout(row4)
        lay.addWidget(g3)
        lay.addStretch(1)
        return panel

    # ------------------------------------------------------------------ 中栏
    def _center_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.video = QVideoWidget()
        self.video.setStyleSheet(
            f"background:#000;border:1px solid {theme.BORDER};border-radius:8px;")
        self.video.setMinimumHeight(420)
        lay.addWidget(self.video, 1)

        bar = QHBoxLayout()
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.seek)
        self.lb_time = QLabel("00:00 / 00:00")
        self.lb_time.setObjectName("subtitle")
        bar.addWidget(self.btn_play)
        bar.addWidget(self.slider, 1)
        bar.addWidget(self.lb_time)
        lay.addLayout(bar)

        self.player = QMediaPlayer(self)
        self.player.setVideoOutput(self.video)
        self.player.setAudioOutput(QAudioOutput(self))
        self.player.positionChanged.connect(self.on_position)
        self.player.durationChanged.connect(self.on_duration)
        self.player.playbackStateChanged.connect(self.on_state)
        return w

    # ------------------------------------------------------------------ 右栏
    def _right_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 0, 0, 0)
        lay.setSpacing(10)

        g = QGroupBox("本次运行指标（规范预判口径）")
        v = QVBoxLayout(g)
        self.card_pred = MetricCard("每帧可预判目标", theme.BLUE)
        self.card_med = MetricCard("1 秒中位误差 (px)", theme.ORANGE)
        self.card_p90 = MetricCard("1 秒 P90 误差 (px)", theme.RED)
        self.card_obj = MetricCard("每帧目标数", theme.GREEN)
        for c in (self.card_pred, self.card_med, self.card_p90, self.card_obj):
            v.addWidget(c)
        lay.addWidget(g)

        g2 = QGroupBox("四个预判梯度")
        v2 = QVBoxLayout(g2)
        for i, (h, note) in enumerate([
                ("0.1", "最短梯度，误差最小，适合瞬时趋势"),
                ("0.5", "半秒，兼顾精度与提前量"),
                ("1.0", "默认梯度，精度与提前量的平衡点"),
                ("2.0", "最长，仅宜作方向示意")]):
            row = QHBoxLayout()
            chip = QLabel()
            chip.setFixedSize(12, 12)
            chip.setStyleSheet(f"background:{VIVID[i]};border-radius:3px;")
            txt = QLabel(f"<b>{h} 秒</b> — {note}")
            txt.setWordWrap(True)
            txt.setObjectName("subtitle")
            row.addWidget(chip)
            row.addWidget(txt, 1)
            v2.addLayout(row)
        lay.addWidget(g2)

        g3 = QGroupBox("报告")
        v3 = QVBoxLayout(g3)
        self.btn_open_out = QPushButton("打开输出目录")
        self.btn_open_out.clicked.connect(self.open_out_dir)
        self.btn_export = QPushButton("导出精度报告（JSON）")
        self.btn_export.clicked.connect(self.export_report)
        self.btn_export.setEnabled(False)
        v3.addWidget(self.btn_open_out)
        v3.addWidget(self.btn_export)
        lay.addWidget(g3)
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------------ 底栏
    def _bottom_bar(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        row = QHBoxLayout()
        self.btn_run = QPushButton("开始预判")
        self.btn_run.setObjectName("primary")
        self.btn_run.setMinimumWidth(130)
        self.btn_run.clicked.connect(self.run)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.job.cancel)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.bar, 1)
        lay.addLayout(row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        self.log.setFixedHeight(120)
        lay.addWidget(self.log)
        return w

    # ------------------------------------------------------------------ 动作
    def pick_video(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "选择视频", str(ROOT / "datasets"), "视频文件 (*.mp4 *.mov *.avi *.mkv *.m4v)")
        if f:
            self.ed_video.setText(f)
            self.player.setSource(QUrl.fromLocalFile(f))
            self.btn_play.setEnabled(True)
            self.slider.setEnabled(True)
            self.statusBar().showMessage(f"已加载：{Path(f).name}")
            self.append_log(f"已加载原视频：{f}")

    def pick_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择输出目录", self.ed_out.text())
        if d:
            self.ed_out.setText(d)

    def toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def seek(self, v: int) -> None:
        self.player.setPosition(v)

    def on_position(self, pos: int) -> None:
        if not self.slider.isSliderDown():
            self.slider.setValue(pos)
        self.lb_time.setText(f"{fmt(pos)} / {fmt(self.player.duration())}")

    def on_duration(self, dur: int) -> None:
        self.slider.setRange(0, max(dur, 1))

    def on_state(self, state) -> None:
        self.btn_play.setText("⏸ 暂停" if state == QMediaPlayer.PlayingState else "▶ 播放")

    def run(self) -> None:
        video = Path(self.ed_video.text().strip())
        if not video.exists():
            QMessageBox.warning(self, "缺少输入", "请先选择一个存在的视频文件。")
            return
        horizon = float(next(h for h, r in self.radios.items() if r.isChecked()))
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.bar.setRange(0, 0)                    # 不确定态，避免假进度
        self.statusBar().showMessage("正在分析…")
        self.job.start(
            video=video, horizon=horizon, out_dir=Path(self.ed_out.text().strip()),
            camera=self.ed_camera.text().strip() or None,
            imgsz=self.sp_imgsz.value(), conf=self.sp_conf.value(),
            static_map=self.chk_static.isChecked(), coast=12 if self.chk_coast.isChecked() else 0,
            max_frames=self.sp_maxframes.value() or None,
            frame_stride=self.sp_stride.value())

    def on_finished(self, ok: bool, info: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.bar.setRange(0, 100)
        self.bar.setValue(100 if ok else 0)
        if ok:
            self.statusBar().showMessage(f"完成：{Path(info).name}")
            self.append_log(f"[完成] 输出视频：{info}")
            self.player.setSource(QUrl.fromLocalFile(info))
            self.btn_play.setEnabled(True)
            self.slider.setEnabled(True)
            self.btn_export.setEnabled(True)
            self.player.play()
        else:
            self.statusBar().showMessage("任务未完成")
            self.append_log(f"[失败] {info}")
            self.btn_export.setEnabled(False)

    def on_metrics(self, metrics: dict) -> None:
        if not metrics:
            return
        def med(key: str):
            m = metrics.get(key) or {}
            return m.get("median_px"), m.get("p90_px"), m.get("samples")
        med1, p90_1, _ = med("1s")
        self.card_med.set(f"{med1:.1f}" if isinstance(med1, (int, float)) else "—")
        self.card_p90.set(f"{p90_1:.1f}" if isinstance(p90_1, (int, float)) else "—")
        self.append_log("[指标] " + "  ".join(
            f"{k}: 中位 {v.get('median_px', '—')} px / {v.get('samples', 0)} 样本"
            for k, v in metrics.items() if v.get("samples")))

    def export_report(self) -> None:
        src = self.job.out_json
        if not src or not src.exists():
            QMessageBox.information(self, "无报告", "本次运行没有生成报告文件。")
            return
        dst, _ = QFileDialog.getSaveFileName(self, "导出报告", str(src.name), "JSON (*.json)")
        if dst:
            Path(dst).write_bytes(src.read_bytes())
            self.append_log(f"[导出] 报告已另存为 {dst}")

    def open_out_dir(self) -> None:
        d = Path(self.ed_out.text().strip())
        d.mkdir(parents=True, exist_ok=True)
        import os
        os.startfile(d)  # noqa: S606  (Windows 打开目录)

    def append_log(self, text: str) -> None:
        self.log.appendPlainText(text)


def fmt(ms: int) -> str:
    s = max(ms, 0) // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("交通视频短时预判软件")
    # 显式解析中文字体：QSS 里的 family 列表在部分平台不生效，
    # 这里按可用性依次回退，避免出现方框字形。
    families = set(QFontDatabase.families())
    for name in ("Microsoft YaHei", "SimSun", "SimHei", "DengXian", "Noto Sans CJK SC"):
        if name in families:
            app.setFont(QFont(name, 10))
            break
    app.setStyleSheet(theme.qss())
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
