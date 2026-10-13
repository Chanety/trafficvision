"""界面主题：配色与技术报告图表严格一致，保证「界面—图表—报告」是同一套视觉语言。

为什么不直接用现成的 Fluent 组件库（如 PyQt-Fluent-Widgets，8.1k★）：
  1. 那个库免费版是 GPL-3.0，会再叠一层传染性许可，且要新增一条依赖进资源清单；
  2. 通用皮肤反而做不到与本文档体系的图表配色一致；
  3. 四梯度同屏对照、视频叠加框这类专用组件本来就要自己写。
"""

from __future__ import annotations

# 与 paper/competition/figures 里的学术配色同源
BLUE = "#2060cc"
RED = "#cc3030"
GREEN = "#208040"
ORANGE = "#cc7020"
PURPLE = "#8040cc"
GOLD = "#b08020"

BG = "#12161c"          # 主背景
BG_PANEL = "#1a2029"    # 面板
BG_INPUT = "#222b36"    # 输入控件
BORDER = "#2c3743"
TEXT = "#e8ecf1"
TEXT_DIM = "#94a3b3"

# 四个预判梯度的固定配色（界面上到处复用，不改）
HORIZON_COLORS = {"0.1": BLUE, "0.5": GREEN, "1.0": ORANGE, "2.0": PURPLE}


def qss() -> str:
    return f"""
QWidget {{ background: {BG}; color: {TEXT};
           font-family: "Microsoft YaHei","SimSun"; font-size: 13px; }}
QGroupBox {{ background: {BG_PANEL}; border: 1px solid {BORDER}; border-radius: 8px;
             margin-top: 14px; padding: 12px 10px 10px 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px;
                    color: {TEXT_DIM}; font-weight: bold; }}
QLabel#title {{ font-size: 19px; font-weight: bold; color: {TEXT}; }}
QLabel#subtitle {{ color: {TEXT_DIM}; font-size: 12px; }}
QLabel#metricValue {{ font-size: 22px; font-weight: bold; color: {BLUE}; }}
QLabel#metricLabel {{ color: {TEXT_DIM}; font-size: 11px; }}
QPushButton {{ background: {BG_INPUT}; border: 1px solid {BORDER}; border-radius: 6px;
               padding: 7px 14px; color: {TEXT}; }}
QPushButton:hover {{ border-color: {BLUE}; }}
QPushButton:disabled {{ color: #5a6672; border-color: #232c36; }}
QPushButton#primary {{ background: {BLUE}; border: none; font-weight: bold; }}
QPushButton#primary:hover {{ background: #2a74e6; }}
QPushButton#primary:disabled {{ background: #2a3a4e; color: #6b7785; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: {BG_INPUT};
    border: 1px solid {BORDER}; border-radius: 6px; padding: 5px 8px; }}
QComboBox QAbstractItemView {{ background: {BG_INPUT}; selection-background-color: {BLUE}; }}
QPlainTextEdit {{ background: #0d1116; border: 1px solid {BORDER}; border-radius: 6px;
                  font-family: Consolas,"Microsoft YaHei"; font-size: 12px; }}
QProgressBar {{ background: {BG_INPUT}; border: none; border-radius: 5px;
                height: 10px; text-align: center; color: {TEXT_DIM}; }}
QProgressBar::chunk {{ background: {BLUE}; border-radius: 5px; }}
QRadioButton, QCheckBox {{ spacing: 6px; }}
QRadioButton::indicator, QCheckBox::indicator {{ width: 15px; height: 15px; }}
QScrollArea {{ border: none; }}
QSlider::groove:horizontal {{ height: 5px; background: {BG_INPUT}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {BLUE}; width: 13px; margin: -5px 0;
                              border-radius: 6px; }}
QStatusBar {{ color: {TEXT_DIM}; }}
"""
