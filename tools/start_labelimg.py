"""启动标注复核工具（labelImg），并自动完成全部预设。

会做的事：
  1. 把 labelImg 的默认格式设为 YOLO（否则默认是 PascalVOC，无法直接读我们的标注）
  2. 打开自动保存
  3. 预设图片目录、类别文件、标注保存目录
  4. 启动图形界面

用法：直接双击 tools\\review_labels.bat，或运行
      python tools/start_labelimg.py
"""

from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "videoset_01_label"
IMAGES = DATASET / "images"
LABELS = DATASET / "labels"
CLASSES = DATASET / "classes.txt"


def write_settings() -> None:
    """预设 labelImg 设置：YOLO 格式 + 自动保存 + 目录。"""
    settings_path = Path(os.path.expanduser("~")) / ".labelImgSettings.pkl"
    data = {}
    if settings_path.exists():
        try:
            with open(settings_path, "rb") as fh:
                data = pickle.load(fh)
        except Exception:
            data = {}
    data["labelFileFormat"] = 2          # LabelFileFormat.YOLO
    data["autosave"] = True
    data["savedir"] = str(LABELS)
    data["lastOpenDir"] = str(IMAGES)
    data["drawSquares"] = False
    data["displayLabel"] = True
    try:
        with open(settings_path, "wb") as fh:
            pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)
        print(f"已写入 labelImg 设置：{settings_path}")
    except Exception as exc:  # 配置写不进去也不该阻止启动
        print(f"[警告] 无法写入设置文件（{exc}）")
        print("        启动后请手动把左上角格式切到 YOLO，并勾选「自动保存」")


def ensure_classes_in_labels() -> None:
    """labelImg 的 YOLO 读取器要求在标注目录里也能找到 classes.txt。

    它会从标注文件所在目录去找 classes.txt，只放在数据集根目录会直接报
    FileNotFoundError 并导致启动失败。
    """
    target = LABELS / "classes.txt"
    if not target.exists():
        shutil.copyfile(CLASSES, target)
        print(f"已补齐类别文件：{target}")


def main() -> None:
    for path in (IMAGES, LABELS, CLASSES):
        if not path.exists():
            print(f"缺少必要文件：{path}")
            print("请先运行 bench/make_label_set.py 生成标注集")
            sys.exit(1)

    write_settings()
    ensure_classes_in_labels()

    exe = Path(sys.executable).parent / "labelImg.exe"
    if exe.exists():
        cmd = [str(exe), str(IMAGES), str(CLASSES), str(LABELS)]
    else:
        cmd = [sys.executable, "-m", "labelImg.labelImg", str(IMAGES), str(CLASSES), str(LABELS)]

    print("启动标注工具：")
    print("  " + " ".join(cmd))
    print(f"\n图片目录：{IMAGES}\n标注目录：{LABELS}\n类别文件：{CLASSES}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n程序退出码：{result.returncode}（非 0 表示启动失败）")
        sys.exit(result.returncode)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log = Path(__file__).resolve().parent / "labelimg_error.log"
        text = traceback.format_exc()
        log.write_text(text, encoding="utf-8")
        print("启动失败，错误详情：\n")
        print(text)
        print(f"（已写入 {log}）")
        sys.exit(1)
