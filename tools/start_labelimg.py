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
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "videoset_01_label"
IMAGES = DATASET / "images"
LABELS = DATASET / "labels"
CLASSES = DATASET / "classes.txt"
LOG = Path(__file__).resolve().parent / "labelimg_startup.log"


def log(message: str) -> None:
    """把启动过程写入日志文件，任何失败都能留痕。"""
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


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
    # 必须存 LabelFileFormat 枚举对象本身。存整数 2 会导致
    # labelImg 内部 get_format_meta() 返回 None 并在启动时抛 TypeError。
    try:
        from libs.labelFile import LabelFileFormat
        data["labelFileFormat"] = LabelFileFormat.YOLO
    except Exception:
        # 取不到枚举就干脆不写这一项，让 labelImg 用默认值（人工点一下切换）
        data.pop("labelFileFormat", None)
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
    log("=" * 60)
    log(f"启动标注工具  Python={sys.version.split()[0]}  可执行文件={sys.executable}")
    log(f"项目目录={ROOT}")
    log(f"图片目录存在={IMAGES.exists()}  标注目录存在={LABELS.exists()}  类别文件存在={CLASSES.exists()}")

    for path in (IMAGES, LABELS, CLASSES):
        if not path.exists():
            log(f"缺少必要文件：{path}")
            log("请先运行 bench/make_label_set.py 生成标注集")
            return 1

    write_settings()
    ensure_classes_in_labels()

    log(f"图片数量={len(list(IMAGES.glob('*.jpg')))}")

    # 不用 labelImg.exe：pip 生成的 exe 会把解释器绝对路径写死，
    # 项目目录改名后（路径变化）它会直接以退出码 1 崩溃且没有任何输出。
    # 改为通过 python -m 调用模块，路径由 sys.executable 动态给出。
    cmd = [sys.executable, "-m", "labelImg.labelImg", str(IMAGES), str(CLASSES), str(LABELS)]

    log("启动命令：" + " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.stdout:
        log("--- 子进程输出 ---")
        for line in result.stdout.splitlines():
            log("  " + line)
    if result.stderr:
        log("--- 子进程错误 ---")
        for line in result.stderr.splitlines():
            log("  " + line)
    log(f"子进程退出码={result.returncode}")
    if result.returncode != 0:
        log("启动失败（退出码非 0）")
    return result.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        text = traceback.format_exc()
        log("启动失败，异常详情：")
        for line in text.splitlines():
            log("  " + line)
        sys.exit(1)
