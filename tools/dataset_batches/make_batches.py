"""把待人工复核的数据集切成固定大小的批次包，供他人下载后各自复核。

用法：
    python tools/dataset_batches/make_batches.py
    python tools/dataset_batches/make_batches.py --source datasets/videoset_02_label2 --name videoset_02_b2

产出（默认写到仓库根的 dataset_batches\，该目录在 main 分支被 .gitignore 忽略）：
    batch_01.zip ...            每个包 = images/ + labels/ + classes.txt + review_order.txt + 标注说明.txt
    manifest.json               批次清单（帧数、体积、sha256、来源视频分布、下载直链）
    下载批次NN.bat / 打开标注_批次NN.bat   供他人双击使用（utf-8 + CRLF）

批次划分：按 review_order.txt 的优先级顺序**顺序切块**，每块 150 张。
因此 batch_01 优先级最高（三轮车→自行车→标志牌→路桩→公交/卡车），最后一包是余数。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCHES_DIR = ROOT / "dataset_batches"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# 下载直链用的配置（与 _batch_config.bat 保持一致）
REPO = "Chanety/ai-software-opensource"
BRANCH = "codex/dataset-batches-v1"

LABEL_GUIDE = """========================================================
 {name} 人工复核说明（{batch_id}，共 {count} 张）
========================================================

【这是什么】
监控视频按每 30 帧取 1 帧抽出的 1080p 图片（同一个视频的帧在时间上连续）。
自动标注已经把框画好了，你的任务是"改错"，不是从零画。

【怎么打开】
  · 装了本项目的人：双击 tools\\dataset_batches\\下载批次{nn}.bat
    下载完再双击 tools\\dataset_batches\\打开标注_批次{nn}.bat
  · 只有这个压缩包的人：用任何 YOLO 格式标注工具打开本目录
    （images/ 是图，labels/ 是 YOLO txt，classes.txt 是类别表）

【要做什么】
  · 框多余 / 框错了 → 删掉（Delete / Backspace）
  · 有目标没框住 → 补画一个
  · 类别错了 → 按数字键改成正确类别
  · 复核完一帧按 D 或右方向键进入下一帧，进度实时保存
  · 已经对的帧直接按 D 跳过，速度优先

【类别（9 类，编号顺序不可改）】
  1 汽车 car          5 自行车 bicycle
  2 卡车 truck        6 行人 person
  3 公交 bus          7 标志牌 sign
  4 摩托 motorcycle   8 路桩 bollard
  （含电动车、踏板车） 9 三轮车 tricycle（三轮车、三轮棚车，不细分）

【三条硬规则】
  1) 先载具、后行人：骑车人、驾驶位里的人只标载具，不标 person。
     按 R 键会自动清掉与载具重叠的行人框。
  2) 静止的东西不预判，但静止类（标志牌 / 路桩）必须保留。
  3) 拿不准的框留着别删——漏标比误检对模型伤害大得多。

【这批素材实测的常见误检，见到就删】
  · 红色招牌 / 灯箱 / 广告牌被标成"标志牌"
  · 花坛边缘 / 路缘石 / 斑马线被标成"路桩"
  · 店面招牌、反光玻璃被反复标成"卡车"
  · 紧贴停放车辆行走的人被标成骑行者（先看脚在不在车上）

【复核标准】
  · 框形差几个像素不用改；"有没有标到"和"类别对不对"才是关键。
  · 被遮挡一半的目标仍然要标，看不见的部分不用猜。
========================================================
"""


def load_order(source: Path) -> tuple[list[str], list[str], dict[str, str]]:
    """返回（按优先级排好的图片名列表, 优先帧集合, 图片名→视频名）。"""
    image_dir = source / "images"
    names = sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not names:
        raise SystemExit(f"没有找到图片：{image_dir}")
    known = set(names)

    order: list[str] = []
    priority: list[str] = []
    order_file = source / "review_order.txt"
    if order_file.exists():
        listed = [ln.strip() for ln in
                  order_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        priority = list(dict.fromkeys(n for n in listed if n in known))   # 去重且保序
        order = list(priority)
    seen = set(order)
    order += [n for n in names if n not in seen]

    # 视频名 = 去掉 _f00030.jpg 这类帧号后缀
    video_of = {n: n.rsplit("_f", 1)[0] if "_f" in n else Path(n).stem for n in names}
    return order, priority, video_of


def split_even(order: list[str], size: int) -> list[list[str]]:
    return [order[i:i + size] for i in range(0, len(order), size)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\r\n" if path.suffix == ".bat" else "\n")


def download_bat(batch_id: str) -> str:
    nn = batch_id.split("_")[1]
    return f"""@echo off
chcp 65001 >nul
setlocal
call "%~dp0_batch_config.bat"
if not defined BATCH_REPO ( echo 缺少 _batch_config.bat & pause & exit /b 1 )

set "BATCH={batch_id}"
set "ZIP=%TEMP%\\%BATCH_NAME%_%BATCH%.zip"
set "URL=https://raw.githubusercontent.com/%BATCH_REPO%/%BATCH_BRANCH%/%BATCH_DIR%/%BATCH%.zip"
set "DEST=%~dp0..\\..\\datasets\\%BATCH_NAME%\\%BATCH%"

echo ============================================================
echo   下载 %BATCH% （%BATCH_NAME%）
echo   来源：%URL%
echo   解压到：%DEST%
echo ============================================================
echo.
echo 正在下载，约 80MB，请等提示"下载完成"再关窗口...

if exist "%ZIP%" del /q "%ZIP%"
curl -fL --retry 3 --retry-delay 2 -o "%ZIP%" "%URL%"
if not exist "%ZIP%" (
  echo curl 下载失败，改用 PowerShell 重试...
  powershell -NoProfile -Command "Invoke-WebRequest -Uri '%URL%' -OutFile '%ZIP%'"
)
if not exist "%ZIP%" (
  echo.
  echo [失败] 没能下载到文件。请检查网络，或手动打开上面的来源链接下载。
  pause
  exit /b 1
)

echo 下载完成，正在解压...
if not exist "%DEST%" mkdir "%DEST%"
tar -xf "%ZIP%" -C "%DEST%" 2>nul
if errorlevel 1 (
  powershell -NoProfile -Command "Expand-Archive -LiteralPath '%ZIP%' -DestinationPath '%DEST%' -Force"
)
del /q "%ZIP%"

echo.
echo [完成] 已解压到 %DEST%
echo        接下来双击 打开标注_批次{nn}.bat 开始复核。
echo.
pause
"""


def open_bat(batch_id: str) -> str:
    return f"""@echo off
chcp 65001 >nul
setlocal
call "%~dp0_batch_config.bat"
cd /d "%~dp0..\\.."

set "BATCH={batch_id}"
set "DATA=datasets\\%BATCH_NAME%\\%BATCH%"
if not exist "%DATA%\\images" (
  echo [缺少数据] 还没下载 %BATCH%，请先双击 下载批次{batch_id.split('_')[1]}.bat
  pause
  exit /b 1
)

set "PY=trafficlab-3d\\env\\Scripts\\python.exe"
if not exist "%PY%" set "PY=python"

echo 正在打开 %DATA% 的复核标注...
echo 滚轮缩放 / 中键或空格拖动 / 拖角改框 / R 清理重叠行人框 / Ctrl+Z 撤销
echo 改完一帧按 D 或右方向键进入下一帧，进度实时写入 labels\\
echo.
"%PY%" tools\\annotate_zh.py "%DATA%"
echo.
echo program exited, errorlevel = %errorlevel%
pause
"""


def make_zip(stage: Path, out: Path) -> None:
    if out.exists():
        out.unlink()
    fixed = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for item in sorted(stage.rglob("*")):
            if not item.is_file():
                continue
            rel = item.relative_to(stage).as_posix()
            info = zipfile.ZipInfo(rel, date_time=fixed)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, item.read_bytes())


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 150 张一包的人工复核批次")
    ap.add_argument("--source", default="datasets/videoset_02_label",
                    help="待复核数据集目录（需含 images/ 与 labels/）")
    ap.add_argument("--size", type=int, default=150, help="每包帧数，默认 150")
    ap.add_argument("--name", default="videoset_02_review", help="解压落地目录名")
    ap.add_argument("--out", default=str(BATCHES_DIR), help="批次包输出目录")
    ap.add_argument("--bats", action="store_true",
                    help="同时生成 下载批次NN.bat / 打开标注_批次NN.bat（放在本脚本旁）")
    args = ap.parse_args()

    source = (ROOT / args.source).resolve() if not Path(args.source).is_absolute() \
        else Path(args.source)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    order, priority, video_of = load_order(source)
    chunks = split_even(order, args.size)
    classes_txt = (source / "classes.txt").read_text(encoding="utf-8")
    stage_root = out_dir / "_stage"
    if stage_root.exists():
        shutil.rmtree(stage_root)

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(source.relative_to(ROOT)).replace("\\", "/"),
        "batch_size": args.size,
        "total_frames": len(order),
        "split_rule": ("按 review_order.txt 的顺序顺序切块（该文件已覆盖全部帧、"
                       "按价值从高到低、跨视频轮流），所以 batch_01 优先帧最多，最后一包是余数"),
        "source_priority_listed": len(priority),
        "branch": BRANCH,
        "extract_to": f"datasets/{args.name}/<batch_id>",
        "batches": [],
    }

    for idx, frames in enumerate(chunks, start=1):
        batch_id = f"batch_{idx:02d}"
        stage = stage_root / batch_id
        (stage / "images").mkdir(parents=True)
        (stage / "labels").mkdir(parents=True)
        (stage / "classes.txt").write_text(classes_txt, encoding="utf-8")
        write_text(stage / "review_order.txt", "\n".join(frames) + "\n")
        write_text(stage / "标注说明.txt",
                   LABEL_GUIDE.format(name=args.name, batch_id=batch_id,
                                      count=len(frames), nn=f"{idx:02d}"))
        for name in frames:
            shutil.copy2(source / "images" / name, stage / "images" / name)
            sidecar = source / "labels" / (Path(name).stem + ".txt")
            if sidecar.exists():
                shutil.copy2(sidecar, stage / "labels" / sidecar.name)

        zip_path = out_dir / f"{batch_id}.zip"
        make_zip(stage, zip_path)
        videos = Counter(video_of[n] for n in frames)
        manifest["batches"].append({
            "id": batch_id,
            "zip": zip_path.name,
            "frames": len(frames),
            "bytes": zip_path.stat().st_size,
            "sha256": sha256(zip_path),
            "videos": dict(sorted(videos.items())),
            "priority_head": frames[0],
            "priority_tail": frames[-1],
            "download_url": f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"
                            f"dataset_batches/{zip_path.name}",
        })
        print(f"{batch_id}: {len(frames)} 帧, "
              f"{zip_path.stat().st_size / 1e6:.1f} MB, {len(videos)} 段视频")

    if args.bats:
        tools_dir = Path(__file__).resolve().parent
        for idx in range(1, len(chunks) + 1):
            batch_id = f"batch_{idx:02d}"
            write_text(tools_dir / f"下载批次{idx:02d}.bat", download_bat(batch_id))
            write_text(tools_dir / f"打开标注_批次{idx:02d}.bat", open_bat(batch_id))
        print(f"已生成 {len(chunks)} 组 bat（下载 / 打开标注）")

    write_text(out_dir / "manifest.json",
               json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    shutil.rmtree(stage_root)
    print(f"清单已写入 {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
