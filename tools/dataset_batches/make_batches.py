"""把待人工复核的数据集切成固定大小的批次包，供他人下载后各自复核。

用法：
    python tools/dataset_batches/make_batches.py
    python tools/dataset_batches/make_batches.py --source datasets/videoset_02_label2 --name videoset_02_b2

产出（默认写到仓库根的 dataset_batches\，该目录在 main 分支被 .gitignore 忽略）：
    batch_01.zip ...            每个包 = images/ + labels/ + classes.txt + review_order.txt
                                          + 标注说明.txt + 口径说明.txt + 离线包说明.txt + 离线标注工具
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
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCHES_DIR = ROOT / "dataset_batches"
TOOLS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(TOOLS_DIR.parent))      # human_task.py 在 tools/ 下
from human_task import NOTICE_FILENAME, notice_text  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# 一起塞进每个批次包、让它自带标注工具的一堆文件
OFFLINE_TOOL = "标注工具_离线版.html"
# 类别 ↔ 权重口径说明（2026-10-10 加：协作者反馈"输出类别和权重口径还没有"）
# 只讲"标哪几类"是不够的——必须同时讲清"软件输出哪几类""这份标注喂给哪版权重"。
SCHEMA_NOTE = "口径说明.txt"

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

【类别（{n_classes} 类，编号顺序不可改）】
{classes_block}

【这 {n_classes} 类就是软件的输出类别——但先读 口径说明.txt】
  包里另有一份《口径说明.txt》，讲清三件事：
    · 软件画面上会出现哪些类别（以及为什么现在默认只出 7 类 + 静态通道补 2 类）
    · 你这批标注最终喂给哪版权重（v9b，10 类口径）
    · 权重文件在哪、怎么核对 SHA-256
  只按 classes.txt 标就可以，但"标完喂给谁"这件事在《口径说明.txt》里，建议看一眼。

【三条硬规则】
  1) 先载具、后行人：**骑车/骑车的人单独标成"非机动车驾驶员 rider"**，
     不要标成 person；只有纯步行的人才标 person。
     （旧规则是"骑车人只标载具、不标人"，本版已改，请按新规则来。）
  2) 静止的东西不预判，但静态类（标志牌 sign / 路桩 bollard / 路灯 roadlamp /
     信号灯 trafficlight）必须保留。
  3) 拿不准的框留着别删——漏标比误检对模型伤害大得多。

【这批素材实测的常见误检，见到就删】
  · 红色招牌 / 灯箱 / 广告牌被标成"信号灯"或"标志牌"
  · 花坛边缘 / 路缘石 / 斑马线被标成"路桩"
  · 店面招牌、反光玻璃被反复标成"卡车"
  · 紧贴停放车辆行走的人被标成 rider（先看脚在不在车上）

【复核标准】
  · 框形差几个像素不用改；"有没有标到"和"类别对不对"才是关键。
  · 被遮挡一半的目标仍然要标，看不见的部分不用猜。
========================================================
"""

OFFLINE_GUIDE = """========================================================
 离线包使用说明（{name} / {batch_id}，共 {count} 张）
========================================================

【最省事的用法：浏览器里标，零安装】
  1. 先把整个文件夹解压出来（别只解压一半）
  2. 双击 标注工具_离线版.html（用 Edge 或 Chrome 打开）
  3. 点左上角"① 选择批次文件夹"，选中本文件夹
     （就是同时含 images 与 labels 的那个）
  4. 空白处左键拖动 = 画框；点框 = 选中；拖四角 = 改大小；
     数字键 = 改类别；Delete = 删框；
     滚轮缩放；中键拖动或空格+左键拖动 = 平移；
     A / ← 上一张，D / → 下一张（自动保存）；R = 清骑车人
  5. 标完直接关窗口即可，标注已经写回本文件夹的 labels\\
     旧浏览器（没有"选择文件夹"能力）会自动退成下载模式：
     改完点 Ctrl+S 存这一张，或点"导出全部改动"批量下载 txt，
     再把 txt 放回 labels\\ 覆盖。

【也可以用别的标注工具】
  本包是标准 YOLO 格式：images\\ 是图，labels\\ 是同名 txt，
  classes.txt 是类别表。labelImg / X-AnyLabeling / Label Studio 都能直接打开。

【标完之后怎么交】
  把 labels\\ 整个文件夹打包发回给负责人（只有 txt，几十 KB）。
  负责人给了 打包复核结果.bat 的话，双击它输入批次号即可一键打包。

【三条硬规则（详细见 标注说明.txt）】
  1) 骑车的人标"非机动车驾驶员 rider"，不要标 person；纯步行的人才标 person
  2) 类别固定 {n_classes} 类，顺序不可改（见 classes.txt）
  3) 拿不准的框留着别删；标志牌 / 路桩 / 路灯 / 信号灯是静止类必须保留
========================================================
"""

# 类别中文名（内部名保持 ASCII，画面与说明用中文）
CLASS_DISPLAY = {
    "car": "汽车", "truck": "卡车", "bus": "公交", "nonmotor": "非机动车（含摩托车/自行车/三轮车）",
    "person": "行人", "sign": "标志牌", "bollard": "路桩", "roadlamp": "路灯",
    "trafficlight": "信号灯", "rider": "非机动车驾驶员", "motorcycle": "摩托车",
    "bicycle": "自行车", "tricycle": "三轮车",
}


def classes_block(source: Path) -> tuple[str, int]:
    """**从数据集的 classes.txt 生成类别说明**——避免说明文件与数据口径脱节
    （上一版硬编码"9 类"，而数据已经换成 10 类）。"""
    names = [n.strip() for n in (source / "classes.txt").read_text(encoding="utf-8").splitlines() if n.strip()]
    width = max((len(CLASS_DISPLAY.get(n, n)) for n in names), default=8) + 2
    lines = []
    for i, name in enumerate(names, start=1):
        label = CLASS_DISPLAY.get(name, name)
        lines.append(f"  {i:>2} {label:<{width}}{name}")
    return "\n".join(lines), len(names)


def load_order(source: Path, order_file: Path | None = None) -> tuple[list[str], list[str], dict[str, str]]:
    """返回（按优先级排好的图片名列表, 优先帧集合, 图片名→视频名）。"""
    image_dir = source / "images"
    names = sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not names:
        raise SystemExit(f"没有找到图片：{image_dir}")
    known = set(names)

    order: list[str] = []
    priority: list[str] = []
    order_file = order_file or (source / "review_order.txt")
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

echo 正在准备复核批次 {batch_id} ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_batch.ps1" ^
  -Batch "{batch_id}" -Repo "%BATCH_REPO%" -Branch "%BATCH_BRANCH%" ^
  -BatDir "%BATCH_DIR%" -DataName "%BATCH_NAME%"
echo.
echo 这一步结束后，双击 打开标注_批次{nn}.bat 开始复核。
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
if not exist "%PY%" (
  where python >nul 2>nul
  if errorlevel 1 (
    echo [缺少标注工具运行环境] 本机既没有项目的 Python 环境，也没装 Python。
    echo    · 装上 Python 与依赖：pip install PyQt5 opencv-python numpy
    echo    · 或者用任何 YOLO 格式标注工具直接打开：
    echo      %CD%\\%DATA%\\images  与  %CD%\\%DATA%\\labels
    echo      （规则见 %CD%\\%DATA%\\标注说明.txt，classes.txt 是类别表）
    pause
    exit /b 1
  )
  set "PY=python"
)

echo 正在打开 %DATA% 的复核标注...
echo 滚轮缩放 / 中键或空格拖动 / 拖角改框 / R 清理重叠行人框 / Ctrl+Z 撤销
echo 改完一帧按 D 或右方向键进入下一帧，进度实时写入 labels\\
echo.
"%PY%" tools\\annotate_zh.py "%DATA%"
if errorlevel 1 (
  echo.
  echo [提示] 标注工具没能启动。多半是缺少 PyQt5 / opencv-python：
  echo        "%PY%" -m pip install PyQt5 opencv-python numpy
  echo        或者换用任何 YOLO 格式标注工具打开上面的 images / labels 目录。
)
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
    ap.add_argument("--bats-only", action="store_true",
                    help="不重打 zip，只用现有 zip 重算清单与 bat（改 bat 模板后用）")
    ap.add_argument("--keep-ratio", type=float, default=0.0,
                    help="自留比例：按优先级顺序取**最前面**这个比例自己处理，"
                         "不放进外发包、也不打人工任务标记（默认 0）")
    ap.add_argument("--keep-out", default=None,
                    help="自留包的输出路径（默认 <out>/batch_self_keep.zip）")
    ap.add_argument("--order-file", default=None,
                    help="优先级顺序文件（source 目录里没有 review_order.txt 时指定）")
    ap.add_argument("--no-tag", action="store_true",
                    help="外发包也不打人工任务标记（默认会打）")
    args = ap.parse_args()

    source = (ROOT / args.source).resolve() if not Path(args.source).is_absolute() \
        else Path(args.source)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    order_file = None
    if args.order_file:
        order_file = Path(args.order_file)
        if not order_file.is_absolute():
            order_file = ROOT / order_file
    order, priority, video_of = load_order(source, order_file)

    # ---- 自留：按优先级取最前面 N%，不打外发标记 ----
    n_keep = int(round(len(order) * max(0.0, min(1.0, args.keep_ratio))))
    keep_frames, order = order[:n_keep], order[n_keep:]
    if keep_frames:
        print(f"自留 {len(keep_frames)} 帧（优先级最前 {args.keep_ratio:.0%}）；"
              f"外发 {len(order)} 帧")
    chunks = split_even(order, args.size)
    classes_txt = (source / "classes.txt").read_text(encoding="utf-8")
    block, n_classes = classes_block(source)
    stage_root = out_dir / "_stage"
    if stage_root.exists():
        shutil.rmtree(stage_root)

    keep_info = None      # 自留包信息，稍后填充（清单里要引用）
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
        "human_task": not args.no_tag,
        "task_type": "annotation-review",
        "self_keep": keep_info,
        "batches": [],
    }

    def build_pack(frames: list[str], pack_id: str, zip_path: Path, *,
                   tag: bool, nn: str) -> None:
        """打一个包。tag=True 时注入人工任务标记（只给外发包打）。"""
        stage = stage_root / pack_id
        (stage / "images").mkdir(parents=True)
        (stage / "labels").mkdir(parents=True)
        (stage / "classes.txt").write_text(classes_txt, encoding="utf-8")
        write_text(stage / "review_order.txt", "\n".join(frames) + "\n")
        write_text(stage / "标注说明.txt",
                   LABEL_GUIDE.format(name=args.name, batch_id=pack_id, count=len(frames),
                                      nn=nn, n_classes=n_classes, classes_block=block))
        write_text(stage / "离线包说明.txt",
                   OFFLINE_GUIDE.format(name=args.name, batch_id=pack_id, count=len(frames),
                                        n_classes=n_classes))
        # 类别 ↔ 权重口径：标哪几类、软件输出哪几类、这份标注喂给哪版权重
        note_src = TOOLS_DIR / SCHEMA_NOTE
        if note_src.exists():
            shutil.copy2(note_src, stage / SCHEMA_NOTE)
        else:
            print(f"提示：没找到 {SCHEMA_NOTE}，本包不含类别/权重口径说明")
        if tag:
            write_text(stage / NOTICE_FILENAME,
                       notice_text(f"\n本包批次号：{pack_id}    共 {len(frames)} 张\n"))
        tool_src = TOOLS_DIR / OFFLINE_TOOL
        if tool_src.exists():
            shutil.copy2(tool_src, stage / OFFLINE_TOOL)
        else:
            print(f"提示：没找到 {OFFLINE_TOOL}，这批不含离线标注工具")
        for name in frames:
            shutil.copy2(source / "images" / name, stage / "images" / name)
            sidecar = source / "labels" / (Path(name).stem + ".txt")
            if sidecar.exists():
                shutil.copy2(sidecar, stage / "labels" / sidecar.name)
        make_zip(stage, zip_path)

    # ---- 自留包（不打人工任务标记）----
    keep_info = None
    if keep_frames:
        keep_zip = Path(args.keep_out) if args.keep_out else (out_dir / "batch_self_keep.zip")
        if args.bats_only:
            # 该模式不重打 zip，但清单里仍要保留自留包的信息（否则会被冲成 null）
            if not keep_zip.exists():
                print(f"提示：--bats-only 下没找到自留包 {keep_zip}，清单里 self_keep 置空")
        else:
            build_pack(keep_frames, "self_keep", keep_zip, tag=False, nn="00")
        if keep_zip.exists():
            keep_info = {
                "zip": keep_zip.name,
                "frames": len(keep_frames),
                "bytes": keep_zip.stat().st_size,
                "sha256": sha256(keep_zip),
                "note": "优先级最前的自有部分，**未打人工任务标记**，供项目方自行复核",
                "priority_head": keep_frames[0],
                "priority_tail": keep_frames[-1],
                "videos": dict(sorted(Counter(video_of[n] for n in keep_frames).items())),
            }
            print(f"自留包: {len(keep_frames)} 帧, {keep_zip.stat().st_size / 1e6:.1f} MB"
                  f"{'（沿用已有包）' if args.bats_only else ''}")

    for idx, frames in enumerate(chunks, start=1):
        batch_id = f"batch_{idx:02d}"
        zip_path = out_dir / f"{batch_id}.zip"
        if args.bats_only:
            if not zip_path.exists():
                raise SystemExit(f"--bats-only 需要已存在的 {zip_path}，请先跑一次完整生成")
        else:
            build_pack(frames, batch_id, zip_path, tag=not args.no_tag, nn=f"{idx:02d}")
        videos = Counter(video_of[n] for n in frames)
        manifest["batches"].append({
            "id": batch_id,
            "zip": zip_path.name,
            "frames": len(frames),
            "bytes": zip_path.stat().st_size,
            "sha256": sha256(zip_path),
            "human_task": not args.no_tag,
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

    # 自留包信息是在清单字典构造之后才算出来的，必须回填，否则写成 null
    manifest["self_keep"] = keep_info
    write_text(out_dir / "manifest.json",
               json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if stage_root.exists():
        shutil.rmtree(stage_root)
    print(f"清单已写入 {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
