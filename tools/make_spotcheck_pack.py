"""抽查包：从"尚未复核"的帧里随机抽 N 帧，供快速抽检（不做逐张复核）。

背景（用户 2026-10-09）：
  自留部分前 40 帧逐张检查后确认"精确度没问题"，因此剩余帧改为**抽样抽检**：
  "剩下的重新抽样 50 张就可以了"。

做法：
  · 从全量帧里排除"已逐张看过的"（默认排除自留名单的前 40 帧）；
  · 按视频**分层抽样**（每段视频按帧数比例分配名额），避免抽到同一段视频扎堆；
  · 固定随机种子，保证结果可复现；
  · 打成与批次包同结构的 zip（含离线标注工具），双击即可看。

用法：
    python tools/make_spotcheck_pack.py --n 50 \
        --exclude dataset_batches/自留帧名单.txt --exclude-first 40
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
OFFLINE_TOOL = "标注工具_离线版.html"


def main() -> None:
    ap = argparse.ArgumentParser(description="从尚未复核的帧里随机抽 N 帧做抽检包")
    ap.add_argument("--source", default="datasets/videoset_02_label3")
    ap.add_argument("--out", default="dataset_batches/spotcheck50.zip")
    ap.add_argument("--list-out", default="dataset_batches/抽样50名单.txt")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--exclude", default="dataset_batches/自留帧名单.txt",
                    help="已逐张看过的帧名单（每行一个文件名）")
    ap.add_argument("--exclude-first", type=int, default=40,
                    help="只排除该名单的前 N 行（默认 40）")
    ap.add_argument("--batch-id", default="spotcheck")
    args = ap.parse_args()

    source = (ROOT / args.source) if not Path(args.source).is_absolute() else Path(args.source)
    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    list_out = (ROOT / args.list_out) if not Path(args.list_out).is_absolute() else Path(args.list_out)

    images = sorted(p.name for p in (source / "images").iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"没有图片：{source / 'images'}")

    exclude = set()
    ex_path = Path(args.exclude)
    if not ex_path.is_absolute():
        ex_path = ROOT / ex_path
    if ex_path.exists():
        lines = [ln.strip() for ln in ex_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        exclude = set(lines[: args.exclude_first] if args.exclude_first > 0 else lines)

    pool = [n for n in images if n not in exclude]
    print(f"总帧 {len(images)}，排除已看过 {len(exclude)}，可抽样 {len(pool)}")
    if len(pool) <= args.n:
        picked = pool
    else:
        # 按视频分层：每段按帧数比例分配，保证跨视频覆盖
        by_video = defaultdict(list)
        for n in pool:
            by_video[n.rsplit("_f", 1)[0]].append(n)
        rng = random.Random(args.seed)
        picked = []
        quota = {k: max(1, round(args.n * len(v) / len(pool))) for k, v in by_video.items()}
        while sum(quota.values()) > args.n:            # 配额超了就按最小者削减
            k = min(quota, key=lambda x: quota[x])
            quota[k] -= 1
        while sum(quota.values()) < args.n:            # 不足就从最大的补
            k = max(quota, key=lambda x: len(by_video[x]) - quota[x])
            if quota[k] >= len(by_video[k]):
                break
            quota[k] += 1
        for v, frames in sorted(by_video.items()):
            picked += rng.sample(frames, min(quota[v], len(frames)))
        picked = sorted(picked)

    by_video_picked = defaultdict(int)
    for n in picked:
        by_video_picked[n.rsplit("_f", 1)[0]] += 1
    print(f"抽到 {len(picked)} 帧，覆盖 {len(by_video_picked)} 段视频（种子 {args.seed}）")
    for v, c in sorted(by_video_picked.items()):
        print(f"   {v}  {c} 帧")

    # 打 zip
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = out.parent / "_stage_spotcheck"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "images").mkdir(parents=True)
    (stage / "labels").mkdir(parents=True)
    (stage / "classes.txt").write_text(
        (source / "classes.txt").read_text(encoding="utf-8"), encoding="utf-8")
    (stage / "review_order.txt").write_text("\n".join(picked) + "\n", encoding="utf-8")
    (stage / "抽检说明.txt").write_text(
        "这是一次【抽样抽检】，不是逐张复核。\n\n"
        f"从尚未复核的 {len(pool)} 帧里按视频分层随机抽了 {len(picked)} 帧（种子 {args.seed}，可复现）。\n"
        "请重点看两类（此前抽检反馈提到的）：\n"
        "  1) 贴边的静态目标（路桩/路灯/信号灯）框得对不对、有没有把画面外也算进去；\n"
        "  2) 行人与非机动车驾驶员的区分（骑车的人应标 rider，纯步行才是 person）。\n\n"
        "操作：双击 标注工具_离线版.html → 点「选择批次文件夹」选本文件夹。\n"
        "数字键改类别（1-9，0=第 10 类 rider），Delete 删框，A/D 翻页自动保存。\n",
        encoding="utf-8")
    tool_src = TOOLS_DIR / "dataset_batches" / OFFLINE_TOOL
    if tool_src.exists():
        shutil.copy2(tool_src, stage / OFFLINE_TOOL)
    for n in picked:
        shutil.copy2(source / "images" / n, stage / "images" / n)
        sidecar = source / "labels" / (Path(n).stem + ".txt")
        if sidecar.exists():
            shutil.copy2(sidecar, stage / "labels" / sidecar.name)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for item in sorted(stage.rglob("*")):
            if item.is_file():
                zf.write(item, item.relative_to(stage).as_posix())
    shutil.rmtree(stage)

    list_out.parent.mkdir(parents=True, exist_ok=True)
    list_out.write_text("\n".join(picked) + "\n", encoding="utf-8")
    print(f"\n抽检包：{out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"名单：{list_out}")


if __name__ == "__main__":
    main()
