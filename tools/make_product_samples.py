"""产品抽检集生成器：抽 N 帧，输出「软件实际画出来的那一帧」+ 对应原始帧。

为什么要有这个：项目的精度指标都是脚本算的，没有人真正逐帧看过软件画出来的画。
这个脚本把"软件此刻看到的画面"原样导出成一帧一图，人工扫一遍就能指出问题，
而且每张带编号，回报问题时可以说"第 7 张"。

产出（out/batch_NN/）：
    01.png … 20.png        软件输出画面（检测框、类型、预判虚框、静态记忆、HUD）
    原始帧/01.png …        同一帧的原始画面（对照用，看有没有漏掉的目标）
    总览.jpg               20 张拼成 4×5 大图，一眼扫完
    索引.md                每张的来源视频、帧号、时间点、目标数与类别分布

用法：
    python tools/make_product_samples.py --batch 1
    python tools/make_product_samples.py --batch 2 --videos 4 --per-video 5

批次是按视频轮换的：第 1 批取前 4 段视频，第 2 批顺延 4 段，
所以连续几批不会重复看同一段素材。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ultralytics 默认往用户目录写配置，本机会失败；必须在 import 前设好
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
CELL_W = 640          # 总览图每格宽度
COLS = 4              # 总览图列数


def pick_videos(videos: list[Path], batch: int, n: int) -> list[Path]:
    """按批次轮换取 n 段视频，取完自动接回开头。"""
    if n >= len(videos):
        return videos
    start = ((batch - 1) * n) % len(videos)
    return [videos[(start + i) % len(videos)] for i in range(n)]


def sample_indices(max_frames: int, offset: int, n: int, rng: np.random.Generator) -> list[int]:
    """在 [offset+warmup, max_frames-margin] 内均匀取 n 个互不相同的帧号。"""
    lo = offset + 15                      # 前 15 帧是跟踪热机期，跳过
    hi = max_frames - 6                   # 末尾留给预判真值
    if hi - lo < n:
        lo = max(0, offset)
        hi = max_frames
    if hi <= lo:
        return []
    slots = np.linspace(lo, hi - 1, n * 3).astype(int)
    picked = sorted({int(v) for v in rng.choice(slots, size=min(n, len(slots)), replace=False)})
    while len(picked) < n:                # 取重了就补齐
        cand = int(rng.integers(lo, hi))
        if cand not in picked:
            picked.append(cand)
    return sorted(picked)[:n]


def frame_stats(frame_obj: list[dict], horizon: float) -> dict:
    """一帧的统计：目标数、按类分布、可预判数、静态记忆数、遮挡续推数。"""
    live = [o for o in frame_obj if not o.get("hidden")]
    dynamics = [o for o in live if not o.get("memory")]
    return {
        "total": len(dynamics),
        "predictable": sum(1 for o in dynamics if o.get("predictable")),
        "memory": sum(1 for o in live if o.get("memory")),
        "coasting": sum(1 for o in dynamics if o.get("coasting")),
        "by_class": Counter(o["cls"] for o in dynamics),
    }


def make_contact_sheet(samples: list[dict], out_path: Path, cols: int = COLS) -> None:
    """把 N 张输出画面拼成一张总览，每格带编号与来源。"""
    tiles = []
    for s in samples:
        img = cv2.imread(str(s["out_png"]))
        if img is None:
            continue
        h = int(img.shape[0] * CELL_W / img.shape[1])
        tile = cv2.resize(img, (CELL_W, h), interpolation=cv2.INTER_AREA)
        bar = np.full((34, CELL_W, 3), 24, np.uint8)
        text = (f'{s["no"]:02d}  {s["video"]}  f{s["frame"]}'
                f'  目标 {s["stats"]["total"]}（预判 {s["stats"]["predictable"]}）')
        cv2.putText(bar, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, tile]))

    if not tiles:
        return
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * th, cols * tw, 3), 12, np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th:r * th + tile.shape[0], c * tw:c * tw + tile.shape[1]] = tile
    # 总览图太宽会看不清，统一压到可在一屏里显示的宽度
    if sheet.shape[1] > 2600:
        scale = 2600 / sheet.shape[1]
        sheet = cv2.resize(sheet, (2600, int(sheet.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> None:
    ap = argparse.ArgumentParser(description="生成产品抽检集（默认 20 张一批）")
    ap.add_argument("--source", default="datasets/videoset_02_origin", help="视频目录")
    ap.add_argument("--out", default="outputs/spotcheck", help="输出根目录")
    ap.add_argument("--batch", type=int, default=1, help="批次号（决定抽哪几段视频）")
    ap.add_argument("--videos", type=int, default=4, help="本批用几段视频")
    ap.add_argument("--per-video", type=int, default=5, help="每段抽几帧")
    ap.add_argument("--frames", type=int, default=150, help="每段处理多少帧（窗口长度）")
    ap.add_argument("--offset", type=int, default=0, help="从第几帧开始处理")
    ap.add_argument("--horizon", type=float, default=1.0, help="预判梯度（秒）")
    ap.add_argument("--weights", default="outputs/train/v7/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=None, help="抽样随机种子（默认按批次取）")
    ap.add_argument("--no-static-map", action="store_true", help="关闭静态目标地图（对照用）")
    args = ap.parse_args()

    from trafficvision.pipeline import probe_video, run_clip

    source = (ROOT / args.source) if not Path(args.source).is_absolute() else Path(args.source)
    videos = sorted(p for p in source.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if not videos:
        raise SystemExit(f"没有视频：{source}")

    picked_videos = pick_videos(videos, args.batch, args.videos)
    out_dir = ((ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)) / f"batch_{args.batch:02d}"
    raw_dir = out_dir / "原始帧"
    tmp_dir = out_dir / "_tmp"
    for d in (out_dir, raw_dir, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed if args.seed is not None else 20261010 + args.batch)
    samples: list[dict] = []
    no = 0

    print(f"批次 {args.batch}：{len(picked_videos)} 段视频 × 每段 {args.per_video} 帧"
          f"（每段处理 {args.frames} 帧，起始第 {args.offset} 帧）")
    for video in picked_videos:
        info = probe_video(video)
        max_frames = min(args.offset + args.frames, int(info["frames"]))
        print(f"\n▶ {video.name}（{info['width']}×{info['height']} {info['fps']:.1f}fps，"
              f"处理前 {max_frames} 帧）")
        clip_mp4 = tmp_dir / f"{video.stem}.mp4"
        clip_json = tmp_dir / f"{video.stem}.json"
        run_clip(
            video=video, horizon=args.horizon, out_video=clip_mp4,
            weights=str((ROOT / args.weights) if not Path(args.weights).is_absolute() else Path(args.weights)),
            imgsz=args.imgsz, conf=0.25, device="0",
            out_json=clip_json, render=True, max_frames_override=max_frames,
            static_map=not args.no_static_map,
            # 只处理一小段窗口，算出来的场景先验质量不可靠，
            # 不落盘，免得污染整段视频的真实先验
            scene_prior="off",
        )
        payload = json.loads(clip_json.read_text(encoding="utf-8"))
        frames = payload["frames"]
        for idx in sample_indices(max_frames, args.offset, args.per_video, rng):
            no += 1
            stat = frame_stats(frames[idx], args.horizon)
            out_png = out_dir / f"{no:02d}.png"
            raw_png = raw_dir / f"{no:02d}.png"

            cap = cv2.VideoCapture(str(clip_mp4))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, rendered = cap.read()
            cap.release()
            if not ok:
                print(f"    ! 第 {idx} 帧读取失败，跳过")
                no -= 1
                continue
            cv2.imwrite(str(out_png), rendered)

            cap = cv2.VideoCapture(str(video))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, raw = cap.read()
            cap.release()
            if ok:
                cv2.imwrite(str(raw_png), raw)

            samples.append({
                "no": no, "video": video.stem, "frame": idx,
                "t": round(idx / float(info["fps"]), 2), "stats": stat,
                "out_png": out_png, "raw_png": raw_png if ok else None,
            })
            print(f"    #{no:02d}  f{idx:<4} 目标 {stat['total']:>2}"
                  f"（可预判 {stat['predictable']}，记忆 {stat['memory']}，续推 {stat['coasting']}）")

    if not samples:
        raise SystemExit("没有抽到任何帧")

    make_contact_sheet(samples, out_dir / "总览.jpg")

    lines = [
        f"# 产品抽检集 · 批次 {args.batch:02d}",
        "",
        f"共 {len(samples)} 张。输出画面在 `{out_dir.name}/NN.png`，"
        f"同一帧的原始画面在 `原始帧/NN.png`。",
        "看的时候按编号说问题即可（例如「07 公交漏了」「12 那根路桩是误检」）。",
        "",
        "| # | 视频 | 帧号 | 时间 | 目标 | 可预判 | 记忆框 | 续推 | 类别分布 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in samples:
        by = "、".join(f"{k}{v}" for k, v in s["stats"]["by_class"].most_common()) or "—"
        lines.append(
            f'| {s["no"]:02d} | {s["video"]} | {s["frame"]} | {s["t"]}s | '
            f'{s["stats"]["total"]} | {s["stats"]["predictable"]} | '
            f'{s["stats"]["memory"]} | {s["stats"]["coasting"]} | {by} |'
        )
    lines += [
        "",
        "> 画面里的标记：实线框＝检测到的目标（`#编号` 是跟踪 ID）；"
        "虚线框＋箭头＝该目标的预判框；密虚线框标「·记忆」＝检测器这一帧没输出、"
        "由整段视频的跨帧投票补回来的静态目标；灰虚线框标「遮挡续推」＝目标当前被挡住，"
        "框是推算出来的。",
    ]
    (out_dir / "索引.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n输出目录：{out_dir}")
    print(f"  输出画面 {len(samples)} 张、原始帧对照 {len(samples)} 张、总览.jpg、索引.md")


if __name__ == "__main__":
    main()
