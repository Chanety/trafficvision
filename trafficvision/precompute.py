"""场景先验预计算：把"这个机位上有哪些固定结构"离线算一次，之后所有视频复用。

这是"把算力从在线搬到离线"这条技术路线的**离线那一半**：

    离线（本模块，每机位只跑一次）
        多提示词扫描 → 跨帧一致性投票 → 语义校验 → 写出 scene_prior.json
    在线（`trafficvision.cli --scene-prior <该文件>`）
        直接加载，跳过扫描与语义校验，只做一次查表注入

支持一次喂多段同机位视频并合并结果——喂得越多，投票统计越稳。

用法：
    python -m trafficvision.precompute \\
        --videos 机位A_1.mp4 机位A_2.mp4 \\
        --camera 路口A \\
        --out outputs/scene_prior/路口A.json

之后：
    python -m trafficvision.cli --video 任意一段.mp4 --horizon 1 --out out.mp4 \\
        --scene-prior outputs/scene_prior/路口A.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .config import (
    STATIC_CLASSES,
    STATIC_CONF,
    STATIC_IMGSZ,
    STATIC_MAP_MIN_FRAMES,
    STATIC_MAP_MIN_RATIO,
    STATIC_STRIDE,
)
from .staticmap import StaticObjectMap, _Cluster, iou
from .pipeline import _verify_static, probe_video, scan_static_channel


def _merge_entry(merged: list[dict], item: dict, iou_thr: float = 0.30) -> None:
    """把一段视频的确认结果并进总表：同类别且位置重叠的算同一个物理目标。"""
    for e in merged:
        if e["cls"] != item["cls"] or iou(e["box"], item["box"]) < iou_thr:
            continue
        # 位置取加权平均（谁看得多谁说话重），计数累加
        w1, w2 = e["visible"], item["visible"]
        tot = max(1, w1 + w2)
        e["box"] = [ (e["box"][i] * w1 + item["box"][i] * w2) / tot for i in range(4) ]
        e["seen"] += item["seen"]
        e["visible"] += item["visible"]
        e["ratio"] = round(e["seen"] / max(1, e["visible"]), 3)
        e["videos"].append(item.get("video"))
        # 语义校验取"更保守"的一个：只要有一路否掉就降级
        order = {"reject": 0, "review": 1, "keep": 2}
        if order.get(item.get("verify", "keep"), 2) < order.get(e.get("verify", "keep"), 2):
            e["verify"] = item.get("verify", "keep")
        return
    merged.append(dict(item, videos=[item.get("video")]))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="离线计算某个机位的场景先验（固定结构位置表）")
    ap.add_argument("--videos", nargs="+", required=True,
                    help="同机位的若干段视频（越多投票越稳）")
    ap.add_argument("--out", required=True, help="输出的场景先验 json 路径")
    ap.add_argument("--camera", default=None, help="机位名（写进先验文件便于追溯）")
    ap.add_argument("--classes", default=",".join(STATIC_CLASSES),
                    help=f"要固化的静态类别，默认 {'/'.join(STATIC_CLASSES)}")
    ap.add_argument("--weights", default=None,
                    help="开放词汇权重，默认取 trafficlab-3d/repo/yolov8s-worldv2.pt")
    ap.add_argument("--conf", type=float, default=STATIC_CONF)
    ap.add_argument("--imgsz", type=int, default=STATIC_IMGSZ)
    ap.add_argument("--stride", type=int, default=STATIC_STRIDE)
    ap.add_argument("--max-frames", type=int, default=0, help="每段最多处理多少帧（0=全部）")
    ap.add_argument("--device", default="0")
    ap.add_argument("--min-ratio", type=float, default=STATIC_MAP_MIN_RATIO)
    ap.add_argument("--min-frames", type=int, default=STATIC_MAP_MIN_FRAMES)
    ap.add_argument("--no-verify", action="store_true", help="跳过 CLIP 语义校验")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    weights = Path(args.weights) if args.weights else root / "trafficlab-3d" / "repo" / "yolov8s-worldv2.pt"
    if not weights.exists():
        raise SystemExit(f"找不到开放词汇权重：{weights}")

    classes = tuple(c.strip() for c in args.classes.split(",") if c.strip())
    merged: list[dict] = []
    frame_size = None
    per_video = []

    for video in args.videos:
        video = Path(video)
        if not video.exists():
            print(f"[跳过] 找不到视频 {video}")
            continue
        info = probe_video(video)
        frames_n = info["frames"]
        if args.max_frames:
            frames_n = min(frames_n, args.max_frames)
        frame_size = (info["width"], info["height"])
        print(f"扫描 {video.name}（{frames_n} 帧，{info['width']}×{info['height']}）…", flush=True)

        scan = scan_static_channel(video, info, weights, frames_n, args.device,
                                   args.conf, args.imgsz, args.stride, classes=classes)
        smap = StaticObjectMap(classes=classes, min_ratio=args.min_ratio,
                               min_frames=args.min_frames)
        for i in sorted(scan):
            smap.observe(i, scan[i], dynamics=[], span=args.stride)
        confirmed = smap.confirmed()

        if not args.no_verify and confirmed:
            decisions = _verify_static(confirmed, video, info["width"], info["height"],
                                       args.device)
            for c in confirmed:
                d, _why, _top = decisions.get(c["idx"], ("keep", "", ""))
                c["verify"] = d
        dropped = 0
        for c in confirmed:
            if c.get("verify") == "reject":
                dropped += 1
                continue
            _merge_entry(merged, {
                "cls": c["cls"], "box": [float(v) for v in c["box"]],
                "seen": c["seen"], "visible": c["visible"], "ratio": c["ratio"],
                "best_conf": c["best_conf"], "verify": c.get("verify", "keep"),
                "video": video.name,
            })
        per_video.append({"video": video.name, "candidates": smap.summary()["clusters"],
                          "confirmed": len(confirmed), "rejected_by_verify": dropped})
        print(f"  候选 {smap.summary()['clusters']} → 确认 {len(confirmed)}"
              f"（语义校验否掉 {dropped}）", flush=True)

    if not merged:
        raise SystemExit("没有任何位置通过一致性检验，先验未生成")

    smap = StaticObjectMap(classes=classes, min_ratio=args.min_ratio,
                           min_frames=args.min_frames)
    for m in merged:
        c = _Cluster(cls=m["cls"], boxes=[m["box"]], forced=True)
        c.meta = {"ratio": m["ratio"], "seen": m["seen"], "visible": m["visible"],
                  "best_conf": m["best_conf"], "verify": m.get("verify", "keep")}
        smap.clusters.append(c)
    path = smap.save_prior(args.out, camera=args.camera, frame_size=frame_size,
                           extra={"source_videos": per_video,
                                  "conf": args.conf, "imgsz": args.imgsz,
                                  "stride": args.stride, "verify": not args.no_verify})

    by_class: dict[str, int] = {}
    for m in merged:
        by_class[m["cls"]] = by_class.get(m["cls"], 0) + 1
    print(f"\n场景先验已写出：{path}")
    print(f"  共 {len(merged)} 个位置：" +
          "  ".join(f"{k}={v}" for k, v in sorted(by_class.items())))
    print(f"  覆盖 {len(per_video)} 段视频，分辨率 {frame_size}")
    print(f"\n使用方式：python -m trafficvision.cli --video X.mp4 --horizon 1 --out Y.mp4 "
          f"--scene-prior {path}")


if __name__ == "__main__":
    main()
