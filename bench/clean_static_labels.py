"""清洗数据集里的静态类标注：跨帧一致性投票 + CLIP 语义校验，写出干净副本。

为什么必须前置（用户 2026-10-09 决策："清洗工作前置"）：
  `videoset_02_label2` 的静态类来自提示词集成 + 低阈值，召回优先，
  里面混着**红色招牌被当成信号灯**这类稳定误检。v9 实测信号灯 AP 只有 0.232，
  而信号灯是验证集里实例最多的类（625 个）——用脏标注继续重训只会把误检学进权重。

清洗分两级（与产品里那套完全一致，只是这里作用于标注）：
  1. 跨帧一致性投票：非遮挡帧里 ≥80% 被识别为同一类 → 确认为静态位置；
  2. CLIP 语义校验：把位置连上下文裁出来判断"到底是什么"，明确不像的删掉。

判决原则与产品一致——**偏向保留**：只有证据确凿的才删，拿不准的降级为"待确认"
（保留在数据里，但列进复核清单）。

用法：
    python bench/clean_static_labels.py \\
        --src datasets/videoset_02_label2 --out datasets/videoset_02_label3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trafficvision.config import MERGED_CLASSES, STATIC_CLASSES  # noqa: E402
from trafficvision.staticmap import StaticObjectMap, iou  # noqa: E402
from trafficvision.verify import StaticVerifier, crop_with_context, decide  # noqa: E402


def read_labels(path: Path, w: int, h: int):
    """返回 [(cid, box_px)]。"""
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            out.append((cid, [(cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def write_labels(path: Path, boxes, w: int, h: int) -> None:
    lines = []
    for cid, b in boxes:
        x1, y1, x2, y2 = b
        lines.append(f"{cid} {((x1 + x2) / 2) / w:.6f} {((y1 + y2) / 2) / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "datasets" / "videoset_02_label2"))
    ap.add_argument("--out", default=str(ROOT / "datasets" / "videoset_02_label3"))
    ap.add_argument("--min-ratio", type=float, default=0.80)
    ap.add_argument("--min-frames", type=int, default=4)
    ap.add_argument("--no-verify", action="store_true", help="只做投票清洗，跳过 CLIP 校验")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))

    by_video = defaultdict(list)
    for m in manifest:
        by_video[m["source_video"]].append(m)
    for v in by_video:
        by_video[v].sort(key=lambda z: z["frame_index"])

    verifier = None if args.no_verify else StaticVerifier()
    removed = Counter()          # 被删掉的静态框（按类）
    kept_static = Counter()
    review = []
    stats = []

    for video, frames in sorted(by_video.items()):
        # ---- 第一级：跨帧一致性投票 ----
        smap = StaticObjectMap(min_ratio=args.min_ratio, min_frames=args.min_frames)
        parsed = []
        for i, m in enumerate(frames):
            w, h = m.get("w", 1920), m.get("h", 1080)
            boxes = read_labels(src / m["label"], w, h)
            parsed.append(boxes)
            statics = [(MERGED_CLASSES[cid], b, 1.0) for cid, b in boxes
                       if MERGED_CLASSES[cid] in STATIC_CLASSES]
            # dynamics 必须来自同帧的**非静态**标注，否则遮挡不计入可见性
            dynamics = [{"bbox": b} for cid, b in boxes
                        if MERGED_CLASSES[cid] not in STATIC_CLASSES]
            smap.observe(i, statics, dynamics=dynamics, span=1)

        confirmed = smap.confirmed()
        allowed: list[dict] = []
        if not confirmed:
            print(f"  [警告] {video}：无位置通过一致性检验，该段静态标注全部保留待复核")
        else:
            # ---- 第二级：CLIP 语义校验 ----
            if verifier is not None:
                for c in confirmed:
                    hit = [i for i, bx in enumerate(parsed)
                           if any(MERGED_CLASSES[cid] == c["cls"] and iou(b, c["box"]) >= 0.3
                                  for cid, b in bx)]
                    picks = sorted({hit[0], hit[len(hit) // 2], hit[-1]}) if hit else []
                    crops = []
                    for i in picks[:3]:
                        img = cv2.imread(str(src / frames[i]["image"]))
                        if img is None:
                            continue
                        crop = crop_with_context(img, c["box"])
                        if crop is not None:
                            crops.append(crop)
                    verdict = verifier.score_many(crops, c["cls"])
                    d, why = decide(c["cls"], c["box"], frames[0].get("w", 1920),
                                    frames[0].get("h", 1080), verdict)
                    c["verify"] = d
                    c["reason"] = why
                    c["top"] = verdict.get("top", "")
            else:
                for c in confirmed:
                    c["verify"], c["reason"], c["top"] = "keep", "", ""

            allowed = [c for c in confirmed if c.get("verify") != "reject"]
            for c in allowed:
                kept_static[c["cls"]] += 1
                if c.get("verify") == "review":
                    review.append({"video": Path(video).stem, "cls": c["cls"],
                                   "box": [round(v) for v in c["box"]],
                                   "ratio": c["ratio"], "reason": c.get("reason", ""),
                                   "clip_top": c.get("top", "")})
            for c in confirmed:
                if c.get("verify") == "reject":
                    removed[c["cls"]] += 1

        # ---- 写出：删掉被否定的静态框 ----
        for i, m in enumerate(frames):
            w, h = m.get("w", 1920), m.get("h", 1080)
            kept = []
            for cid, b in parsed[i]:
                name = MERGED_CLASSES[cid]
                if name in STATIC_CLASSES and confirmed:
                    ok = any(c["cls"] == name and c.get("verify") != "reject"
                             and iou(b, c["box"]) >= 0.3 for c in confirmed)
                    if not ok:
                        continue
                kept.append((cid, b))
            write_labels(out / "labels" / f"{Path(m['image']).stem}.txt", kept, w, h)
            dst_img = out / "images" / Path(m["image"]).name
            if not dst_img.exists():
                try:
                    os.link(src / m["image"], dst_img)
                except OSError:
                    shutil.copyfile(src / m["image"], dst_img)

        summary = smap.summary()
        n_removed = sum(1 for c in confirmed if c.get("verify") == "reject")
        stats.append({"video": Path(video).stem, "candidates": summary["clusters"],
                      "confirmed": len(confirmed), "kept": len(allowed),
                      "removed": n_removed})
        print(f"  {Path(video).name}: 候选 {summary['clusters']} → 确认 {len(confirmed)}"
              f" → 保留 {len(allowed)}（删 {n_removed}）")

    for name in ("classes.txt", "data.yaml"):
        if (src / name).exists():
            (out / name).write_text((src / name).read_text(encoding="utf-8"), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    (out / "clean_report.json").write_text(json.dumps({
        "src": str(src), "out": str(out), "min_ratio": args.min_ratio,
        "verify": not args.no_verify,
        "removed_static_positions": dict(removed),
        "kept_static_positions": dict(kept_static),
        "review_positions": review,
        "per_video": stats,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 清洗结果（位置级）===")
    for k in sorted(set(removed) | set(kept_static)):
        print(f"  {k:<13} 保留 {kept_static.get(k, 0):>3}  删除 {removed.get(k, 0):>3}")
    print(f"  待确认（保留在数据里，需人工看一眼）：{len(review)} 个")
    print(f"\n干净副本：{out}")
    print(f"明细：{out / 'clean_report.json'}")


if __name__ == "__main__":
    main()
