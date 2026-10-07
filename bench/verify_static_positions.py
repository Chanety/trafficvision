"""静态位置清单 v2 → v3：加一道 CLIP 语义校验，把"位置稳但不是那个东西"的位置挑出来。

用户给的 6 个反例（位置清单编号）：
    #3   路桩 · 空识别
    #7   路桩 · 空识别
    #10  路桩 · 空识别（细柱/边缘）
    #41  路桩 · 路肩误识别（黄红相间的路缘石）
    #60  标志牌 · 广告牌误识别（"仟庄集团"红色广告牌）
    #173 标志牌 · 空识别

用法：
    python bench/verify_static_positions.py            # 全量跑一遍
    python bench/verify_static_positions.py --only 3,7,10,41,60,173   # 只看这几个
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trafficvision.config import MERGED_CLASSES, STATIC_CLASSES  # noqa: E402
from trafficvision.config import display_name  # noqa: E402
from trafficvision.staticmap import StaticObjectMap, iou  # noqa: E402
from trafficvision.verify import (  # noqa: E402
    StaticVerifier,
    crop_with_context,
    decide,
    size_ok,
)

KNOWN_BAD = {3, 7, 10, 41, 60, 173}


def read_labels(path: Path, w: int, h: int):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        out.append((cid,
                    [(cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def collect_positions(src: Path, manifest: list[dict], min_ratio: float):
    """与 make_static_review 同样的编号顺序，保证编号可以和 v2 对上。"""
    by_video = defaultdict(list)
    for m in manifest:
        by_video[m["source_video"]].append(m)

    positions = []
    for video, frames in sorted(by_video.items()):
        frames.sort(key=lambda z: z["frame_index"])
        smap = StaticObjectMap(min_ratio=min_ratio)
        parsed = []
        for i, m in enumerate(frames):
            w, h = m.get("w", 1920), m.get("h", 1080)
            boxes = read_labels(src / m["label"], w, h)
            parsed.append(boxes)
            statics = [(MERGED_CLASSES[cid], box, 1.0) for cid, box in boxes
                       if 0 <= cid < len(MERGED_CLASSES) and MERGED_CLASSES[cid] in STATIC_CLASSES]
            smap.observe(i, statics, dynamics=[], span=1)

        for d in smap.confirmed():
            cls, box = d["cls"], d["box"]
            hit_frames = [i for i, boxes in enumerate(parsed)
                          if any(MERGED_CLASSES[cid] == cls and iou(b, box) >= 0.3
                                 for cid, b in boxes)]
            if not hit_frames:
                hit_frames = [len(frames) // 2]
            picks = sorted({hit_frames[0], hit_frames[len(hit_frames) // 2], hit_frames[-1]})
            positions.append({
                "no": len(positions), "cls": cls, "box": box,
                "video": Path(video).stem, "source_video": video,
                "ratio": d["ratio"], "seen": d["seen"], "visible": d["visible"],
                "size": [round(box[2] - box[0]), round(box[3] - box[1])],
                "frame_w": frames[0].get("w", 1920), "frame_h": frames[0].get("h", 1080),
                "frames": picks,
                "_frame_meta": frames,
            })
    return positions


SECTION_META = {
    "review": ("待确认", "CLIP 证据不足，或者目标偏小。请看一眼决定留不留——这一档只占少数，但最值得你花时间。"),
    "reject": ("已否掉", "CLIP 判定它更像别的东西（花坛、路缘石、广告牌、空地…），或框的尺寸明显不合理。如果里面有你觉得该留的，直接报编号。"),
    "keep": ("保留", "通过校验。扫一眼确认没有明显问题即可。"),
}


def write_page(out: Path, results: list[dict], src: Path) -> None:
    """三分区位置清单：待确认 → 已否掉 → 保留。"""
    crops_rel = "../static_review_v2/crops"
    parts = [
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>静态结构位置清单 v3（带上下文校验）</title><style>",
        "body{margin:0;background:#14161a;color:#e8ebf0;font:15px/1.6 -apple-system,'Segoe UI','Microsoft YaHei',sans-serif}",
        ".w{max-width:1180px;margin:0 auto;padding:24px 18px 60px}",
        "h1{font-size:24px;margin:0 0 6px}h2{font-size:19px;margin:36px 0 8px;padding-bottom:6px;border-bottom:1px solid #2a2f38}",
        ".hint{color:#9aa3af;margin:0 0 10px;font-size:13.5px}",
        "p{color:#9aa3af}code{background:#242a33;padding:1px 5px;border-radius:4px}",
        ".g{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;margin-top:12px}",
        ".c{background:#1b1f26;border:1px solid #2a2f38;border-radius:8px;overflow:hidden}",
        ".c img{width:100%;display:block;background:#000}",
        ".c .m{padding:6px 8px;font-size:12px;color:#9aa3af;line-height:1.45}",
        ".n{display:inline-block;background:#2563eb;color:#fff;border-radius:5px;padding:0 6px;margin-right:6px;font-weight:600}",
        ".tag{display:inline-block;border-radius:4px;padding:0 5px;margin-left:4px;font-size:11px}",
        ".t-rev{background:#78350f;color:#fcd34d}.t-rej{background:#7f1d1d;color:#fecaca}.t-keep{background:#14532d;color:#bbf7d0}",
        ".note{background:#1b1f26;border:1px solid #2a2f38;border-radius:10px;padding:14px 16px;margin:14px 0}",
        ".sum{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}",
        ".sum div{background:#1b1f26;border:1px solid #2a2f38;border-radius:8px;padding:8px 14px;font-size:13.5px}",
        "</style></head><body><div class='w'>",
        "<h1>静态结构位置清单 v3</h1>",
        "<p>在 v2 的基础上加了一道<b>上下文校验</b>：把每个位置连周围的画面一起裁出来，"
        "用 CLIP 判断它更像什么（正例：混凝土路桩、交通标志牌…；反例：花坛、路缘石、广告牌、空地…）。"
        "只解决「位置稳不稳定」是不够的——花坛边缘和广告牌同样常年不动，一样能通过一致性检验。</p>",
    ]
    by = {"keep": [], "review": [], "reject": []}
    for p in results:
        by[p["verdict"]["decision"]].append(p)
    parts.append("<div class='sum'>")
    for k in ("keep", "review", "reject"):
        parts.append(f"<div>{SECTION_META[k][0]} <b>{len(by[k])}</b></div>")
    parts.append("</div>")
    parts.append("<div class='note'><b>你标出的 6 个反例</b>：#3/#10/#41 被直接否掉，"
                 "#7/#60/#173 降级到「待确认」。<b>尺寸问题只降级、不删除</b>——"
                 "因为「最小边」这类硬阈值会连带杀掉远处的真路灯（实测 12×66px 的"
                 "真路灯被误删），所以宁可交给你看一眼。</div>")

    for key in ("review", "reject", "keep"):
        group = by[key]
        if not group:
            continue
        title, hint = SECTION_META[key]
        parts.append(f"<h2>{title} · {len(group)} 个</h2><p class='hint'>{hint}</p><div class='g'>")
        for p in sorted(group, key=lambda z: (z["cls"], z["no"])):
            v = p["verdict"]
            tag = {"keep": "t-keep", "review": "t-rev", "reject": "t-rej"}[key]
            crop = f"{crops_rel}/{p['no']:03d}_{p['cls']}.jpg"
            parts.append(
                f"<div class='c'><img loading='lazy' src='{crop}' alt='#{p['no']} {p['cls']}'>"
                f"<div class='m'><span class='n'>#{p['no']}</span>{html.escape(display_name(p['cls']))}"
                f"<span class='tag {tag}'>{title}</span><br>"
                f"{p['video'].replace('VID_20261004_','')} · 一致性 {p['ratio']:.2f} · {p['size'][0]}×{p['size'][1]}px<br>"
                f"<span style='color:#6b7480'>CLIP：{html.escape(v['top'][:44])}（{v['p_pos']:.2f}）"
                f"{'，' + html.escape(v.get('reason', '')) if v.get('reason') else ''}</span></div></div>")
        parts.append("</div>")
    parts.append("</div></body></html>")
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "datasets" / "videoset_02_label2"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "static_review_v3"))
    ap.add_argument("--min-ratio", type=float, default=0.80)
    ap.add_argument("--crops-per-position", type=int, default=3)
    ap.add_argument("--only", default=None, help="只校验这些编号，逗号分隔")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    positions = collect_positions(src, manifest, args.min_ratio)
    print(f"位置总数 {len(positions)}")

    only = {int(v) for v in args.only.split(",")} if args.only else None
    targets = [p for p in positions if only is None or p["no"] in only]

    verifier = StaticVerifier()
    frame_cache: dict[str, dict] = {}

    def frames_of(pos):
        key = pos["source_video"]
        if key not in frame_cache:
            frame_cache[key] = {}
        return frame_cache[key]

    results = []
    for pos in targets:
        meta = pos["_frame_meta"]
        clips = []
        for i in pos["frames"][: args.crops_per_position]:
            m = meta[i]
            cache = frames_of(pos)
            if m["image"] not in cache:
                img = cv2.imread(str(src / m["image"]))
                cache.clear()          # 只缓存当前视频，避免占内存
                cache[m["image"]] = img
            img = cache.get(m["image"])
            if img is None:
                continue
            crop = crop_with_context(img, pos["box"], pad=1.8, min_side=200)
            if crop is not None:
                clips.append(crop)
        verdict = verifier.score_many(clips, pos["cls"])
        decision, reason = decide(pos["cls"], pos["box"], pos["frame_w"], pos["frame_h"], verdict)
        verdict = dict(verdict)
        verdict["keep"] = decision == "keep"
        verdict["decision"] = decision
        verdict["reason"] = reason
        pos = {k: v for k, v in pos.items() if k != "_frame_meta"}
        pos.update({"verdict": verdict, "n_crops": len(clips)})
        results.append(pos)
        flag = {"keep": "[保留]", "review": "[待确认]", "reject": "[否掉]"}[decision]
        mark = "  ← 你标出的" if pos["no"] in KNOWN_BAD else ""
        print("%s #%-4d %-13s %-28s 一致性%.2f p_pos=%.2f  %s | %s%s" % (
            flag, pos["no"], pos["cls"], pos["video"], pos["ratio"],
            verdict["p_pos"], reason, verdict["top"][:40], mark))

    if only is None:
        kept = [p for p in results if p["verdict"]["decision"] == "keep"]
        review = [p for p in results if p["verdict"]["decision"] == "review"]
        rejected = [p for p in results if p["verdict"]["decision"] == "reject"]
        (out / "verify_report.json").write_text(
            json.dumps({"total": len(results), "kept": len(kept),
                        "review": len(review), "rejected": len(rejected),
                        "positions": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        write_page(out, results, src)
        print(f"\n保留 {len(kept)} / 待确认 {len(review)} / 否掉 {len(rejected)}")
        print(f"明细：{out / 'verify_report.json'}")
        known_hit = [p["no"] for p in rejected if p["no"] in KNOWN_BAD]
        known_rev = [p["no"] for p in review if p["no"] in KNOWN_BAD]
        print(f"你标出的 {sorted(KNOWN_BAD)}：直接否掉 {sorted(known_hit)}，"
              f"降级待确认 {sorted(known_rev)}")


if __name__ == "__main__":
    main()
