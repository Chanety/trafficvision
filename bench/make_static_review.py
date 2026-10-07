"""静态结构"位置清单"：把要复核的单位从"332 张图"压成"几十个位置"。

用户没法逐张标注，但可以对着几十张裁剪小图说"这几个不对"。
本脚本把静态类（标志牌/路桩/路灯/信号灯）在整段视频里做跨帧投票，
把通过 ≥80% 一致性检验的位置逐个裁出来编号，生成一个离线清单页。

这样复核的粒度就从 332 帧 × 每帧几十个框，降到"每个类几十个位置"。

用法：
    python bench/make_static_review.py --src datasets/videoset_02_label2 \
        --out outputs/static_review_v2
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

from trafficvision.config import (  # noqa: E402
    MERGED_CLASSES,
    STATIC_CLASSES,
    display_name,
)
from trafficvision.staticmap import StaticObjectMap, iou  # noqa: E402

FONT = r"C:\Windows\Fonts\msyh.ttc"


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "datasets" / "videoset_02_label2"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "static_review_v2"))
    ap.add_argument("--pad", type=float, default=1.6, help="裁剪时向外扩的倍数")
    ap.add_argument("--tile", type=int, default=260, help="缩略图边长")
    ap.add_argument("--min-side", type=int, default=260,
                    help="裁剪区域短边的最小像素数（小目标也能看清）")
    ap.add_argument("--min-ratio", type=float, default=0.80)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    by_video = defaultdict(list)
    for m in manifest:
        by_video[m["source_video"]].append(m)

    items = []
    for video, frames in sorted(by_video.items()):
        frames.sort(key=lambda z: z["frame_index"])
        smap = StaticObjectMap(min_ratio=args.min_ratio)
        parsed = []
        for i, m in enumerate(frames):
            w, h = m.get("w", 1920), m.get("h", 1080)
            boxes = read_labels(src / m["label"], w, h)
            parsed.append(boxes)
            statics = [(MERGED_CLASSES[cid], box, 1.0)
                       for cid, box in boxes
                       if 0 <= cid < len(MERGED_CLASSES) and MERGED_CLASSES[cid] in STATIC_CLASSES]
            smap.observe(i, statics, dynamics=[], span=1)

        confirmed = smap.confirmed()
        print(f"{video:<30} 确认位置 {len(confirmed)} 个")
        # 用中间帧裁剪（目标在整段都在，取哪帧都一样）
        mid = frames[len(frames) // 2]
        img = cv2.imread(str(src / mid["image"]))
        if img is None:
            continue
        H, W = img.shape[:2]

        for k, d in enumerate(confirmed):
            cls, box = d["cls"], d["box"]
            idx = len(items)
            cw, ch = box[2] - box[0], box[3] - box[1]
            mx, my = cw * args.pad, ch * args.pad
            # 小目标裁出来根本看不清，统一保证裁剪区域短边至少 min_side 像素
            ccx, ccy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            half = max(mx, args.min_side / 2)
            half_v = max(my, args.min_side / 2)
            x1 = max(0, int(ccx - half)); x2 = min(W, int(ccx + half))
            y1 = max(0, int(ccy - half_v)); y2 = min(H, int(ccy + half_v))
            crop = img[y1:y2, x1:x2].copy()
            cv2.rectangle(crop, (int(box[0] - x1), int(box[1] - y1)),
                          (int(box[2] - x1), int(box[3] - y1)), (0, 0, 255), 3)
            name = f"{idx:03d}_{cls}.jpg"
            cv2.imwrite(str(crops_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
            items.append({
                "no": idx, "cls": cls, "crop": f"crops/{name}",
                "video": Path(video).stem, "ratio": d["ratio"],
                "seen": d["seen"], "visible": d["visible"],
                "box": [round(v) for v in box],
                "size": [round(cw), round(ch)],
            })

    order = [c for c in ("bollard", "sign", "roadlamp", "trafficlight") if c in STATIC_CLASSES]
    parts = [
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>静态结构位置清单</title><style>",
        "body{margin:0;background:#14161a;color:#e8ebf0;font:15px/1.6 -apple-system,'Segoe UI','Microsoft YaHei',sans-serif}",
        ".w{max-width:1180px;margin:0 auto;padding:24px 18px 60px}",
        "h1{font-size:24px;margin:0 0 6px}h2{font-size:18px;margin:32px 0 10px;border-bottom:1px solid #2a2f38;padding-bottom:6px}",
        "p{color:#9aa3af}code{background:#242a33;padding:1px 5px;border-radius:4px}",
        ".g{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px;margin-top:12px}",
        ".c{background:#1b1f26;border:1px solid #2a2f38;border-radius:8px;overflow:hidden}",
        ".c img{width:100%;display:block;background:#000}",
        ".c .m{padding:6px 8px;font-size:12.5px;color:#9aa3af}",
        ".c .n{display:inline-block;background:#2563eb;color:#fff;border-radius:5px;padding:0 6px;margin-right:6px;font-weight:600}",
        ".note{background:#1b1f26;border:1px solid #2a2f38;border-radius:10px;padding:14px 16px;margin:14px 0}",
        "</style></head><body><div class='w'>",
        "<h1>静态结构位置清单</h1>",
        f"<p>共 <b>{len(items)}</b> 个位置，全部通过了跨帧一致性检验"
        f"（非遮挡帧里 ≥{int(args.min_ratio*100)}% 被识别为同一类）。"
        "红框是自动标注给出的位置，图是原始画面裁出来的。</p>",
        "<div class='note'><b>怎么用</b>：不用逐张标。对着下面每张小图看红框里是不是它标的那类东西，"
        "把编号记下来即可，例如：<code>信号灯 #12、#17 不是信号灯，是红色招牌</code>、"
        "<code>路灯 #3 漏了一半</code>。编号就是左上角那个蓝底数字。<br>"
        "原始大图在复核包里；这一页只负责让你能一眼扫完所有位置。</div>",
    ]
    for cls in order:
        group = [it for it in items if it["cls"] == cls]
        if not group:
            continue
        parts.append(f"<h2>{html.escape(display_name(cls))}（{cls}） · {len(group)} 个位置</h2>")
        parts.append("<div class='g'>")
        for it in group:
            parts.append(
                f"<div class='c'><img loading='lazy' src='{it['crop']}' alt='#{it['no']} {cls}'>"
                f"<div class='m'><span class='n'>#{it['no']}</span>{it['video']}"
                f"<br>一致性 {it['ratio']:.2f}（{it['seen']}/{it['visible']}）"
                f" · 框 {it['size'][0]}×{it['size'][1]}px</div></div>")
        parts.append("</div>")
    parts.append("</div></body></html>")
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")
    (out / "positions.json").write_text(json.dumps(items, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"\n位置清单：{out / 'index.html'}（{len(items)} 个位置）")


if __name__ == "__main__":
    main()
