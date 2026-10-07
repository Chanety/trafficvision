"""生成"人类可读"的标注复核包：把自动标注画到图上，做成可翻阅的画廊。

用途：用户不做逐张标注，只需要浏览 + 指出问题。本脚本把
`datasets/videoset_02_label` 的 332 帧自动标注渲染成带框的图片，
并为每段源视频生成一张总览图，最后写一个离线 HTML 画廊。

用法：
    python bench/make_review_pack.py
    python bench/make_review_pack.py --src datasets/videoset_02_label --out outputs/review_videoset_02
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 与产品保持一致的类别颜色 / 中文名
CLASS_COLORS_BGR = {
    "car": (80, 220, 80), "truck": (0, 150, 255), "bus": (255, 130, 40),
    "motorcycle": (220, 60, 220), "bicycle": (220, 220, 60), "person": (60, 220, 255),
    "sign": (255, 255, 255), "bollard": (150, 150, 255), "tricycle": (0, 215, 255),
    "nonmotor": (220, 60, 220), "roadlamp": (0, 200, 200), "trafficlight": (60, 60, 255),
    "rider": (255, 200, 0),
}
CLASS_DISPLAY = {
    "car": "汽车", "truck": "卡车", "bus": "公交", "motorcycle": "摩托",
    "bicycle": "自行车", "person": "行人", "sign": "标志牌", "bollard": "路桩",
    "tricycle": "三轮车", "nonmotor": "非机动车", "roadlamp": "路灯",
    "trafficlight": "信号灯", "rider": "非机动车驾驶员",
}
try:  # 优先用产品配置，保证颜色/名称一致
    from trafficvision.config import CLASS_COLORS as _C, CLASS_DISPLAY_NAMES as _D
    CLASS_COLORS_BGR = dict(_C)
    CLASS_DISPLAY = dict(_D)
except Exception as exc:  # 独立运行时也能工作
    print(f"[提示] 未能加载 trafficvision.config（{exc}），使用内置的 9 类名称表")

CLASSES = ("car", "truck", "bus", "motorcycle", "bicycle", "person", "sign", "bollard", "tricycle")

FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
)


def rgb(cls_name: str) -> tuple[int, int, int]:
    b, g, r = CLASS_COLORS_BGR.get(cls_name, (180, 180, 180))
    return (r, g, b)


def load_font(size: int):
    from PIL import ImageFont
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def load_classes(src: Path) -> tuple[str, ...]:
    """优先读数据集自带的 classes.txt，兼容 9 类旧集与 10 类合并新集。"""
    p = src / "classes.txt"
    if p.exists():
        names = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if names:
            return tuple(names)
    return CLASSES


def parse_label(path: Path, classes: tuple[str, ...]) -> list[tuple[str, float, float, float, float]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cid = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            continue
        name = classes[cid] if 0 <= cid < len(classes) else f"class{cid}"
        out.append((name, x, y, w, h))
    return out


def draw_frame(img, boxes, header: str):
    """在图上画框 + 中文类名，并在左上角贴一条信息带。"""
    from PIL import ImageDraw
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    lw = max(2, H // 380)
    fs = max(16, H // 42)
    hfs = max(15, H // 46)
    font = load_font(fs)
    hfont = load_font(hfs)

    for name, x, y, w, h in boxes:
        x1 = (x - w / 2) * W
        y1 = (y - h / 2) * H
        x2 = (x + w / 2) * W
        y2 = (y + h / 2) * H
        col = rgb(name)
        d.rectangle([x1, y1, x2, y2], outline=col + (255,), width=lw)
        text = CLASS_DISPLAY.get(name, name)
        try:
            tw = d.textlength(text, font=font)
        except Exception:
            tw = len(text) * fs * 0.6
        pad = max(2, fs // 6)
        ty = y1 - fs - 2 * pad
        if ty < 0:
            ty = y2 + pad
        d.rectangle([x1, ty, x1 + tw + 2 * pad, ty + fs + 2 * pad], fill=col + (215,))
        d.text((x1 + pad, ty + pad), text, font=font, fill=(0, 0, 0, 255))

    try:
        hw = d.textlength(header, font=hfont)
    except Exception:
        hw = len(header) * hfs * 0.6
    d.rectangle([0, 0, hw + 24, hfs + 16], fill=(0, 0, 0, 175))
    d.text((12, 8), header, font=hfont, fill=(255, 255, 255, 255))
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "datasets" / "videoset_02_label"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "review_videoset_02"))
    ap.add_argument("--quality", type=int, default=74)
    ap.add_argument("--tile-w", type=int, default=640)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    from PIL import Image

    src = Path(args.src)
    out = Path(args.out)
    classes = load_classes(src)
    frames_dir = out / "frames"
    sheets_dir = out / "sheets"
    frames_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    by_video: dict[str, list[dict]] = defaultdict(list)
    for item in manifest:
        by_video[item["source_video"]].append(item)
    for v in by_video:
        by_video[v].sort(key=lambda it: it["frame_index"])

    total_boxes = 0
    sheet_files: list[tuple[str, str, int]] = []

    for video, items in sorted(by_video.items()):
        tiles = []
        for it in items:
            img_path = src / it["image"]
            if not img_path.exists():
                continue
            boxes = parse_label(src / it["label"], classes)
            total_boxes += len(boxes)
            img = Image.open(img_path).convert("RGB")
            header = f"{img_path.stem}   t={it['time_s']:.1f}s   框数 {len(boxes)}"
            draw_frame(img, boxes, header)
            img.save(frames_dir / f"{img_path.stem}.jpg", quality=args.quality, optimize=True)
            tiles.append(img)

        if not tiles:
            continue

        # 每段视频一张总览：把该段的帧拼成网格
        tw = args.tile_w
        th = int(tw * tiles[0].height / tiles[0].width)
        cols = args.cols
        rows = (len(tiles) + cols - 1) // cols
        gap = 6
        top = int(th * 0.06)
        sheet = Image.new("RGB", (cols * tw + (cols + 1) * gap, top + rows * th + (rows + 1) * gap), (18, 18, 20))
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            sheet.paste(t.resize((tw, th)), (gap + c * (tw + gap), top + gap + r * (th + gap)))
        from PIL import ImageDraw
        d = ImageDraw.Draw(sheet)
        d.text((14, int(top * 0.22)), f"{video}   共 {len(tiles)} 帧（每帧间隔 1 秒）",
               font=load_font(max(20, top // 2)), fill=(255, 255, 255))
        sheet_name = f"{Path(video).stem}.jpg"
        sheet.save(sheets_dir / sheet_name, quality=80, optimize=True)
        sheet_files.append((video, sheet_name, len(tiles)))
        print(f"  {video}: {len(tiles)} 帧 -> sheets/{sheet_name}")

    # ---- 离线画廊 ----
    parts = [
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>videoset_02 标注复核包</title><style>",
        "body{margin:0;background:#14161a;color:#e8ebf0;font:15px/1.6 -apple-system,'Segoe UI','Microsoft YaHei',sans-serif}",
        ".w{max-width:1200px;margin:0 auto;padding:24px 18px 60px}",
        "h1{font-size:24px;margin:0 0 6px}h2{font-size:18px;margin:34px 0 10px;border-bottom:1px solid #2a2f38;padding-bottom:6px}",
        "p{color:#9aa3af}.card{background:#1b1f26;border:1px solid #2a2f38;border-radius:10px;padding:14px 16px;margin:14px 0}",
        "img{width:100%;border-radius:8px;display:block;border:1px solid #2a2f38}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px;margin-top:12px}",
        ".grid a{display:block;background:#1b1f26;border:1px solid #2a2f38;border-radius:6px;overflow:hidden;text-decoration:none}",
        ".grid img{border:0;border-radius:0}.grid span{display:block;font-size:11.5px;color:#9aa3af;padding:4px 6px}",
        "code{background:#242a33;padding:1px 5px;border-radius:4px;font-size:.9em}",
        "ol{color:#9aa3af}li{margin:4px 0}b{color:#e8ebf0}",
        "</style></head><body><div class='w'>",
        "<h1>videoset_02 标注复核包</h1>",
        f"<p>共 {sum(n for _, _, n in sheet_files)} 帧、{total_boxes} 个自动标注框，来自 "
        f"{len(sheet_files)} 段视频。框的颜色与产品一致，标签为中文类别名。</p>",
        "<div class='card'><b>怎么看</b><ol>"
        "<li>先看每段视频的<b>总览图</b>（下面每节的第一张），快速扫一遍有没有明显问题。</li>"
        "<li>发现问题就点开对应的<b>单帧大图</b>确认。</li>"
        "<li>把问题写进 <code>问题记录模板.md</code>（同目录），只要写清楚「哪段视频 / 哪一帧 / 什么问题」就行，不用逐张标。</li>"
        "</ol><p style='margin-bottom:0'>复核优先顺序（也可以按你自己的习惯来）："
        "<b>三轮车 → 自行车 → 标志牌 → 路桩 → 公交 / 卡车 → 其余</b>。"
        "原始视频在同项目 <code>datasets/videoset_02_origin/</code>。</p></div>",
    ]
    for video, sheet_name, n in sheet_files:
        stem = Path(video).stem
        parts.append(f"<h2>{html.escape(stem)} <span style='color:#6b7480;font-size:14px'>（{n} 帧）</span></h2>")
        parts.append(f"<a href='sheets/{sheet_name}' target='_blank'><img src='sheets/{sheet_name}' alt='{html.escape(stem)} 总览'></a>")
        parts.append("<div class='grid'>")
        for it in by_video[video]:
            stem_f = Path(it["image"]).stem
            parts.append(
                f"<a href='frames/{stem_f}.jpg' target='_blank'>"
                f"<img loading='lazy' src='frames/{stem_f}.jpg' alt='{stem_f}'>"
                f"<span>t={it['time_s']:.1f}s · {stem_f.rsplit('_f', 1)[-1]}</span></a>"
            )
        parts.append("</div>")
    parts.append("</div></body></html>")
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")

    print(f"\n完成：{out}")
    print(f"  画廊      {out / 'index.html'}")
    print(f"  总览图    {sheets_dir}（{len(sheet_files)} 张，每段视频一张）")
    print(f"  单帧大图  {frames_dir}（{sum(n for _, _, n in sheet_files)} 张）")
    print(f"  总框数    {total_boxes}")


if __name__ == "__main__":
    main()
