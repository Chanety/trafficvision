"""把"帧对标注包"转成评测用的 MOT 格式人工真值。

配合 tools/make_truth_pairs.py 使用：
  标注包里的文件名自带全部信息 —— `p03_a_VID_xxx_f00150.jpg`
  分别表示 第 3 组 / a=起始帧 / 来源视频 / 真实帧号。

配对规则（与标注说明一致）：**每对帧里同一类别最多一个目标**，
所以跨帧配对直接按类别唯一确定，不需要人工编 ID。

输出 MOT 行：`frame,id,x,y,w,h,cls`

用法：
    python bench/pairs_to_truth.py --pack datasets/truth_pairs_labeled --out truth/truth.mot.txt
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
NAME_RE = re.compile(r"^(p\d+)_(a|b)_(.+)_f(\d+)\.(?:jpe?g|png)$", re.I)


def load_classes(pack: Path) -> list[str]:
    f = pack / "classes.txt"
    if not f.exists():
        raise SystemExit(f"缺少 classes.txt：{pack}")
    return [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]


def read_label(path: Path, w: int, h: int):
    """YOLO 归一化 → 像素 (x1,y1,x2,y2)。"""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        out.append((cid, [(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="标完的帧对包目录（含 images/ labels/ classes.txt）")
    ap.add_argument("--out", default=None, help="输出 MOT 文本")
    args = ap.parse_args()

    pack = Path(args.pack)
    if not pack.is_absolute():
        pack = ROOT / pack
    classes = load_classes(pack)

    pairs: dict[str, dict[str, tuple[int, str, list]]] = defaultdict(dict)
    for img in sorted((pack / "images").glob("*")):
        m = NAME_RE.match(img.name)
        if not m:
            continue
        pid, tag, video, frame = m.group(1), m.group(2).lower(), m.group(3), int(m.group(4))
        data = cv2.imread(str(img))
        if data is None:
            continue
        h, w = data.shape[:2]
        boxes = read_label(pack / "labels" / (img.stem + ".txt"), w, h)
        pairs[pid][tag] = (frame, video, boxes)

    lines, stats = [], []
    n_pairs_used = 0
    for pid in sorted(pairs):
        both = pairs[pid]
        if "a" not in both or "b" not in both:
            continue
        fa, va, ba = both["a"]
        fb, vb, bb = both["b"]
        by_cls_a = {cid: box for cid, box in ba}
        by_cls_b = {cid: box for cid, box in bb}
        common = sorted(set(by_cls_a) & set(by_cls_b))
        used = 0
        for cid in common:
            name = classes[cid] if 0 <= cid < len(classes) else f"c{cid}"
            tid = int(pid[1:]) * 100 + cid          # 组号×100 + 类别号：稳定唯一
            for f, box in ((fa, by_cls_a[cid]), (fb, by_cls_b[cid])):
                x1, y1, x2, y2 = box
                lines.append(f"{f},{tid},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{name}")
            used += 1
        n_pairs_used += 1
        stats.append({"pair": pid, "video": va, "frame_a": fa, "frame_b": fb,
                      "matched_classes": [classes[c] if c < len(classes) else c for c in common],
                      "samples": used})

    if not lines:
        raise SystemExit(
            "没有生成任何真值。常见原因：\n"
            "  · labels/ 还是空的（还没标）；\n"
            "  · 同一组里同一类别标了多个框（协议要求每类最多一个）；\n"
            "  · 文件名被改过，配对信息丢失。")

    out = Path(args.out) if args.out else (pack / "truth.mot.txt")
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total = sum(s["samples"] for s in stats)
    print(f"用了 {n_pairs_used} 组帧对，共 {total} 个「目标 × 1 秒」配对样本")
    print(f"{'组':<6}{'视频':<24}{'类别'}")
    for s in stats:
        print(f"{s['pair']:<6}{s['video']:<24}{'、'.join(s['matched_classes']) or '（两端类别不一致，已跳过）'}")
    print(f"\nMOT 真值：{out}")
    (out.parent / "pairs_summary.json").write_text(
        json.dumps({"pairs": stats, "samples": total}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("\n下一步：python bench/eval_manual_truth.py --video <那段视频> --truth "
          f"{out.name} --horizon 1")


if __name__ == "__main__":
    main()
