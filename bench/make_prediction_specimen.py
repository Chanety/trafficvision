"""生成"预判标本"：左＝原图原框，右＝1 秒后的图 + 预判框。

用途：肉眼核对预判到底准不准。左图是本帧的检测结果，右图是同一批目标
**1 秒后**的画面，上面画的是模型在本帧给出的预判框；同时用细线标出
该目标 1 秒后的**实测**位置，两者的差距就是预判误差。

用法：
    python bench/make_prediction_specimen.py --video datasets/videoset_02_origin/VID_20261004_112912.mp4 \
        --frame 300 --n 5 --out outputs/specimen/112912_f300.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trafficvision.config import display_name  # noqa: E402
from trafficvision.pipeline import probe_video, run_clip  # noqa: E402
from trafficvision.render import class_color  # noqa: E402

PANEL_W = 1280          # 每个面板的宽度（高度按比例）


def dashed_rect(img, box, color, thickness=3, dash=14, gap=9):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    step = dash + gap
    for x in range(x1, x2, step):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness, cv2.LINE_AA)
    for y in range(y1, y2, step):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness, cv2.LINE_AA)


def label(img, text, org, color, scale=0.7):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = int(org[0]), max(th + 6, int(org[1]))
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 8, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (15, 15, 15), 2, cv2.LINE_AA)


def center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=300, help="起始帧号 t")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--horizon", type=float, default=1.0)
    ap.add_argument("--weights", default=str(ROOT / "outputs" / "train" / "v7" / "weights" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel-w", type=int, default=PANEL_W)
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_absolute():
        video = ROOT / video
    info = probe_video(video)
    fps = info["fps"]
    gap = int(round(args.horizon * fps))

    # 跑预判（不渲染视频，只要结构化结果）
    tmp = Path(args.out).with_suffix(".pred.json")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    print(f"跑预判：{video.name}  第 {args.frame} 帧起，1 秒梯度…")
    run_clip(video=video, horizon=args.horizon, out_video=tmp.with_suffix(".mp4"),
             weights=args.weights, imgsz=args.imgsz, device="0", render=False,
             out_json=tmp, max_frames_override=args.frame + gap + 5, static_map=False)
    data = json.loads(tmp.read_text(encoding="utf-8"))
    frames = data["frames"]
    t = args.frame
    if t >= len(frames) or t + gap >= len(frames):
        raise SystemExit(f"帧号越界：视频共 {len(frames)} 帧")

    # 选目标：可预判、框够大、优先"在动"的
    cands = []
    W, H = info["width"], info["height"]
    for o in frames[t]:
        if not o.get("predictable"):
            continue
        b = o["bbox"]
        h = b[3] - b[1]
        w = b[2] - b[0]
        if h < 45 or h > 320:          # 太小看不清、太大则贴脸且常被裁切
            continue
        m = 12                         # 贴边的目标（出画/被裁）不作为标本
        if b[0] < m or b[1] < m or b[2] > W - m or b[3] > H - m:
            continue
        cands.append(o)
    cands.sort(key=lambda o: -(o["speed_px_s"] or 0))
    picked = cands[: args.n]
    if not picked:
        raise SystemExit("这一帧没有合适的可预判目标，换一帧试试")
    print(f"选中 {len(picked)} 个目标：" +
          "、".join(f"#{i+1}{display_name(o['cls'])}" for i, o in enumerate(picked)))

    # 取这两帧画面
    cap = cv2.VideoCapture(str(video))
    imgs = {}
    for tag, f in (("a", t), ("b", t + gap)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            raise SystemExit(f"读不到第 {f} 帧")
        imgs[tag] = fr
    cap.release()

    scale = args.panel_w / info["width"]
    ph = int(info["height"] * scale)
    A = cv2.resize(imgs["a"], (args.panel_w, ph))
    B = cv2.resize(imgs["b"], (args.panel_w, ph))
    S = lambda b: [v * scale for v in b]        # noqa: E731

    errs = []
    for i, o in enumerate(picked, start=1):
        color = class_color(o["cls"])
        name = display_name(o["cls"])
        # 左：原图原框
        b = S(o["bbox"])
        cv2.rectangle(A, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 4, cv2.LINE_AA)
        label(A, f"#{i} {name}", (b[0], b[1] - 4), color, 0.75)

        # 右：预判框（虚线）
        pred = (o.get("preds") or {}).get(str(args.horizon)) or (o.get("preds") or {}).get(str(float(args.horizon)))
        if pred is None:
            continue
        pb = S(pred)
        dashed_rect(B, pb, color, 4)
        label(B, f"#{i} 预判", (pb[0], pb[1] - 4), color, 0.75)

        # 真实位置（同 id 在 t+gap 的实测框）
        act = next((x for x in frames[t + gap] if x["id"] == o["id"]), None)
        if act is not None and not act.get("coasting"):
            ab = S(act["bbox"])
            cv2.rectangle(B, (int(ab[0]), int(ab[1])), (int(ab[2]), int(ab[3])),
                          (255, 255, 255), 2, cv2.LINE_AA)
            pc, ac = center(pb), center(ab)
            cv2.line(B, (int(pc[0]), int(pc[1])), (int(ac[0]), int(ac[1])),
                     (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(B, (int(ac[0]), int(ac[1])), 5, (255, 255, 255), -1, cv2.LINE_AA)
            e = float(np.hypot(pc[0] - ac[0], pc[1] - ac[1]) / scale)
            errs.append(e)
            # 误差数字放到实测框下方，避免和预判标签叠在一起
            label(B, f"#{i} 实测 · 误差 {e:.0f}px",
                  (ab[0], ab[3] + 4), (255, 255, 255), 0.6)

    # 标题条
    def banner(img, text, color):
        bar = np.zeros((56, img.shape[1], 3), np.uint8)
        bar[:] = color
        cv2.putText(bar, text, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return np.vstack([bar, img])

    wa = banner(A, f"左：第 {t} 帧（原图 + 检测框）  t = {t / fps:.2f}s", (40, 40, 40))
    wb = banner(B, f"右：第 {t + gap} 帧（1 秒后）· 彩色虚线 = 预判框 · 白框 = 实测位置", (40, 40, 90))
    comp = np.hstack([wa, wb])

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), comp, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if errs:
        print(f"\n本标本 {len(errs)} 个样本：误差中位 {np.median(errs):.1f}px，"
              f"最大 {max(errs):.1f}px（按原图分辨率计）")
    print(f"标本：{out}  ({comp.shape[1]}×{comp.shape[0]})")


if __name__ == "__main__":
    main()
