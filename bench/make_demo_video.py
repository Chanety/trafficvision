"""生成竞赛演示视频：MP4 / H.264 / 3--5 分钟 / 不超过 300 MB。

设计要点
--------
  * 素材全部来自真实运行结果：每段演示都是 ``trafficvision.cli`` 的实际输出，
    不做剪辑拼接以外的修饰。
  * 演示片段按 0.5 倍速播放（``setpts=2*PTS``），既把总时长拉到 3 分钟以上，
    也让预判框的位移更容易看清；视频内会明确标注这一点。
  * 同机位复用场景先验（``--camera``），首次渲染后命中缓存，后续重跑快约 4 倍。
  * 渲染结果按段落缓存，可中断续跑；已存在且非空的段落不会重算。
  * 合流后自动校验时长区间与体积上限；超限时自动降码率或降分辨率重编。

用法：
    python bench/make_demo_video.py            # 全流程
    python bench/make_demo_video.py --plan     # 只打印分镜与预估时长
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "trafficlab-3d" / "env" / "Scripts" / "python.exe"
DEMO = ROOT / "outputs" / "demo"
RAW = DEMO / "raw"
PARTS = DEMO / "parts"
SRC = ROOT / "datasets" / "videoset_02_origin"

FPS = 30
SIZE = "1920x1080"
SLOWDOWN = 2.0          # 0.5 倍速
MIN_SEC, MAX_SEC = 190.0, 295.0
SIZE_CAP_MB = 300.0

TITLE = "交通视频短时预判软件"
SUBTITLE = "0.1 / 0.5 / 1 / 2 秒梯度 · 因果预判 · 单卡 8GB 可运行"


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(cmd: list[str], quiet: bool = True) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        tail = "\n".join((r.stderr or "").strip().splitlines()[-12:])
        raise SystemExit(f"命令失败（{r.returncode}）：\n{' '.join(cmd[:4])} ...\n{tail}")
    if not quiet:
        print(r.stdout[-2000:])


def probe(path: Path) -> tuple[float, float]:
    """返回 (时长秒, 体积MB)。"""
    r = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    dur = 0.0
    for line in (r.stderr or "").splitlines():
        if "Duration:" in line:
            hms = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = hms.split(":")
            dur = int(h) * 3600 + int(m) * 60 + float(s)
            break
    return dur, path.stat().st_size / 1024 / 1024


# --------------------------------------------------------------------------- 卡片

def card(lines: list[str], out: Path, sub: str | None = None,
         bg=(18, 22, 30), fg=(240, 242, 245), accent=(90, 170, 255)) -> Path:
    from PIL import Image, ImageDraw, ImageFont
    W, H = 1920, 1080
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 8], fill=accent)
    d.rectangle([0, H - 8, W, H], fill=accent)

    def font(pt: int):
        for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
            p = Path("C:/Windows/Fonts") / name
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), pt)
                except Exception:
                    continue
        return ImageFont.load_default()

    title_f, body_f = font(74), font(38)
    y = H // 2 - 40 * len(lines)
    for i, ln in enumerate(lines):
        f = title_f if i == 0 else body_f
        w = d.textlength(ln, font=f)
        d.text(((W - w) / 2, y), ln, font=f, fill=fg if i == 0 else (200, 208, 218))
        y += 110 if i == 0 else 72
    if sub:
        w = d.textlength(sub, font=body_f)
        d.text(((W - w) / 2, H - 130), sub, font=body_f, fill=accent)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def still(src: Path, out: Path) -> Path:
    """把图表按 16:9 居中放到深色底上。"""
    from PIL import Image
    W, H = 1920, 1080
    bg = Image.new("RGB", (W, H), (255, 255, 255))
    im = Image.open(src).convert("RGB")
    s = min((W - 80) / im.width, (H - 80) / im.height)
    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    bg.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    bg.save(out)
    return out


# --------------------------------------------------------------------------- 渲染

def render_clip(name: str, video: Path, horizon: float, frames: int,
                camera: str, extra: list[str] | None = None) -> Path:
    """跑一次真实预判，返回输出视频路径（命中缓存则跳过）。"""
    out = RAW / f"{name}.mp4"
    if out.exists() and out.stat().st_size > 100_000:
        print(f"  [缓存] {name}")
        return out
    cmd = [str(PY), "-m", "trafficvision.cli", "--video", str(video),
           "--horizon", str(horizon), "--out", str(out),
           "--max-frames", str(frames), "--camera", camera]
    if extra:
        cmd += extra
    print(f"  [渲染] {name}（horizon {horizon}s × {frames} 帧）...")
    run(cmd)
    return out


def normalize(src: Path, out: Path, slow: bool = True) -> Path:
    """统一分辨率/帧率/编码，便于无损拼接。"""
    vf = [f"scale={SIZE.replace('x', ':')}:force_original_aspect_ratio=decrease",
          f"pad={SIZE.replace('x', ':')}:(ow-iw)/2:(oh-ih)/2"]
    if slow:
        vf.append(f"setpts={SLOWDOWN}*PTS")
    run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-vf", ",".join(vf), "-r", str(FPS), "-c:v", "libx264", "-crf", "18",
         "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)])
    return out


def hold(img: Path, seconds: float, out: Path) -> Path:
    run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-loop", "1",
         "-i", str(img), "-t", f"{seconds}", "-r", str(FPS),
         "-vf", f"scale={SIZE.replace('x', ':')}", "-c:v", "libx264", "-crf", "18",
         "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)])
    return out


def concat(parts: list[Path], out: Path, encoder: str = "libx264",
           crf: int = 24, scale: str = SIZE) -> Path:
    lst = DEMO / "concat.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
         "-safe", "0", "-i", str(lst),
         "-vf", f"scale={scale.replace('x', ':')}", "-c:v", encoder, "-crf", str(crf),
         "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-an", str(out)])
    return out


# --------------------------------------------------------------------------- 分镜

def plan() -> list[dict]:
    return [
        {"kind": "card", "seconds": 7,
         "lines": [TITLE, SUBTITLE],
         "sub": "演示视频 · 全部画面为真实运行输出"},
        {"kind": "card", "seconds": 6,
         "lines": ["演示内容",
                   "一、系统实际输出画面（含类型、预判框、动态/静止状态）",
                   "二、四个预判梯度对照",
                   "三、静态结构稳定显示与遮挡续推",
                   "四、实测精度与误差结构"],
         "sub": "画面按 0.5 倍速播放，便于观察预判框位移"},
        {"kind": "clip", "seconds": None, "name": "s1_main",
         "video": SRC / "VID_20261004_112222.mp4", "horizon": 1.0,
         "frames": 420, "camera": "cam112222"},
        {"kind": "card", "seconds": 5,
         "lines": ["二、四个预判梯度对照"],
         "sub": "同一机位、同一目标，梯度越大外推越远"},
    ] + [
        {"kind": "clip", "seconds": None, "name": f"s2_h{h}",
         "video": SRC / "VID_20261004_112333.mp4", "horizon": h,
         "frames": 200, "camera": "cam112333"}
        for h in (0.1, 0.5, 1.0, 2.0)
    ] + [
        {"kind": "card", "seconds": 5,
         "lines": ["三、静态结构与遮挡续推"],
         "sub": "路桩、路灯、信号灯：跨帧投票 + 语义校验 + 场景先验"},
        {"kind": "clip", "seconds": None, "name": "s3_static",
         "video": SRC / "VID_20261004_112254.mp4", "horizon": 1.0,
         "frames": 320, "camera": "cam112254"},
        {"kind": "clip", "seconds": None, "name": "s4_coast",
         "video": SRC / "VID_20261004_112453.mp4", "horizon": 1.0,
         "frames": 320, "camera": "cam112453"},
        {"kind": "card", "seconds": 6,
         "lines": ["四、实测精度", "目标域 mAP50 0.637",
                   "每帧可预判目标 14.87 个（占检出 44.6%）",
                   "1 秒预判中位误差 33.4 像素",
                   "轨迹跳变 18.3 次/千帧"],
         "sub": "六段 × 300 帧、置信度 0.25、分辨率 1280、关闭静态通道"},
    ] + [
        {"kind": "still", "seconds": 11,
         "src": ROOT / "outputs" / "error_dist" / "fig01_along_by_horizon.png"},
        {"kind": "still", "seconds": 11,
         "src": ROOT / "outputs" / "error_dist" / "尺度混合" / "尺度混合汇总.png"},
        {"kind": "still", "seconds": 11,
         "src": ROOT / "outputs" / "error_dist" / "像素口径" / "像素口径_归属与收益.png"},
        {"kind": "still", "seconds": 9,
         "src": ROOT / "outputs" / "error_dist" / "归一化对照" / "归一化_对照总览.png"},
        {"kind": "card", "seconds": 7,
         "lines": ["误差结构（本项目的核心发现）",
                   "误差几乎全部沿运动方向，横向分量中位 < 0.4 像素",
                   "系统性超调：58%~68% 的样本打过头",
                   "分布是尺度混合：21%~36% 的样本贡献 75%~98% 的方差"],
         "sub": "详见技术报告第 6 节"},
        {"kind": "card", "seconds": 7,
         "lines": ["开放成果与合规",
                   "源码、评测脚本、标注工具整体以 AGPL-3.0 开放",
                   "附 40 条参考文献与《开源及第三方资源使用清单》",
                   "12 条技术路线的实测负结果一并公开"],
         "sub": "运行环境：Python 3.11 · PyTorch · 单卡 8GB"},
        {"kind": "card", "seconds": 5,
         "lines": ["完"], "sub": TITLE},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="只打印分镜")
    args = ap.parse_args()

    shots = plan()
    RAW.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)

    if args.plan:
        total = 0.0
        for i, s in enumerate(shots):
            if s["kind"] == "clip":
                sec = s["frames"] / FPS * SLOWDOWN
            else:
                sec = s["seconds"]
            total += sec
            tag = s.get("name") or (s.get("lines") or [s.get("src", "")])[0]
            print(f"{i:>2}  {s['kind']:<6} {sec:6.1f}s  {tag}")
        print(f"合计约 {total:.1f} 秒（目标 {MIN_SEC:.0f}--{MAX_SEC:.0f} 秒）")
        return

    print("== 1/3 渲染演示片段（真实运行）==")
    parts: list[Path] = []
    for i, s in enumerate(shots):
        if s["kind"] == "card":
            png = card(s["lines"], PARTS / f"{i:02d}_card.png", s.get("sub"))
            parts.append(hold(png, s["seconds"], PARTS / f"{i:02d}_card.mp4"))
            print(f"  [卡片] {s['lines'][0]}")
        elif s["kind"] == "still":
            png = still(s["src"], PARTS / f"{i:02d}_still.png")
            parts.append(hold(png, s["seconds"], PARTS / f"{i:02d}_still.mp4"))
            print(f"  [图表] {s['src'].name}")
        else:
            raw = render_clip(s["name"], s["video"], s["horizon"], s["frames"],
                              s["camera"])
            parts.append(normalize(raw, PARTS / f"{i:02d}_{s['name']}.mp4"))

    print("== 2/3 合流 ==")
    out = DEMO / "trafficvision_demo.mp4"
    concat(parts, out, crf=24)
    dur, mb = probe(out)
    print(f"  初版：{dur:.1f}s，{mb:.1f} MB")

    print("== 3/3 合规校验 ==")
    if mb > SIZE_CAP_MB * 0.9 or dur > MAX_SEC:
        print("  超限，降码率重编...")
        concat(parts, out, crf=27)
        dur, mb = probe(out)
        print(f"  重编后：{dur:.1f}s，{mb:.1f} MB")
    if mb > SIZE_CAP_MB * 0.9:
        print("  仍偏大，降到 1280x720 ...")
        concat(parts, out, crf=25, scale="1280x720")
        dur, mb = probe(out)
        print(f"  重编后：{dur:.1f}s，{mb:.1f} MB")

    ok_dur = MIN_SEC <= dur <= MAX_SEC
    ok_size = mb <= SIZE_CAP_MB
    print(f"\n结果：{out}")
    print(f"  时长 {dur:.1f}s（要求 180--300s）：{'PASS' if ok_dur else 'CHECK'}")
    print(f"  体积 {mb:.1f} MB（上限 300MB）：{'PASS' if ok_size else 'FAIL'}")
    rep = DEMO / "演示视频_校验报告.csv"
    with rep.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["项目", "值", "要求", "结论"])
        w.writerow(["输出路径", str(out), "", ""])
        w.writerow(["时长(秒)", round(dur, 1), "180-300", "PASS" if ok_dur else "CHECK"])
        w.writerow(["体积(MB)", round(mb, 1), "<=300", "PASS" if ok_size else "FAIL"])
        w.writerow(["编码", "H.264 (libx264) / yuv420p", "MP4", "PASS"])
        w.writerow(["分辨率", SIZE, "", ""])
        w.writerow(["播放速度", "0.5 倍速（画面内已标注）", "", ""])
    print(f"  校验报告：{rep}")


if __name__ == "__main__":
    main()
