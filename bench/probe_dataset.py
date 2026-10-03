"""数据集技术体检：分辨率、标称帧率、实际帧间隔、可变帧率(VFR)判定、时长。

用法：python bench/probe_dataset.py <数据集目录> [输出 json]

为什么要查"实际帧间隔"：手机录制常见可变帧率（VFR），容器里的标称帧率
（如 30）与真实时间戳不一致。预判按"秒"计算，必须用真实时间戳，
否则 1 秒梯度的位移量会算错。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"file": path.name, "error": "无法打开"}

    info = {
        "file": path.name,
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps_nominal": round(float(cap.get(cv2.CAP_PROP_FPS)), 3),
        "frames_header": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    info["codec"] = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))

    # 逐帧读取，采集真实时间戳
    stamps = []
    count = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        stamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        count += 1
    cap.release()

    info["frames"] = count
    if count >= 2 and stamps[-1] > stamps[0]:
        intervals = np.diff(np.array(stamps, dtype=float))
        intervals = intervals[intervals > 0]
        info["duration_s"] = round((stamps[-1] - stamps[0]) / 1000.0, 2)
        info["interval_mean_ms"] = round(float(intervals.mean()), 2)
        info["interval_std_ms"] = round(float(intervals.std()), 3)
        info["interval_min_ms"] = round(float(intervals.min()), 2)
        info["interval_max_ms"] = round(float(intervals.max()), 2)
        info["fps_effective"] = round(1000.0 / float(intervals.mean()), 2)
        # VFR 判定：帧间隔波动超过 2ms 或存在多种间隔取值
        unique = np.unique(np.round(intervals, 1))
        info["interval_unique_values"] = int(unique.size)
        info["vfr"] = bool(intervals.std() > 2.0 or unique.size > 3)
        info["fps_vs_nominal_pct"] = round(
            (info["fps_effective"] - info["fps_nominal"]) / info["fps_nominal"] * 100, 1
        ) if info["fps_nominal"] else None
    else:
        info["duration_s"] = None
        info["vfr"] = None
    return info


def probe_image(path: Path) -> dict:
    from PIL import Image

    with Image.open(path) as im:
        info = {"file": path.name, "size_mb": round(path.stat().st_size / 1e6, 2),
                "width": im.width, "height": im.height, "mode": im.mode}
        exif = im.getexif()
        if exif:
            info["exif_datetime"] = str(exif.get(306, ""))
            info["exif_model"] = str(exif.get(272, ""))
            info["exif_focal_length"] = str(exif.get(37386, ""))
    return info


def main() -> None:
    root = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    files = sorted(p for p in root.iterdir() if p.is_file())

    videos = [probe_video(p) for p in files if p.suffix.lower() in VIDEO_EXT]
    images = [probe_image(p) for p in files if p.suffix.lower() in IMAGE_EXT]

    print(f"=== 视频 {len(videos)} 段 ===")
    print(f"{'文件':<28}{'分辨率':>12}{'标称fps':>9}{'实测fps':>9}{'帧数':>7}"
          f"{'时长s':>8}{'间隔波动':>9}{'VFR':>6}")
    for v in videos:
        print(f"{v['file']:<28}{str(v['width']) + 'x' + str(v['height']):>12}"
              f"{v.get('fps_nominal', 0):>9}{v.get('fps_effective', 0):>9}"
              f"{v.get('frames', 0):>7}{v.get('duration_s', 0):>8}"
              f"{v.get('interval_std_ms', 0):>9}{str(v.get('vfr')):>6}")

    print(f"\n=== 场景图 {len(images)} 张 ===")
    for i in images:
        print(f"{i['file']:<28}{str(i['width']) + 'x' + str(i['height']):>12}"
              f"  {i.get('exif_datetime', '')}")

    fps_values = sorted({v.get("fps_effective") for v in videos if v.get("fps_effective")})
    print(f"\n实测帧率取值：{fps_values}")
    print(f"VFR（可变帧率）视频数：{sum(1 for v in videos if v.get('vfr'))}/{len(videos)}")

    if out_path:
        out_path.write_text(
            json.dumps({"videos": videos, "images": images}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"manifest -> {out_path}")


if __name__ == "__main__":
    main()
