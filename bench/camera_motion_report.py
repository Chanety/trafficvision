"""相机运动体检：估计手持/云台抖动造成的整幅画面位移。

原理：相位相关（phase correlation）估计相邻帧之间的全局平移量，
累加得到画面累计位移。若该值与预判误差高度相关，说明"相机自己在动"
是误差的主要来源，必须做运动补偿。

用法：python bench/camera_motion_report.py <数据集目录> [每段采样帧数]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def global_translation(prev_gray, gray) -> tuple[float, float]:
    shift, _ = cv2.phaseCorrelate(np.float32(prev_gray), np.float32(gray))
    return float(shift[0]), float(shift[1])


def probe(path: Path, want: int = 200) -> dict:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, total // want) if total > 0 else 1
    prev = None
    mags = []
    magnitudes = []
    cumulative = 0.0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            gray = cv2.cvtColor(cv2.resize(frame, (480, 270)), cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            if prev is not None:
                dx, dy = global_translation(prev, gray)
                magnitude = float(np.hypot(dx, dy))
                mags.append(magnitude)
                # 归一化到"每秒"尺度，便于跨视频比较
                magnitudes.append(magnitude)
                cumulative += magnitude
            prev = gray
        idx += 1
    cap.release()

    if not mags:
        return {"file": path.name, "error": "样本不足"}
    arr = np.array(mags)
    return {
        "file": path.name,
        "frames_sampled": len(mags),
        "camera_shift_mean_px": round(float(arr.mean()), 2),
        "camera_shift_p90_px": round(float(np.percentile(arr, 90)), 2),
        "camera_shift_max_px": round(float(arr.max()), 2),
        "camera_shift_total_px": round(cumulative, 1),
        "moving_frame_ratio": round(float((arr > 0.5).mean()), 3),
    }


def main() -> None:
    root = Path(sys.argv[1])
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    rows = [probe(p, want) for p in sorted(root.iterdir())
            if p.suffix.lower() in VIDEO_EXT]
    print(f"{'文件':<28}{'平均位移':>10}{'P90':>8}{'最大':>8}{'累计':>10}{'有运动帧占比':>14}")
    for r in rows:
        if "error" in r:
            continue
        print(f"{r['file']:<28}{r['camera_shift_mean_px']:>10}{r['camera_shift_p90_px']:>8}"
              f"{r['camera_shift_max_px']:>8}{r['camera_shift_total_px']:>10}"
              f"{r['moving_frame_ratio']:>14}")
    if out:
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告 -> {out}")


if __name__ == "__main__":
    main()
