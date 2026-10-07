"""抖动敏感性实验：给稳定视频注入不同强度的全局抖动，量它对预判的影响。

动机：用户要自采数据集但没有三脚架，需要知道"手抖到什么程度还算能用"。

做法：取一段最稳的素材（112416，实测抖动 0.03 px/帧），
用 AR(1) 平滑随机过程生成全局平移，叠加到每一帧（同时放大 12% 避免黑边），
合成出不同抖动强度的视频，再走完整预判流程对比指标。

抖动强度用**和项目其它报告相同的口径**（相邻帧全局位移均值，px/帧）标定，
生成后再用相机运动工具实测确认，保证数字可比。

用法：python bench/shake_sensitivity.py [每段帧数=300]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trafficvision.pipeline import run_clip  # noqa: E402

SOURCE = ROOT / "datasets" / "videoset_02_origin" / "VID_20261004_112416.mp4"
WEIGHTS = ROOT / "outputs" / "train" / "v7" / "weights" / "best.pt"
OUT = ROOT / "outputs" / "shake_test"
# 目标抖动强度（相邻帧全局位移均值，px/帧）
LEVELS = [0.0, 0.5, 2.0, 5.0]


def synth_video(src: Path, dst: Path, target_shift: float, frames: int, seed: int = 0) -> None:
    """AR(1) 平滑抖动：rho=0.9 时，帧间位移 σ = σ_ε·√(2(1−ρ))，E|Δ| ≈ 0.8σ。"""
    import cv2

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(seed)
    zoom = 1.12
    rho = 0.9
    sigma = (target_shift / 0.8) / np.sqrt(2 * (1 - rho)) if target_shift > 0 else 0.0
    dx = dy = 0.0
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        if target_shift > 0:
            dx = rho * dx + rng.normal(0, sigma)
            dy = rho * dy + rng.normal(0, sigma)
        m = np.array([[zoom, 0.0, -(zoom - 1) * w / 2 + dx],
                      [0.0, zoom, -(zoom - 1) * h / 2 + dy]], dtype=np.float32)
        writer.write(cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE))
    cap.release()
    writer.release()


def main() -> None:
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"源视频 {SOURCE.name}（实测最稳的一段，0.03 px/帧）")
    print(f"注入抖动强度（px/帧）：{LEVELS}\n")
    print(f"{'注入强度':>9}{'实测抖动':>10}{'1s中位':>9}{'1s均值':>9}{'1sP90':>9}"
          f"{'可预判/帧':>11}{'跳变/千帧':>11}")

    for level in LEVELS:
        video = OUT / f"shake_{level:g}.mp4"
        json_path = OUT / f"shake_{level:g}.json"
        synth_video(SOURCE, video, level, frames)
        run_clip(video=video, horizon=1.0, out_video=OUT / "unused.mp4",
                 weights=str(WEIGHTS), imgsz=1280, conf=0.25, device="0", world=False,
                 out_json=json_path, render=False, max_frames_override=frames,
                 tracker="bytetrack.yaml", guard=True, coast_frames=12)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        m = payload["metrics"].get("1s", {})
        objs = [o for f in payload["frames"] for o in f]
        # 简易跳变率：相邻帧同一 id 位移 > max(40, 4×中位) 的次数
        seq: dict[int, list] = {}
        for i, f in enumerate(payload["frames"]):
            for o in f:
                if o["id"] > 0 and not o.get("coasting"):
                    seq.setdefault(o["id"], []).append((i, o["centroid"][0], o["centroid"][1]))
        jumps = total_steps = 0
        for pts in seq.values():
            if len(pts) < 4:
                continue
            pts.sort()
            steps = [np.hypot(pts[k][1] - pts[k - 1][1], pts[k][2] - pts[k - 1][2])
                     for k in range(1, len(pts))]
            med = float(np.median(steps))
            jumps += sum(1 for s in steps if s > max(40.0, 4 * med))
            total_steps += len(steps)
        # 用相机运动工具实测这次合成的真实抖动
        import subprocess
        meas = subprocess.run(
            [str(ROOT / "trafficlab-3d" / "env" / "Scripts" / "python.exe"),
             str(ROOT / "bench" / "camera_motion_report.py"), str(OUT), "60",
             str(OUT / "cam.json")], capture_output=True, text=True)
        realized = None
        cam_file = OUT / "cam.json"
        if cam_file.exists():
            for r in json.loads(cam_file.read_text(encoding="utf-8")):
                if r["file"] == video.name:
                    realized = r["camera_shift_mean_px"]
        print(f"{level:>9.1f}{(realized if realized is not None else float('nan')):>10.2f}"
              f"{m.get('median_px', float('nan')):>9.1f}{m.get('mean_px', float('nan')):>9.1f}"
              f"{m.get('p90_px', float('nan')):>9.1f}"
              f"{sum(1 for o in objs if o.get('predictable')) / max(1, len(payload['frames'])):>11.2f}"
              f"{jumps / max(1, len(payload['frames'])) * 1000:>11.1f}")


if __name__ == "__main__":
    main()
