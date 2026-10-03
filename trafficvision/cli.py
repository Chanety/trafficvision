"""命令行入口。

示例：
    python -m trafficvision.cli --video clip.mp4 --horizon 1 --out result.mp4
"""

from __future__ import annotations

import argparse
import json

from .config import DEFAULT_CLASSES, HORIZONS
from .pipeline import run_clip


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="交通视频预判：输入视频 + 时间梯度，输出带预判位置的视频")
    p.add_argument("--video", required=True, help="输入视频（2~30 秒）")
    p.add_argument("--horizon", type=float, required=True, choices=list(HORIZONS),
                   help="预判时间梯度（秒）：0.1 / 0.5 / 1 / 2")
    p.add_argument("--out", required=True, help="输出视频路径")
    p.add_argument("--weights", default="trafficlab-3d/repo/yolov8s-worldv2.pt",
                   help="检测权重路径")
    p.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                   help="目标类型，逗号分隔")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="0", help="0 表示第一块 GPU，cpu 表示用 CPU")
    p.add_argument("--no-world", action="store_true", help="权重为普通检测模型时使用")
    p.add_argument("--motion-model", choices=["cv", "ca", "scale", "auto"], default="auto",
                   help="运动模型：auto 按梯度自动分工（默认）/ cv 图像匀速 / "
                        "ca 匀加速 / scale 强制维度判定")
    p.add_argument("--accel-std", type=float, default=400.0,
                   help="过程噪声（像素/秒²）：越小速度估计越平滑，越大跟随越快")
    p.add_argument("--accel-prior", type=float, default=0.0,
                   help="速度变化采纳权重 0~1（驾驶经验先验）：0 表示完全匀速")
    p.add_argument("--json", dest="out_json", default=None, help="逐帧数据与精度报告输出路径")
    p.add_argument("--gif", default=None, help="额外输出 GIF 预览路径")
    p.add_argument("--gif-stride", type=int, default=3, help="GIF 抽帧间隔（越大文件越小）")
    p.add_argument("--max-frames", type=int, default=None, help="只处理前 N 帧（调试用）")
    p.add_argument("--no-render", action="store_true", help="只做检测与预判，不写输出视频")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   help="跟踪器配置：bytetrack / botsort / ocsort / deepocsort / fasttrack / tracktrack")
    p.add_argument("--no-timestamps", action="store_true",
                   help="忽略真实时间戳，退回按帧号/帧率计时（对照实验用）")
    return p


def main() -> None:
    args = build_parser().parse_args()
    result = run_clip(
        video=args.video,
        horizon=args.horizon,
        out_video=args.out,
        weights=args.weights,
        classes=tuple(c.strip() for c in args.classes.split(",") if c.strip()),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        world=not args.no_world,
        out_json=args.out_json,
        gif=args.gif,
        gif_stride=args.gif_stride,
        motion_model=args.motion_model,
        accel_std=args.accel_std,
        accel_prior=args.accel_prior,
        max_frames_override=args.max_frames,
        use_timestamps=not args.no_timestamps,
        render=not args.no_render,
        tracker=args.tracker,
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    if result.get("y_horizon") is not None:
        print(f"地平线估计: y = {result['y_horizon']:.1f} px"
              f"（维度判定生效 {result.get('scale_predictions', 0)} 处）")


if __name__ == "__main__":
    main()
