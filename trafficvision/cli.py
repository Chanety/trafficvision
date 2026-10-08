"""命令行入口。

示例：
    python -m trafficvision.cli --video clip.mp4 --horizon 1 --out result.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    COAST_FRAMES,
    DEFAULT_CLASSES,
    HORIZONS,
    STATIC_CLASSES,
    STATIC_CONF,
    STATIC_IMGSZ,
    STATIC_STRIDE,
    MIN_TARGET_AREA_PX,
    MIN_TARGET_SIDE_PX,
    WORLD_PROMPT_CLASSES,
)
from .pipeline import run_clip


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="交通视频预判：输入视频 + 时间梯度，输出带预判位置的视频")
    p.add_argument("--video", required=True, help="输入视频（2~30 秒）")
    p.add_argument("--horizon", type=float, required=True, choices=list(HORIZONS),
                   help="预判时间梯度（秒）：0.1 / 0.5 / 1 / 2")
    p.add_argument("--out", required=True, help="输出视频路径")
    # 默认权重改为微调后的合并模型：它自带 7 类（含 nonmotor）类别表，
    # 也就是用户要求的"摩托车/自行车/三轮车 → 非机动车"。
    # 旧默认是开放词汇 yolov8s-worldv2 + 9 类提示词，输出里仍会分成
    # 摩托/自行车/三轮车三类——这正是"合并没有执行"的观感来源。
    p.add_argument("--weights", default="outputs/train/v7/weights/best.pt",
                   help="检测权重路径（默认微调合并模型 v7；传 world 权重则自动走开放词汇模式）")
    p.add_argument("--classes", default=",".join(WORLD_PROMPT_CLASSES),
                   help="开放词汇模式下的目标类型提示词，逗号分隔（仅在 --world 生效）")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="推理分辨率（默认 1280；实测优于 960，1600 反而变差）")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="0", help="0 表示第一块 GPU，cpu 表示用 CPU")
    p.add_argument("--world", action=argparse.BooleanOptionalAction, default=None,
                   help="权重是否为开放词汇（YOLO-World）模型；默认按文件名自动判断")
    p.add_argument("--merge-names", action=argparse.BooleanOptionalAction, default=True,
                   help="把 motorcycle/bicycle/tricycle 合并输出为 nonmotor（默认开）")
    p.add_argument("--motion-model", choices=["cv", "ca", "scale", "auto"], default="auto",
                   help="运动模型：auto 按梯度自动分工（默认，仅 2 秒档用维度判定）/ "
                         "cv 图像匀速（推荐对照）/ ca 匀加速（实测差 3~8 倍，不建议）/ "
                         "scale 强制维度判定")
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
    p.add_argument("--coast", type=int, default=None,
                   help=f"遮挡续推帧数：目标被遮挡后继续外推显示的帧数（默认 {COAST_FRAMES}，0 关闭）")
    p.add_argument("--no-guard", action="store_true",
                   help="关闭串轨防护（轨迹跳变切断）")
    p.add_argument("--gate-maneuver", action="store_true",
                   help="对机动（转弯/变道）中的目标不给该梯度预判（默认只标低置信）")
    # ---------------------------------------------------------- 静态目标地图
    p.add_argument("--static-map", action=argparse.BooleanOptionalAction, default=True,
                   help="开启静态目标回档机制：整段视频里 ≥80% 可见帧被识别的"
                        "路桩/标志牌/路灯/信号灯，写入隐标记并在漏检帧补回（默认开）")
    p.add_argument("--static-weights", default=None,
                   help="静态通道用的开放词汇权重（默认自动找 trafficlab-3d/repo/yolov8s-worldv2.pt）")
    p.add_argument("--static-conf", type=float, default=STATIC_CONF,
                   help=f"静态通道置信度阈值（默认 {STATIC_CONF}，召回优先）")
    p.add_argument("--static-imgsz", type=int, default=STATIC_IMGSZ,
                   help=f"静态通道推理分辨率（默认 {STATIC_IMGSZ}）")
    p.add_argument("--static-stride", type=int, default=STATIC_STRIDE,
                   help=f"静态通道每 N 帧扫一次（默认 {STATIC_STRIDE}，即每帧）")
    p.add_argument("--static-classes", default=",".join(STATIC_CLASSES),
                   help="只对哪些静态类写入隐标记，逗号分隔"
                        "（默认全部：sign,bollard,roadlamp,trafficlight）")
    p.add_argument("--static-verify", action=argparse.BooleanOptionalAction, default=True,
                   help="对隐标记位置做一次上下文校验（CLIP 判断它到底更像什么），"
                        "明确不像目标物的位置不画；拿不准的照旧画出来（默认开）")
    p.add_argument("--scene-prior", default="auto",
                   help="场景先验缓存：auto（默认）在同一机位首次运行时计算并落盘、"
                        "之后直接加载；off 每次重算；也可给显式文件路径")
    p.add_argument("--camera", default=None,
                   help="机位名。同一机位的多段视频共用一份场景先验，"
                        "例如 --camera 路口A（默认取视频文件名）")
    p.add_argument("--min-side", type=float, default=MIN_TARGET_SIDE_PX,
                   help=f"过小目标不识别：短边下限，像素（默认 {MIN_TARGET_SIDE_PX:g}）")
    p.add_argument("--min-area", type=float, default=MIN_TARGET_AREA_PX,
                   help=f"过小目标不识别：面积下限，px²（默认 {MIN_TARGET_AREA_PX:g}）")
    return p


def main() -> None:
    args = build_parser().parse_args()
    world = args.world
    if world is None:
        world = "world" in Path(args.weights).name.lower()
    result = run_clip(
        video=args.video,
        horizon=args.horizon,
        out_video=args.out,
        weights=args.weights,
        classes=tuple(c.strip() for c in args.classes.split(",") if c.strip()),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        world=world,
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
        guard=not args.no_guard,
        coast_frames=COAST_FRAMES if args.coast is None else args.coast,
        gate_maneuver=args.gate_maneuver,
        static_map=args.static_map,
        static_weights=args.static_weights,
        static_conf=args.static_conf,
        static_imgsz=args.static_imgsz,
        static_stride=args.static_stride,
        static_classes=tuple(c.strip() for c in args.static_classes.split(",") if c.strip()),
        static_verify=args.static_verify,
        scene_prior=args.scene_prior,
        camera=args.camera,
        merge_names=args.merge_names,
        min_side=args.min_side,
        min_area=args.min_area,
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    if result.get("y_horizon") is not None:
        print(f"地平线估计: y = {result['y_horizon']:.1f} px"
              f"（维度判定生效 {result.get('scale_predictions', 0)} 处）")
    sm = result.get("static_map") or {}
    if sm.get("enabled"):
        if "error" in sm:
            print(f"静态目标地图: 失败（{sm['error']}）")
        else:
            print(f"静态目标地图: 候选 {sm['clusters']} 个 → 确认 {sm['confirmed']} 个"
                  f"（否决 {sm['rejected']}），补回 {sm['injected_boxes']} 个隐标记框")
            v = sm.get("verify") or {}
            if v.get("enabled"):
                if "error" in v:
                    print(f"    上下文校验: 失败（{v['error']}）")
                else:
                    print(f"    上下文校验: 保留 {v['keep']} / 待确认 {v['review']}"
                          f" / 否掉 {v['reject']}"
                          f"（否掉的不画，待确认的照旧画出来）")
                    for it in v.get("rejected_items", [])[:6]:
                        print(f"      × {it['cls']:<13} {it['reason']} ← CLIP: {it['top'][:40]}")
            for item in sm.get("items", [])[:12]:
                print(f"    {item['cls']:<13} 检出率 {item['ratio']:.2f}"
                      f"（{item['seen']}/{item['visible']} 帧可见）")
    elif sm:
        print(f"静态目标地图: 未启用（{sm.get('reason', '')}）")


if __name__ == "__main__":
    main()
