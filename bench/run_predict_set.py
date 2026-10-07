"""批量跑预判（只写 JSON，不渲染视频），用于跟踪器/参数对比实验。

用法：
    python bench/run_predict_set.py <视频目录> <输出目录> <权重> [选项]
选项：
    --tracker bytetrack.yaml   跟踪器配置（bytetrack / botsort / ocsort / 自定义 yaml 路径）
    --frames 300               每段最多处理帧数
    --limit 5                  最多处理几段视频
    --world                    权重是开放词汇模型（需要 set_classes）
    --label 前缀               输出文件名的前缀，便于区分不同配置
    --conf 0.25                检测置信度阈值（ByteTrack 的低分召回需要放低到 0.1 左右才生效）
    --coast 12                 遮挡续推帧数（0 关闭）
    --no-guard                 关闭串轨防护
    --motion auto              运动模型：cv（图像匀速）/ ca（匀加速）/ scale（维度判定）/ auto
    --gate-maneuver            机动目标不给预判（默认只标低置信）
    --imgsz 1280               推理分辨率（实测 1280 优于 960，见 docs/11）
    --no-static-map            关闭静态目标地图（对照实验用）
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trafficvision.pipeline import run_clip  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def arg(name: str, default: str | None = None) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        return
    video_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    weights = sys.argv[3]
    tracker = arg("--tracker", "bytetrack.yaml") or "bytetrack.yaml"
    max_frames = int(arg("--frames", "300") or 300)
    limit = int(arg("--limit", "5") or 5)
    label = arg("--label", "") or ""
    conf = float(arg("--conf", "0.25") or 0.25)
    world = "--world" in sys.argv
    coast = int(arg("--coast", "12") or 12)
    guard = "--no-guard" not in sys.argv
    motion = arg("--motion", "auto") or "auto"
    gate_maneuver = "--gate-maneuver" in sys.argv
    static_map = "--no-static-map" not in sys.argv
    imgsz = int(arg("--imgsz", "1280") or 1280)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(p for p in video_dir.rglob("*") if p.suffix.lower() in VIDEO_EXT)[:limit]
    print(f"跟踪器 {tracker}｜conf {conf}｜串轨防护 {'开' if guard else '关'}｜"
          f"遮挡续推 {coast} 帧｜运动模型 {motion}｜权重 {Path(weights).name}｜"
          f"imgsz {imgsz}｜每段 {max_frames} 帧｜{len(videos)} 段")
    for video in videos:
        json_path = out_dir / f"{label}{video.stem}.json"
        run_clip(
            video=video, horizon=1.0, out_video=out_dir / "unused.mp4", weights=str(weights),
            imgsz=imgsz, conf=conf, device="0", world=world, out_json=json_path,
            render=False, max_frames_override=max_frames, tracker=tracker,
            guard=guard, coast_frames=coast,
            motion_model=motion,
            gate_maneuver=gate_maneuver,
            static_map=static_map,
        )
        print(f"  {video.name} -> {json_path.name}")


if __name__ == "__main__":
    main()
