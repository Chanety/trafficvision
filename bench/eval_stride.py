"""跳帧的严格精度对照：真值固定为 stride=1 的稠密记录。

为什么需要这个脚本
------------------
管线自带的评估用**同一次运行**的未来帧当真值。跳帧时，被评估的预判和真值同时被
稀疏化，相邻观测相隔 stride/30 秒，"1 秒后"的真值是在更宽的区间上线性插值的，
插值会把真值抹平——于是 stride 越大误差读数越小，这是评估口径造成的假象。

本脚本把真值锁死为 stride=1 的稠密观测，只让预判来自不同 stride 的运行，
因此三个配置的误差**可直接比较**。

ID 不可比的问题
---------------
跟踪器在每次运行时自行分配 track id，稀疏运行的 id 与稠密运行无关。因此用
**同一时刻的空间最近邻**（同类别、中心距离最小、阈值内）把稀疏预测挂到稠密轨迹上，
再用该轨迹在 t+h 时刻的稠密观测（线性插值）作为真值。

用法：python bench/eval_stride.py [视频] [输出目录]
"""

from __future__ import annotations

import json
import subprocess
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "trafficlab-3d" / "env" / "Scripts" / "python.exe"
OUT = ROOT / "outputs" / "stride_eval"
DEFAULT_VIDEO = ROOT / "datasets" / "videoset_02_origin" / "VID_20261004_112222.mp4"
HORIZONS = ("0.1", "0.5", "1.0", "2.0")
SRC_FRAMES = 600          # 每个配置统一消耗的源帧数
MATCH_TOL = 15.0          # 空间最近邻匹配阈值（像素）


def load(path: Path):
    """返回 (times, per_frame_id2c, objects)。时间轴对空帧做线性补齐。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload["frames"]
    fps = float(payload["video_info"]["fps"]) or 30.0
    n = len(frames)
    times = np.full(n, np.nan)
    for i, f in enumerate(frames):
        if f:
            times[i] = f[0]["t"]
    if np.isnan(times[0]):
        times[0] = 0.0
    known = np.where(~np.isnan(times))[0]
    for i in range(n):
        if np.isnan(times[i]):
            prev = known[known < i]
            times[i] = (times[prev[-1]] + (i - prev[-1]) / fps) if prev.size else times[known[0]]
    # 真值池：续推位置是推算值，不能当真值（与管线内评估一致）
    pool = [{o["id"]: tuple(o["centroid"]) for o in f if not o.get("coasting")}
            for f in frames]
    return times, pool, frames


def truth_at(times, pool, t_target, tid):
    i1 = int(np.searchsorted(times, t_target))
    if i1 >= len(pool):
        return None
    i0 = max(0, i1 - 1)
    span = times[i1] - times[i0]
    frac = (t_target - times[i0]) / span if span > 1e-9 else 0.0
    a, b = pool[i0].get(tid), pool[i1].get(tid)
    if a is not None and b is not None:
        return (a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac)
    if a is not None and frac < 0.5:
        return a
    if b is not None and frac >= 0.5:
        return b
    return None


def match_dense(times, pool, frames, t, cls, centroid):
    """把稀疏预测挂到稠密轨迹：同一时刻、同类别、中心最近且在阈值内。"""
    i = int(np.searchsorted(times, t))
    i = min(max(i, 0), len(pool) - 1)
    best, best_d = None, MATCH_TOL
    for o in frames[i]:
        if o.get("coasting") or o["cls"] != cls:
            continue
        cx, cy = o["centroid"]
        d = float(np.hypot(cx - centroid[0], cy - centroid[1]))
        if d < best_d:
            best, best_d = o["id"], d
    return best


def evaluate(sparse_json: Path, dense):
    d_times, d_pool, d_frames = dense
    times, _, frames = load(sparse_json)
    errs = {h: [] for h in HORIZONS}
    matched = miss = 0
    for i, objs in enumerate(frames):
        t = times[i]
        for o in objs:
            if not o.get("predictable") or o.get("coasting"):
                continue
            tid = match_dense(d_times, d_pool, d_frames, t, o["cls"], o["centroid"])
            if tid is None:
                miss += 1
                continue
            matched += 1
            for h in HORIZONS:
                pred = o["preds"].get(h)
                if pred is None:
                    continue
                truth = truth_at(d_times, d_pool, t + float(h), tid)
                if truth is None:
                    continue
                pc = ((pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0)
                errs[h].append(float(np.hypot(pc[0] - truth[0], pc[1] - truth[1])))
    return errs, matched, miss


def run_cli(video: Path, stride: int, nf: int, js: Path) -> float:
    import time
    cmd = [str(PY), "-m", "trafficvision.cli", "--video", str(video), "--horizon", "1",
           "--out", str(OUT / "unused.mp4"), "--max-frames", str(nf),
           "--frame-stride", str(stride), "--imgsz", "1280",
           "--no-render", "--no-static-map", "--json", str(js)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"stride={stride} 运行失败：\n{' '.join(r.stderr.splitlines()[-5:])}")
    return time.time() - t0


def main() -> None:
    global OUT
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO
    OUT = OUT / video.stem          # 每段视频独立目录，避免复用上一段的缓存
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"视频 {video.name}；每配置消耗 {SRC_FRAMES} 个源帧\n")

    configs = [(1, SRC_FRAMES), (2, SRC_FRAMES // 2), (3, SRC_FRAMES // 3)]
    rows = []
    dense = None
    for stride, nf in configs:
        js = OUT / f"stride{stride}.json"
        if not js.exists():
            run_cli(video, stride, nf, js)
        if stride == 1:
            dense = load(js)
    if dense is None:
        raise SystemExit("缺少 stride=1 的稠密记录")

    for stride, nf in configs:
        js = OUT / f"stride{stride}.json"
        errs, matched, miss = evaluate(js, dense)
        row = {"stride": stride, "匹配上": matched, "未匹配": miss}
        for h in HORIZONS:
            a = np.array(errs[h]) if errs[h] else np.array([np.nan])
            row[f"{h}s 中位"] = round(float(np.median(a)), 1) if a.size else np.nan
            row[f"{h}s P90"] = round(float(np.percentile(a, 90)), 1) if a.size else np.nan
        rows.append(row)
        print(f"stride={stride}  匹配 {matched} 未匹配 {miss}")

    # 与基线比较
    base = rows[0]
    print("\n=== 真值固定为 stride=1 稠密记录 ===")
    hdr = f"{'配置':<10}{'1s中位':>9}{'相对基线':>10}{'1s P90':>9}{'2s中位':>9}"
    print(hdr)
    for r in rows:
        d = (r["1.0s 中位"] - base["1.0s 中位"]) / base["1.0s 中位"] * 100
        tag = "基线" if r["stride"] == 1 else f"{d:+.1f}%"
        print(f"stride={r['stride']:<3}{r['1.0s 中位']:>9.1f}{tag:>10}"
              f"{r['1.0s P90']:>9.1f}{r['2.0s 中位']:>9.1f}")

    (OUT / "stride_eval.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {OUT / 'stride_eval.json'}")


if __name__ == "__main__":
    main()
