"""跟踪器配对对比：只在"同一批物理目标"上比误差，消除样本集合不同的干扰。

为什么需要它：换跟踪器会改变"哪些目标被跟踪"，样本集合一变，
误差中位数就没有可比性（跟踪更多难目标的一方天然吃亏）。

做法：以基准配置的样本为参考，用 (视频, 帧号, 类别, IoU≥0.5) 把其它配置的样本配上去，
只在配对成功的子集上重算各配置的误差与异常率。

用法：python bench/tracker_matched_bench.py <基准目录> <目录2> <目录3> ...
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_diagnostics import analyze  # noqa: E402


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_samples(out_dir: Path) -> dict[tuple[str, int, str], list[dict]]:
    buckets: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in analyze(payload, 1.0):
            buckets[(row["video"], row["frame_idx"], row["cls"])].append(row)
    return buckets


def match(reference: dict, other: dict, iou_thr: float = 0.5):
    """返回 (参考误差, 对方误差, 参考异常标记, 对方异常标记) 的配对数组。"""
    pairs = []
    for key, rows in reference.items():
        others = other.get(key)
        if not others:
            continue
        used = set()
        for r in rows:
            best, best_iou = None, iou_thr
            for i, o in enumerate(others):
                if i in used:
                    continue
                score = iou(r["bbox"], o["bbox"])
                if score >= best_iou:
                    best, best_iou = i, score
            if best is None:
                continue
            used.add(best)
            pairs.append((r["error"], others[best]["error"],
                          r["implied_ratio"] > 2.0, others[best]["implied_ratio"] > 2.0))
    return pairs


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        return
    base_dir = Path(sys.argv[1])
    base = load_samples(base_dir)
    n_base = sum(len(v) for v in base.values())
    print(f"基准 {base_dir.name}：{n_base} 个样本\n")
    header = (f"{'配置':<24}{'配对样本':>9}{'配对率':>8}{'基准中位':>10}{'该配置中位':>12}"
              f"{'基准均值':>10}{'该配置均值':>12}{'异常率变化':>12}")
    print(header)
    print("-" * len(header))
    for item in sys.argv[2:]:
        path, _, label = item.partition("=")
        other_dir = Path(path)
        other = load_samples(other_dir)
        pairs = match(base, other)
        if not pairs:
            print(f"{(label or other_dir.name):<24} 没有配对样本")
            continue
        arr = np.array(pairs)
        base_err, other_err = arr[:, 0], arr[:, 1]
        base_anom, other_anom = arr[:, 2], arr[:, 3]
        print(f"{(label or other_dir.name):<24}{len(pairs):>9}"
              f"{len(pairs) / n_base * 100:>7.0f}%{np.median(base_err):>10.1f}"
              f"{np.median(other_err):>12.1f}{base_err.mean():>10.1f}{other_err.mean():>12.1f}"
              f"{base_anom.mean() * 100:>6.1f}%→{other_anom.mean() * 100:.1f}%")
    print("\n配对率＝基准样本里能在该配置中找到同一目标的占比；")
    print("异常率＝窗口内真实位移换算速度 > 估计速度 2 倍的样本比例（ID 跳变特征）。")


if __name__ == "__main__":
    main()
