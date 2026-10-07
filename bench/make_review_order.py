"""生成人工复核的优先顺序（review_order.txt）。

复核是整条链路最贵的一环，所以顺序要按"信息量"排：
  1. 含稀有类的帧优先（三轮车 > 自行车 > 标志牌/路桩 > 公交/卡车）；
  2. 同分情况下目标多的帧优先（难例）；
  3. 轮流从各视频取，避免前几十帧全来自同一段路。

标注工具 tools/annotate_zh.py 会读取数据集目录下的 review_order.txt，
按这个顺序翻页，因此复核者可以随时停下，先做完的都是最值钱的帧。

用法：python bench/make_review_order.py <标注集目录>
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

WEIGHTS = {8: 100.0,   # tricycle
           4: 40.0,    # bicycle
           6: 12.0,    # sign
           7: 6.0,     # bollard
           2: 10.0,    # bus
           1: 8.0,     # truck
           3: 3.0}     # motorcycle


def main() -> None:
    dataset = Path(sys.argv[1])
    labels_dir = dataset / "labels"
    if not labels_dir.exists():
        print(f"找不到标注目录：{labels_dir}")
        return

    per_video: dict[str, list[tuple[float, int, str]]] = collections.defaultdict(list)
    for label in sorted(labels_dir.glob("*.txt")):
        score = 0.0
        count = 0
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            count += 1
            score += WEIGHTS.get(int(parts[0]), 0.0)
        video = label.stem.split("_f")[0]
        per_video[video].append((score + count * 0.1, count, label.name))

    for video in per_video:
        per_video[video].sort(key=lambda item: (-item[0], -item[1]))

    order: list[str] = []
    queues = {v: list(items) for v, items in per_video.items()}
    while queues:
        for video in sorted(queues):
            if queues[video]:
                order.append(queues[video].pop(0)[2].replace(".txt", ".jpg"))
            if not queues[video]:
                del queues[video]

    out = dataset / "review_order.txt"
    out.write_text("\n".join(order) + "\n", encoding="utf-8")
    print(f"{len(order)} 帧的复核顺序 -> {out}")
    print("前 12 帧（最优先）: " + ", ".join(order[:12]))


if __name__ == "__main__":
    main()
