"""构建第二版训练集：人工复核样本 + 从自采视频挖掘的稀有类补充样本。

验证集保持与第一版完全一致，这样两版指标可以直接对比。
"""

from __future__ import annotations

import collections
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "datasets"
V1 = BASE / "train_v1"
V2 = BASE / "train_v2"
MINED = BASE / "supplement_mined"
NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "sign", "bollard"]


def val_videos() -> set[str]:
    """验证集所用的视频。补充样本必须排除这些视频，否则造成数据泄漏。"""
    folder = V1 / "labels" / "val"
    if not folder.exists():
        return set()
    return {path.stem.split("_f")[0] for path in folder.glob("*.txt")}


def main() -> None:
    blocked = val_videos()
    print(f"验证集视频（补充样本须排除）：{sorted(blocked)}")
    if V2.exists():
        shutil.rmtree(V2)
    counts = collections.Counter()
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (V2 / kind / split).mkdir(parents=True, exist_ok=True)
            for path in (V1 / kind / split).iterdir():
                shutil.copyfile(path, V2 / kind / split / path.name)
        counts[split] = len(list((V2 / "images" / split).glob("*.jpg")))

    added = 0
    skipped = 0
    if MINED.exists():
        for image in (MINED / "images").glob("*.jpg"):
            if image.stem.split("_f")[0] in blocked:
                skipped += 1
                continue
            label = MINED / "labels" / (image.stem + ".txt")
            if not label.exists():
                continue
            shutil.copyfile(image, V2 / "images" / "train" / image.name)
            shutil.copyfile(label, V2 / "labels" / "train" / label.name)
            added += 1

    (V2 / "data.yaml").write_text(
        f"path: {V2.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)),
        encoding="utf-8",
    )
    print(f"训练集：{counts['train']} 张人工复核 + {added} 张稀有类补充 = {counts['train'] + added} 张")
    print(f"（已排除来自验证视频的补充样本 {skipped} 帧，避免数据泄漏）")
    print(f"验证集：{counts['val']} 张（与第一版一致，指标可直接对比）")
    print(f"data.yaml -> {V2 / 'data.yaml'}")


if __name__ == "__main__":
    main()
