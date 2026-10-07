"""构建第四版数据集：人工复核 + 扩充后的稀有类样本（验证集与训练集同时扩充）。

与第三版的区别：
  · 验证集从 58 张扩到 116 张（新增帧同样来自那 3 段验证视频，优先含自行车）
  · 训练集加入第二次挖掘的 220 帧（严格排除验证视频，避免泄漏）

注意：新增的验证帧是模型自动标注的，对自行车这类弱项存在循环论证风险，
结论需谨慎解读；如需可信结论，应人工复核这批验证帧。
"""

from __future__ import annotations

import collections
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "datasets"
V1 = BASE / "train_v1"
V4 = BASE / "train_v4"
SOURCES = {
    "train": [BASE / "supplement_mined", BASE / "supplement_mined_extra"],
    "val": [BASE / "supplement_val_extra"],
}
NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "sign", "bollard"]


def val_videos() -> set[str]:
    return {p.stem.split("_f")[0] for p in (V1 / "labels" / "val").glob("*.txt")}


def main() -> None:
    blocked = val_videos()
    if V4.exists():
        shutil.rmtree(V4)
    counts = collections.Counter()
    skipped = 0

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (V4 / kind / split).mkdir(parents=True, exist_ok=True)
        for path in (V1 / "images" / split).iterdir():
            shutil.copyfile(path, V4 / "images" / split / path.name)
            shutil.copyfile(V1 / "labels" / split / (path.stem + ".txt"),
                            V4 / "labels" / split / (path.stem + ".txt"))
            counts[split] += 1

        for source in SOURCES[split]:
            if not source.exists():
                continue
            for image in (source / "images").glob("*.jpg"):
                # 训练集必须排除来自验证视频的帧，防止数据泄漏
                if split == "train" and image.stem.split("_f")[0] in blocked:
                    skipped += 1
                    continue
                label = source / "labels" / (image.stem + ".txt")
                if not label.exists():
                    continue
                if (V4 / "images" / split / image.name).exists():
                    continue
                shutil.copyfile(image, V4 / "images" / split / image.name)
                shutil.copyfile(label, V4 / "labels" / split / label.name)
                counts[split] += 1

    (V4 / "data.yaml").write_text(
        f"path: {V4.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)),
        encoding="utf-8",
    )
    print(f"训练集 {counts['train']} 张（已排除验证视频样本 {skipped} 帧）")
    print(f"验证集 {counts['val']} 张")
    print(f"data.yaml -> {V4 / 'data.yaml'}")


if __name__ == "__main__":
    main()
