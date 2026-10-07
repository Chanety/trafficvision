"""构建第六版数据集：以新素材 videoset_02 为主体，混入旧数据防止遗忘。

划分原则（按视频划，不按帧划）：
  · 验证集：112844 / 113037 / 114022 这 3 段整段留出，
    保证模型在训练中从未见过这些视频的任何一帧；
  · 训练集：其余 11 段的自动标注草图（人工复核后再升级为复核版）。

配比：
  · 新素材 267 帧 ×3 过采样（用 train.txt 重复行实现）≈ 801
  · 旧街面 train_v4 训练集 473 帧
  · VisDrone 俯拍抽样 250 帧（三轮车这一类目前只有它有真值）
  合计约 1524，新素材占一半以上，避免被旧域淹没。

注意：本版用的是"自动标注草图"，验证集标注也是自动的，
所以 v6 的绝对指标会偏乐观（自己给自己打分）；相对比较（v5 vs v6）
仍有意义。人工复核后重跑 build_train_v6.py --reviewed 即可换成真值。

用法：python bench/build_train_v6.py [--visdrone 250] [--repeat 3] [--reviewed]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "datasets"
NEW = BASE / "videoset_02_label"          # 自动标注草图的复核副本
OLD = BASE / "train_v4" / "images" / "train"
VIS = BASE / "visdrone_tricycle"
OUT = BASE / "train_v6"
VAL_VIDEOS = {"VID_20261004_112844", "VID_20261004_113037", "VID_20261004_114022"}
NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "sign", "bollard", "tricycle"]


def main() -> None:
    visdrone_n = 250
    repeat = 3
    if "--visdrone" in sys.argv:
        visdrone_n = int(sys.argv[sys.argv.index("--visdrone") + 1])
    if "--repeat" in sys.argv:
        repeat = int(sys.argv[sys.argv.index("--repeat") + 1])

    new_images = sorted((NEW / "images").glob("*.jpg"))
    train_new = [p for p in new_images if p.stem.split("_f")[0] not in VAL_VIDEOS]
    val_new = [p for p in new_images if p.stem.split("_f")[0] in VAL_VIDEOS]
    if not train_new or not val_new:
        print("新素材目录为空，先跑 bench/make_label_set.py 生成自动标注草图")
        return

    old_images = sorted(OLD.glob("*.jpg"))
    vis_all = sorted((VIS / "images").glob("visdrone_train_*.jpg"))
    step = max(1, len(vis_all) // visdrone_n)
    vis_images = vis_all[::step][:visdrone_n]

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    train_lines: list[str] = [str(p) for p in train_new] * repeat
    train_lines += [str(p) for p in old_images]
    train_lines += [str(p) for p in vis_images]
    (OUT / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (OUT / "val.txt").write_text("\n".join(str(p) for p in val_new) + "\n", encoding="utf-8")
    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: train.txt\nval: val.txt\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)),
        encoding="utf-8",
    )

    print(f"新素材：训练 {len(train_new)} 帧（{len(VAL_VIDEOS)} 段留作验证）+ 验证 {len(val_new)} 帧"
          f" = {len(new_images)}")
    print(f"训练组合：新素材 {len(train_new)}×{repeat}={len(train_new) * repeat}"
          f" + 旧街面 {len(old_images)} + 俯拍 {len(vis_images)}"
          f" = {len(train_lines)} 条")
    print(f"data.yaml -> {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
