"""从 COCO 补充训练样本：只下载含指定类别的图片，转成我们的 YOLO 格式。

为什么需要：在 219 张自采图上微调属于小样本训练，稀有类（自行车、摩托）
会发生灾难性遗忘——预训练时本来认识，微调后反而退化。
用外部数据做"复习"可以显著改善稀有类。

做法：
  1. 下载 COCO 标注文件（约 25 MB）
  2. 只挑含目标类别的图片（按需限定数量），逐张下载
  3. 按我们的类别顺序转换并写出 YOLO 标注

用法：python bench/fetch_coco_supplement.py <输出目录> [每类上限]
"""

from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

ANNOT_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_URL = "http://images.cocodataset.org/val2017/{}.jpg"

# COCO 类别名 -> 我们的类别编号
COCO_TO_OURS = {
    "person": 5,
    "bicycle": 4,
    "car": 0,
    "motorcycle": 3,
    "bus": 2,
    "truck": 1,
    "stop sign": 6,      # 归入我们的 sign（标志牌）
}
OUR_NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "sign", "bollard"]
# 稀有类优先：自行车与摩托是当前最弱的
PRIORITY = ["bicycle", "motorcycle", "truck", "sign", "bus", "car", "person"]


def download(url: str, dest: Path, timeout: int = 120) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return True
    except Exception as exc:
        print(f"  下载失败 {url}: {exc}")
        return False


def main() -> None:
    out = Path(sys.argv[1])
    per_class = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    cache = out / "_cache"
    images_dir = out / "images"
    labels_dir = out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    zip_path = cache / "annotations_trainval2017.zip"
    if not download(ANNOT_URL, zip_path):
        print("标注文件下载失败，退出")
        return
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("annotations/instances_val2017.json") as fh:
            data = json.load(fh)

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    images = {img["id"]: img for img in data["images"]}
    anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        name = cat_id_to_name.get(ann["category_id"])
        if name in COCO_TO_OURS and ann.get("iscrowd", 0) == 0:
            anns_by_image[ann["image_id"]].append((name, ann["bbox"]))

    # 按优先级挑选图片，稀有类先选
    chosen: list[int] = []
    seen: set[int] = set()
    for target in PRIORITY:
        count = 0
        for image_id, anns in anns_by_image.items():
            if count >= per_class or image_id in seen:
                continue
            if any(name == target for name, _ in anns):
                chosen.append(image_id)
                seen.add(image_id)
                count += 1
        print(f"优先类别 {target:<11} 选中 {count} 张")

    print(f"共选中 {len(chosen)} 张图片，开始下载…")
    written = 0
    for index, image_id in enumerate(chosen):
        info = images[image_id]
        name = f"{info['file_name'].replace('.jpg','')}"
        image_path = images_dir / f"coco_{name}.jpg"
        if not image_path.exists():
            if not download(IMAGE_URL.format(name), image_path):
                continue
        w, h = info["width"], info["height"]
        lines = []
        for cls_name, (x, y, bw, bh) in anns_by_image[image_id]:
            cid = COCO_TO_OURS[cls_name]
            lines.append(f"{cid} {(x + bw/2)/w:.6f} {(y + bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}")
        (labels_dir / f"coco_{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1
        if (index + 1) % 100 == 0:
            print(f"  已处理 {index + 1}/{len(chosen)}")

    print(f"完成：{written} 张图片与标注写入 {out}")
    print("类别顺序：" + ", ".join(f"{i}={n}" for i, n in enumerate(OUR_NAMES)))


if __name__ == "__main__":
    main()
