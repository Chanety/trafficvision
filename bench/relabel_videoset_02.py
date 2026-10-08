"""按 2026-10-06 的新口径重新自动标注 videoset_02。

相对旧标注（datasets/videoset_02_label）改了四件事：

  1. **提示词集成**：路桩不再只用单词 "bollard"（实测对混凝土路桩只有
     0.02~0.16 分），改成 bollard / concrete bollard / barrier post /
     traffic bollard 四个提示词投票 + 同类 NMS。这是"左边两根完全标不出来"
     的直接修复。
  2. **类别合并**：摩托车 / 自行车 / 三轮车 → nonmotor（非机动车）。
     用户要求的口径，旧的自动标注一直是分开的三类。
  3. **新增两类静态结构**：roadlamp（路灯）、trafficlight（信号灯）。
  4. **骑乘状态**：骑车人不再删除，而是单独标成 rider（非机动车驾驶员），
     与纯步行的 person 区分。

输出 10 类（合并方案）：
    0 car  1 truck  2 bus  3 nonmotor  4 person  5 sign  6 bollard
    7 roadlamp  8 trafficlight  9 rider

用法：
    python bench/relabel_videoset_02.py
    python bench/relabel_videoset_02.py --src datasets/videoset_02_label \
        --out datasets/videoset_02_label2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]

import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from trafficvision.config import (  # noqa: E402
    MERGED_CLASSES,
    MERGE_MAP,
    STATIC_CLASSES,
    STATIC_PROMPTS,
)
from trafficvision.labelrules import classify_riders  # noqa: E402

WORLD = ROOT / "trafficlab-3d" / "repo" / "yolov8s-worldv2.pt"

# 动目标提示词（原词）
CORE_PROMPTS = ("car", "truck", "bus", "motorcycle", "bicycle", "person", "tricycle")

# 每个概念允许的最低置信度：静态结构放宽（靠跨帧投票去伪），动目标收紧
MIN_CONF = {
    "car": 0.15, "truck": 0.15, "bus": 0.15, "motorcycle": 0.15,
    "bicycle": 0.15, "person": 0.15, "tricycle": 0.15,
    "sign": 0.05, "bollard": 0.05, "roadlamp": 0.05, "trafficlight": 0.05,
}

# 最小框面积（像素²）：低于此值基本是噪声
MIN_AREA = 300.0


def build_prompts():
    prompts: list[str] = []
    mapping: dict[str, str] = {}
    for word in CORE_PROMPTS:
        prompts.append(word)
        mapping[word] = word
    for cls, words in STATIC_PROMPTS.items():
        for word in words:
            prompts.append(word)
            mapping[word] = cls
    # 去重，保持顺序
    seen, uniq = set(), []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq, mapping


def nms(items, iou_thr=0.5):
    kept = []
    for conf, box in sorted(items, key=lambda z: -z[0]):
        drop = False
        for _, kb in kept:
            ix1, iy1 = max(box[0], kb[0]), max(box[1], kb[1])
            ix2, iy2 = min(box[2], kb[2]), min(box[3], kb[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0:
                continue
            aa = (box[2] - box[0]) * (box[3] - box[1])
            ab = (kb[2] - kb[0]) * (kb[3] - kb[1])
            union = aa + ab - inter
            if union > 0 and inter / union >= iou_thr:
                drop = True
                break
        if not drop:
            kept.append((conf, box))
    return kept


def to_yolo(cls_id: int, box, w: int, h: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [float(cls_id), ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h,
            (x2 - x1) / w, (y2 - y1) / h]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "datasets" / "videoset_02_label"))
    ap.add_argument("--out", default=str(ROOT / "datasets" / "videoset_02_label2"))
    ap.add_argument("--weights", default=str(WORLD))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.05, help="底层阈值，之后按类再卡一次")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    src, out = Path(args.src), Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    prompts, mapping = build_prompts()
    print(f"提示词 {len(prompts)} 个：{prompts}")

    model = YOLO(args.weights)
    model.set_classes(prompts)

    old_counts, new_counts = Counter(), Counter()
    old_dir = src / "labels"
    riders_total = 0
    new_manifest = []

    # 分批读图 + 推理
    imgs, metas = [], []
    def flush():
        nonlocal riders_total
        if not imgs:
            return
        results = model.predict(imgs, imgsz=args.imgsz, conf=args.conf, iou=0.6,
                                device=args.device, verbose=False, batch=4)
        for meta, res in zip(metas, results):
            h, w = res.orig_shape
            per: dict[str, list] = {}
            for b in res.boxes:
                name = res.names[int(b.cls)]
                cls = mapping.get(name)
                if cls is None:
                    continue
                conf = float(b.conf)
                if conf < MIN_CONF.get(cls, 0.15):
                    continue
                box = [float(v) for v in b.xyxy[0]]
                if (box[2] - box[0]) * (box[3] - box[1]) < MIN_AREA:
                    continue
                per.setdefault(cls, []).append((conf, box))

            base = []          # 基础类别编号（含 person=5 / rider=11）
            base_names = ["car", "truck", "bus", "motorcycle", "bicycle", "person",
                          "sign", "bollard", "tricycle", "roadlamp", "trafficlight", "rider"]
            for cls, items in per.items():
                for _conf, box in nms(items):
                    base.append(to_yolo(base_names.index(cls), box, w, h))

            # 骑乘判定必须在合并之前做（合并后就分不出"在骑什么"了）
            base, changed = classify_riders(base)
            riders_total += changed

            merged = []
            for b in base:
                cid = MERGE_MAP[int(b[0])]
                merged.append([float(cid)] + [float(v) for v in b[1:]])
                new_counts[MERGED_CLASSES[cid]] += 1

            stem = Path(meta["image"]).stem
            (out / "labels" / f"{stem}.txt").write_text(
                "\n".join(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in merged)
                + ("\n" if merged else ""), encoding="utf-8")

            new_manifest.append({
                "image": f"images/{stem}.jpg",
                "label": f"labels/{stem}.txt",
                "source_video": meta["source_video"],
                "frame_index": meta["frame_index"],
                "time_s": meta["time_s"],
                "w": int(w),
                "h": int(h),
                "merged_boxes": len(merged),
            })
            # 图片：硬链接，不额外占盘
            src_img = src / meta["image"]
            dst_img = out / "images" / f"{stem}.jpg"
            if not dst_img.exists():
                try:
                    os.link(src_img, dst_img)
                except OSError:
                    shutil.copy2(src_img, dst_img)
        imgs.clear()
        metas.clear()

    for i, meta in enumerate(manifest, 1):
        img = cv2.imread(str(src / meta["image"]))
        if img is None:
            continue
        imgs.append(img)
        metas.append(meta)
        if len(imgs) >= 16:
            flush()
            print(f"  {i}/{len(manifest)}", flush=True)
    flush()

    # 旧标注整体统计（基础 9 类口径）
    old_names = ["car", "truck", "bus", "motorcycle", "bicycle", "person",
                 "sign", "bollard", "tricycle"]
    for meta in manifest:
        p = old_dir / f"{Path(meta['image']).stem}.txt"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                old_counts[old_names[int(float(parts[0]))]] += 1

    (out / "classes.txt").write_text("\n".join(MERGED_CLASSES) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images\nval: images\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(MERGED_CLASSES)),
        encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(new_manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    (out / "relabel_report.json").write_text(json.dumps({
        "prompts": prompts,
        "min_conf": MIN_CONF,
        "min_area": MIN_AREA,
        "old_counts": dict(old_counts),
        "new_counts": dict(new_counts),
        "riders_reclassified": riders_total,
        "frames": len(new_manifest),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    def merged_view(c: Counter) -> Counter:
        m = Counter()
        for k, v in c.items():
            key = {"motorcycle": "nonmotor", "bicycle": "nonmotor",
                   "tricycle": "nonmotor"}.get(k, k)
            m[key] += v
        return m

    pruned = prune_static(out, new_manifest)

    # 清理后重新统计（前面的计数是清理前的）
    new_counts = Counter()
    for m in new_manifest:
        p = out / m["label"]
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                new_counts[MERGED_CLASSES[int(float(parts[0]))]] += 1

    old_m, new_m = merged_view(old_counts), merged_view(new_counts)
    print("\n=== 合并口径下的新旧对比（框数）===")
    print(f"{'类别':<14}{'旧自动标注':>12}{'新自动标注':>12}")
    for cls in MERGED_CLASSES:
        print(f"{cls:<14}{old_m.get(cls, 0):>12}{new_m.get(cls, 0):>12}")
    print(f"{'合计':<14}{sum(old_m.values()):>12}{sum(new_m.values()):>12}")
    print(f"\n静态类跨帧投票清理：删掉 {pruned} 个偶发框（保留通过 80% 一致性检验的位置）")
    print(f"\n骑乘状态改判为 rider 的框：{riders_total}")
    print(f"输出目录：{out}")


def prune_static(out: Path, manifest: list[dict]) -> int:
    """用"回档查看"的跨帧投票，把静态类的偶发误检从标注里删掉。

    开放词汇模型对 traffic light / street light 这类词非常敏感（路口的红色
    灯箱、招牌都会被响应），一次推理能给出 20~30 个框。但真正的静态结构会
    在整段视频的每一帧都出现在同一位置，偶发误检不会——所以直接套用产品里
    那套 ≥80% 一致性判据，就能把标注里的噪声顺手清掉。
    """
    from trafficvision.staticmap import StaticObjectMap

    by_video: dict[str, list[dict]] = {}
    for m in manifest:
        by_video.setdefault(m["source_video"], []).append(m)

    dropped = 0
    unpruned = 0
    for video, items in by_video.items():
        items.sort(key=lambda z: z["frame_index"])
        smap = StaticObjectMap()
        frames = []
        for i, m in enumerate(items):
            w, h = m.get("w", 1920), m.get("h", 1080)
            boxes = []
            p = out / m["label"]
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cid = int(float(parts[0]))
                    cx, cy, bw, bh = (float(v) for v in parts[1:])
                    name = MERGED_CLASSES[cid] if 0 <= cid < len(MERGED_CLASSES) else f"c{cid}"
                    box = [(cx - bw / 2) * w, (cy - bh / 2) * h,
                           (cx + bw / 2) * w, (cy + bh / 2) * h]
                    boxes.append((cid, name, box))
            frames.append(boxes)
            statics = [(name, box, 1.0) for _cid, name, box in boxes if name in STATIC_CLASSES]
            # dynamics 必须来自同帧的**非静态**标注，否则遮挡永远不计入
            # 可见性判断——被车/人挡住的真静态目标会因为分母错误而被当成漏检删掉。
            dynamics = [{"bbox": box} for _cid, name, box in boxes
                        if name not in STATIC_CLASSES]
            smap.observe(i, statics, dynamics=dynamics, span=1)

        confirmed = smap.confirmed()
        if not confirmed:
            # 一个位置都没通过一致性检验，通常意味着这批标注的静态类基本是偶发误检
            # （或全部被遮挡）。此时静默跳过等于"一个都不清"，必须显式提示。
            print(f"[警告] {video}：没有任何静态位置通过一致性检验，"
                  f"该段未做清理，请人工确认")
            unpruned += 1
            continue
        from trafficvision.staticmap import iou as _iou

        for i, m in enumerate(items):
            keep, changed = [], False
            for cid, name, box in frames[i]:
                if name in STATIC_CLASSES:
                    ok = any(d["cls"] == name and _iou(box, d["box"]) >= 0.30
                             for d in confirmed)
                    if not ok:
                        dropped += 1
                        changed = True
                        continue
                keep.append((cid, box, m.get("w", 1920), m.get("h", 1080)))
            if changed:
                (out / m["label"]).write_text(
                    "".join("%d %.6f %.6f %.6f %.6f\n" % tuple(
                        [c] + _to_yolo_norm(b, w, h)) for c, b, w, h in keep),
                    encoding="utf-8")
                m["merged_boxes"] = len(keep)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    if unpruned:
        print(f"[提示] 共 {unpruned} 段视频未做清理（没有任何位置通过一致性检验），"
              f"这些段的静态标注保持原样，需人工复核")
    return dropped


def _to_yolo_norm(box, w, h):
    x1, y1, x2, y2 = box
    return [((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h, (x2 - x1) / w, (y2 - y1) / h]


if __name__ == "__main__":
    main()
