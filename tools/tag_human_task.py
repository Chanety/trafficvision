"""给已生成的人工任务产物打标签（幂等，可重复运行）。

处理对象：
  1. `dataset_batches/batch_*.zip` —— 往包里注入《★人工任务包_请先读我.txt》，
     并在包内《离线包说明.txt》《标注说明.txt》顶部加标记；同步更新 manifest 的
     SHA-256（注入会改变文件哈希）与人工任务字段。
  2. 目录形式的复核产物（如 outputs/static_review_v3）—— 落一份标记文件。

用法：
    python tools/tag_human_task.py            # 处理上面列出的对象
    python tools/tag_human_task.py --check    # 只检查有没有漏打标签
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from human_task import BADGE, NOTICE_FILENAME, has_tag, notice_text  # noqa: E402

BATCH_DIR = ROOT / "dataset_batches"
MANIFEST = BATCH_DIR / "manifest.json"
# 包内需要在顶部加标记的说明文件
INNER_DOCS = ("离线包说明.txt", "标注说明.txt")
# 目录形式的人工任务产物
DIR_TARGETS = [
    ROOT / "outputs" / "static_review_v3",
    ROOT / "outputs" / "review_videoset_02_v2",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tag_zip(zip_path: Path) -> dict:
    """注入标记文件 + 给内部说明加标记；返回统计。

    采用"重写"而不是"追加"：追加同名条目会在包里留下重复项，
    不同解压工具取哪一个不确定，容易埋坑。重写一次几十 MB，代价可接受。
    """
    added_notice = tagged_docs = 0
    tmp = zip_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(zip_path) as src, \
            zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as dst:
        names = set(src.namelist())
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename in INNER_DOCS:
                text = data.decode("utf-8", errors="ignore")
                if not has_tag(text):
                    data = (BADGE + "\n" + text).encode("utf-8")
                    tagged_docs += 1
            dst.writestr(item, data)          # 保留原始条目元数据
        if NOTICE_FILENAME not in names:
            dst.writestr(NOTICE_FILENAME, notice_text())
            added_notice = 1
    tmp.replace(zip_path)
    return {"zip": zip_path.name, "notice_added": added_notice, "docs_tagged": tagged_docs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查，不修改")
    args = ap.parse_args()

    report = {"zips": [], "dirs": []}

    for zp in sorted(BATCH_DIR.glob("batch_*.zip")):
        if args.check:
            with zipfile.ZipFile(zp) as zf:
                has = NOTICE_FILENAME in zf.namelist()
            report["zips"].append({"zip": zp.name, "tagged": has})
            print(f"  {'已标记' if has else '缺标记'}  {zp.name}")
            continue
        info = tag_zip(zp)
        info["sha256"] = sha256(zp)
        info["bytes"] = zp.stat().st_size
        report["zips"].append(info)
        print(f"  {zp.name}: 注入标记 {info['notice_added']}，"
              f"说明加标记 {info['docs_tagged']}，新 SHA-256 {info['sha256'][:16]}…")

    for d in DIR_TARGETS:
        if not d.exists():
            continue
        if args.check:
            ok = (d / NOTICE_FILENAME).exists()
            report["dirs"].append({"dir": str(d.relative_to(ROOT)), "tagged": ok})
            print(f"  {'已标记' if ok else '缺标记'}  {d.relative_to(ROOT)}")
            continue
        path = d / NOTICE_FILENAME
        path.write_text(notice_text(), encoding="utf-8")
        report["dirs"].append({"dir": str(d.relative_to(ROOT)), "tagged": True})
        print(f"  {d.relative_to(ROOT)}: 已落 {NOTICE_FILENAME}")

    if not args.check and MANIFEST.exists():
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        data["human_task"] = True
        data["task_type"] = "annotation-review"
        data["notice"] = (f"每个包内已包含《{NOTICE_FILENAME}》：明确声明这是待人工复核的"
                          f"任务包，不是自动标注成品；并要求 AI 助手主动提醒其用户。")
        by_id = {b["id"]: b for b in data.get("batches", [])}
        for info in report["zips"]:
            bid = info["zip"].replace(".zip", "")
            if bid in by_id and "sha256" in info:
                by_id[bid]["sha256"] = info["sha256"]
                by_id[bid]["bytes"] = info["bytes"]
                by_id[bid]["human_task"] = True
        MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  manifest.json 已更新（含 human_task 字段与新的 SHA-256）")

    out = ROOT / "outputs" / "human_task_tagging.json"
    if not args.check:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告：{out}")


if __name__ == "__main__":
    main()
