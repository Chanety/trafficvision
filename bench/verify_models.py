"""按 models/manifest.json 校验本地权重是否完整、是否与记录一致。

为什么需要它：权重不随仓库分发，靠 Release/网盘下载。下载被截断、
代理缓存了旧版本、或拿错文件时，**程序不会报错，指标会静默变化**——
这类问题在复现赛里是致命的。

用法：
    python bench/verify_models.py            # 只校验清单里列出的文件
    python bench/verify_models.py --strict   # 有缺失或不匹配时返回非零退出码
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "models" / "manifest.json"))
    ap.add_argument("--strict", action="store_true",
                    help="存在缺失或不匹配时以非零退出码结束（适合放进 CI）")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    entries = list(manifest.get("权重", [])) + list(
        manifest.get("上游基础权重（需另行下载，CC-BY/AGPL 归 Ultralytics）", []))
    if not entries:
        raise SystemExit(f"清单里没有权重条目：{args.manifest}")

    missing = mismatch = ok = 0
    print(f"{'状态':<8}{'大小':>10}  文件")
    print("-" * 78)
    for item in entries:
        path = ROOT / item["文件"]
        if not path.exists():
            missing += 1
            print(f"{'缺失':<8}{'-':>10}  {item['文件']}")
            continue
        actual = sha256(path)
        want = item.get("sha256", "")
        mb = path.stat().st_size / 1048576
        if actual == want:
            ok += 1
            print(f"{'正常':<8}{mb:>9.2f}M  {item['文件']}")
        else:
            mismatch += 1
            print(f"{'不匹配':<8}{mb:>9.2f}M  {item['文件']}")
            print(f"{'':<8}{'':>10}  期望 {want[:16]}…  实际 {actual[:16]}…")

    print("-" * 78)
    print(f"正常 {ok} · 缺失 {missing} · 不匹配 {mismatch} · 合计 {len(entries)}")
    if missing or mismatch:
        print("\n提示：缺失或不匹配的权重不能用于正式评测，否则指标无法复现。")
    if args.strict and (missing or mismatch):
        sys.exit(1)


if __name__ == "__main__":
    main()
