"""打包"作品提交/成果链接"用的压缩包，并统计真实体积。

用法：

    python tools/build_submission_pack.py --list-only
    python tools/build_submission_pack.py                       # 生成 dist/ 下的软件包
    python tools/build_submission_pack.py --sample-video 路径   # 额外带一段示例视频

默认包含：源码 + 文档 + 默认权重 v7。数据集、outputs 实验产物、CLIP 权重
（337 MB，官方可直接下载）都不进包。`--no-docs-images` 可剔除 docs 下的
UI 截图（13.6 MB，压缩后仍占大头）。
"""

from __future__ import annotations

import argparse
import fnmatch
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GROUPS: dict[str, list[str]] = {
    "源码": ["trafficvision", "bench", "tools", "vendor", "models", "schemas"],
    "顶层文件": [
        "README.md",
        "给协作者.md",
        "requirements.txt",
        "LICENSE",
        "NOTICE",
        ".gitignore",
        ".gitattributes",
    ],
    "文档": ["docs"],
    "默认权重": ["outputs/train/v7/weights/best.pt"],
}

DOC_IMAGE_GLOBS = ["docs/*.png", "docs/**/*.png"]

SKIP_DIRS = {"__pycache__", ".git", ".cache", "env", ".venv", "backups"}
SKIP_GLOBS = ["*.pyc", "*.pyo", "*.log", "*.pt", "*.engine", "*.onnx", "*.mp4"]
# *.pt 默认排除（权重单独按路径点名进包），示例视频由 --sample-video 显式加入。


def iter_files(patterns: list[str], extra: list[str] | None = None) -> list[Path]:
    out: list[Path] = []
    for pat in list(patterns) + list(extra or []):
        target = ROOT / pat
        if target.is_file():
            out.append(target)
            continue
        if not target.is_dir():
            print(f"  [跳过] 不存在：{pat}")
            continue
        for p in sorted(target.rglob("*")):
            if not p.is_file():
                continue
            if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
                continue
            rel = p.relative_to(ROOT).as_posix()
            if any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(p.name, g) for g in SKIP_GLOBS):
                continue
            out.append(p)
    return out


def collect(args) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for name, patterns in GROUPS.items():
        files = iter_files(patterns)
        if name == "默认权重":  # 权重被 SKIP_GLOBS 挡掉，显式加回
            files = [ROOT / p for p in patterns if (ROOT / p).is_file()]
        if name == "文档" and args.no_docs_images:
            files = [
                f
                for f in files
                if not any(fnmatch.fnmatch(f.relative_to(ROOT).as_posix(), g) for g in DOC_IMAGE_GLOBS)
            ]
        groups[name] = files
    extras = list(args.sample_video or []) + list(args.extra or [])
    if extras:
        groups["附加文件"] = [Path(v).resolve() if Path(v).is_absolute() else (ROOT / v) for v in extras]
    return groups


def human(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:,.1f} {unit}"
        num /= 1024
    return f"{num:,.1f} TB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist/trafficvision-v0.2-software-pack.zip")
    ap.add_argument("--no-docs-images", action="store_true", help="剔除 docs 下的 UI 截图（约 13.6 MB）")
    ap.add_argument("--sample-video", action="append", help="额外打进包的示例视频（可重复）")
    ap.add_argument("--extra", action="append", help="其他附加文件（如开放词汇权重）；包外文件放进 附加文件/ 目录")
    ap.add_argument("--list-only", action="store_true", help="只统计，不生成压缩包")
    args = ap.parse_args()

    groups = collect(args)
    print(f"{'分组':<10}{'文件数':>8}{'原始体积':>14}")
    raw_total = 0
    seen: set[Path] = set()
    counts: dict[str, int] = {}
    raws: dict[str, int] = {}
    for name, files in groups.items():
        uniq = [f for f in files if f not in seen]
        seen.update(uniq)
        size = sum(f.stat().st_size for f in uniq)
        counts[name], raws[name] = len(uniq), size
        raw_total += size
        print(f"{name:<10}{len(uniq):>8}{human(size):>14}")
    print(f"{'合计':<10}{sum(counts.values()):>8}{human(raw_total):>14}")

    if args.list_only:
        return

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for files in groups.values():
            for f in files:
                try:
                    zf.write(f, f.relative_to(ROOT).as_posix())
                except ValueError:  # 包外文件（例如桌面上的示例视频）
                    zf.write(f, f"附加文件/{f.name}")

    zip_mb = out.stat().st_size / 1024 / 1024
    print(f"\n生成：{out}  ({zip_mb:,.1f} MB)")
    print(f"压缩率：{zip_mb * 1024 * 1024 / raw_total:.0%}（原始 {human(raw_total)}）")
    print("校验：", zipfile.ZipFile(out).testzip() or "OK")


if __name__ == "__main__":
    main()
