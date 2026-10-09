"""类别表加载与一致性校验——防止"权重与数据集口径不匹配却不报错"。

背景（2026-10-08 仓库审计 P0-01）：仓库里同时存在 9 / 10 / 12 类三套编号，
而**用错口径不会报错，只会静默错位**。最典型的一次是 v9 评测：
`sign` 从 0.866 掉到 0.240，看起来像模型退化，实际是验证真值从 16 个实例
变成了 139 个——同一个类名，两套真值。

本模块提供两件事：
  1. `load_schema()`  从 `schemas/classes_v3_10.yaml` 读唯一权威类别表；
  2. `check_compatible()` 比较"权重的类别"与"数据集的类别"，不匹配就明确失败。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "schemas" / "classes_v3_10.yaml"


def load_schema(path: str | Path | None = None) -> dict:
    """读类别表。缺 PyYAML 或文件时给出明确原因，不静默回退到错误的类别表。"""
    path = Path(path) if path else DEFAULT_SCHEMA
    if not path.exists():
        raise SystemExit(f"找不到类别表 schema：{path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"需要 PyYAML 才能读取类别表：{exc}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "classes" not in data:
        raise SystemExit(f"类别表格式不对（缺 classes 段）：{path}")
    data["classes"] = {int(k): str(v) for k, v in data["classes"].items()}
    return data


def schema_names(schema: dict | None = None) -> list[str]:
    schema = schema or load_schema()
    return [schema["classes"][i] for i in sorted(schema["classes"])]


def names_from_data_yaml(path: str | Path) -> list[str]:
    """从 YOLO 数据配置里读类别名（支持 names 为 dict 或 list）。"""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, list):
        return [str(n) for n in names]
    raise SystemExit(f"数据配置里读不到 names：{path}")


def check_compatible(model_names, data_names, *, where: str = "",
                     strict_order: bool = True) -> dict:
    """比较权重类别与数据集类别，返回结论文档；不匹配时抛 SystemExit。

    strict_order=True 时要求逐位相同（最安全）；False 时只要求集合相同，
    并返回 name→index 的重排映射。
    """
    model_names = [str(n) for n in model_names]
    data_names = [str(n) for n in data_names]
    report = {"where": where, "model": model_names, "data": data_names}

    if len(model_names) != len(data_names):
        report["verdict"] = "数量不一致"
        msg = (f"类别数量不匹配{'（' + where + '）' if where else ''}："
               f"权重 {len(model_names)} 类 {model_names} vs "
               f"数据集 {len(data_names)} 类 {data_names}\n"
               f"继续评测会得到**静默错位**的指标。请先统一口径"
               f"（权威类别表见 schemas/classes_v3_10.yaml）。")
        raise SystemExit(msg)

    if model_names == data_names:
        report["verdict"] = "完全一致"
        return report

    if set(model_names) == set(data_names):
        report["verdict"] = "类别相同但顺序不同"
        if strict_order:
            raise SystemExit(
                f"类别顺序不一致{'（' + where + '）' if where else ''}：\n"
                f"  权重 {model_names}\n  数据集 {data_names}\n"
                f"顺序不同会导致编号错位，请对齐后再评测。")
        report["remap_data_to_model"] = {i: model_names.index(n)
                                         for i, n in enumerate(data_names)}
        return report

    only_model = [n for n in model_names if n not in data_names]
    only_data = [n for n in data_names if n not in model_names]
    raise SystemExit(
        f"类别名不一致{'（' + where + '）' if where else ''}：\n"
        f"  仅权重有：{only_model}\n  仅数据集有：{only_data}\n"
        f"权威类别表见 schemas/classes_v3_10.yaml。")


def compare_with_schema(model_names, *, where: str = "") -> dict:
    """检查一份权重是否属于当前生产口径（10 类）。"""
    return check_compatible(model_names, schema_names(), where=where)


def main() -> None:
    """命令行自检：打印当前权威类别表，并可校验给定的权重/数据配置。"""
    import argparse

    ap = argparse.ArgumentParser(description="类别表 schema 自检与一致性校验")
    ap.add_argument("--weights", default=None, help="要检查的权重路径")
    ap.add_argument("--data", default=None, help="要检查的 YOLO data.yaml")
    args = ap.parse_args()

    schema = load_schema()
    print(f"权威类别表：schema_version={schema['schema_version']} "
          f"name={schema['name']}（{len(schema['classes'])} 类）")
    for i in sorted(schema["classes"]):
        print(f"  {i}: {schema['classes'][i]}")

    if args.weights:
        from ultralytics import YOLO
        names = [str(YOLO(args.weights).names[i]) for i in sorted(YOLO(args.weights).names)]
        print(f"\n权重 {args.weights}：{names}")
        try:
            compare_with_schema(names, where="权重 vs 权威表")
            print("  → 与生产口径一致")
        except SystemExit as exc:
            print(f"  → 不一致：{exc}")

    if args.data:
        names = names_from_data_yaml(args.data)
        print(f"\n数据集 {args.data}：{names}")
        if args.weights:
            from ultralytics import YOLO
            wnames = [str(YOLO(args.weights).names[i]) for i in sorted(YOLO(args.weights).names)]
            check_compatible(wnames, names, where="权重 vs 数据集")
            print("  → 权重与数据集兼容")


if __name__ == "__main__":
    main()
