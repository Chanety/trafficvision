"""filterpy.common 的最小替身：IMM.py 只在 __repr__ 里用到 pretty_str。

原本 `from filterpy.common import pretty_str` 会连带拉起 filterpy 的其它模块，
而这里只需要这一个纯字符串格式化函数，因此做一个最小实现，
避免为了一个 repr 引入整个依赖树（也不改动 IMM.py 的算法部分）。
"""

from __future__ import annotations


def pretty_str(label: str, value) -> str:
    return f"{label} = {value}"
