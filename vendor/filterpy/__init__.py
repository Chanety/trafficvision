"""Vendored 自 https://github.com/rlabbe/filterpy （MIT License，见同目录 LICENSE）。

只保留了 IMM.py（交互式多模型估计器）与其需要的 pretty_str；
算法部分未做任何改动，仅补了一个最小 common.py 以免引入整棵依赖树。
"""

__all__ = ["IMMEstimator"]

from .IMM import IMMEstimator  # noqa: F401
