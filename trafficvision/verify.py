"""静态目标的上下文校验：判断"这个位置到底是什么东西"。

用户反馈（2026-10-06）：
  "路桩出现了空识别与路肩误识别，路牌出现了广告牌误识别。
   这两个我觉得是对周遭的物体考量不清晰导致的。"

诊断完全正确。前面那套"跨帧投票"只回答了一个问题——**这个位置稳不稳定**，
它回答不了**这个位置到底是什么**：花坛边缘、路缘石、红色广告牌都是常年不动的，
任意一个弱提示词稳定地打在上面，就会一路通过 80% 一致性检验。

所以再加一道**语义校验**：把该位置连周围的画面一起裁出来，用 CLIP 零样本判断
"这更像什么"。候选词里既有正例（混凝土路桩、交通标志牌…），也有反例
（花坛、路缘石、广告牌、空地…）。反例得分更高就否掉。

这一步是"看周遭"：判据不是框内的纹理，而是框外那片上下文——
路桩立在人行道上、下面是硬质地面；路缘石是贴着花坛的一条长条；
广告牌长在建筑立面上、字大且彩色。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIP = ROOT / "weights" / "clip" / "ViT-B-32.pt"

# 用户第二批反馈里的一个特殊失败模式：公交车停靠在同一位置很久，
# 车窗立柱这种竖直结构被跨帧投票当成了"路灯"。它既不是目标物，
# 也不属于任何类别——所以在所有静态类的反例里都加上"车辆部件"。
VEHICLE_PART_NEG = (
    "the window pillar of a bus",
    "the side of a bus",
    "a bus window frame",
    "part of a vehicle",
)

# 每个类别：正例（是这种东西）+ 反例（长得很像但不是）
LABELS: dict[str, dict[str, tuple[str, ...]]] = {
    "bollard": {
        "pos": (
            "a concrete bollard post standing on the ground",
            "a short vertical stone post barrier on a sidewalk",
            "a traffic bollard with a warning stripe",
        ),
        "neg": (
            "a painted curb along a flower bed",
            "a road curb at the edge of a sidewalk",
            "a flower bed full of orange flowers",
            "empty asphalt road surface",
            "a road surface with white pedestrian crossing stripes",
            "a shop sign on a building",
            "a tall street lamp post",
            "a traffic signal pole",
            "a car on the road",
            "a person walking on the street",
            "the branch of a tree",
            "a distant building",
        ),
    },
    "sign": {
        "pos": (
            "a traffic sign on a pole",
            "a road direction sign board",
            "a traffic warning sign with a symbol",
        ),
        "neg": (
            "a shop billboard with large text on a building",
            "a commercial storefront advertisement sign",
            "a red decorative sculpture on a traffic island",
            "a building facade with windows",
            "empty sky between buildings",
            "a traffic light with colored lamps",
            "a tree",
            "a car on the road",
        ),
    },
    "roadlamp": {
        "pos": (
            "a tall street lamp post with a lamp head",
            "a street light pole along a road",
            "a decorative street lamp post",
        ),
        "neg": (
            # 注意：不能写 "a traffic signal pole with a signal head" ——
            # 实测它会把真正的路灯（灯杆+灯头）整片抢走，路桩那一类
            # 被它误伤的代价太大。只保留"确实不是路灯"的几种。
            "a shop sign on a building",
            "a tree",
            "empty sky",
            "a building facade",
            "a utility pole with cables",
            "a person walking on the street",
        ),
    },
    "trafficlight": {
        "pos": (
            "a traffic light with three colored lamps",
            "a traffic signal head mounted on a pole",
        ),
        "neg": (
            "a red shop sign on a building",
            "a billboard on a building",
            "a street lamp",
            "empty sky",
            "a building window",
            "a dark gap under a gantry",
            "a car on the road",
            "a tree",
        ),
    },
}

# 尺寸合理性：静态街道设施不可能无限大/无限小。
# 实测拿到的反例里，有 397×730 像素（占画面 14%）的"路桩"，明显是整片背景；
# 也有 34×17 像素的"标志牌"，那个尺寸连人都认不出是什么。
MAX_AREA_FRAC = {"bollard": 0.02, "sign": 0.10, "roadlamp": 0.30, "trafficlight": 0.30}
MIN_SIDE_PX = {"bollard": 14.0, "sign": 18.0, "roadlamp": 16.0, "trafficlight": 12.0}


def size_ok(cls: str, box, frame_w: int, frame_h: int) -> tuple[bool, str]:
    """尺寸合理性检查，返回（是否合理, 原因）。"""
    w = box[2] - box[0]
    h = box[3] - box[1]
    from .config import min_side_for
    if min(w, h) < min(min_side_for(cls), MIN_SIDE_PX.get(cls, 14.0)):
        return False, f"过小 {w:.0f}×{h:.0f}px"
    frac = (w * h) / float(frame_w * frame_h)
    if frac > MAX_AREA_FRAC.get(cls, 0.30):
        return False, f"过大 占画面 {frac * 100:.1f}%"
    return True, ""


# ---------------------------------------------------------------- 判决规则
# 这里的原则是**代价不对称**，必须写清楚，否则很容易被调歪：
#
#   · 误删一个真目标 —— 正是用户最初抱怨的那个问题（路桩看不见），代价高；
#   · 留着一个误检   —— 它会出现在位置清单里，用户扫一眼就能划掉，代价低。
#
# 因此规则整体偏向"保留"，只对证据确凿的情况直接否掉，
# 拿不准的一律降级成"待确认"，交给人工扫清单，而不是替用户做决定。
#
# 特别是尺寸：一条规则叫"最小边 ≥16px"会把远处真路灯（12×66px）全杀掉——
# 实测 #58/#92/#114/#115 都是这种，CLIP 明确判成路灯（p_pos 0.57~0.75），
# 却被尺寸卡掉。所以尺寸问题只降级、不删除；只有"长边都不到 20px"这种
# 两个方向都小到没有信息量的框才直接否掉。
POS_ABSENT = 0.25        # 正例总概率低于此 → 直接否掉
NEG_DECISIVE = 0.60      # 反例拿到这么高的单类概率 → 直接否掉
TINY_LONG_SIDE = 20.0    # 长边都不到 20px → 直接否掉
GRAY_POS = 0.45          # 正例总概率低于此但高于 POS_ABSENT → 待确认


def decide(cls: str, box, frame_w: int, frame_h: int, verdict: dict) -> tuple[str, str]:
    """综合 CLIP 置信度与几何合理性，给出 keep / review / reject。

    review = 保留在数据集里，但在位置清单上单独列出，请人看一下。
    """
    if not verdict:
        return "review", "无校验结果"
    w = box[2] - box[0]
    h = box[3] - box[1]
    frac = (w * h) / float(frame_w * frame_h)
    spec = LABELS.get(cls)
    top_is_pos = bool(spec) and verdict.get("top") in spec["pos"]
    p_pos = float(verdict.get("p_pos", 0.0))
    p_top = float(verdict.get("p_top", 0.0))

    if frac > MAX_AREA_FRAC.get(cls, 0.30):
        return "reject", f"框过大，占画面 {frac * 100:.1f}%"
    # 用户要求：过小目标一律不识别（动态静态同一标准），这里先按全局尺寸门槛否掉
    from .config import target_too_small
    if target_too_small(box, cls):
        return "reject", f"过小（{w:.0f}×{h:.0f}px，面积 {w * h:.0f}px²）"
    if p_pos <= POS_ABSENT:
        return "reject", f"正例证据缺失（正例总概率 {p_pos:.2f}）"
    if (not top_is_pos) and p_top >= NEG_DECISIVE:
        return "reject", f"反例决定性胜出（{p_top:.2f}）"
    if max(w, h) < TINY_LONG_SIDE:
        return "reject", f"框极小（{w:.0f}×{h:.0f}px）"
    if not top_is_pos:
        return "review", "反例略占优，证据不足"
    if p_pos < GRAY_POS:
        return "review", f"正例概率偏低（{p_pos:.2f}）"
    if min(w, h) < MIN_SIDE_PX.get(cls, 14.0):
        return "review", f"目标偏小（{w:.0f}×{h:.0f}px）"
    return "keep", ""

# 所有类别共用的兜底反例
COMMON_NEG = (
    "empty background with nothing in it",
    "a plain wall",
    "blurred out of focus area",
)


class StaticVerifier:
    """CLIP 零样本校验器。构造时加载一次模型，之后按位置反复调用。"""

    def __init__(self, weights: str | Path | None = None, device: str = "cuda",
                 temperature: float = 100.0) -> None:
        import clip  # 与 YOLO-World 共用同一份 ViT-B-32 权重
        import torch

        self.torch = torch
        self.device = self._normalize_device(device, torch)
        self.model, self.preprocess = clip.load(str(weights or DEFAULT_CLIP), device=self.device, jit=False)
        self.model.eval()
        self.temperature = temperature
        self._text_cache: dict[tuple[str, ...], "torch.Tensor"] = {}

    @staticmethod
    def _normalize_device(device, torch) -> str:
        """把 CLI 习惯的 "0" 规范化成 "cuda:0"——clip.load 不认纯数字。"""
        if not torch.cuda.is_available():
            return "cpu"
        s = str(device).strip().lower()
        if s in ("", "cuda", "gpu"):
            return "cuda:0"
        if s.isdigit():
            return f"cuda:{s}"
        if s.startswith("cuda"):
            return s
        return "cpu"

    def _text_features(self, labels: tuple[str, ...]):
        key = tuple(labels)
        if key not in self._text_cache:
            import clip as _clip
            with self.torch.no_grad():
                feats = self.model.encode_text(_clip.tokenize(list(labels)).to(self.device))
                feats = feats / feats.norm(dim=-1, keepdim=True)
            self._text_cache[key] = feats
        return self._text_cache[key]

    def score(self, crop_bgr, cls: str) -> dict:
        """对单张裁剪图打分。返回 {keep, top, p_top, p_pos, probs}。"""
        import cv2
        from PIL import Image

        spec = LABELS.get(cls)
        if spec is None:
            return {"keep": True, "top": "(未登记类别)", "p_top": 1.0, "p_pos": 1.0, "probs": {}}
        labels = tuple(spec["pos"]) + tuple(spec["neg"]) + VEHICLE_PART_NEG + COMMON_NEG
        n_pos = len(spec["pos"])

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        txt = self._text_features(labels)
        with self.torch.no_grad():
            img = self.model.encode_image(tensor)
            img = img / img.norm(dim=-1, keepdim=True)
            logits = (self.temperature * img @ txt.T).softmax(dim=-1)[0]
        probs = logits.detach().cpu().numpy()
        order = np.argsort(-probs)
        p_pos = float(probs[:n_pos].sum())
        top = labels[int(order[0])]
        keep = int(order[0]) < n_pos
        return {"keep": bool(keep), "top": top, "p_top": float(probs[order[0]]),
                "p_pos": p_pos, "n_pos": n_pos,
                "probs": {labels[i]: float(probs[i]) for i in range(len(labels))}}

    def score_many(self, crops: list, cls: str) -> dict:
        """多帧裁剪一起判断：把所有标签的概率一起取均值，再用均值做判定。

        早前的实现在这里出了错——判定用"每帧各投一票"（多数票），
        而打印出来的 top 用的是均值，两者会打架：真正的路桩 #33 平均下来
        top 就是正例，却因为逐帧票数不过半被否掉。改成统一以均值为准。
        """
        if not crops:
            return {"keep": True, "top": "(无裁剪)", "p_top": 0.0, "p_pos": 1.0, "probs": {}}
        results = [self.score(c, cls) for c in crops]
        keys = list(results[0]["probs"].keys())
        avg = {k: float(np.mean([r["probs"].get(k, 0.0) for r in results])) for k in keys}
        n_pos = results[0]["n_pos"]
        pos_labels = set(keys[:n_pos])
        top = max(avg, key=avg.get)
        p_pos = float(sum(avg[k] for k in pos_labels))
        # 判定：平均意义上的第一名必须是正例，且正例总概率够高
        keep = (top in pos_labels) and (p_pos >= 0.40)
        return {
            "keep": bool(keep),
            "top": top,
            "p_top": float(avg.get(top, 0.0)),
            "p_pos": p_pos,
            "per_crop_keep": sum(1 for r in results if r["keep"]),
            "n_crops": len(results),
            "probs": avg,
        }


def crop_with_context(img, box, pad: float = 1.8, min_side: int = 200):
    """连上下文一起裁：框外留足够大的余量，CLIP 才能看见"它站在什么上面"。"""
    h, w = img.shape[:2]
    cw, ch = box[2] - box[0], box[3] - box[1]
    half = max(cw * pad, min_side / 2)
    half_v = max(ch * pad, min_side / 2)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x1 = max(0, int(cx - half)); x2 = min(w, int(cx + half))
    y1 = max(0, int(cy - half_v)); y2 = min(h, int(cy + half_v))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return img[y1:y2, x1:x2].copy()
