# coding=utf-8
# Reconstruction Cycle 阶段检测器（泛化，支持多种推理模型）
#
# 设计目标：
# - 思维链全程监控不变；本模块仅负责判断“是否已进入 Reconstruction Cycle 阶段”。
# - 通过正则标记 + 轻量置信度，泛化适配 DeepSeek / Qwen / GPT / Claude 等模型。
# - 一旦判定进入重构阶段即返回 True（调用方据此打开注入开关）。

import re
from typing import List


class ReconstructionPhaseDetector:
    """检测思维链是否进入 Reconstruction Cycle（重构/反思）阶段。"""

    # 泛化的重构标记：英文 / 中文 / 通用
    RECONSTRUCTION_MARKERS = {
        "english": [
            r"\bWait\b",
            r"\bHmm+\b",
            r"\bActually\b",
            r"\bHold on\b",
            r"\bBut wait\b",
            r"\bOn second thought\b",
            r"\bAlternatively\b",
            r"\bAnother (?:way|approach|option)\b",
            r"\bLet me (?:reconsider|re-?examine|re-?think|verify|double[- ]?check)\b",
            r"\bIs there (?:another|a better)\b",
            r"\bWhat if\b",
            r"\bMaybe I should\b",
            r"\bRe-?thinking\b",
        ],
        "chinese": [
            r"等等",
            r"等一下",
            r"慢着",
            r"不对",
            r"不过",
            r"实际上",
            r"重新(?:考虑|思考|审视|检查)",
            r"让我(?:再|重新)(?:想想|考虑|检查|确认)",
            r"换(?:个|一)(?:角度|思路|方法)",
            r"或者说",
            r"也许(?:应该|可以)",
            r"会不会",
        ],
        "universal": [
            r"(?:^|\n)\s*(?:Wait|Hmm|Actually|But)\b",  # 段首反思
            r"\?\s*(?:Wait|Actually|Hmm)\b",            # 自问后反思
        ],
    }

    # 强标记：单次命中即可较高置信度判定进入重构阶段
    STRONG_MARKERS = (
        "wait", "actually", "but wait", "on second thought",
        "alternatively", "let me reconsider", "rethink", "re-examine",
        "等等", "不对", "重新考虑", "重新思考", "换个思路", "换个角度",
    )

    def __init__(self, model_type: str = "auto", min_confidence: float = 0.6):
        """
        Args:
            model_type: 'auto' | 'deepseek' | 'qwen' | 'gpt' | 'claude'
            min_confidence: 进入重构阶段的最小置信度（0~1）
        """
        self.model_type = (model_type or "auto").lower()
        self.min_confidence = float(min_confidence)
        self._compile()

    def _compile(self):
        if self.model_type == "qwen":
            groups = ["english", "chinese", "universal"]
        elif self.model_type in ("deepseek", "gpt", "claude"):
            groups = ["english", "universal"]
        else:  # auto
            groups = ["english", "chinese", "universal"]

        patterns: List[str] = []
        for g in groups:
            patterns.extend(self.RECONSTRUCTION_MARKERS[g])

        self._pattern = re.compile(
            "|".join(f"(?:{p})" for p in patterns),
            re.IGNORECASE | re.MULTILINE,
        )

    def is_reconstruction(self, text: str) -> bool:
        """判断给定思维链文本是否已进入 Reconstruction Cycle 阶段。"""
        if not text:
            return False

        match = None
        for match in self._pattern.finditer(text):
            pass  # 取最后一个命中，更接近“当前正在发生”的反思
        if match is None:
            return False

        return self._confidence(text, match) >= self.min_confidence

    def _confidence(self, text: str, match) -> float:
        """基于标记强度 / 位置 / 上下文 / 思考长度的轻量置信度。"""
        confidence = 0.45
        matched = match.group(0).lower()
        start = match.start()

        if any(s in matched for s in self.STRONG_MARKERS):
            confidence += 0.25

        before = text[max(0, start - 40):start]
        if (not before.strip()) or before.rstrip().endswith(("\n",)):
            confidence += 0.15
        if before[-2:].strip()[-1:] in (".", "?", "。", "？", ";"):
            confidence += 0.1

        # 已经过一定长度的铺垫，更可能是真正的反思而非开场白
        if start > 120:
            confidence += 0.1

        return min(confidence, 1.0)
