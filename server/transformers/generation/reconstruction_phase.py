# coding=utf-8
# Reconstruction Cycle phase detector for reasoning models.
#
# Design goals:
# - Keep full-chain reasoning monitoring unchanged; this module only determines
#   whether the output has entered the Reconstruction Cycle phase.
# - Use regex markers plus a lightweight confidence score to support common
#   reasoning models such as DeepSeek, Qwen, GPT, and Claude.
# - Return True once the reconstruction phase is detected so the caller can
#   enable the intervention.

import re
from typing import List


class ReconstructionPhaseDetector:
    """Detect whether chain-of-thought text has entered the Reconstruction Cycle phase."""

    # General reconstruction markers.
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
        "universal": [
            r"(?:^|\n)\s*(?:Wait|Hmm|Actually|But)\b",  # Reflection at a paragraph start.
            r"\?\s*(?:Wait|Actually|Hmm)\b",            # Reflection after a self-question.
        ],
    }

    # Strong markers can trigger a high-confidence reconstruction decision.
    STRONG_MARKERS = (
        "wait", "actually", "but wait", "on second thought",
        "alternatively", "let me reconsider", "rethink", "re-examine",
    )

    def __init__(self, model_type: str = "auto", min_confidence: float = 0.6):
        """
        Args:
            model_type: 'auto' | 'deepseek' | 'qwen' | 'gpt' | 'claude'
            min_confidence: minimum confidence for entering the reconstruction phase (0-1)
        """
        self.model_type = (model_type or "auto").lower()
        self.min_confidence = float(min_confidence)
        self._compile()

    def _compile(self):
        groups = ["english", "universal"]

        patterns: List[str] = []
        for g in groups:
            patterns.extend(self.RECONSTRUCTION_MARKERS[g])

        self._pattern = re.compile(
            "|".join(f"(?:{p})" for p in patterns),
            re.IGNORECASE | re.MULTILINE,
        )

    def is_reconstruction(self, text: str) -> bool:
        """Return True if the given reasoning text has entered the Reconstruction Cycle phase."""
        if not text:
            return False

        match = None
        for match in self._pattern.finditer(text):
            pass  # Use the last match, which is closest to the current reflection.
        if match is None:
            return False

        return self._confidence(text, match) >= self.min_confidence

    def _confidence(self, text: str, match) -> float:
        """Compute a lightweight confidence score from marker strength, position, context, and reasoning length."""
        confidence = 0.45
        matched = match.group(0).lower()
        start = match.start()

        if any(s in matched for s in self.STRONG_MARKERS):
            confidence += 0.25

        before = text[max(0, start - 40):start]
        if (not before.strip()) or before.rstrip().endswith(("\n",)):
            confidence += 0.15
        if before[-2:].strip()[-1:] in (".", "?", ";", ":"):
            confidence += 0.1

        # A longer lead-in is more likely to be a true reflection than an opening phrase.
        if start > 120:
            confidence += 0.1

        return min(confidence, 1.0)
