"""NARS truth-value support for HBMA memory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

# Lean4 mapping: formalization/Extensions.lean NARS truth theorems.


@dataclass
class NarsTruth:
    """NARS truth-value: (frequency, confidence) pair."""

    freq: float = 0.5
    conf: float = 0.0

    EVIDENTIAL_HORIZON: ClassVar[float] = 1.0
    MAX_CONFIDENCE: ClassVar[float] = 0.99
    RELIANCE: ClassVar[float] = 0.9

    def __post_init__(self):
        self.freq = max(0.0, min(1.0, self.freq))
        self.conf = max(0.0, min(self.MAX_CONFIDENCE, self.conf))

    def evidential_weight(self) -> float:
        if self.conf <= 0.0:
            return 0.0
        if self.conf >= self.MAX_CONFIDENCE:
            return 1e6
        return self.EVIDENTIAL_HORIZON * self.conf / (1.0 - self.conf)

    def expectation(self) -> float:
        return self.conf * (self.freq - 0.5) + 0.5

    @staticmethod
    def w2c(w: float, horizon: float = 1.0) -> float:
        if w <= 0.0:
            return 0.0
        return min(NarsTruth.MAX_CONFIDENCE, w / (w + horizon))

    @staticmethod
    def c2w(c: float, horizon: float = 1.0) -> float:
        if c <= 0.0:
            return 0.0
        if c >= NarsTruth.MAX_CONFIDENCE:
            return 1e6
        return horizon * c / (1.0 - c)

    @staticmethod
    def revision(a: NarsTruth, b: NarsTruth, horizon: float = 1.0) -> NarsTruth:
        w1 = a.evidential_weight()
        w2 = b.evidential_weight()
        total_w = w1 + w2
        if total_w <= 0.0:
            return NarsTruth(freq=0.5, conf=0.0)
        freq = (w1 * a.freq + w2 * b.freq) / total_w
        conf = NarsTruth.w2c(total_w, horizon)
        conf = max(conf, a.conf, b.conf)
        return NarsTruth(freq=freq, conf=min(conf, NarsTruth.MAX_CONFIDENCE))

    @staticmethod
    def projection(conf: float, time_diff: float, decay: float = 0.8) -> float:
        return conf * (decay ** abs(time_diff))

    def __repr__(self) -> str:
        return f"NarsTruth(f={self.freq:.3f}, c={self.conf:.3f})"
