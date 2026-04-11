"""NARS Truth-Value: (frequency, confidence) pair for evidential reasoning.

Borrowed from OpenNARS-for-Applications (Truth.h, Truth.c).
- frequency: proportion of positive evidence [0,1]
- confidence: total evidence strength w/(w+h) [0, MAX_CONFIDENCE]
- evidential_horizon h=1.0 (NARS default)
- MAX_CONFIDENCE = 0.99 (prevents absolute certainty)
"""

from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class NarsTruth:
    """NARS truth-value: (frequency, confidence) pair.

    frequency = proportion of positive evidence.
    confidence = w/(w+h) where w = evidential weight, h = horizon.
    """

    freq: float = 0.5
    conf: float = 0.0

    EVIDENTIAL_HORIZON: float = 1.0
    MAX_CONFIDENCE: float = 0.99
    RELIANCE: float = 0.9  # weight of new evidence in revision

    def __post_init__(self):
        self.freq = max(0.0, min(1.0, self.freq))
        self.conf = max(0.0, min(self.MAX_CONFIDENCE, self.conf))

    def evidential_weight(self) -> float:
        """w = h * c / (1 - c). Returns 0 when conf=0."""
        if self.conf <= 0.0:
            return 0.0
        if self.conf >= self.MAX_CONFIDENCE:
            return 1e6  # effectively infinite
        return self.EVIDENTIAL_HORIZON * self.conf / (1.0 - self.conf)

    def expectation(self) -> float:
        """E = c * (f - 0.5) + 0.5. NARS decision quantity."""
        return self.conf * (self.freq - 0.5) + 0.5

    @staticmethod
    def w2c(w: float, horizon: float = 1.0) -> float:
        """Evidence weight to confidence: c = w / (w + h)."""
        if w <= 0.0:
            return 0.0
        return min(NarsTruth.MAX_CONFIDENCE, w / (w + horizon))

    @staticmethod
    def c2w(c: float, horizon: float = 1.0) -> float:
        """Confidence to evidence weight: w = h * c / (1 - c)."""
        if c <= 0.0:
            return 0.0
        if c >= NarsTruth.MAX_CONFIDENCE:
            return 1e6
        return horizon * c / (1.0 - c)

    @staticmethod
    def revision(a: NarsTruth, b: NarsTruth, horizon: float = 1.0) -> NarsTruth:
        """Truth revision: merge two truth values without double-counting.

        f = (w1*f1 + w2*f2) / (w1 + w2)
        c = w2c(w1 + w2)
        """
        w1 = a.evidential_weight()
        w2 = b.evidential_weight()
        total_w = w1 + w2
        if total_w <= 0.0:
            return NarsTruth(freq=0.5, conf=0.0)
        freq = (w1 * a.freq + w2 * b.freq) / total_w
        conf = NarsTruth.w2c(total_w, horizon)
        conf = max(conf, a.conf, b.conf)  # revision cannot reduce confidence
        return NarsTruth(freq=freq, conf=min(conf, NarsTruth.MAX_CONFIDENCE))

    @staticmethod
    def projection(conf: float, time_diff: float, decay: float = 0.8) -> float:
        """Temporal projection: c *= decay^timeDiff."""
        return conf * (decay ** abs(time_diff))

    def __repr__(self) -> str:
        return f"NarsTruth(f={self.freq:.3f}, c={self.conf:.3f})"
