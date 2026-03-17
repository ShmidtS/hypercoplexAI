"""Models package for HypercoplexAI — HDIM neural network architectures."""

from src.models.hdim_model import (
    HDIMConfig,
    HDIMModel,
    HDIMRuntimeConfig,
)
from src.models.text_hdim_model import (
    SimpleTextEncoder,
    TextHDIMModel,
    TextPairScoreResult,
)

__all__ = [
    "HDIMConfig",
    "HDIMModel",
    "HDIMRuntimeConfig",
    "SimpleTextEncoder",
    "TextHDIMModel",
    "TextPairScoreResult",
]
