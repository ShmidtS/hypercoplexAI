"""Models package for HypercoplexAI — HDIM neural network architectures."""

from src.models.hdim_model import (
    HDIMAuxState,
    HDIMConfig,
    HDIMModel,
    HDIMRuntimeConfig,
    HDIMTextConfig,
)
from src.models.text_hdim_model import (
    SimpleTextEncoder,
    TextHDIMModel,
    TextPairScoreResult,
)

__all__ = [
    "HDIMAuxState",
    "HDIMConfig",
    "HDIMModel",
    "HDIMRuntimeConfig",
    "HDIMTextConfig",
    "SimpleTextEncoder",
    "TextHDIMModel",
    "TextPairScoreResult",
]
