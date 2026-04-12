"""Models package for HypercoplexAI — HDIM neural network architectures."""

from src.models.hdim_model import (
    HDIMConfig,
    HDIMModel,
    HDIMRuntimeConfig,
)
from src.models.results import (
    CoreResult,
    ForwardResult,
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
    "CoreResult",
    "ForwardResult",
    "SimpleTextEncoder",
    "TextHDIMModel",
    "TextPairScoreResult",
]
