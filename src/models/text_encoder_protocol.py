"""HDIM — TextEncoder Protocol.

Protocol for text encoding components (SimpleTextEncoder, SBERTEncoder, ModernBertEncoder).
Moved from the old core text encoder interface to keep MoE focused on routing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol
from typing import runtime_checkable

if TYPE_CHECKING:
    import torch


@runtime_checkable
class TextEncoder(Protocol):
    """Protocol for text encoding components."""

    def forward(self, texts: list, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor: ...
    def encode(self, texts: list) -> torch.Tensor: ...
