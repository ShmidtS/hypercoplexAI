"""HDIM — TextEncoder Protocol.

Protocol for text encoding components (SimpleTextEncoder, SBERTEncoder, ModernBertEncoder).
Moved from the old core text encoder interface to keep MoE focused on routing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class TextEncoder(Protocol):
    """Protocol for text encoding components."""

    def forward(self, texts: list, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor: ...
    def encode(self, texts: list) -> torch.Tensor: ...
