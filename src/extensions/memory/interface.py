"""Unified memory interface for HDIM pipeline.

Defines the MemoryInterface ABC and MemoryResult dataclass that all
memory systems conform to, enabling memory-agnostic pipeline code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class MemoryResult:
    """Unified memory retrieval result."""
    output: torch.Tensor       # memory-augmented representation [B, D]
    loss: torch.Tensor         # auxiliary memory loss (scalar)
    updated: bool              # whether memory was updated this step
    alpha: Optional[torch.Tensor] = None  # memory blend gate (Titans)
    eta: Optional[torch.Tensor] = None    # momentum learning rate (Titans)
    theta: Optional[torch.Tensor] = None  # gradient step size (Titans)
    surprise: Optional[torch.Tensor] = None  # surprise signal (Titans)


class MemoryInterface(nn.Module, ABC):
    """
    Abstract interface for all memory systems in HDIM.

    Unified contract:
      forward(x, update_memory=False) -> MemoryResult
        - x: input tensor [B, D]
        - Returns memory-augmented output and auxiliary loss

      reset(strategy='geometric') -> None
        - Reset memory state

      memory_loss() -> Tensor
        - Current auxiliary loss (for monitoring)
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        update_memory: bool = False,
    ) -> MemoryResult:
        """Run memory retrieval (+ optional update). Returns MemoryResult."""
        ...

    def reset(self, strategy: str = 'geometric') -> None:
        """Reset memory state. Default: no-op."""
        pass

    @abstractmethod
    def memory_loss(self) -> torch.Tensor:
        """Current auxiliary memory loss."""
        ...
