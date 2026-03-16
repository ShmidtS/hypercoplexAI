"""
Unified memory interface for HDIM pipeline.

Bridges TitansMemoryModule (k, v API) and HBMAMemory (single-input API)
through a common ABC, enabling memory-agnostic pipeline code.

Design:
  - MemoryInterface: ABC defining the unified contract
  - TitansAdapter: wraps TitansMemoryModule to conform to MemoryInterface
  - HBMAMemory already conforms via forward(x) + memory_loss() + reset()
  - MemoryState: shared dataclass for retrieval results
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

    @abstractmethod
    def reset(self, strategy: str = 'geometric') -> None:
        """Reset memory state."""
        ...

    def memory_loss(self) -> torch.Tensor:
        """Current auxiliary memory loss. Default: 0."""
        return torch.tensor(0.0)


class TitansAdapter(MemoryInterface):
    """
    Adapts TitansMemoryModule to MemoryInterface.

    Absorbs key projection and gated retrieval into the adapter,
    presenting a clean single-input interface.
    """

    def __init__(
        self,
        titans_module: nn.Module,
        clifford_dim: int,
        memory_key_dim: int,
    ):
        super().__init__()
        self.titans = titans_module
        self.key_proj = nn.Linear(clifford_dim, memory_key_dim)
        self.gate = nn.Sequential(
            nn.Linear(clifford_dim, clifford_dim // 4),
            nn.ReLU(),
            nn.Linear(clifford_dim // 4, 1),
        )
        self._last_loss = torch.tensor(0.0)

    def forward(
        self,
        x: torch.Tensor,
        update_memory: bool = False,
    ) -> MemoryResult:
        k = self.key_proj(x)
        retrieved, loss = self.titans(k, x, update_memory=update_memory)
        self._last_loss = loss.detach()
        gate_val = torch.sigmoid(self.gate(x))
        gated_output = x + gate_val * retrieved
        return MemoryResult(
            output=gated_output,
            loss=loss,
            updated=update_memory,
        )

    def reset(self, strategy: str = 'geometric') -> None:
        if hasattr(self.titans, 'reset_memory'):
            self.titans.reset_memory(strategy=strategy)
        else:
            if hasattr(self.titans, 'memory') and hasattr(self.titans.memory, 'weight'):
                self.titans.memory.weight.data.zero_()
            if hasattr(self.titans, 'momentum_S'):
                self.titans.momentum_S.data.zero_()

    reset_memory = reset

    def memory_loss(self) -> torch.Tensor:
        return self._last_loss


class HBMAMemoryAdapter(MemoryInterface):
    """
    Adapts HBMAMemory (or CLSMemory) to MemoryInterface.

    HBMAMemory already has forward(x) -> Tensor and memory_loss(),
    so this is a thin wrapper that adds MemoryResult wrapping.
    """

    def __init__(self, hbma_module: nn.Module):
        super().__init__()
        self.hbma = hbma_module

    def forward(
        self,
        x: torch.Tensor,
        update_memory: bool = False,
    ) -> MemoryResult:
        # HBMA manages its own internal update logic
        output = self.hbma(x)
        loss = self.hbma.memory_loss() if hasattr(self.hbma, 'memory_loss') else torch.tensor(0.0, device=x.device)
        return MemoryResult(
            output=output,
            loss=loss,
            updated=True,  # HBMA always updates internally
        )

    def reset(self, strategy: str = 'geometric') -> None:
        if hasattr(self.hbma, 'reset'):
            self.hbma.reset()
        elif hasattr(self.hbma, 'reset_memory'):
            self.hbma.reset_memory(strategy=strategy)

    def memory_loss(self) -> torch.Tensor:
        if hasattr(self.hbma, 'memory_loss'):
            return self.hbma.memory_loss()
        return torch.tensor(0.0)


def wrap_memory(module: nn.Module, memory_type: str) -> MemoryInterface:
    """
    Factory: wrap any memory module to MemoryInterface.

    Args:
        module: TitansMemoryModule, HBMAMemory, or CLSMemory instance
        memory_type: 'titans', 'hbma', 'cls', 'hippocampus'

    Returns:
        MemoryInterface-wrapped module
    """
    if memory_type == 'titans':
        return TitansAdapter(module)
    else:
        # hbma, cls, hippocampus, neocortex — all use HBMA interface
        return HBMAMemoryAdapter(module)
