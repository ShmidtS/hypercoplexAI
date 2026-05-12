"""Plugin base for HBMA memory subsystems."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .context import ConsolidationContext


class MemorySubsystemPlugin(nn.Module, ABC):
    """Base class for HBMA memory subsystem plugins."""

    name: str = "plugin"
    priority: int = 10

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def on_consolidate(self, ctx: ConsolidationContext) -> None:
        pass

    def reset(self) -> None:
        pass

    def auxiliary_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, dtype=torch.float32)
