"""TransferState — состояние кроссдоменного переноса.

Вынесен в отдельный модуль для избежания циклических импортов.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .nars_truth import NarsTruth


@dataclass
class TransferState:
    """Состояние трансфера между доменами."""
    g_source: Optional[torch.Tensor]
    u_inv: torch.Tensor
    u_mem: torch.Tensor
    u_route: torch.Tensor
    g_target: torch.Tensor
    output: torch.Tensor
    memory_loss: torch.Tensor
    memory_retrieved: torch.Tensor
    memory_updated: bool
    memory_alpha: Optional[torch.Tensor]
    memory_eta: Optional[torch.Tensor]
    memory_theta: Optional[torch.Tensor]
    router_state: Dict[str, Any]
    memory_mode: str
    update_memory: bool
    input_is_invariant: bool
    transfer_truth: Optional[NarsTruth] = None

    @property
    def routing_weights(self) -> torch.Tensor:
        return self.router_state["gate_weights"]

    @property
    def raw_invariant(self) -> torch.Tensor:
        return self.u_inv

    @property
    def memory_augmented_invariant(self) -> torch.Tensor:
        return self.u_mem

    @property
    def exported_invariant(self) -> torch.Tensor:
        return self.u_route

    @property
    def invariant(self) -> torch.Tensor:
        return self.exported_invariant

    def to_dict(self) -> Dict[str, Any]:
        return {
            "g_source": self.g_source,
            "u_inv": self.u_inv,
            "u_mem": self.u_mem,
            "u_route": self.u_route,
            "g_target": self.g_target,
            "output": self.output,
            "memory_loss": self.memory_loss,
            "memory_retrieved": self.memory_retrieved,
            "memory_updated": self.memory_updated,
            "memory_alpha": self.memory_alpha,
            "memory_eta": self.memory_eta,
            "memory_theta": self.memory_theta,
            "router_state": self.router_state,
            "routing_weights": self.routing_weights,
            "memory_mode": self.memory_mode,
            "update_memory": self.update_memory,
            "input_is_invariant": self.input_is_invariant,
            "raw_invariant": self.raw_invariant,
            "memory_augmented_invariant": self.memory_augmented_invariant,
            "exported_invariant": self.exported_invariant,
            "invariant": self.invariant,
            "transfer_truth": self.transfer_truth,
        }
