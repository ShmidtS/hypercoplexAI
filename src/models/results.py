"""Typed result containers for HDIM model forward paths."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import torch


def _warn_deprecated(name: str) -> None:
    warnings.warn(
        f"HDIMAuxState.{name} is deprecated in core mode",
        DeprecationWarning,
        stacklevel=3,
    )


class HDIMAuxState:
    """Core HDIM aux state with deprecated legacy compatibility fields."""

    def __init__(
        self,
        *,
        raw_invariant: torch.Tensor,
        exported_invariant: torch.Tensor,
        matches: list[list[Any]],
        training_invariant: torch.Tensor | None = None,
        memory_mode: str = "none",
        update_memory: bool = False,
        memory_updated: bool = False,
    ) -> None:
        self.raw_invariant = raw_invariant
        self.exported_invariant = exported_invariant
        self.matches = matches
        self.training_invariant = training_invariant if training_invariant is not None else exported_invariant
        self.memory_mode = memory_mode
        self.update_memory = update_memory
        self.memory_updated = memory_updated

    @property
    def memory_augmented_invariant(self) -> torch.Tensor:
        return self.raw_invariant

    @property
    def memory_loss(self) -> torch.Tensor:
        _warn_deprecated("memory_loss")
        return self.raw_invariant.new_tensor(0.0)

    @property
    def router_loss(self) -> torch.Tensor:
        _warn_deprecated("router_loss")
        return self.raw_invariant.new_tensor(0.0)

    @property
    def routing_weights(self) -> torch.Tensor:
        _warn_deprecated("routing_weights")
        return self.raw_invariant.new_zeros(self.raw_invariant.shape[0], 0)

    @property
    def hallucination_risk(self) -> float:
        _warn_deprecated("hallucination_risk")
        return 0.0

    @property
    def online_loss(self) -> torch.Tensor:
        _warn_deprecated("online_loss")
        return self.raw_invariant.new_tensor(0.0)

    @property
    def topk_idx(self) -> torch.Tensor:
        return torch.empty(self.raw_invariant.shape[0], 0, device=self.raw_invariant.device, dtype=torch.long)

    @property
    def topk_gate_weights(self) -> torch.Tensor:
        return self.raw_invariant.new_zeros(self.raw_invariant.shape[0], 0)

    @property
    def train_scores_snapshot(self) -> torch.Tensor:
        return self.raw_invariant.new_zeros(0)

    @property
    def expert_usage(self) -> torch.Tensor:
        return self.raw_invariant.new_zeros(0)

    @property
    def routing_entropy(self) -> torch.Tensor:
        return self.raw_invariant.new_tensor(0.0)

    @property
    def z_loss(self) -> torch.Tensor:
        return self.raw_invariant.new_tensor(0.0)

    @property
    def memory_surprise(self) -> None:
        return None

    @property
    def feedback_action(self) -> None:
        return None

    @property
    def online_updated(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_invariant": self.raw_invariant,
            "exported_invariant": self.exported_invariant,
            "matches": self.matches,
            "training_invariant": self.training_invariant,
            "memory_augmented_invariant": self.memory_augmented_invariant,
            "memory_loss": self.memory_loss,
            "router_loss": self.router_loss,
            "routing_weights": self.routing_weights,
            "topk_idx": self.topk_idx,
            "topk_gate_weights": self.topk_gate_weights,
            "train_scores_snapshot": self.train_scores_snapshot,
            "expert_usage": self.expert_usage,
            "routing_entropy": self.routing_entropy,
            "z_loss": self.z_loss,
            "memory_updated": self.memory_updated,
            "memory_mode": self.memory_mode,
            "update_memory": self.update_memory,
            "hallucination_risk": self.hallucination_risk,
            "memory_surprise": self.memory_surprise,
            "feedback_action": self.feedback_action,
            "online_loss": self.online_loss,
            "online_updated": self.online_updated,
        }


@dataclass
class CoreResult:
    """Return type for HDIMModel._forward_core()."""

    output: torch.Tensor
    raw_invariant: torch.Tensor
    exported_invariant: torch.Tensor
    matches: list[list[Any]]
    routing_weights: torch.Tensor
    slot_outputs: torch.Tensor | None = None

    @property
    def memory_augmented_invariant(self) -> torch.Tensor:
        return self.raw_invariant

    @property
    def router_loss(self) -> torch.Tensor:
        return self.raw_invariant.new_tensor(0.0)

    @property
    def memory_loss(self) -> torch.Tensor:
        return self.raw_invariant.new_tensor(0.0)

    @property
    def z_loss(self) -> torch.Tensor:
        return self.raw_invariant.new_tensor(0.0)

    @property
    def memory_updated(self) -> bool:
        return False


@dataclass
class ForwardResult:
    """Return type for HDIMModel.forward() and TextHDIMModel.forward_texts()."""

    output: torch.Tensor
    routing_weights: torch.Tensor
    invariant: torch.Tensor
    slot_outputs: torch.Tensor | None
    aux_state: Any | None
    encodings: torch.Tensor | None = None
