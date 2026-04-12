"""Typed result containers for HDIM model forward paths.

Replaces positional tuple returns with named dataclass fields,
eliminating silent misalignment bugs from incorrect unpack order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class CoreResult:
    """Return type for HDIMModel._forward_core() — replaces 20 positional values."""

    output: torch.Tensor
    routing_weights: torch.Tensor
    raw_invariant: torch.Tensor
    memory_augmented_invariant: torch.Tensor
    exported_invariant: torch.Tensor
    topk_idx: torch.Tensor
    topk_gate_weights: torch.Tensor
    train_scores_snapshot: torch.Tensor
    expert_usage: torch.Tensor
    routing_entropy: torch.Tensor
    memory_loss: torch.Tensor
    router_loss: torch.Tensor
    z_loss: torch.Tensor
    memory_updated: bool
    slot_outputs: Optional[torch.Tensor]
    hallucination_risk: float
    memory_surprise: Optional[float]
    feedback_action: Optional[str]
    online_loss: torch.Tensor
    online_updated: bool


@dataclass
class ForwardResult:
    """Return type for HDIMModel.forward() and TextHDIMModel.forward_texts()."""

    output: torch.Tensor
    routing_weights: torch.Tensor
    invariant: torch.Tensor
    slot_outputs: Optional[torch.Tensor]
    aux_state: Optional[Any]
    encodings: Optional[torch.Tensor] = None
