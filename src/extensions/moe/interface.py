"""
HDIM — MoE Router Abstract Interface.

Defines the common interface that all MoE implementations must follow.
This enables polymorphic usage of SoftMoERouter, MoEKernel, and future implementations.

Architecture:
- MoERouter (abstract base): defines contract for all routers
- SoftMoERouter: soft routing without token dropping
- MoEKernel: domain-specific experts with advanced features
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, TypedDict

import torch
from torch import Tensor, nn


class RouterState(TypedDict, total=False):
    """Structured routing state returned by MoE routers.

    Unified state format for both MoEKernel and SoftMoERouter.
    All MoE implementations must populate at least the required fields.
    """

    # Required fields
    router_loss: torch.Tensor
    z_loss: torch.Tensor
    expert_usage: torch.Tensor
    routing_entropy: torch.Tensor
    gate_weights: torch.Tensor
    topk_idx: torch.Tensor
    topk_gate_weights: torch.Tensor
    train_scores_snapshot: torch.Tensor
    dispatch_weights: torch.Tensor
    combine_weights: torch.Tensor

    # Optional fields (present in MoEKernel, may be absent in SoftMoERouter)
    expert_weights: torch.Tensor
    ortho_loss: torch.Tensor
    expert_names: list
    top_expert_idx: torch.Tensor
    expert_reliability: torch.Tensor
    slot_outputs: torch.Tensor
    alignment: torch.Tensor
    u_route: torch.Tensor
    g_target: torch.Tensor


class MoERouter(nn.Module, ABC):
    """Abstract interface for Mixture-of-Experts routers.

    All MoE implementations (SoftMoERouter, MoEKernel) must
    inherit from this class to ensure API compatibility.

    This enables:
    1. Polymorphic routing in HDIMPipeline
    2. Easy swapping of MoE implementations
    3. Consistent testing interface

    Attributes:
        num_experts: Number of experts in the router
        num_slots: Total number of slots (num_experts * slots_per_expert)
    """

    num_experts: int
    num_slots: int

    @abstractmethod
    def forward(self, x: Tensor) -> tuple[Tensor, Dict[str, Any]]:
        """Route input through experts.

        Args:
            x: Input tensor with shape:
               - [batch, dim] for single vectors
               - [batch, seq, dim] for sequences

        Returns:
            output: Processed tensor with same shape as input
            info: Dict with routing information:
                - expert_load: Tensor[num_experts] - load per expert
                - aux_loss: Optional[Tensor] - auxiliary loss if applicable
                - router_loss: Tensor - load balancing loss
                - expert_usage: Tensor[num_experts] - usage statistics
        """
        ...

    @abstractmethod
    def get_expert_load(self) -> Tensor:
        """Return current expert load statistics.

        Returns:
            Tensor[num_experts]: EMA or current load for each expert.
            Values should sum to approximately 1.0 for balanced routing.
        """
        ...

    def reset_training_state(self) -> None:
        """Reset EMA and training statistics.

        Override this method if the router maintains training state
        that needs to be reset between training runs.

        Default implementation does nothing.
        """
        pass

    @property
    def expert_load(self) -> Tensor:
        """Alias for get_expert_load() for convenience."""
        return self.get_expert_load()
