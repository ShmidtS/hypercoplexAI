"""
HDIM — MoE Router Abstract Interface.

Defines the common interface that all MoE implementations must follow.
This enables polymorphic usage of SoftMoERouter, MoEKernel, and future implementations.

Architecture:
- MoERouter (abstract base): defines contract for all routers
- SoftMoERouter: soft routing without token dropping
- MoEKernel: domain-specific experts with advanced features
- MoEKernelAdapter: wraps MoEKernel to MoERouter interface
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol, TypedDict, runtime_checkable

import torch
from torch import Tensor, nn


@runtime_checkable
class TextEncoder(Protocol):
    """Protocol for text encoding components (SimpleTextEncoder, SBERTEncoder, ModernBertEncoder)."""

    def forward(self, texts: list, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor: ...
    def encode(self, texts: list) -> torch.Tensor: ...


class RouterState(TypedDict, total=False):
    """Structured routing state returned by MoE routers."""

    routing_weights: torch.Tensor
    expert_indices: torch.Tensor
    diversity_loss: torch.Tensor
    z_loss: torch.Tensor
    entropy: torch.Tensor
    domain_logits: torch.Tensor


class MoERouter(nn.Module, ABC):
    """Abstract interface for Mixture-of-Experts routers.

    All MoE implementations (SoftMoERouter, MoEKernel via adapter) must
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

    @abstractmethod
    def expert_orthogonalization_loss(self) -> Tensor:
        """Return orthogonalization loss for expert diversity.

        This loss encourages experts to learn different representations
        by penalizing similarity between expert weight matrices.

        Returns:
            Tensor: Scalar loss value (0 if orthogonalization not enabled)
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
