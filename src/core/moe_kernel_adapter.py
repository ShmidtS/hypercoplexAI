"""
HDIM — MoE Kernel Adapter.

Adapter that wraps MoEKernel to be compatible with MoERouter interface.
This enables polymorphic usage of MoEKernel in HDIMPipeline.

Architecture:
    MoEKernel → MoEKernelAdapter → MoERouter interface

The adapter translates:
- MoEKernelState → Dict[str, Any]
- MoEKernel methods → MoERouter abstract methods
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor

from .moe_interface import MoERouter
from .moe_kernel import MoEKernel, MoEKernelState


class MoEKernelAdapter(MoERouter):
    """Adapter wrapping MoEKernel to MoERouter interface.

    This adapter enables MoEKernel to be used anywhere MoERouter is expected,
    providing seamless integration with HDIMPipeline and other components.

    Example:
        >>> from src.core import MoEKernel, MoEKernelConfig, MoEKernelAdapter
        >>> config = MoEKernelConfig(input_dim=128, num_experts=4)
        >>> kernel = MoEKernel(config)
        >>> router = MoEKernelAdapter(kernel)
        >>> # Now router can be used as MoERouter
        >>> output, info = router(torch.randn(2, 128))

    Attributes:
        kernel: The wrapped MoEKernel instance
        num_experts: Number of experts (delegates to kernel)
        num_slots: Number of slots (same as num_experts for MoEKernel)
    """

    def __init__(self, kernel: MoEKernel):
        """Initialize adapter with a MoEKernel instance.

        Args:
            kernel: The MoEKernel to wrap
        """
        super().__init__()
        # Must call super().__init__() BEFORE assigning submodules
        # because nn.Module tracks submodules for parameters/grads
        self.kernel = kernel
        self.num_experts = kernel.num_experts
        self.num_slots = kernel.num_experts  # MoEKernel uses 1 slot per expert

    def forward(self, x: Tensor) -> tuple[Tensor, Dict[str, Any]]:
        """Route input through MoEKernel experts.

        Args:
            x: Input tensor [batch, dim] or [batch, seq, dim]

        Returns:
            output: Processed tensor with same shape as input
            info: Dict with routing information
        """
        output, state = self.kernel(x)

        # Convert MoEKernelState to dict for MoERouter interface
        info: Dict[str, Any] = {
            "expert_load": state.expert_usage,
            "aux_loss": state.router_loss,
            "router_loss": state.router_loss,
            "z_loss": state.z_loss,
            "ortho_loss": state.ortho_loss,
            "expert_usage": state.expert_usage,
            "routing_entropy": state.routing_entropy,
            "expert_weights": state.expert_weights,
            "dispatch_weights": state.dispatch_weights,
            "combine_weights": state.combine_weights,
            "expert_names": state.expert_names,
            "top_expert_idx": state.top_expert_idx,
            "total_loss": state.total_loss(),
            "dominant_expert_names": state.dominant_expert_names(),
        }

        return output, info

    def get_expert_load(self) -> Tensor:
        """Return current expert load statistics from MoEKernel.

        Returns:
            Tensor[num_experts]: EMA train_scores for each expert
        """
        return self.kernel.train_scores.clone()

    def expert_orthogonalization_loss(self) -> Tensor:
        """Return orthogonalization loss from MoEKernel.

        Returns:
            Tensor: Scalar orthogonalization loss
        """
        return self.kernel.expert_orthogonalization_loss()

    def reset_training_state(self) -> None:
        """Reset MoEKernel training state (EMA scores and bias)."""
        # Reset EMA train_scores to uniform distribution
        self.kernel.train_scores.fill_(1.0 / self.kernel.num_experts)
        # Reset per-expert bias if present
        self.kernel.reset_bias()

    def train(self, mode: bool = True) -> "MoEKernelAdapter":
        """Set training mode on kernel."""
        self.kernel.train(mode)
        # Also call parent train() to set self.training attribute
        return super().train(mode)

    def eval(self) -> "MoEKernelAdapter":
        """Set evaluation mode on kernel."""
        self.kernel.eval()
        # Also call parent eval() to set self.training attribute
        return super().eval()
