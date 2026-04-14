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
        self.top_k = min(2, kernel.num_experts)

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

        # Alias keys expected by _forward_core
        # Compute proper top-k from expert_weights -> (..., top_k)
        top_k = min(2, self.kernel.num_experts)
        # expert_weights may be (B, E) or (B, T, E) — always topk along last dim
        topk_weights, topk_indices = state.expert_weights.topk(top_k, dim=-1)  # (..., top_k)

        info.update({
            "gate_weights": state.expert_weights,
            "topk_idx": topk_indices,
            "topk_gate_weights": topk_weights,
            "train_scores_snapshot": self.kernel.train_scores.detach().clone(),
        })

        return output, info

    @property
    def expert_names(self) -> list:
        """Proxy: delegate expert_names to kernel."""
        return self.kernel.expert_names

    @property
    def train_scores(self) -> Tensor:
        """Proxy: delegate train_scores to kernel (in-place ops like fill_/mul_/add_ work)."""
        return self.kernel.train_scores

    def enable_shared_expert(self) -> None:
        """Enable shared expert at runtime.

        MoEKernel reads use_shared_expert at __init__; this sets the
        runtime flag AND creates the shared_expert module if missing.
        """
        if getattr(self.kernel, 'shared_expert', None) is not None:
            return
        self.kernel.use_shared_expert = True
        self.kernel.config.use_shared_expert = True
        input_dim = self.kernel.config.input_dim
        hidden_dim = self.kernel.config.expert_hidden_dim
        self.kernel.shared_expert = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def enable_aux_loss_free(self, aux_lr: float = 0.01) -> None:
        """Enable aux-loss-free load balancing at runtime.

        MoEKernel stores bias as _expert_bias buffer and uses _aux_lr.
        When disabled at init, _expert_bias is registered as None —
        replace it with a real tensor.
        """
        self.kernel.config.use_aux_loss_free = True
        self.kernel.config.use_bias_balancing = True
        self.kernel._aux_lr = aux_lr
        bias = getattr(self.kernel, '_expert_bias', None)
        if bias is None:
            self.kernel._expert_bias = torch.zeros(self.kernel.num_experts)

    def enable_expert_ortho(self) -> None:
        """Enable expert orthogonalization loss at runtime."""
        self.kernel.use_expert_ortho = True
        self.kernel.config.use_expert_ortho = True

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
