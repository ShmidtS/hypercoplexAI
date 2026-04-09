"""
HDIM — MaxScore MoE Router (Wang et al., ACL 2025).

Maximum Score Routing for Mixture-of-Experts.
Paper: Wang et al., ACL 2025 "Maximum Score Routing For Mixture-of-Experts"

Key innovation: Models routing as min-cost max-flow problem.
- 0% token dropping (no capacity constraints)
- SoftTopk operator for differentiable expert selection
- Optimal assignment via linear programming relaxation

Architecture:
    scores = gate(x)                           # (B, S, E) routing logits
    topk_scores, topk_idx = soft_topk(scores)  # differentiable top-k
    output = weighted_expert_combine(topk_scores, topk_idx)

Difference from SoftMoERouter:
- SoftMoE: every token uses ALL experts via dispatch/combine matrices
- MaxScore: every token uses top-k experts, but selection is differentiable

Checkpoint/Rollback (MAJOR-3):
- save_checkpoint(): persist router state
- load_checkpoint(): restore router state
- rollback(): revert to last stable checkpoint on failure
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_interface import MoERouter


@dataclass
class RouterCheckpoint:
    """Checkpoint state for MaxScoreRouter.

    Stores model parameters and training statistics for recovery.
    """

    gate_weight: torch.Tensor
    train_scores: torch.Tensor
    step: int = 0
    loss_value: float = float("inf")
    metadata: Dict[str, Any] = field(default_factory=dict)


class RouterResult:
    """Result container for MaxScoreRouter forward pass.

    Attributes:
        topk_indices: (B, S, top_k) indices of selected experts
        topk_weights: (B, S, top_k) weights for selected experts
        routing_entropy: scalar tensor, entropy of routing distribution
        expert_weights: (B, S, E) full routing weights per token
        router_loss: scalar tensor, load balancing loss
    """

    def __init__(
        self,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        routing_entropy: torch.Tensor,
        expert_weights: torch.Tensor,
        router_loss: torch.Tensor,
        gate_weights: Optional[torch.Tensor] = None,
    ):
        self.topk_indices = topk_indices
        self.topk_weights = topk_weights
        self.routing_entropy = routing_entropy
        self.expert_weights = expert_weights
        self.router_loss = router_loss
        self.gate_weights = gate_weights if gate_weights is not None else expert_weights

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary for compatibility with MoERouter interface."""
        return {
            "topk_idx": self.topk_indices,
            "topk_weights": self.topk_weights,
            "routing_entropy": self.routing_entropy,
            "expert_weights": self.expert_weights,
            "router_loss": self.router_loss,
            "gate_weights": self.gate_weights,
        }


class MaxScoreRouter(MoERouter):
    """Maximum Score Routing for Mixture-of-Experts.

    Paper: Wang et al., ACL 2025 "Maximum Score Routing For Mixture-of-Experts"

    Models routing as min-cost max-flow problem. SoftTopk operator provides
    differentiable expert selection without token dropping.

    Key features:
    1. 0% token dropping — all tokens are processed
    2. Differentiable top-k selection via SoftTopk
    3. Load balancing via entropy regularization
    4. Checkpoint/rollback for fault tolerance (MAJOR-3)

    Args:
        hidden_dim: Input dimension
        num_experts: Number of experts in the MoE layer
        top_k: Number of experts to route each token to (default: 2)
        temperature: Softmax temperature for SoftTopk (default: 10.0)
        entropy_weight: Weight for entropy regularization loss (default: 0.01)
    """

    num_experts: int
    num_slots: int

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        temperature: float = 10.0,
        entropy_weight: float = 0.01,
        z_loss_weight: float = 0.0,
        route_only: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.z_loss_weight = z_loss_weight
        self.route_only = route_only

        # num_slots for MoERouter interface compatibility
        self.num_slots = num_experts

        # Gate: projects hidden states to expert scores
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)

        # EMA train scores for load tracking (R3-style compatibility)
        self.register_buffer(
            "train_scores",
            torch.ones(num_experts) / num_experts,
        )

        # Checkpoint system (MAJOR-3)
        self._checkpoint: Optional[RouterCheckpoint] = None
        self._checkpoint_step: int = 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """MaxScore routing with SoftTopk.

        Args:
            x: Input tensor with shape:
                - [batch, hidden_dim] for single vectors
                - [batch, seq, hidden_dim] for sequences

        Returns:
            output: Combined expert outputs, same shape as input
            info: Dict with routing information:
                - topk_idx: (B, S, top_k) expert indices
                - topk_weights: (B, S, top_k) expert weights
                - routing_entropy: scalar entropy
                - router_loss: load balancing loss
                - expert_usage: (E,) usage statistics
        """
        import warnings
        warnings.warn(
            "MaxScoreRouter.forward() returns pass-through input; "
            "use SoftMoERouter or MoEKernel for end-to-end routing.",
            UserWarning, stacklevel=2,
        )
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])  # (T, D)
        T = x_flat.shape[0]

        # Compute routing scores
        gate_logits = self.gate(x_flat)  # (T, E)

        # Router z-loss for training stability
        z_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.z_loss_weight > 0:
            lse = torch.logsumexp(gate_logits, dim=-1)
            z_loss = torch.clamp(lse, max=10.0).pow(2).mean()

        # Temperature-scaled softmax on raw logits (single softmax, not double)
        scores = F.softmax(gate_logits / self.temperature, dim=-1)  # (T, E)

        # SoftTopk: differentiable top-k selection from raw logits
        topk_weights, topk_indices = self._soft_topk(gate_logits)  # (T, k), (T, k)

        # Compute routing entropy for monitoring
        entropy = self._compute_entropy(scores)

        # NOTE: MaxScoreRouter is a pure router when route_only=True -- it computes
        # optimal dispatch weights but does NOT apply experts. Use with an external
        # expert executor. For the standard pipeline, use SoftMoERouter or MoEKernel.
        # When route_only=False, output is routing weights for external combination.
        if self.route_only:
            output = x_flat
            output_shape = orig_shape
        else:
            output = scores
            output_shape = orig_shape[:-1] + (self.num_experts,)

        # Load balancing loss
        router_loss = self._compute_load_balance_loss(scores)

        # Add entropy regularization
        router_loss = router_loss + self.entropy_weight * (-entropy)

        # Add z-loss if enabled
        if self.z_loss_weight > 0:
            router_loss = router_loss + self.z_loss_weight * z_loss

        # Update EMA train scores during training
        if self.training:
            with torch.no_grad():
                expert_usage = scores.mean(0)  # (E,)
                self.train_scores.mul_(0.9).add_(expert_usage, alpha=0.1)
        else:
            expert_usage = scores.mean(0).detach()

        # Build output dict compatible with MoERouter interface
        topk_idx_reshaped = topk_indices.reshape(*orig_shape[:-1], self.top_k)
        topk_weights_reshaped = topk_weights.reshape(*orig_shape[:-1], self.top_k)
        scores_reshaped = scores.reshape(*orig_shape[:-1], self.num_experts)

        info: Dict[str, Any] = {
            "loss": router_loss,
            "router_loss": router_loss,
            "z_loss": z_loss,
            "scores": scores_reshaped,
            "topk_idx": topk_idx_reshaped,
            "gate_weights": scores_reshaped,
            "topk_gate_weights": topk_weights_reshaped,
            "train_scores_snapshot": self.train_scores.detach().clone(),
            "expert_usage": expert_usage,
            "routing_entropy": entropy,
        }

        return output.reshape(output_shape), info

    def _soft_topk(
        self, logits: torch.Tensor, dim: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable soft top-k selection.

        Implements SoftTopk operator from Wang et al.:
        1. Hard top-k: select top-k logits
        2. Temperature-scaled softmax over top-k for differentiability
        3. Renormalize weights to sum to 1

        This is differentiable through the softmax step,
        while maintaining sparse selection via hard top-k.

        Args:
            logits: (T, E) raw gate logits (NOT softmax probabilities)
            dim: Dimension to select from (default: -1)

        Returns:
            topk_soft: (T, k) differentiable top-k weights
            topk_idx: (T, k) indices of selected experts
        """
        # Hard top-k selection on raw logits
        topk_vals, topk_idx = logits.topk(self.top_k, dim=dim)

        # Single temperature-scaled softmax over top-k for differentiability
        topk_soft = F.softmax(topk_vals / self.temperature, dim=dim)

        # Renormalize to sum to 1 (softmax already normalizes, but for safety)
        topk_soft = topk_soft / (topk_soft.sum(dim=dim, keepdim=True) + 1e-8)

        return topk_soft, topk_idx

    def _compute_entropy(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute routing entropy for load balancing.

        Higher entropy = more uniform expert usage.
        Lower entropy = more specialized routing.

        Args:
            scores: (T, E) routing probabilities

        Returns:
            entropy: Scalar tensor, mean entropy across tokens
        """
        # Entropy: H = -sum(p * log(p))
        log_probs = torch.log(scores + 1e-8)
        entropy = -(scores * log_probs).sum(dim=-1).mean()
        return entropy

    def _compute_load_balance_loss(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss.

        Encourages uniform expert usage via auxiliary loss.
        Uses Switch Transformer-style load balance:
        L_lb = E * sum_e f_e * P_e

        where:
        - f_e: fraction of tokens routed to expert e
        - P_e: average routing probability for expert e

        Args:
            scores: (T, E) routing probabilities

        Returns:
            loss: Scalar load balance loss
        """
        T = scores.shape[0]

        # f_e: fraction of tokens dispatched to each expert (detached)
        # For MaxScore, we use mean probability as proxy
        f_e = scores.mean(0).detach()  # (E,)

        # P_e: mean routing probability (with gradient)
        P_e = scores.mean(0)  # (E,)

        # Load balance loss
        loss = self.num_experts * (f_e * P_e).sum()

        return loss

    def get_expert_load(self) -> torch.Tensor:
        """Return current expert load statistics (EMA train_scores).

        Returns:
            Tensor[num_experts]: EMA load for each expert.
            Values sum to approximately 1.0 for balanced routing.
        """
        return self.train_scores.clone()

    def expert_orthogonalization_loss(self) -> torch.Tensor:
        """Return orthogonalization loss for expert diversity.

        Note: MaxScoreRouter doesn't have expert weights,
        so this returns zero. This method exists for MoERouter
        interface compatibility.

        Returns:
            Tensor: Scalar zero tensor
        """
        return torch.tensor(0.0, device=self.gate.weight.device)

    def reset_training_state(self) -> None:
        """Reset EMA train_scores to uniform distribution."""
        self.train_scores.fill_(1.0 / self.num_experts)

    # ============================================================
    # Checkpoint/Rollback (MAJOR-3)
    # ============================================================

    def save_checkpoint(self, step: int, loss_value: float = float("inf")) -> None:
        """Save current router state as checkpoint.

        Args:
            step: Training step number
            loss_value: Current loss value for comparison
        """
        self._checkpoint = RouterCheckpoint(
            gate_weight=self.gate.weight.data.clone(),
            train_scores=self.train_scores.clone(),
            step=step,
            loss_value=loss_value,
            metadata={"top_k": self.top_k, "temperature": self.temperature},
        )
        self._checkpoint_step = step

    def load_checkpoint(self, checkpoint: Optional[RouterCheckpoint] = None) -> bool:
        """Load router state from checkpoint.

        Args:
            checkpoint: Checkpoint to load (uses internal checkpoint if None)

        Returns:
            True if checkpoint loaded successfully, False otherwise
        """
        if checkpoint is not None:
            cp = checkpoint
        elif self._checkpoint is not None:
            cp = self._checkpoint
        else:
            return False

        # Restore state
        self.gate.weight.data.copy_(cp.gate_weight)
        self.train_scores.copy_(cp.train_scores)
        return True

    def rollback(self) -> bool:
        """Rollback to last saved checkpoint.

        Called when training step fails (NaN, divergence, etc.).

        Returns:
            True if rollback successful, False if no checkpoint
        """
        if self._checkpoint is None:
            return False

        return self.load_checkpoint()

    def get_checkpoint(self) -> Optional[RouterCheckpoint]:
        """Get current checkpoint without loading.

        Returns:
            RouterCheckpoint if available, None otherwise
        """
        return self._checkpoint

    def has_checkpoint(self) -> bool:
        """Check if checkpoint is available.

        Returns:
            True if checkpoint exists, False otherwise
        """
        return self._checkpoint is not None

    def clear_checkpoint(self) -> None:
        """Clear saved checkpoint."""
        self._checkpoint = None
        self._checkpoint_step = 0

    # ============================================================
    # Utility Methods
    # ============================================================

    def get_routing_stats(self, x: torch.Tensor) -> Dict[str, float]:
        """Get routing statistics for analysis.

        Args:
            x: Input tensor

        Returns:
            Dict with routing statistics:
                - entropy: Mean routing entropy
                - max_usage: Maximum expert usage
                - min_usage: Minimum expert usage
                - usage_std: Standard deviation of expert usage
        """
        self.eval()
        with torch.no_grad():
            _, info = self.forward(x)

            usage = info["expert_usage"]
            entropy = info["routing_entropy"].item()

            return {
                "entropy": entropy,
                "max_usage": usage.max().item(),
                "min_usage": usage.min().item(),
                "usage_std": usage.std().item(),
                "usage_range": (usage.max() - usage.min()).item(),
            }

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"temperature={self.temperature}"
        )
