"""State containers for the MoE kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class MoEKernelState:
    """Состояние одного forward-прохода MoE-ядра."""

    output: torch.Tensor
    router_loss: torch.Tensor
    z_loss: torch.Tensor
    ortho_loss: torch.Tensor
    expert_weights: torch.Tensor
    expert_usage: torch.Tensor
    routing_entropy: torch.Tensor
    dispatch_weights: torch.Tensor
    combine_weights: torch.Tensor
    expert_names: List[str]
    top_expert_idx: torch.Tensor
    expert_reliability: torch.Tensor
    slot_outputs: Optional[torch.Tensor] = None

    def total_loss(self) -> torch.Tensor:
        return self.router_loss + self.z_loss + self.ortho_loss

    def dominant_expert_names(self) -> List[str]:
        idx = self.top_expert_idx[..., 0] if self.top_expert_idx.dim() >= 2 else self.top_expert_idx
        return [self.expert_names[int(i)] for i in idx.flatten().tolist()]

    def to_dict(self, orig_shape: torch.Size, num_experts: int, slots_per_expert: int, top_k: int) -> Dict[str, Any]:
        """Convert to MoERouter-compatible dict."""
        expert_weights = self.expert_weights.reshape(*orig_shape[:-1], num_experts)
        topk_weights, topk_indices = self.expert_weights.topk(top_k, dim=-1)
        topk_weights_norm = topk_weights / topk_weights.sum(-1, keepdim=True).clamp_min(1e-8)

        return {
            "expert_load": self.expert_usage,
            "aux_loss": self.router_loss,
            "router_loss": self.router_loss,
            "z_loss": self.z_loss,
            "ortho_loss": self.ortho_loss,
            "expert_usage": self.expert_usage,
            "routing_entropy": self.routing_entropy,
            "expert_weights": expert_weights,
            "dispatch_weights": self.dispatch_weights.reshape(*orig_shape[:-1], num_experts * slots_per_expert),
            "combine_weights": self.combine_weights.reshape(*orig_shape[:-1], num_experts * slots_per_expert),
            "expert_names": self.expert_names,
            "top_expert_idx": self.top_expert_idx.reshape(*orig_shape[:-1], top_k),
            "total_loss": self.total_loss(),
            "dominant_expert_names": self.dominant_expert_names(),
            "slot_outputs": self.slot_outputs,
            "gate_weights": expert_weights,
            "topk_idx": topk_indices.reshape(*orig_shape[:-1], top_k),
            "topk_gate_weights": topk_weights_norm.reshape(*orig_shape[:-1], top_k),
            "train_scores_snapshot": self.expert_reliability.detach().clone(),
        }
