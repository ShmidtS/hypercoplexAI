"""
HDIM — MoE Router с R3-стабилизацией.
Marshalling of Experts через Rollout Routing Replay.

Математика R3:
 s_infer_i — инференс-скор i-го эксперта
 s_train_i — обучающий скор (сохранённый)
 I_infer = TopKMask(s_infer, K) — маска K лучших экспертов
 g_i = I_infer_i * exp(s_train_i) / Σ_j(I_infer_j * exp(s_train_j))

R3 использует ИНФЕРЕНС-скоры для выбора экспертов, но ОБУЧАЮЩИЕ скоры для взвешивания.
Это устраняет training-inference mismatch в MoE.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Tuple


class R3MoERouter(nn.Module):
    """
    Маршрутизатор экспертов с Rollout Routing Replay (R3).

    Args:
        input_dim: размерность входного вектора
        num_experts: количество экспертов
        top_k: количество активируемых экспертов
        expert_dim: размерность каждого эксперта (hidden)
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        expert_dim: int = 256,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dim = expert_dim
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, input_dim),
            )
            for _ in range(num_experts)
        ])
        self.register_buffer(
            "train_scores",
            torch.ones(num_experts) / num_experts,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x: входной тензор (..., input_dim)
        Returns:
            (output, router_state): выход MoE и реальное состояние роутера.
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])

        scores = self.router(x_flat)
        _, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        mask = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)

        if self.training:
            with torch.no_grad():
                mean_scores = scores.detach().mean(0)
                self.train_scores.mul_(0.9).add_(0.1 * mean_scores)

        train_scores_snapshot = self.train_scores.detach().clone()
        train_weights = torch.softmax(train_scores_snapshot, dim=-1)
        gate_weights = mask * train_weights.unsqueeze(0)
        gate_sum = gate_weights.sum(-1, keepdim=True).clamp(min=1e-8)
        gate_weights = gate_weights / gate_sum
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            expert_out = expert(x_flat)
            output += gate_weights[:, i:i + 1] * expert_out
        router_loss = self._load_balance_loss(scores, mask)
        router_state = {
            "loss": router_loss,
            "router_loss": router_loss,
            "scores": scores.reshape(*orig_shape[:-1], self.num_experts),
            "topk_idx": topk_idx.reshape(*orig_shape[:-1], self.top_k),
            "gate_weights": gate_weights.reshape(*orig_shape[:-1], self.num_experts),
            "train_scores_snapshot": train_scores_snapshot,
        }
        return output.reshape(orig_shape), router_state

    def _load_balance_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Switch Transformer load balancing loss.
        L_lb = E * Σ_i f_i * p_i
        f_i — доля токенов направленных к эксперту i
        p_i — средний softmax-скор эксперта i
        """
        num_experts = self.num_experts
        f = mask.mean(0)
        p = torch.softmax(scores, dim=-1).mean(0)
        return num_experts * (f * p).sum()
