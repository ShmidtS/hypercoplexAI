"""
HDIM — Soft MoE Router
Мягкая маршрутизация экспертов без token dropping (Puigcerver et al., ICLR 2024).

Отличие от hard top-k:
  Hard MoE: каждый токен → K лучших экспертов (token dropping при перегрузке)
  Soft MoE: каждый токен получает взвешенную смесь ВСЕХ экспертов через dispatch/combine матрицы

Архитектура:
  Φ = softmax(X · Θ)                  # dispatch weights [T × E*S]
  X̃_es = Φ[:, e*S+s]ᵀ · X           # slot inputs [E*S × D]
  ỹ_es = Expert_e(X̃_es)              # expert outputs
  y_t = Σ_{e,s} Φ[t, e*S+s] · ỹ_es  # combine

Где E = num_experts, S = slots_per_expert (обычно 1).
Roller Routing Replay совместим: train_scores обновляются через EMA.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class SoftRouterState:
    """Typed state для SoftMoERouter — совместим с R3MoERouter RouterState API."""
    loss: torch.Tensor
    router_loss: torch.Tensor
    scores: torch.Tensor
    topk_idx: torch.Tensor
    gate_weights: torch.Tensor
    train_scores_snapshot: torch.Tensor
    topk_gate_weights: torch.Tensor
    expert_usage: torch.Tensor
    routing_entropy: torch.Tensor
    dispatch_weights: torch.Tensor   # Soft MoE specific: полные dispatch weights


class SoftMoERouter(nn.Module):
    """
    Soft Mixture of Experts Router.

    Преимущества перед hard top-k:
    1. Нет token dropping — все токены используют всех экспертов
    2. Дифференцируемый routing — более стабильные градиенты
    3. Нет load imbalance — все эксперты получают примерно равную нагрузку
    4. Совместим с R3-идеей через EMA train_scores

    Args:
        input_dim: размерность входа
        num_experts: количество экспертов
        expert_dim: размерность скрытого слоя каждого эксперта
        slots_per_expert: количество слотов на эксперт (обычно 1)
        top_k: для совместимости с R3MoERouter API (не используется в soft routing)
        temperature: температура softmax для dispatch/combine
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        expert_dim: int = 256,
        slots_per_expert: int = 1,
        top_k: int = 2,  # kept for API compatibility
        temperature: float = 1.0,
        z_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.slots_per_expert = slots_per_expert
        self.top_k = min(top_k, num_experts)  # API compat
        self.temperature = temperature
        self.num_slots = num_experts * slots_per_expert
        self.z_loss_weight = z_loss_weight
        self.use_similarity_balance = False  # ICLR 2026 Similarity-Preserving Router

        # Dispatch parameter matrix: (input_dim, num_slots)
        self.dispatch_proj = nn.Linear(input_dim, self.num_slots, bias=False)
        nn.init.normal_(self.dispatch_proj.weight, std=0.02)

        # Expert networks — kept for parameter registration and state_dict compat
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(expert_dim, input_dim),
            )
            for _ in range(num_experts)
        ])

        # Stacked expert weights for batched einsum execution.
        # These ARE the trainable parameters; the individual expert Linear
        # weights are frozen to avoid duplicate-parameter divergence.
        self.W1_stack = nn.Parameter(torch.stack([e[0].weight for e in self.experts]))
        self.b1_stack = nn.Parameter(torch.stack([e[0].bias for e in self.experts]))
        self.W2_stack = nn.Parameter(torch.stack([e[3].weight for e in self.experts]))
        self.b2_stack = nn.Parameter(torch.stack([e[3].bias for e in self.experts]))

        # Freeze original expert params — stacked params are the source of truth
        for e in self.experts:
            for p in e.parameters():
                p.requires_grad_(False)

        # R3-style EMA train scores (for API compatibility)
        self.register_buffer(
            "train_scores",
            torch.ones(num_experts) / num_experts,
        )

    def _compute_dispatch_combine(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычисляет dispatch и combine матрицы.

        Returns:
            dispatch: (T, num_slots) — веса для агрегации токенов в слоты
            combine:  (T, num_slots) — веса для агрегации выходов экспертов
        """
        # logits: (T, num_slots)
        logits = self.dispatch_proj(x) / self.temperature

        # Router z-loss (ST-MoE): penalize large logit magnitudes
        if self.z_loss_weight > 0:
            z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()
        else:
            z_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        self._z_loss = z_loss

        T = x.shape[0]
        # C1 FIX: guard for T=1 — dim=0 softmax with single row returns all-ones
        if T == 1:
            dispatch = torch.ones(1, self.num_slots, device=x.device, dtype=x.dtype) / self.num_slots
        else:
            # dispatch: нормализация по токенам (каждый слот получает mix токенов)
            dispatch = F.softmax(logits, dim=0)   # (T, num_slots)
        # combine: нормализация по слотам (каждый токен получает mix слотов)
        combine = F.softmax(logits, dim=-1)   # (T, num_slots)
        return dispatch, combine

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Soft MoE forward pass.

        Args:
            x: (batch, input_dim) или (batch, seq, input_dim)
        Returns:
            (output, router_state)
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])  # (T, D)
        T, D = x_flat.shape

        dispatch, combine = self._compute_dispatch_combine(x_flat)
        # dispatch: (T, num_slots), combine: (T, num_slots)

        # Aggregate tokens into expert slots: X̃ = dispatch.T @ X
        # Shape: (num_slots, D)
        slot_inputs = dispatch.T @ x_flat  # (num_slots, D)

        # Batched expert execution via stacked weights + einsum
        # Avoids per-expert GPU kernel launches from the sequential for-loop
        x_exp = slot_inputs.view(self.num_experts, self.slots_per_expert, -1)  # (E, S, D)
        # Expert layer 1: (E, S, D) × (E, H, D) → (E, S, H)
        h = torch.einsum('esd,ehd->esh', x_exp, self.W1_stack) + self.b1_stack.unsqueeze(1)
        h = F.gelu(h)
        h = F.dropout(h, p=0.1, training=self.training)
        # Expert layer 2: (E, S, H) × (E, D, H) → (E, S, D)
        out = torch.einsum('esh,edh->esd', h, self.W2_stack) + self.b2_stack.unsqueeze(1)
        slot_outputs = out.reshape(-1, slot_inputs.shape[-1])  # (E*S, D)

        # Combine: y = combine @ slot_outputs
        # Shape: (T, D)
        output = combine @ slot_outputs

        # Update EMA train scores (для R3 совместимости)
        if self.training:
            with torch.no_grad():
                # Используем среднюю нагрузку каждого эксперта как прокси
                expert_load = combine.reshape(T, self.num_experts, self.slots_per_expert)
                expert_load = expert_load.sum(-1).mean(0)  # (num_experts,)
                self.train_scores.mul_(0.9).add_(0.1 * expert_load)

        # Router loss: entropy regularization для равномерного использования
        if self.use_similarity_balance:
            router_loss = self._similarity_preserving_loss(x_flat, dispatch)
        else:
            router_loss = self._entropy_load_balance_loss(combine)

        # Build topk indices для совместимости API
        # Используем combine weights для нахождения top-k экспертов на токен
        expert_weights = combine.reshape(T, self.num_experts, self.slots_per_expert).mean(-1)
        topk_weights, topk_idx = expert_weights.topk(self.top_k, dim=-1)  # (T, top_k)
        topk_weights_norm = topk_weights / topk_weights.sum(-1, keepdim=True).clamp_min(1e-8)

        gate_weights = expert_weights.reshape(*orig_shape[:-1], self.num_experts)
        topk_idx_view = topk_idx.reshape(*orig_shape[:-1], self.top_k)
        topk_weights_view = topk_weights_norm.reshape(*orig_shape[:-1], self.top_k)

        train_scores_snapshot = self.train_scores.detach().clone()
        expert_usage = expert_weights.mean(0).detach()  # (num_experts,)
        routing_entropy = -(gate_weights * (gate_weights + 1e-8).log()).sum(dim=-1).mean()

        # Add z_loss to router loss for stability
        z_loss = getattr(self, '_z_loss', torch.zeros((), device=x.device, dtype=x.dtype))
        router_loss = router_loss + self.z_loss_weight * z_loss

        router_state: Dict[str, Any] = {
            "loss": router_loss,
            "router_loss": router_loss,
            "z_loss": z_loss,
            "scores": expert_weights.reshape(*orig_shape[:-1], self.num_experts),
            "topk_idx": topk_idx_view,
            "gate_weights": gate_weights,
            "train_scores_snapshot": train_scores_snapshot,
            "topk_gate_weights": topk_weights_view,
            "expert_usage": expert_usage,
            "routing_entropy": routing_entropy,
            "dispatch_weights": dispatch.reshape(*orig_shape[:-1], self.num_slots),
        }

        return output.reshape(orig_shape), router_state

    def _entropy_load_balance_loss(
        self,
        combine: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dynamic Switch Transformer load balance loss (C2 FIX).
        Uses runtime expert fraction f_e (not static weight-based p).
        L_lb = E * Σ_e f_e.detach() * mean_usage_e
        Gradient flows only through mean_usage (not f_e) — standard Switch Transformer.
        """
        T = combine.shape[0]
        expert_weights = combine.reshape(T, self.num_experts, self.slots_per_expert).mean(-1)
        # f_e: fraction of tokens dispatched to each expert (detached — no gradient)
        dispatch_per_expert = expert_weights  # (T, E) — combine weights proxy for dispatch
        f_e = dispatch_per_expert.mean(0).detach()  # (E,) — stop-gradient on fraction
        mean_usage = expert_weights.mean(0)  # (E,) — gradient flows here
        return self.num_experts * (f_e * mean_usage).sum()

    def _similarity_preserving_loss(
        self,
        tokens: torch.Tensor,
        dispatch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Similarity-Preserving Router (Omi et al., ICLR 2026)
        Encourages similar tokens to route to similar experts.
        L = -Σ_i Σ_j sim(x_i, x_j) * sim(r_i, r_j)
        """
        # Normalize tokens for cosine similarity
        tokens_norm = F.normalize(tokens, dim=-1)
        token_sim = torch.mm(tokens_norm, tokens_norm.t())  # (T, T)

        # Routing probability vectors (dispatch shape: T, num_slots)
        route_norm = F.normalize(dispatch, dim=-1)
        route_sim = torch.mm(route_norm, route_norm.t())  # (T, T)

        # Negative correlation (minimize = maximize correlation)
        return -torch.sum(token_sim * route_sim) / (tokens.shape[0] ** 2)
