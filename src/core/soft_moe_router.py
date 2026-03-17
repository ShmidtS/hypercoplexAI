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
from typing import Any, Dict, Tuple


class SoftMoERouter(nn.Module):
    """
    Soft Mixture of Experts Router.

    Преимущества перед hard top-k:
    1. Нет token dropping — все токены используют всех экспертов
    2. Дифференцируемый routing — более стабильные градиенты
    3. Нет load imbalance — все эксперты получают примерно равную нагрузку
    4. Совместим с R3-идеей через EMA train_scores

    Phase 26 нововведения:
    - Shared Expert (DeepSeek-V3): всегда-включённый FFN обрабатывает ВСЕ входы
    - Auxiliary-Loss-Free Balancing (DeepSeek-V3): bias-based балансировка вместо loss
    - Expert Orthogonalization: эксперты учатся разным представлениям
    - Sigmoid Gating: опциональная замена softmax на per-expert sigmoid

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

        # Dispatch parameter matrix: (input_dim, num_slots)
        self.dispatch_proj = nn.Linear(input_dim, self.num_slots, bias=False)
        nn.init.normal_(self.dispatch_proj.weight, std=0.02)

        # Stacked expert weights for batched einsum execution.
        W1 = torch.stack([nn.Linear(input_dim, expert_dim).weight for _ in range(num_experts)])
        b1 = torch.zeros(num_experts, expert_dim)
        W2 = torch.stack([nn.Linear(expert_dim, input_dim).weight for _ in range(num_experts)])
        b2 = torch.zeros(num_experts, input_dim)
        nn.init.kaiming_uniform_(W1.reshape(-1, input_dim))
        nn.init.kaiming_uniform_(W2.reshape(-1, expert_dim))
        self.W1_stack = nn.Parameter(W1)
        self.b1_stack = nn.Parameter(b1)
        self.W2_stack = nn.Parameter(W2)
        self.b2_stack = nn.Parameter(b2)

        # R3-style EMA train scores (for API compatibility)
        self.register_buffer(
            "train_scores",
            torch.ones(num_experts) / num_experts,
        )

        # Phase 26: Shared Expert (DeepSeek-V3)
        self.use_shared_expert = False
        self._shared_expert = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(expert_dim, input_dim),
        )

        # Phase 26: Auxiliary-Loss-Free Balancing (DeepSeek-V3)
        # Per-expert bias terms that dynamically adjust routing
        self.use_aux_loss_free = False
        self._expert_bias = nn.Parameter(torch.zeros(num_experts))
        self._aux_lr = 0.001  # bias adjustment rate
        self.register_buffer("_target_load", torch.ones(num_experts) / num_experts)

        # Phase 26: Expert Orthogonalization loss flag
        self.use_expert_ortho = False

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

        # Phase 26: Auxiliary-Loss-Free balancing — add per-expert bias
        if self.use_aux_loss_free:
            bias = self._expert_bias  # (num_experts,)
            bias_expanded = bias.repeat_interleave(self.slots_per_expert)  # (num_slots,)
            logits = logits + bias_expanded.unsqueeze(0)  # broadcast over batch

        # Router z-loss (ST-MoE): penalize large logit magnitudes
        if self.z_loss_weight > 0:
            self._z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()
        else:
            self._z_loss = None

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

    def _evaluate_experts(self, slot_inputs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all experts on slot inputs using batched matrix multiplications.

        Args:
            slot_inputs: (num_slots, D) aggregated slot representations
        Returns:
            slot_outputs: (num_slots, D) expert outputs
        """
        x_exp = slot_inputs.view(self.num_experts, self.slots_per_expert, -1)
        h = torch.einsum('esd,ehd->esh', x_exp, self.W1_stack) + self.b1_stack.unsqueeze(1)
        h = F.gelu(h)
        h = F.dropout(h, p=0.1, training=self.training)
        out = torch.einsum('esh,edh->esd', h, self.W2_stack) + self.b2_stack.unsqueeze(1)
        # Clamp to prevent fp16 overflow under AMP
        out = torch.clamp(out, min=-10.0, max=10.0)
        return out.reshape(-1, slot_inputs.shape[-1])

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
        slot_outputs = self._evaluate_experts(slot_inputs)  # (E*S, D)

        output = combine @ slot_outputs  # (T, D)
        # NaN/Inf protection: clamp to prevent overflow, preserve gradients
        output = torch.nan_to_num(output, nan=0.0, posinf=10.0, neginf=-10.0)
        output = torch.clamp(output, min=-10.0, max=10.0)

        # Phase 26: Shared Expert (DeepSeek-V3) — always-on FFN
        if self.use_shared_expert:
            shared_out = self._shared_expert(x_flat)
            output = output + shared_out

        combine_reshaped = combine.reshape(T, self.num_experts, self.slots_per_expert)
        expert_weights = combine_reshaped.mean(-1)  # (T, E) — computed once

        # Update EMA train scores (для R3 совместимости)
        if self.training:
            with torch.no_grad():
                expert_load = combine_reshaped.sum(-1).mean(0)  # (num_experts,)
                self.train_scores.mul_(0.9).add_(0.1 * expert_load)

                # Phase 26: Auxiliary-Loss-Free bias update (DeepSeek-V3)
                if self.use_aux_loss_free:
                    delta = torch.sign(expert_load - self._target_load)
                    self._expert_bias.data -= self._aux_lr * delta

        # Router loss: load balance
        router_loss = self._entropy_load_balance_loss(combine)

        # Build topk indices для совместимости API
        topk_weights, topk_idx = expert_weights.topk(self.top_k, dim=-1)  # (T, top_k)
        topk_weights_norm = topk_weights / topk_weights.sum(-1, keepdim=True).clamp_min(1e-8)

        gate_weights = expert_weights.reshape(*orig_shape[:-1], self.num_experts)
        topk_idx_view = topk_idx.reshape(*orig_shape[:-1], self.top_k)
        topk_weights_view = topk_weights_norm.reshape(*orig_shape[:-1], self.top_k)

        train_scores_snapshot = self.train_scores.detach().clone()
        expert_usage = expert_weights.mean(0).detach()  # (num_experts,)
        routing_entropy = -(gate_weights * (gate_weights + 1e-8).log()).sum(dim=-1).mean()

        # Add z_loss to router loss for stability
        z_loss = self._z_loss if self._z_loss is not None else torch.zeros((), device=x.device, dtype=x.dtype)
        if self.z_loss_weight > 0 and self._z_loss is not None:
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

    def expert_orthogonalization_loss(self) -> torch.Tensor:
        """Phase 26: Expert Orthogonalization loss (arXiv:2505.22323).

        Penalizes expert weight matrices for being too similar.
        L_o = ||W1 @ W1^T - I||^2 + ||W2 @ W2^T - I||^2

        Encourages experts to learn truly orthogonal representations.
        """
        # W1_stack: (E, expert_dim, input_dim)
        # W2_stack: (E, input_dim, expert_dim)
        E = self.num_experts
        device = self.W1_stack.device
        I = torch.eye(E, device=device)

        # Normalize each expert's weight vector
        w1_norm = F.normalize(self.W1_stack.reshape(E, -1), dim=-1)  # (E, D1)
        gram1 = w1_norm @ w1_norm.T  # (E, E)
        loss1 = ((gram1 - I) ** 2).mean()

        w2_norm = F.normalize(self.W2_stack.reshape(E, -1), dim=-1)  # (E, D2)
        gram2 = w2_norm @ w2_norm.T  # (E, E)
        loss2 = ((gram2 - I) ** 2).mean()

        return (loss1 + loss2) * 0.5

    def enable_shared_expert(self) -> None:
        """Enable DeepSeek-V3 always-on shared expert."""
        self.use_shared_expert = True

    def enable_aux_loss_free(self, aux_lr: float = 0.001) -> None:
        """Enable Auxiliary-Loss-Free balancing (DeepSeek-V3)."""
        self.use_aux_loss_free = True
        self._aux_lr = aux_lr

    def enable_expert_ortho(self) -> None:
        """Enable expert orthogonalization loss."""
        self.use_expert_ortho = True
