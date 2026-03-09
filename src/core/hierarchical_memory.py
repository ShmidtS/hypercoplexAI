"""
HDIM — Hierarchical Titans Memory
Двухуровневая ассоциативная память с surprise-based обновлением.

Архитектура:
  Level 1 (Working): быстрая память малого объёма, обновляется на каждом шаге
  Level 2 (Long-term): медленная память большого объёма, обновляется при высоком surprise

Математика (Titans + расширение):
  surprise_t = || v_t - M_{t-1}(k_t) ||²  — удивление = ошибка предсказания
  update_gate = σ(W_gate * [k_t, surprise_t])  — gate на основе surprise

  Level 1 (working memory — всегда обновляется):
    L1 = || M1_{t-1}(k_t) - v_t ||²
    S1_t = η1 * S1_{t-1} - θ1 * ∇L1
    M1_t = (1 - α1_t) * M1_{t-1} + S1_t

  Level 2 (long-term memory — обновляется при surprise > threshold):
    L2 = || M2_{t-1}(k_t) - v_t ||²
    S2_t = η2 * S2_{t-1} - θ2 * ∇L2  (только если surprise > θ_surprise)
    M2_t = (1 - α2_t * gate) * M2_{t-1} + gate * S2_t

  Retrieval: m_t = α_blend * M1(k_t) + (1-α_blend) * M2(k_t)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HierarchicalMemoryState:
    retrieved: torch.Tensor
    loss: torch.Tensor
    updated: bool
    working_surprise: torch.Tensor
    longterm_gate: torch.Tensor
    blend_alpha: torch.Tensor


class HierarchicalTitansMemory(nn.Module):
    """
    Двухуровневая Titans-подобная память с surprise-based routing.

    Args:
        key_dim: размерность ключей
        val_dim: размерность значений
        hidden_dim: размерность проекции гейтов
        surprise_threshold: порог surprise для обновления long-term памяти
        blend_init: начальное значение blend коэффициента (доля working memory)
    """

    def __init__(
        self,
        key_dim: int,
        val_dim: int,
        hidden_dim: int = 64,
        surprise_threshold: float = 0.5,
        blend_init: float = 0.7,
    ):
        super().__init__()
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.surprise_threshold = surprise_threshold

        # --- Level 1: Working Memory ---
        self.working_memory = nn.Linear(key_dim, val_dim, bias=False)
        nn.init.zeros_(self.working_memory.weight)
        self.register_buffer("working_momentum", torch.zeros(val_dim, key_dim))

        # --- Level 2: Long-term Memory ---
        self.longterm_memory = nn.Linear(key_dim, val_dim, bias=False)
        nn.init.zeros_(self.longterm_memory.weight)
        self.register_buffer("longterm_momentum", torch.zeros(val_dim, key_dim))

        # --- Гейты для working memory: α1, η1, θ1 ---
        self.working_gates = nn.Linear(key_dim, 3, bias=True)
        nn.init.zeros_(self.working_gates.weight)
        nn.init.constant_(self.working_gates.bias, 0.5)

        # --- Гейты для long-term memory: α2, η2, θ2 ---
        self.longterm_gates = nn.Linear(key_dim, 3, bias=True)
        nn.init.zeros_(self.longterm_gates.weight)
        nn.init.constant_(self.longterm_gates.bias, 0.3)

        # --- Surprise gate: determines long-term update intensity ---
        # вход: [key_dim + 1 (surprise scalar)]
        self.surprise_gate = nn.Sequential(
            nn.Linear(key_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # --- Blend coefficient: learned mix of working vs long-term ---
        self.blend_proj = nn.Linear(key_dim, 1)
        nn.init.zeros_(self.blend_proj.weight)
        nn.init.constant_(self.blend_proj.bias, blend_init)

    def _aggregate_key(self, k: torch.Tensor) -> torch.Tensor:
        """Агрегирует батч ключей в один вектор для гейтов."""
        return k.mean(0) if k.dim() > 1 else k

    def _compute_surprise(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет surprise как среднюю ошибку рабочей памяти.
        surprise = mean(|| working_memory(k) - v ||²)
        """
        with torch.no_grad():
            pred = self.working_memory(k.detach())
            surprise = F.mse_loss(pred, v.detach())
        return surprise

    def _update_level(
        self,
        memory: nn.Linear,
        momentum_buf: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate_proj: nn.Linear,
        scale: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Один TTT шаг для одного уровня памяти.
        Возвращает (alpha, eta, theta) для диагностики.
        """
        # Use float32 for stability in mixed-precision contexts
        k32 = k.detach().float()
        v32 = v.detach().float()
        k_agg = self._aggregate_key(k32)
        gates = torch.sigmoid(gate_proj(k_agg.to(gate_proj.weight.dtype)))
        alpha, eta, theta = gates[0], gates[1], gates[2]

        if scale is not None:
            theta = theta * scale.clamp(0.0, 1.0)

        # TTT gradient in float32
        mem_weight_fp32 = memory.weight.float()
        pred = k32 @ mem_weight_fp32.T
        loss_ttt = F.mse_loss(pred, v32)
        if torch.isnan(loss_ttt) or torch.isinf(loss_ttt):
            return alpha.detach(), eta.detach(), theta.detach()
        grad = torch.autograd.grad(
            loss_ttt,
            mem_weight_fp32,
            retain_graph=False,
            create_graph=False,
        )[0]

        # Clamp gradient to prevent explosive TTT update
        grad_clamped = torch.clamp(grad.detach(), min=-0.5, max=0.5)

        # Momentum update in float32
        mom_fp32 = momentum_buf.float()
        alpha_f = alpha.float()
        eta_f = eta.float()
        theta_f = theta.float()
        new_momentum = eta_f * mom_fp32 - 0.01 * theta_f * grad_clamped
        # Clamp momentum norm
        mom_norm = new_momentum.norm()
        if mom_norm > 5.0:
            new_momentum = new_momentum * (5.0 / (mom_norm + 1e-8))
        momentum_buf.copy_(new_momentum.to(momentum_buf.dtype))

        # Memory update with max-norm constraint
        new_weight = (1 - alpha_f) * mem_weight_fp32 + new_momentum
        weight_norm = new_weight.norm()
        if weight_norm > 10.0:
            new_weight = new_weight * (10.0 / (weight_norm + 1e-8))
        memory.weight.data.copy_(new_weight.to(memory.weight.dtype))
        return alpha.detach(), eta.detach(), theta.detach()


    def retrieve_and_update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        update_memory: bool = True,
    ):
        """Unified retrieve-and-optionally-update interface compatible with
        TitansMemoryModule.  Returns a MemoryState so that HDIMPipeline can
        treat both memory implementations uniformly.
        """
        from src.core.titans_memory import MemoryState  # local import to avoid circularity

        h_state = self.retrieve(keys, values)
        if update_memory and self.training:
            self.update(keys, values)
        # MemoryState fields: retrieved, loss, updated — same contract as TitansMemoryModule
        return MemoryState(
            retrieved=h_state.retrieved,
            loss=h_state.loss,
            updated=update_memory and self.training,
        )

    def retrieve(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> HierarchicalMemoryState:
        """Только retrieval, без обновления памяти."""
        working_pred = self.working_memory(k)
        longterm_pred = self.longterm_memory(k)

        # Blend coefficient из ключа
        k_agg = self._aggregate_key(k.detach())
        blend_alpha = torch.sigmoid(self.blend_proj(k_agg)).squeeze()  # scalar

        # Взвешенное объединение
        retrieved = blend_alpha * working_pred + (1.0 - blend_alpha) * longterm_pred

        loss = F.mse_loss(retrieved, v.detach())
        surprise = self._compute_surprise(k, v)

        # Фиктивный gate для retrieve mode
        longterm_gate = torch.zeros(1, device=k.device, dtype=k.dtype)

        return HierarchicalMemoryState(
            retrieved=retrieved,
            loss=loss,
            updated=False,
            working_surprise=surprise,
            longterm_gate=longterm_gate,
            blend_alpha=blend_alpha.detach(),
        )

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TTT обновление обоих уровней памяти.
        Returns: (surprise, longterm_gate)
        """
        surprise = self._compute_surprise(k, v)

        # --- Обновляем working memory (всегда) ---
        self._update_level(
            self.working_memory,
            self.working_momentum,
            k, v,
            self.working_gates,
        )

        # --- Вычисляем gate для long-term update ---
        k_agg = self._aggregate_key(k.detach())
        surprise_scalar = surprise.unsqueeze(0)  # (1,)
        gate_input = torch.cat([k_agg.reshape(1, -1), surprise_scalar.reshape(1, 1)], dim=-1).squeeze(0)
        longterm_gate = self.surprise_gate(gate_input.unsqueeze(0)).squeeze()

        # --- Обновляем long-term memory с масштабированием на gate ---
        self._update_level(
            self.longterm_memory,
            self.longterm_momentum,
            k, v,
            self.longterm_gates,
            scale=longterm_gate,
        )

        return surprise, longterm_gate

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        update_memory: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieval + optional update.

        Returns:
            (retrieved, loss): извлечённое значение и потеря памяти
        """
        # Retrieve
        working_pred = self.working_memory(k)
        longterm_pred = self.longterm_memory(k)
        k_agg = self._aggregate_key(k.detach())
        blend_alpha = torch.sigmoid(self.blend_proj(k_agg)).squeeze()
        retrieved = blend_alpha * working_pred + (1.0 - blend_alpha) * longterm_pred
        loss = F.mse_loss(retrieved, v.detach())

        # Update
        if update_memory and self.training:
            self.update(k, v)

        return retrieved, loss

    def reset_memory(self):
        """Сбрасывает всю память и momentum к нулям."""
        with torch.no_grad():
            self.working_memory.weight.zero_()
            self.working_momentum.zero_()
            self.longterm_memory.weight.zero_()
            self.longterm_momentum.zero_()
