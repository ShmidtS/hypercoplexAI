"""
HDIM — Modular MoE Router с динамическим управлением экспертами.
Phase 7: можно добавлять/удалять любое количество экспертов в runtime.

Ключевые свойства:
- add_expert(config) — добавить нового эксперта без перестройки модели
- remove_expert(id) — удалить эксперта, переиндексировать
- routing_type='soft' — SoftMoE (дифференцируемый, без token dropping)
- routing_type='hard' — Hard top-k с R3 EMA stabilization
- Полностью совместим с R3MoERouter/SoftMoERouter API (одинаковые router_state ключи)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Expert configuration
# ---------------------------------------------------------------------------

@dataclass
class ExpertConfig:
    """Конфигурация одного эксперта."""
    hidden_dim: int = 256
    dropout: float = 0.0
    activation: str = 'gelu'  # 'gelu' | 'relu' | 'silu'
    layer_count: int = 2      # 1, 2, или 3 слоя
    use_residual: bool = False  # добавить residual connection если input_dim == hidden_dim


# ---------------------------------------------------------------------------
# Single expert module
# ---------------------------------------------------------------------------

_ACTIVATIONS = {
    'gelu': nn.GELU,
    'relu': nn.ReLU,
    'silu': nn.SiLU,
}


class ExpertModule(nn.Module):
    """Один эксперт с конфигурируемой глубиной и активацией."""

    def __init__(self, input_dim: int, config: ExpertConfig) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        act_cls = _ACTIVATIONS.get(config.activation, nn.GELU)

        layers: List[nn.Module] = []
        in_dim = input_dim

        for i in range(config.layer_count):
            out_dim = config.hidden_dim if i < config.layer_count - 1 else input_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < config.layer_count - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(act_cls())
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))
            in_dim = out_dim

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Малая инициализация для стабильного старта при добавлении экспертов."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.config.use_residual and out.shape == x.shape:
            return x + out
        return out


# ---------------------------------------------------------------------------
# Modular MoE Router
# ---------------------------------------------------------------------------


class ModularMoERouter(nn.Module):
    """
    Модульный Mixture-of-Experts роутер с динамическим управлением экспертами.

    Поддерживает два режима маршрутизации:
    - 'soft': SoftMoE (дифференцируемый dispatch/combine, без token dropping)
    - 'hard': Hard top-k с R3 EMA stabilization

    Args:
        input_dim: размерность входа/выхода
        experts: список ExpertConfig или int (количество одинаковых экспертов)
        top_k: число активных экспертов (только для hard routing)
        routing_type: 'soft' или 'hard'
        temperature: температура softmax для soft routing
    """

    def __init__(
        self,
        input_dim: int,
        experts: List[ExpertConfig] | int = 4,
        top_k: int = 2,
        routing_type: str = 'soft',
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.top_k_value = top_k
        self.routing_type = routing_type
        self.temperature = temperature

        # Список конфигов для воспроизводимости при add_expert
        self._expert_configs: List[ExpertConfig] = []

        # Реестр экспертов как ModuleList (динамически перестраивается)
        self.experts = nn.ModuleList()

        # Роутер (линейный слой) — перестраивается при add/remove
        self._router: Optional[nn.Linear] = None

        # EMA train scores для R3 compatibility
        # Регистрируем как non-persistent buffer — пересоздаём при add/remove
        self._ema_scores: torch.Tensor = torch.empty(0)

        # Инициализация начальными экспертами
        if isinstance(experts, int):
            default_cfg = ExpertConfig(hidden_dim=input_dim * 2)
            init_configs = [default_cfg] * experts
        else:
            init_configs = list(experts)

        for cfg in init_configs:
            self._add_expert_internal(cfg)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    @property
    def top_k(self) -> int:
        return min(self.top_k_value, self.num_experts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_expert_internal(self, config: ExpertConfig) -> None:
        """Добавляет эксперта без обновления роутера."""
        expert = ExpertModule(self.input_dim, config)
        self.experts.append(expert)
        self._expert_configs.append(config)

    def _rebuild_router(self) -> None:
        """Перестраивает роутер под текущее кол-во экспертов."""
        n = self.num_experts
        if n == 0:
            self._router = None
            self._ema_scores = torch.empty(0)
            return

        device = next(self.parameters(), torch.tensor(0.0)).device
        dtype = next(self.parameters(), torch.tensor(0.0)).dtype

        old_router = self._router
        new_router = nn.Linear(self.input_dim, n, bias=False)
        nn.init.normal_(new_router.weight, std=0.02)

        # Preserve weights for existing experts if possible
        if old_router is not None:
            old_n = old_router.out_features
            copy_n = min(old_n, n)
            with torch.no_grad():
                new_router.weight.data[:copy_n] = old_router.weight.data[:copy_n]

        self._router = new_router.to(device=device, dtype=dtype)

        # Rebuild EMA scores buffer
        self._ema_scores = torch.ones(n, device=device, dtype=dtype) / n

    # ------------------------------------------------------------------
    # Public API: add / remove experts
    # ------------------------------------------------------------------

    def add_expert(self, config: Optional[ExpertConfig] = None) -> int:
        """Добавляет нового эксперта. Возвращает его индекс."""
        if config is None:
            config = ExpertConfig(hidden_dim=self.input_dim * 2)
        self._add_expert_internal(config)
        self._rebuild_router()
        return self.num_experts - 1

    def remove_expert(self, expert_id: int) -> None:
        """Удаляет эксперта по индексу и перестраивает роутер."""
        if expert_id < 0 or expert_id >= self.num_experts:
            raise IndexError(f"expert_id {expert_id} out of range [0, {self.num_experts})")

        # Перестраиваем ModuleList без удалённого эксперта
        remaining = [e for i, e in enumerate(self.experts) if i != expert_id]
        self.experts = nn.ModuleList(remaining)
        self._expert_configs.pop(expert_id)
        self._rebuild_router()

    # ------------------------------------------------------------------
    # Routing implementations
    # ------------------------------------------------------------------

    def _soft_routing(
        self, x_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """SoftMoE: дифференцируемый dispatch/combine."""
        T, D = x_flat.shape
        E = self.num_experts

        logits = self._router(x_flat) / self.temperature  # (T, E)
        dispatch = F.softmax(logits, dim=0)   # (T, E): каждый эксперт получает mix токенов
        combine  = F.softmax(logits, dim=-1)  # (T, E): каждый токен получает mix экспертов

        # Slot inputs: X_e = dispatch[:,e]^T @ X  →  (E, D)
        slot_inputs = dispatch.T @ x_flat  # (E, D)

        # Apply experts
        slot_outputs = torch.stack(
            [self.experts[e](slot_inputs[e].unsqueeze(0)).squeeze(0) for e in range(E)],
            dim=0,
        )  # (E, D)

        # Combine
        output = combine @ slot_outputs  # (T, D)

        # Update EMA
        if self.training:
            with torch.no_grad():
                usage = combine.mean(0)  # (E,)
                self._ema_scores = self._ema_scores * 0.9 + usage.detach() * 0.1

        # Load balance loss
        mean_usage = combine.mean(0)  # (E,)
        p = F.softmax(self._router.weight.mean(-1), dim=0)  # (E,)
        router_loss = E * (mean_usage * p).sum()

        # Build top-k indices for API compatibility
        topk_w, topk_idx = combine.topk(self.top_k, dim=-1)  # (T, top_k)
        topk_w_norm = topk_w / topk_w.sum(-1, keepdim=True).clamp_min(1e-8)
        entropy = -(combine * (combine + 1e-8).log()).sum(-1).mean()

        state: Dict[str, Any] = {
            'loss': router_loss,
            'router_loss': router_loss,
            'scores': logits,
            'topk_idx': topk_idx,
            'gate_weights': combine,
            'train_scores_snapshot': self._ema_scores.detach().clone(),
            'topk_gate_weights': topk_w_norm,
            'expert_usage': combine.mean(0).detach(),
            'routing_entropy': entropy,
        }
        return output, state

    def _hard_routing(
        self, x_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Hard top-k routing с R3 EMA stabilization."""
        T, D = x_flat.shape
        E = self.num_experts
        K = self.top_k

        scores = self._router(x_flat)  # (T, E)
        _, topk_idx = torch.topk(scores, K, dim=-1)  # (T, K)
        mask = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)  # (T, E)

        # R3: update EMA with inference scores
        if self.training:
            with torch.no_grad():
                mean_scores = scores.detach().mean(0)
                self._ema_scores = self._ema_scores * 0.9 + mean_scores * 0.1

        snap = self._ema_scores.detach().clone()
        train_weights = F.softmax(snap, dim=-1)  # (E,)
        gate_weights = mask * train_weights.unsqueeze(0)  # (T, E)
        gate_sum = gate_weights.sum(-1, keepdim=True).clamp_min(1e-8)
        gate_weights = gate_weights / gate_sum

        output = torch.zeros_like(x_flat)
        for e in range(E):
            output += gate_weights[:, e:e+1] * self.experts[e](x_flat)

        # Load balance loss
        f = mask.mean(0)  # (E,)
        p = F.softmax(scores, dim=-1).mean(0)  # (E,)
        router_loss = E * (f * p).sum()

        topk_gate_w = torch.gather(gate_weights, -1, topk_idx)  # (T, K)
        entropy = -(gate_weights * (gate_weights + 1e-8).log()).sum(-1).mean()

        state: Dict[str, Any] = {
            'loss': router_loss,
            'router_loss': router_loss,
            'scores': scores,
            'topk_idx': topk_idx,
            'gate_weights': gate_weights,
            'train_scores_snapshot': snap,
            'topk_gate_weights': topk_gate_w,
            'expert_usage': mask.float().mean(0).detach(),
            'routing_entropy': entropy,
        }
        return output, state

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x: (..., input_dim)
        Returns:
            (output, router_state)
        """
        if self.num_experts == 0:
            raise RuntimeError('ModularMoERouter has no experts. Add at least one expert.')
        if self._router is None:
            self._rebuild_router()

        # Sync EMA buffer device
        device = x.device
        if self._ema_scores.device != device:
            self._ema_scores = self._ema_scores.to(device=device, dtype=x.dtype)

        orig_shape = x.shape
        x_flat = x.reshape(-1, self.input_dim)

        if self.routing_type == 'soft':
            output_flat, state = self._soft_routing(x_flat)
        else:
            output_flat, state = self._hard_routing(x_flat)

        # Reshape state tensors back to original batch shape
        batch_dims = orig_shape[:-1]
        E = self.num_experts
        K = self.top_k
        state['scores']         = state['scores'].reshape(*batch_dims, E)
        state['topk_idx']       = state['topk_idx'].reshape(*batch_dims, K)
        state['gate_weights']   = state['gate_weights'].reshape(*batch_dims, E)
        state['topk_gate_weights'] = state['topk_gate_weights'].reshape(*batch_dims, K)

        return output_flat.reshape(orig_shape), state


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_modular_moe(
    input_dim: int,
    num_experts: int = 4,
    top_k: int = 2,
    routing_type: str = 'soft',
    expert_hidden_dim: Optional[int] = None,
    expert_dropout: float = 0.0,
) -> ModularMoERouter:
    """Быстрая сборка ModularMoERouter с однородными экспертами."""
    hidden = expert_hidden_dim or input_dim * 2
    configs = [
        ExpertConfig(hidden_dim=hidden, dropout=expert_dropout)
        for _ in range(num_experts)
    ]
    return ModularMoERouter(
        input_dim=input_dim,
        experts=configs,
        top_k=top_k,
        routing_type=routing_type,
    )
