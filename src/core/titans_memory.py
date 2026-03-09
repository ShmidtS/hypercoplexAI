"""
HDIM — Titans Memory Module
Нейронная долгосрочная память с обновлением через тест-тайм обучение (TTT).

Математика (из архитектуры Titans):
  L_memory = || M_{t-1}(k_t) - v_t ||²
  S_t = η_t * S_{t-1} - θ_t * ∇L_memory   (momentum gradient step)
  M_t = (1 - α_t) * M_{t-1} + S_t          (memory update)
  α_t, η_t, θ_t — обучаемые скаляры (sigmoid-гейты из входа)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


@dataclass
class MemoryState:
    retrieved: torch.Tensor
    loss: torch.Tensor
    updated: bool
    alpha: torch.Tensor | None = None
    eta: torch.Tensor | None = None
    theta: torch.Tensor | None = None


class TitansMemoryModule(nn.Module):
    """
    Нейронная ассоциативная память с Test-Time Training (TTT) обновлением.

    Память M хранится как линейный слой без bias: M: R^{key_dim} → R^{val_dim}
    Обновляется онлайн через градиентный шаг по L_memory при каждом forward.

    Args:
        key_dim: размерность ключей
        val_dim: размерность значений
        hidden_dim: размерность проекции для вычисления гейтов α, η, θ
    """

    def __init__(self, key_dim: int, val_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.key_dim = key_dim
        self.val_dim = val_dim

        # Память M реализована как линейный слой (матрица весов)
        self.memory = nn.Linear(key_dim, val_dim, bias=False)
        nn.init.zeros_(self.memory.weight)  # инициализация нулями

        # Momentum state S (не параметр, а буфер)
        self.register_buffer('momentum_S', torch.zeros(val_dim, key_dim))

        # Гейты α (forget), η (momentum), θ (lr) — проекции из входа
        self.gate_proj = nn.Linear(key_dim, 3, bias=True)  # 3 скаляра
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 0.5)  # начальные значения ~0.5

    def retrieve(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> MemoryState:
        retrieved = self.memory(k)
        loss_memory = F.mse_loss(retrieved, v.detach())
        return MemoryState(retrieved=retrieved, loss=loss_memory, updated=False)

    # Максимальная норма весов памяти — предотвращает TTT взрыв
    _MEMORY_MAX_NORM: float = 10.0
    # Масштаб шага TTT — уменьшен для стабильности
    _TTT_LR_SCALE: float = 0.01

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_ttt = F.mse_loss(self.memory(k.detach()), v.detach())
        gates = torch.sigmoid(self.gate_proj(k.detach().mean(0) if k.dim() > 1 else k.detach()))
        alpha, eta, theta = gates[..., 0], gates[..., 1], gates[..., 2]
        grad = torch.autograd.grad(
            loss_ttt,
            self.memory.weight,
            retain_graph=False,
            create_graph=False,
        )[0]
        # Clamp gradient to prevent explosive TTT update
        grad_clamped = torch.clamp(grad.detach(), min=-1.0, max=1.0)
        self.momentum_S = eta * self.momentum_S.detach() - self._TTT_LR_SCALE * theta * grad_clamped
        new_weight = (1 - alpha) * self.memory.weight.data + self.momentum_S.data
        # Max-norm constraint: prevent memory weight from exploding
        weight_norm = new_weight.norm()
        if weight_norm > self._MEMORY_MAX_NORM:
            new_weight = new_weight * (self._MEMORY_MAX_NORM / (weight_norm + 1e-8))
        self.memory.weight.data.copy_(new_weight)
        return alpha.detach(), eta.detach(), theta.detach()

    def retrieve_and_update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        update_memory: bool = True,
    ) -> MemoryState:
        state = self.retrieve(k, v)
        if update_memory and self.training:
            alpha, eta, theta = self.update(k, v)
            state.updated = True
            state.alpha = alpha
            state.eta = eta
            state.theta = theta
        return state

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        update_memory: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Выполняет поиск в памяти и опционально обновляет память.

        Args:
            k: ключ (..., key_dim)
            v: целевое значение (..., val_dim)
            update_memory: если True — обновить память через TTT шаг

        Returns:
            (retrieved, loss): извлечённое значение и потеря памяти
        """
        state = self.retrieve_and_update(k, v, update_memory=update_memory)
        return state.retrieved, state.loss

    def reset_memory(self):
        """Сбрасывает память и momentum state к нулям."""
        with torch.no_grad():
            self.memory.weight.zero_()
            self.momentum_S.zero_()
