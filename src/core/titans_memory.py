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
    _MEMORY_MAX_NORM: float = 5.0
    # Масштаб шага TTT — снижен для стабильности (Phase 19)
    _TTT_LR_SCALE: float = 0.005

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # C4+A7 FIX: explicit fp32 TTT path — avoids AMP fp16 overflow
        # Use detached fp32 leaf tensor so grad does NOT flow into main graph
        k32 = k.detach().float()
        v32 = v.detach().float()
        # Create fp32 leaf copy of memory weights (detached from main param)
        mem_w = self.memory.weight.detach().float().requires_grad_(True)
        k_agg = k32.mean(0) if k32.dim() > 1 else k32
        gates = torch.sigmoid(self.gate_proj(k_agg.to(self.gate_proj.weight.dtype)))
        alpha = gates[..., 0].float()
        eta   = gates[..., 1].float()
        theta = gates[..., 2].float()
        pred = k32 @ mem_w.T
        loss_ttt = F.mse_loss(pred, v32)
        (grad,) = torch.autograd.grad(
            loss_ttt,
            mem_w,
            retain_graph=False,
            create_graph=False,
        )
        # Clamp gradient to prevent explosive TTT update
        grad_clamped = grad.detach().clamp(-1.0, 1.0)
        mom_fp32 = self.momentum_S.detach().float()
        new_momentum = eta * mom_fp32 - self._TTT_LR_SCALE * theta * grad_clamped
        # Clamp momentum norm
        momentum_norm = new_momentum.norm()
        if momentum_norm > self._MEMORY_MAX_NORM:
            new_momentum = new_momentum * (self._MEMORY_MAX_NORM / (momentum_norm + 1e-8))
        self.momentum_S.copy_(new_momentum.to(self.momentum_S.dtype))
        new_weight = (1 - alpha) * mem_w.detach() + new_momentum
        # Max-norm constraint: prevent memory weight from exploding
        weight_norm = new_weight.norm()
        if weight_norm > self._MEMORY_MAX_NORM:
            new_weight = new_weight * (self._MEMORY_MAX_NORM / (weight_norm + 1e-8))
        self.memory.weight.data.copy_(new_weight.to(self.memory.weight.dtype))
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

    def reset_memory(self, strategy: str = 'geometric', decay_window: float = 50.0) -> None:
        """Умный reset памяти — не обнуляет полностью, сохраняет важное.

        Стратегии:
            'hard'      — полный сброс в нули (только при инициализации/epoch=1)
            'geometric' — экспоненциальное затухание весов (сохраняет паттерны)
            'stabilize' — нормировка momentum без изменения весов памяти

        Аргументы:
            strategy:     тип сброса
            decay_window: характерное время затухания для 'geometric' (эпох)
        """
        with torch.no_grad():
            if strategy == 'hard':
                self.memory.weight.zero_()
                self.momentum_S.zero_()
            elif strategy == 'geometric':
                # Экспоненциальное затухание — сохраняет паттерны, убирает шум
                # decay_factor ≈ 0.607 при decay_window=50 (медленное забывание)
                decay = torch.exp(torch.tensor(-1.0 / max(decay_window, 1.0)))
                self.memory.weight.mul_(decay)
                self.momentum_S.mul_(decay * 0.5)  # momentum затухает быстрее
            elif strategy == 'stabilize':
                # Только нормировка momentum — не трогаем веса памяти
                norm = self.momentum_S.norm()
                if norm > self._MEMORY_MAX_NORM:
                    self.momentum_S.mul_(self._MEMORY_MAX_NORM / (norm + 1e-8))

    def stabilize_momentum(self) -> bool:
        """Принудительная нормировка momentum_S. Вызывать при LR restarts.

        Returns True если была применена нормировка.
        """
        with torch.no_grad():
            norm = self.momentum_S.norm()
            if norm > self._MEMORY_MAX_NORM * 0.5:
                self.momentum_S.mul_(self._MEMORY_MAX_NORM * 0.5 / (norm + 1e-8))
                return True
        return False
