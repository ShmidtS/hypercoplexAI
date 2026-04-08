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

    def __init__(
        self,
        key_dim: int,
        val_dim: int,
        hidden_dim: int = 64,
        memory_max_norm: float = 5.0,
        ttt_lr_scale: float = 0.005,
    ):
        super().__init__()
        self.key_dim = key_dim
        self.val_dim = val_dim

        # Параметры конфигурации (ранее hardcoded)
        self.memory_max_norm = memory_max_norm
        self.ttt_lr_scale = ttt_lr_scale

        # Память M реализована как линейный слой (матрица весов)
        self.memory = nn.Linear(key_dim, val_dim, bias=False)
        nn.init.normal_(self.memory.weight, std=0.01) # small random init

        # Momentum state S (не параметр, а буфер)
        self.register_buffer('momentum_S', torch.zeros(val_dim, key_dim))

        # Гейты α (forget), η (momentum), θ (lr) — проекции из входа
        self.gate_proj = nn.Linear(key_dim + val_dim, 3, bias=True) # 3 скаляра
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 0.5) # начальные значения ~0.5

        # Phase 22: gradient-based surprise + adaptive forgetting
        self.use_gradient_surprise: bool = False
        self.use_adaptive_forgetting: bool = False
        self._last_surprise: float = 0.0

        # Phase 29: RAG-compatible freeze state
        self._frozen: bool = False

    # ==================== Phase 29: RAG-compatible API ====================

    def freeze_memory(self) -> None:
        """Freeze memory weights for RAG inference.

        После вызова:
        - retrieve() работает без обновления
        - Memory weights не изменяются
        - Embeddings детерминированы
        """
        self.memory.weight.requires_grad_(False)
        self._frozen = True

    def unfreeze_memory(self) -> None:
        """Unfreeze for training."""
        self.memory.weight.requires_grad_(True)
        self._frozen = False

    def is_frozen(self) -> bool:
        """Check if memory is frozen."""
        return self._frozen

    def retrieve_only(self, k: torch.Tensor) -> torch.Tensor:
        """RAG-compatible retrieval without memory update.

        Equivalent to forward with update_memory=False, but returns
        only the retrieved tensor (no loss) for clean RAG inference.
        No gradients flow through this method by design.

        Args:
            k: key tensor (..., key_dim)

        Returns:
            Retrieved value tensor (..., val_dim)
        """
        with torch.no_grad():
            return self.memory(k)

    # Backward compatibility properties (tests access these)
    @property
    def _MEMORY_MAX_NORM(self) -> float:
        return self.memory_max_norm

    @property
    def _TTT_LR_SCALE(self) -> float:
        return self.ttt_lr_scale

    # =========================================================================

    def retrieve(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> MemoryState:
        retrieved = self.memory(k)
        loss_memory = F.mse_loss(retrieved, v.detach())
        return MemoryState(retrieved=retrieved, loss=loss_memory, updated=False)

    def _compute_surprise(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Gradient norm as surprise metric (Titans, NeurIPS 2025)."""
        k32 = k.detach().float().requires_grad_(True)
        pred = self.memory(k32)
        loss = F.mse_loss(pred, v.detach().float())
        grad = torch.autograd.grad(loss, k32, retain_graph=False)[0]
        return grad.norm(dim=-1).mean()  # scalar surprise

    def _adaptive_alpha(self, surprise: torch.Tensor, base_alpha: torch.Tensor) -> torch.Tensor:
        """High surprise → less forgetting."""
        surprise_norm = torch.sigmoid(surprise - 1.0)  # centered sigmoid
        return base_alpha * (1.0 - 0.5 * surprise_norm)  # 50-100% of base alpha

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
        # TTT inner loop: detach creates a separate leaf tensor for local gradient updates. Gradient still flows through retrieve() path.
        mem_w = self.memory.weight.detach().float().requires_grad_(True)
        k_agg = k32.mean(0) if k32.dim() > 1 else k32
        v_agg = v32.mean(0) if v32.dim() > 1 else v32
        kv_agg = torch.cat([k_agg, v_agg], dim=-1)
        gates = torch.sigmoid(self.gate_proj(kv_agg.to(self.gate_proj.weight.dtype)))
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
        new_momentum = eta * mom_fp32 - self.ttt_lr_scale * theta * grad_clamped
        # Clamp momentum norm
        momentum_norm = new_momentum.norm()
        if momentum_norm > self.memory_max_norm:
            new_momentum = new_momentum * (self.memory_max_norm / (momentum_norm + 1e-8))
        self.momentum_S.copy_(new_momentum.to(self.momentum_S.dtype))
        # Phase 22: gradient-based surprise + adaptive forgetting
        effective_alpha = alpha
        if self.use_gradient_surprise:
            surprise = self._compute_surprise(k32, v32)
            self._last_surprise = surprise.item()
            if self.use_adaptive_forgetting:
                effective_alpha = self._adaptive_alpha(surprise, alpha)
        new_weight = (1 - effective_alpha) * mem_w.detach() + new_momentum
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
        # Phase 29: Force disable update when frozen (RAG mode)
        if self._frozen and update_memory:
            update_memory = False

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

