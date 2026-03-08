"""
HDIM — Titans Memory Module
Нейронная долгосрочная память с обновлением через тест-тайм обучение (TTT).

Математика (из архитектуры Titans):
  L_memory = || M_{t-1}(k_t) - v_t ||²
  S_t = η_t * S_{t-1} - θ_t * ∇L_memory   (momentum gradient step)
  M_t = (1 - α_t) * M_{t-1} + S_t          (memory update)
  α_t, η_t, θ_t — обучаемые скаляры (sigmoid-гейты из входа)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


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

    def forward(
        self,
        k: torch.Tensor,           # (..., key_dim) — ключ
        v: torch.Tensor,           # (..., val_dim) — значение (цель)
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
        # Поиск: retrieved = M(k)
        retrieved = self.memory(k)  # (..., val_dim)

        # Потеря памяти: || M(k) - v ||²
        loss_memory = F.mse_loss(retrieved, v.detach())

        if update_memory and self.training:
            # TTT update на detached входах — изолируем от внешнего графа
            loss_ttt = F.mse_loss(self.memory(k.detach()), v.detach())

            # Вычисляем гейты из ключа (усредняем по batch)
            gates = torch.sigmoid(self.gate_proj(k.detach().mean(0) if k.dim() > 1 else k.detach()))
            alpha, eta, theta = gates[..., 0], gates[..., 1], gates[..., 2]

            # Градиент по весам памяти (изолирован от внешнего графа)
            grad = torch.autograd.grad(
                loss_ttt,
                self.memory.weight,
                retain_graph=False,
                create_graph=False,
            )[0]  # (val_dim, key_dim)

            # Обновление momentum: S_t = η * S_{t-1} - θ * ∇L
            self.momentum_S = eta * self.momentum_S.detach() - theta * grad.detach()

            # Обновление памяти через copy_ (не inplace на граф)
            # .data.copy_ не инкрементирует версию тензора в autograd,
            # поэтому не нарушает граф внешнего backward-прохода
            self.memory.weight.data.copy_(
                (1 - alpha) * self.memory.weight.data + self.momentum_S.data
            )

        return retrieved, loss_memory

    def reset_memory(self):
        """Сбрасывает память и momentum state к нулям."""
        with torch.no_grad():
            self.memory.weight.zero_()
            self.momentum_S.zero_()
