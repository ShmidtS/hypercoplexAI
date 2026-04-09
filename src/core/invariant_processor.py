"""InvariantProcessor — инкапсулирует обработку инвариантов через память.

Отвечает за:
- Применение memory к инварианту
- Управление режимами памяти (none/retrieve/update)
- Возврат MemoryState с метаданными

Контракт:
    process(u_inv, update_memory, memory_mode) -> (u_mem, MemoryState)

    где:
    - u_inv: входной инвариант [B, clifford_dim]
    - u_mem: memory-augmented инвариант [B, clifford_dim]
    - MemoryState: retrieved, loss, updated, alpha, eta, theta
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .memory_interface import MemoryInterface, MemoryResult


@dataclass
class InvariantMemoryState:
    """Состояние памяти после обработки инварианта."""
    retrieved: torch.Tensor  # retrieved memory component [B, D]
    loss: torch.Tensor       # auxiliary memory loss (scalar)
    updated: bool            # whether memory was updated
    alpha: Optional[torch.Tensor] = None  # memory blend gate (Titans)
    eta: Optional[torch.Tensor] = None    # momentum learning rate (Titans)
    theta: Optional[torch.Tensor] = None  # gradient step size (Titans)
    surprise: Optional[torch.Tensor] = None  # surprise signal (Titans)


class InvariantProcessor(nn.Module):
    """Обрабатывает инварианты через memory system.

    Поддерживаемые режимы памяти:
    - 'none': память не используется, u_mem = u_inv
    - 'retrieve': только чтение из памяти
    - 'update': чтение + запись в память

    Unified path: все memory types проходят через MemoryInterface.
    """

    VALID_MEMORY_MODES = {"none", "retrieve", "update"}

    def __init__(self, memory: MemoryInterface):
        """Инициализация InvariantProcessor.

        Args:
            memory: memory system (TitansAdapter, HBMAMemoryAdapter, etc.)
        """
        super().__init__()
        self.memory = memory

    def process(
        self,
        u_inv: torch.Tensor,
        update_memory: bool = True,
        memory_mode: str = "update",
    ) -> tuple[torch.Tensor, InvariantMemoryState]:
        """Обрабатывает инвариант через memory system.

        Args:
            u_inv: входной инвариант [B, clifford_dim]
            update_memory: флаг разрешения обновления памяти
            memory_mode: режим памяти (none/retrieve/update)

        Returns:
            u_mem: memory-augmented инвариант [B, clifford_dim]
            state: InvariantMemoryState с метаданными

        Raises:
            ValueError: если memory_mode не поддерживается
        """
        if memory_mode not in self.VALID_MEMORY_MODES:
            raise ValueError(
                f"Unsupported memory_mode: {memory_mode}. "
                f"Valid modes: {self.VALID_MEMORY_MODES}"
            )

        # Mode 'none': bypass memory entirely
        if memory_mode == "none":
            empty_state = InvariantMemoryState(
                retrieved=torch.zeros_like(u_inv),
                loss=torch.zeros((), device=u_inv.device, dtype=u_inv.dtype),
                updated=False,
                surprise=torch.zeros(1, device=u_inv.device, dtype=u_inv.dtype),
            )
            return u_inv, empty_state

        # Determine if we should update memory
        do_update = update_memory and memory_mode == "update"

        # Unified path through MemoryInterface
        result: MemoryResult = self.memory(u_inv, update_memory=do_update)

        # Convert MemoryResult to InvariantMemoryState
        state = InvariantMemoryState(
            retrieved=result.output - u_inv,  # actual retrieved component
            loss=result.loss,
            updated=result.updated,
            alpha=result.alpha,
            eta=result.eta,
            theta=result.theta,
            surprise=result.surprise,
        )

        return result.output, state

    def forward(
        self,
        u_inv: torch.Tensor,
        update_memory: bool = True,
        memory_mode: str = "update",
    ) -> tuple[torch.Tensor, InvariantMemoryState]:
        """Алиас для process()."""
        return self.process(u_inv, update_memory, memory_mode)

    def reset_memory(self, strategy: str = "geometric") -> None:
        """Сбрасывает память.

        Args:
            strategy: стратегия сброса (geometric/hard)
        """
        self.memory.reset(strategy=strategy)

    def get_memory_loss(self) -> torch.Tensor:
        """Возвращает текущий auxiliary loss памяти.

        Returns:
            Скалярный тензор с memory loss
        """
        return self.memory.memory_loss()
