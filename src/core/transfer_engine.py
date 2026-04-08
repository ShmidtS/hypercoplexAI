"""TransferEngine — инкапсулирует логику кроссдоменного переноса.

Отвечает за:
- MoE routing инварианта
- Sandwich transfer через доменные роторы
- Декодирование в выходной домен
- Сборку TransferState

Контракт:
    transfer(u_mem, source_domain, target_domain, ...) -> (output, state_dict)

    где:
    - u_mem: memory-augmented инвариант [B, clifford_dim]
    - output: выходной тензор [B, output_dim]
    - state_dict: словарь с метаданными трансфера
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .hypercomplex import CliffordAlgebra
from .domain_operators import DomainRotationOperator, sandwich_transfer
from .transfer_state import TransferState


class TransferEngine(nn.Module):
    """Выполняет кроссдоменный перенос через MoE routing.

    Pipeline:
        u_mem -> moe -> u_route
        u_route -> sandwich_transfer -> g_target
        g_target -> decoder -> output
    """

    def __init__(
        self,
        clifford_dim: int,
        output_dim: int,
        algebra: CliffordAlgebra,
        num_experts: int = 4,
        top_k: int = 2,
        router_cls: Optional[type] = None,
    ):
        """Инициализация TransferEngine.

        Args:
            clifford_dim: размерность мультивектора Клиффорда
            output_dim: размерность выходного вектора
            algebra: алгебра Клиффорда для операций
            num_experts: количество экспертов MoE
            top_k: топ-k маршрутизации
            router_cls: класс роутера MoE (None = SoftMoERouter)
        """
        super().__init__()

        self.algebra = algebra
        self.clifford_dim = clifford_dim
        self.output_dim = output_dim

        # Import here to avoid circular dependency
        from .soft_moe_router import SoftMoERouter
        from .hdim_pipeline import HDIMDecoder

        if router_cls is None:
            router_cls = SoftMoERouter

        # MoE router: soft routing к доменным экспертам
        self.moe = router_cls(
            input_dim=clifford_dim,
            num_experts=num_experts,
            expert_dim=clifford_dim * 2,
            top_k=top_k,
        )

        # Decoder: мультивектор -> выход
        self.decoder = HDIMDecoder(clifford_dim, output_dim)

        # Gradient checkpointing flag
        self._use_gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on MoE forward."""
        self._use_gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self._use_gradient_checkpointing = False

    def transfer(
        self,
        u_mem: torch.Tensor,
        source_rotor: DomainRotationOperator,
        target_rotor: DomainRotationOperator,
        g_source: Optional[torch.Tensor] = None,
        input_is_invariant: bool = False,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Выполняет перенос из source в target домен.

        Args:
            u_mem: memory-augmented инвариант [B, clifford_dim]
            source_rotor: ротор source домена
            target_rotor: ротор target домена
            g_source: исходный мультивектор (опционально)
            input_is_invariant: True если u_mem уже является инвариантом

        Returns:
            output: выходной тензор [B, output_dim]
            router_state: словарь с состоянием MoE router
        """
        # MoE routing with optional gradient checkpointing
        if self._use_gradient_checkpointing and self.training:
            result = checkpoint(
                self.moe, u_mem,
                use_reentrant=False,
            )
            u_route, router_state = result  # type: ignore[misc]
        else:
            u_route, router_state = self.moe(u_mem)

        # Sandwich transfer through domain rotors
        _, g_target = sandwich_transfer(
            self.algebra,
            u_route,
            source_rotor,
            target_rotor,
            invariant_override=u_route,
        )

        # Decode to output
        output = self.decoder(g_target)

        router_state = dict(router_state)
        router_state["u_route"] = u_route
        router_state["g_target"] = g_target

        return output, router_state

    def forward(
        self,
        u_mem: torch.Tensor,
        source_rotor: DomainRotationOperator,
        target_rotor: DomainRotationOperator,
        g_source: Optional[torch.Tensor] = None,
        input_is_invariant: bool = False,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Алиас для transfer()."""
        return self.transfer(
            u_mem, source_rotor, target_rotor, g_source, input_is_invariant
        )

    def decode(self, g_target: torch.Tensor) -> torch.Tensor:
        """Декодирует мультивектор в выход.

        Args:
            g_target: мультивектор [B, clifford_dim]

        Returns:
            output: выходной тензор [B, output_dim]
        """
        return self.decoder(g_target)

    @property
    def num_experts(self) -> int:
        """Количество экспертов MoE."""
        return self.moe.num_experts

    @property
    def top_k(self) -> int:
        """Топ-k маршрутизации."""
        return self.moe.top_k
