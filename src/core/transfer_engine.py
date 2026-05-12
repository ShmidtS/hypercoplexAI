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


class _CoreTruth:
    """Minimal truth-value placeholder for core transfer metadata."""

    def __init__(self, freq: float = 1.0, conf: float = 1.0):
        self.freq = freq
        self.conf = conf


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
        router: Optional[nn.Module] = None,
        router_cls: Optional[type] = None,
        z_loss_weight: float = 0.0,
        n_shared_experts: int = 0,
    ):
        """Инициализация TransferEngine.

        Args:
            clifford_dim: размерность мультивектора Клиффорда
            output_dim: размерность выходного вектора
            algebra: алгебра Клиффорда для операций
            num_experts: количество экспертов MoE
            top_k: топ-k маршрутизации
            router: optional MoE router module; None uses identity routing
            router_cls: optional router class for explicit legacy MoE construction
            z_loss_weight: weight for router z-loss regularization
            n_shared_experts: number of DeepSeek-V3 style shared experts
        """
        super().__init__()

        self.algebra = algebra
        self.clifford_dim = clifford_dim
        self.output_dim = output_dim

        # Import here to avoid circular dependency
        from .hdim_pipeline import HDIMDecoder

        if router is not None:
            self.router: Optional[nn.Module] = router
        elif router_cls is not None:
            self.router = router_cls(
                input_dim=clifford_dim,
                num_experts=num_experts,
                expert_dim=clifford_dim * 2,
                top_k=top_k,
                z_loss_weight=z_loss_weight,
                n_shared_experts=n_shared_experts,
            )
        else:
            self.router = None
        self.moe = self.router

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

    def _rotor_quality(self, rotor: DomainRotationOperator) -> torch.Tensor:
        R = rotor._normalized_R()
        R_rev = rotor.get_inverse()
        quad = self.algebra.geometric_product(R, R_rev)[..., 0]
        return (1.0 / (1.0 + (quad.abs() - 1.0).abs())).clamp(0.0, 1.0)

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
        # Optional MoE routing with identity fallback for core mode
        if self.router is None:
            u_route = u_mem
            router_state = {}
        elif self._use_gradient_checkpointing and self.training:
            result = checkpoint(
                self.router, u_mem,
                use_reentrant=False,
            )
            u_route, router_state = result  # type: ignore[misc]
        else:
            u_route, router_state = self.router(u_mem)

        if g_source is not None:
            R_src_inv = source_rotor.get_inverse()
            R_src_n = source_rotor._normalized_R()
            step = self.algebra.geometric_product(R_src_inv.expand(*g_source.shape), g_source)
            U_inv = self.algebra.geometric_product(step, R_src_n.expand(*g_source.shape))
        elif input_is_invariant:
            U_inv = u_mem
        else:
            U_inv, _ = sandwich_transfer(
                self.algebra,
                u_mem,
                source_rotor,
                target_rotor,
            )

        R_tgt_n = target_rotor._normalized_R()
        R_tgt_inv = target_rotor.get_inverse()
        step = self.algebra.geometric_product(R_tgt_n.expand(*U_inv.shape), U_inv)
        g_target = self.algebra.geometric_product(step, R_tgt_inv.expand(*U_inv.shape))

        # MoE adaptation: residual applied separately after transfer
        moe_delta = u_route - u_mem
        step = self.algebra.geometric_product(R_tgt_n.expand(*moe_delta.shape), moe_delta)
        moe_residual = self.algebra.geometric_product(step, R_tgt_inv.expand(*moe_delta.shape))
        g_target = g_target + moe_residual

        # Compute alignment confidence from Clifford rotor quality
        if hasattr(source_rotor, '_normalized_R') and hasattr(target_rotor, '_normalized_R'):
            alignment = (self._rotor_quality(source_rotor) * self._rotor_quality(target_rotor)).clamp(0.0, 1.0)
        else:
            alignment = torch.tensor(1.0, device=u_mem.device)

        # Decode to output
        output = self.decoder(g_target)

        router_state = dict(router_state)
        router_state["u_route"] = u_route
        router_state["u_invariant"] = U_inv
        router_state["g_target"] = g_target
        router_state["alignment"] = alignment

        return output, router_state

    def reverse_transfer(
        self,
        u_target: torch.Tensor,
        source_rotor: DomainRotationOperator,
        target_rotor: DomainRotationOperator,
    ) -> tuple[torch.Tensor, "_CoreTruth"]:
        """Abductive inference: project from target back to source domain.

        NARS backward inference: from conclusion to premises.
        Uses inverse sandwich: sandwich(target_rotor^{-1}, u_target, unit=True)
        then project through source_rotor^{-1}.
        """
        inv_rotor = target_rotor.get_inverse()
        U = self.algebra.sandwich(inv_rotor, u_target, unit=True)
        R_src_n = source_rotor._normalized_R()
        R_src_inv = source_rotor.get_inverse()
        step = self.algebra.geometric_product(R_src_n.expand(*U.shape), U)
        g_source = self.algebra.geometric_product(step, R_src_inv.expand(*U.shape))

        alignment = (self._rotor_quality(source_rotor) * self._rotor_quality(target_rotor)).clamp(0.0, 1.0)

        conf_val = alignment.item() if isinstance(alignment, torch.Tensor) else alignment
        return g_source, _CoreTruth(freq=1.0, conf=conf_val)

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
        return getattr(self.router, "num_experts", 0)

    @property
    def top_k(self) -> int:
        """Топ-k маршрутизации."""
        return getattr(self.router, "top_k", 0)
