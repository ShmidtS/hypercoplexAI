"""
HDIM — Domain Operators.
Операторы доменных вращений, инвариантного извлечения и кроссдоменного переноса.

Математическая основа:
  R — ротор домена (верзор алгебры Клиффорда)
  U_inv = R^{-1} ⊗_Cl G ⊗_Cl R — структурный инвариант
  G_target = R_target ⊗ U_inv ⊗ R_target^{-1} — перенос в целевой домен
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .hypercomplex import CliffordAlgebra


class DomainRotationOperator(nn.Module):
    """Обучаемый ротор домена.

    Математически (согласно теоремам HDIM.lean):
      R — единичный ротор: ||R|| = 1
      R⁻¹ = ~R (для единичного ротора reverse = inverse)
      sandwich(R, x) = R ⊗ x ⊗ R⁻¹

    Численно (для стабильности градиентов):
      epsilon используется только в _normalized_R при делении на ||R||,
      чтобы избежать NaN когда ||R|| ≈ 0 во время обучения.
      get_inverse НЕ добавляет epsilon — она принимает уже нормализованный ротор.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        domain_name: str = "default",
        init_identity: bool = True,
    ):
        super().__init__()
        self.algebra = algebra
        self.domain_name = domain_name
        bivector_indices = [i for i in range(algebra.dim) if i.bit_count() == 2]
        self.register_buffer("bivector_indices", torch.tensor(bivector_indices, dtype=torch.long))
        self.R = nn.Parameter(torch.zeros(len(bivector_indices)))

    def _bivector(self) -> torch.Tensor:
        B = torch.zeros(self.algebra.dim, dtype=self.R.dtype, device=self.R.device)
        B.scatter_(0, self.bivector_indices.to(device=self.R.device), self.R)
        return B

    def _normalized_R(self) -> torch.Tensor:
        """Construct unit rotor R = exp(-B/2) from trainable bivector B."""
        B = -0.5 * self._bivector()
        B_sq = self.algebra.geometric_product(B, B)[..., 0]
        B_norm = torch.sqrt(torch.clamp(B_sq.abs(), min=1e-12))
        sin_over_norm = torch.where(
            B_norm > 1e-6,
            torch.sin(B_norm) / B_norm,
            1.0 - (B_norm * B_norm) / 6.0,
        )
        R = sin_over_norm * B
        R = R.clone()
        R[0] = torch.cos(B_norm)
        R_rev = self.algebra.reverse(R)
        quad_form = self.algebra.geometric_product(R, R_rev)[..., 0]
        norm = torch.sqrt(torch.clamp(quad_form.abs(), min=1e-12))
        return R / norm

    def get_inverse(self) -> torch.Tensor:
        """Обратный ротор для unit rotor: R⁻¹ = ~R."""
        return self.algebra.reverse(self._normalized_R())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # unit=False: epsilon для численной стабильности под AMP (fp16 optimizer
        # может деградировать норму ротора к ~0, unit=True даёт деление на 0 → inf)
        return self.algebra.sandwich(self._normalized_R(), x, unit=False)

    def apply_inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.algebra.sandwich(self.get_inverse(), x, unit=False)


class InvariantExtractor(nn.Module):
    """Извлекает структурный инвариант U_inv = R^{-1} ⊗_Cl G ⊗_Cl R."""

    def __init__(self, algebra: CliffordAlgebra):
        super().__init__()
        self.algebra = algebra

    def forward(
        self,
        G_source: torch.Tensor,
        R: DomainRotationOperator,
    ) -> torch.Tensor:
        R_inv = R.get_inverse()
        R_n = R._normalized_R()
        step1 = self.algebra.geometric_product(R_inv.expand(*G_source.shape), G_source)
        return self.algebra.geometric_product(step1, R_n.expand(*G_source.shape))


def sandwich_transfer(
    algebra: CliffordAlgebra,
    G_source: torch.Tensor,
    R_source: DomainRotationOperator,
    R_target: DomainRotationOperator,
    invariant_override: Optional[torch.Tensor] = None,
):
    """Переносит мультивектор через общий инвариант в целевой домен."""
    if invariant_override is None:
        R_src_inv = R_source.get_inverse()
        R_src_n = R_source._normalized_R()
        step1 = algebra.geometric_product(R_src_inv.expand(*G_source.shape), G_source)
        U_inv = algebra.geometric_product(step1, R_src_n.expand(*G_source.shape))
    else:
        U_inv = invariant_override

    R_tgt_n = R_target._normalized_R()
    step2 = algebra.geometric_product(R_tgt_n.expand(*U_inv.shape), U_inv)
    G_target = algebra.geometric_product(step2, R_target.get_inverse().expand(*U_inv.shape))
    return U_inv, G_target


class DomainIndexEmbedding(nn.Module):
    """Sinusoidal positional embedding over domain indices.

    Each domain (e.g. math=0, language=1, code=2, science=3) gets a
    distinct positional signal via sin/cos at different frequencies,
    analogous to Transformer positional encodings but applied to the
    discrete domain_id rather than sequence position.
    """

    def __init__(self, dim: int, max_domains: int = 4):
        super().__init__()
        self.dim = dim
        self.max_domains = max_domains
        half_dim = dim // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        positions = torch.arange(max_domains, dtype=torch.float32).unsqueeze(1)
        angles = positions * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(max_domains, 1)], dim=1)
        self.register_buffer("embedding", emb)

    def forward(self, domain_id: torch.Tensor) -> torch.Tensor:
        """Look up sinusoidal embedding for each domain_id in the batch.

        Args:
            domain_id: (B,) long tensor of domain indices.

        Returns:
            (B, dim) sinusoidal embedding tensor.
        """
        return self.embedding[domain_id]
