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
        self.R = nn.Parameter(torch.zeros(algebra.dim))
        if init_identity:
            self.R.data[0] = 1.0

    def _normalized_R(self) -> torch.Tensor:
        """Нормализация R → R/||R|| (epsilon только для защиты от ||R||=0).
        Результат: ||_normalized_R()|| ≈ 1."""
        norm = self.algebra.norm(self.R)
        return self.R / (norm + 1e-4)

    def get_inverse(self) -> torch.Tensor:
        """Обратный ротор: R⁻¹ = ~R / <R*~R>_0.
        Использует скалярную часть геометрического произведения R*~R
        напрямую (с сохранением знака), а не norm()², которая теряет
        знак для timelike-роторов в Cl(p,q) с q>0."""
        eps = 1e-8
        R_n = self._normalized_R()
        R_rev = self.algebra.reverse(R_n)
        quad_form = self.algebra.geometric_product(R_n, R_rev)[..., 0]
        return R_rev / (quad_form + eps)

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
