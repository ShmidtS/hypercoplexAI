"""
HDIM — focused invariant extraction and transfer helpers.
"""


import torch
import torch.nn as nn

from .algebra import CliffordAlgebra
from .rotors import DomainRotationOperator


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
    invariant_override: torch.Tensor | None = None,
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
