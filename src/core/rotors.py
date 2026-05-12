"""
HDIM — focused domain rotor operators.
"""

import torch
import torch.nn as nn

from .algebra import CliffordAlgebra


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
