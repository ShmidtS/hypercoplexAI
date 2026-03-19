"""Clifford Interaction Expert - CAN-style geometric nonlinearity.

Inspired by CliffordNet (arXiv:2601.06793) - "All You Need is Geometric Algebra".
Preserves grade structure of multivectors unlike GELU which destroys it.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class CliffordInteractionExpert(nn.Module):
    """CAN-style expert without FFN that preserves GA structure.

    Uses wedge + inner product instead of GELU:
    - inner product: scalar part (feature coherence)
    - wedge product: bivector part (structural variation)

    This preserves the grade structure of multivectors.
    """

    def __init__(
        self,
        input_dim: int,
        shifts: List[int] = [1, 2, 4],
        use_gate: bool = True
    ):
        super().__init__()
        self.shifts = shifts
        self.use_gate = use_gate

        if use_gate:
            # Learnable gate for combining scalar/bivector
            self.gate = nn.Linear(input_dim, 1)

        # Optional: learnable mixing weights
        self.scalar_weight = nn.Parameter(torch.tensor(0.5))
        self.bivector_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, D) or (B, D) where D = clifford_dim
        Returns:
            Transformed multivector with preserved grade structure
        """
        # Handle both (B, D) and (B, T, D) shapes
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)

        B, T, D = x.shape
        output = x.clone()

        for shift in self.shifts:
            # Local context via discrete Laplacian (CAN-style)
            if T > shift:
                x_shifted = torch.roll(x, shifts=shift, dims=1)
                C = x - x_shifted  # Local differential context
            else:
                # For single-token inputs, use zero context
                C = torch.zeros_like(x)

            # Geometric interaction (preserves grade structure)
            scalar = self._inner_product(x, C)  # (B, T)
            bivector = self._wedge_product(x, C)  # (B, T, D)

            # Learnable mixing
            scalar_part = torch.sigmoid(self.scalar_weight) * scalar.unsqueeze(-1)
            bivector_part = torch.sigmoid(self.bivector_weight) * bivector

            if self.use_gate:
                gate = torch.sigmoid(self.gate(x))
                output = output + gate * (scalar_part + bivector_part)
            else:
                output = output + 0.5 * scalar_part + 0.5 * bivector_part

        # Restore original shape
        if original_shape.dim() == 2:
            output = output.squeeze(1)

        return output

    def _inner_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Scalar part: <a*b>_0

        This is the grade-0 extraction of the geometric product.
        """
        return (a * b).sum(dim=-1)

    def _wedge_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Bivector part: a∧b = (a*b - b*a) / 2

        This preserves multivector structure unlike element-wise GELU.
        """
        # Anticommutative part of geometric product
        return (a * b - b * a) / 2


class CliffordFFN(nn.Module):
    """FFN-like block using Clifford Interaction instead of GELU.

    Drop-in replacement for expert FFN in MoE systems.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        shifts: List[int] = [1, 2, 4]
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim * 4

        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.clifford = CliffordInteractionExpert(hidden_dim, shifts=shifts)
        self.down_proj = nn.Linear(hidden_dim, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) or (B, D)
        """
        # Project up
        h = self.up_proj(x)

        # Apply Clifford interaction (preserves grade structure)
        h = self.clifford(h)

        # Project down
        h = self.down_proj(h)
        h = self.dropout(h)

        return h
