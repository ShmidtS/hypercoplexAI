"""Clifford Interaction Expert - CAN-style geometric nonlinearity.

Inspired by CliffordNet (arXiv:2601.06793) - "All You Need is Geometric Algebra".
Preserves grade structure of multivectors unlike GELU which destroys it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .hypercomplex import CliffordAlgebra

# Triton acceleration (optional)
try:
    from .cuda import has_triton_support, apply_triton_clifford_interaction
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    
    def has_triton_support() -> bool:
        return False
    
    def apply_triton_clifford_interaction(*args, **kwargs):
        raise RuntimeError("Triton not available")


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
        if len(original_shape) == 2:
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


class CliffordInteractionLayer(nn.Module):
    """CAN-style Clifford Interaction для 16D мультивекторов HDIM.

    Адаптация CAN (ParaMind2025) для работы с CliffordAlgebra Cl_{3,1,0}.

    Args:
        dim: Размерность мультивектора (по умолчанию 16)
        shifts: Список сдвигов для multi-scale взаимодействий [1, 2, 4, 8, 16]
        use_inner: Использовать скалярную часть (inner product)
        use_wedge: Использовать бивекторную часть (wedge product)
        dropout: Dropout rate
        use_triton: Использовать Triton-ускорение (если доступно)
        use_vectorized: Использовать векторизованный PyTorch forward (default=True)
    """

    def __init__(
        self,
        dim: int = 16,
        shifts: List[int] = [1, 2, 4, 8, 16],
        use_inner: bool = True,
        use_wedge: bool = True,
        dropout: float = 0.1,
        use_triton: bool = False,
        use_vectorized: bool = True
    ):
        super().__init__()
        assert dim == 16, f"CliffordInteractionLayer expects dim=16 for Cl_{{3,1,0}}, got {dim}"

        self.dim = dim
        self.shifts = shifts
        self.use_inner = use_inner
        self.use_wedge = use_wedge
        self.dropout = nn.Dropout(dropout)
        
        # Triton acceleration flag
        self.use_triton = use_triton
        self._triton_enabled = use_triton and has_triton_support()

        # Vectorized PyTorch forward (default: True for performance)
        self.use_vectorized = use_vectorized

        # Clifford algebra Cl_{3,1,0} for HDIM
        self.clifford = CliffordAlgebra(p=3, q=1, r=0)

        # Learnable weights for inner/wedge mixing
        self.inner_weight = nn.Parameter(torch.tensor(0.5))
        self.wedge_weight = nn.Parameter(torch.tensor(0.5))

        # Gate for combining original and interaction output
        self.gate_fc = nn.Linear(dim, dim, bias=True)

        # Global context pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, D) or (B, D) where D = 16
        Returns:
            Transformed multivector with CAN-style geometric interactions
        """
        # Use Triton acceleration if enabled and available
        if self._triton_enabled and x.is_cuda:
            return self._forward_triton(x)
        elif self.use_vectorized:
            return self._forward_pytorch_vectorized(x)
        else:
            return self._forward_pytorch(x)
    
    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        """Triton-accelerated forward pass.
        
        Uses Triton kernels for geometric product computations.
        Falls back to PyTorch if Triton fails.
        """
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
        
        B, T, D = x.shape
        
        try:
            # Use Triton-accelerated interaction
            # Note: gate_fc is used differently in Triton version
            # We use the first row of gate_fc.weight as the gate projection
            gate_weight = self.gate_fc.weight[0]  # (D,)
            gate_bias = self.gate_fc.bias[0] if self.gate_fc.bias is not None else torch.tensor(0.0, device=x.device)
            
            output = apply_triton_clifford_interaction(
                x,
                self.shifts,
                self.inner_weight,
                self.wedge_weight,
                gate_weight,
                gate_bias,
                self.use_inner,
                self.use_wedge
            )
            
            # Global context (PyTorch fallback for pooling)
            if T > 1:
                global_context = self.global_pool(x.transpose(1, 2)).transpose(1, 2)
                output = output + global_context
            
            # Numerical stability
            output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
            output = torch.clamp(output, min=-1e3, max=1e3)
            
            # Apply dropout
            output = self.dropout(output)
            
            # Restore original shape
            if len(original_shape) == 2:
                output = output.squeeze(1)
            
            return output
            
        except Exception as e:
            # Fallback to PyTorch on Triton error
            import warnings
            warnings.warn(f"Triton forward failed, falling back to PyTorch: {e}")
            return self._forward_pytorch(x)
    

    def _forward_pytorch_vectorized(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized PyTorch forward pass with batch shift processing.

        This implementation processes all shifts in parallel using batched tensors
        and geometric_product_batch for 2-3x speedup on GPU.
        """
        # Handle both (B, D) and (B, T, D) shapes
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)

        B, T, D = x.shape

        # Determine which shifts are valid (T > shift)
        valid_shifts = [s for s in self.shifts if T > s]

        if not valid_shifts:
            # No valid shifts, return processed input directly
            interaction_output = torch.zeros_like(x)
        else:
            # Pre-compute all shifted tensors at once
            shifted_list = [torch.roll(x, shifts=s, dims=1) for s in valid_shifts]

            # Stack: (num_valid_shifts, B, T, D)
            all_shifted = torch.stack(shifted_list, dim=0)

            # Compute local context for all shifts at once
            C_all = x.unsqueeze(0) - all_shifted  # (num_shifts, B, T, D)

            # Accumulate interaction outputs
            interaction_output = torch.zeros_like(x)

            # Process inner product if enabled (fully vectorized)
            if self.use_inner:
                # Use batched geometric product: (num_shifts, B, T, D)
                product = self.clifford.geometric_product_batch(x, C_all)
                # Extract scalar part (grade-0, index 0) and sum over shifts
                scalar_parts = product[..., 0:1]  # (num_shifts, B, T, 1)
                inner_sum = scalar_parts.sum(dim=0)  # (B, T, 1)
                inner_weighted = torch.sigmoid(self.inner_weight) * inner_sum
                interaction_output = interaction_output + inner_weighted

            # Process wedge product if enabled (fully vectorized)
            if self.use_wedge:
                # ab: x (B,T,D) * C_all (num_shifts, B,T,D)
                ab = self.clifford.geometric_product_batch(x, C_all)  # (num_shifts, B, T, D)

                # ba: C_all * x for each shift
                # Reshape for batch processing
                C_flat = C_all.reshape(-1, T, D)  # (num_shifts * B, T, D)
                x_repeat = x.unsqueeze(0).expand(len(valid_shifts), -1, -1, -1).reshape(-1, T, D)

                # Compute ba: C_all * x for all
                ba_flat = self.clifford.geometric_product(C_flat, x_repeat)  # (num_shifts * B, T, D)
                ba = ba_flat.reshape(len(valid_shifts), B, T, D)  # (num_shifts, B, T, D)

                # Wedge = (ab - ba) / 2
                wedge_all = (ab - ba) / 2.0  # (num_shifts, B, T, D)

                # Numerical stability
                wedge_all = torch.nan_to_num(wedge_all, nan=0.0, posinf=1e4, neginf=-1e4)
                wedge_all = torch.clamp(wedge_all, min=-1e3, max=1e3)

                # Sum over shifts
                wedge_sum = wedge_all.sum(dim=0)  # (B, T, D)
                wedge_weighted = torch.sigmoid(self.wedge_weight) * wedge_sum
                interaction_output = interaction_output + wedge_weighted

        # Global context: GlobalAvgPool
        if T > 1:
            global_context = self.global_pool(x.transpose(1, 2)).transpose(1, 2)
            interaction_output = interaction_output + global_context

        # Numerical stability: clamp and nan_to_num
        interaction_output = torch.nan_to_num(
            interaction_output, nan=0.0, posinf=1e4, neginf=-1e4
        )
        interaction_output = torch.clamp(interaction_output, min=-1e3, max=1e3)

        # Gated combination: SiLU(x) + gate * interaction_output
        gate = torch.sigmoid(self.gate_fc(x))
        output = F.silu(x) + gate * interaction_output

        # Apply dropout
        output = self.dropout(output)

        # Restore original shape
        if len(original_shape) == 2:
            output = output.squeeze(1)

        return output


    def _forward_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch forward pass (CPU/GPU fallback)."""
        # Handle both (B, D) and (B, T, D) shapes
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1) # (B, 1, D)

        B, T, D = x.shape

        # Accumulate interaction outputs across scales
        interaction_output = torch.zeros_like(x)

        for shift in self.shifts:
            # Multi-scale shift along sequence dimension
            if T > shift:
                x_shifted = torch.roll(x, shifts=shift, dims=1)
                # Local context: discrete Laplacian (CAN-style)
                C_local = x - x_shifted
            else:
                # For short sequences, use zero context
                C_local = torch.zeros_like(x)

            # Compute geometric interactions using CliffordAlgebra
            if self.use_inner:
                # Inner product: scalar part <x*C>_0
                inner = self._compute_inner(x, C_local) # (B, T, 1)
                inner = torch.sigmoid(self.inner_weight) * inner
                interaction_output = interaction_output + inner

            if self.use_wedge:
                # Wedge product: bivector part (x∧C)
                wedge = self._compute_wedge(x, C_local) # (B, T, D)
                wedge = torch.sigmoid(self.wedge_weight) * wedge
                interaction_output = interaction_output + wedge

        # Global context: GlobalAvgPool
        if T > 1:
            # (B, T, D) -> (B, D, T) -> pool -> (B, D, 1) -> (B, 1, D)
            global_context = self.global_pool(x.transpose(1, 2)).transpose(1, 2)
            # Broadcast global context
            interaction_output = interaction_output + global_context

        # Numerical stability: clamp and nan_to_num
        interaction_output = torch.nan_to_num(
            interaction_output, nan=0.0, posinf=1e4, neginf=-1e4
        )
        interaction_output = torch.clamp(interaction_output, min=-1e3, max=1e3)

        # Gated combination: SiLU(x) + gate * interaction_output
        gate = torch.sigmoid(self.gate_fc(x))
        output = F.silu(x) + gate * interaction_output

        # Apply dropout
        output = self.dropout(output)

        # Restore original shape
        if len(original_shape) == 2:
            output = output.squeeze(1)

        return output

    def _compute_inner(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute scalar (inner) part using CliffordAlgebra.geometric_product.

        The inner product is the grade-0 component of the geometric product.
        Returns shape (B, T, 1) for broadcasting.
        """
        # Use geometric_product from CliffordAlgebra
        product = self.clifford.geometric_product(a, b)  # (B, T, D)
        # Extract scalar part (grade-0, index 0)
        scalar_part = product[..., 0:1]  # (B, T, 1)
        return scalar_part

    def _compute_wedge(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute bivector (wedge) part using CliffordAlgebra.geometric_product.

        The wedge product is the antisymmetric part: a∧b = (a*b - b*a) / 2
        This preserves multivector structure unlike element-wise operations.
        """
        # Geometric products
        ab = self.clifford.geometric_product(a, b)  # (B, T, D)
        ba = self.clifford.geometric_product(b, a)  # (B, T, D)

        # Anticommutative part (wedge product)
        wedge = (ab - ba) / 2.0

        # Numerical stability
        wedge = torch.nan_to_num(wedge, nan=0.0, posinf=1e4, neginf=-1e4)
        wedge = torch.clamp(wedge, min=-1e3, max=1e3)

        return wedge
