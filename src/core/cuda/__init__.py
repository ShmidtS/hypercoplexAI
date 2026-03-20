"""Triton-accelerated kernels for Clifford operations.

This module provides CUDA kernels via Triton for:
- Geometric product in Cl_{3,1,0} (16D multivectors)
- Clifford Interaction forward/backward passes

Graceful fallback to PyTorch when Triton/CUDA unavailable.
"""

import torch

# Try to import Triton, gracefully fallback
TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    pass

CUDA_AVAILABLE = torch.cuda.is_available()

from .clifford_interaction_kernels import (
    TritonCliffordInteraction,
    triton_geometric_product,
    triton_clifford_interaction_forward,
    has_triton_support,
)

__all__ = [
    "TRITON_AVAILABLE",
    "CUDA_AVAILABLE",
    "TritonCliffordInteraction",
    "triton_geometric_product",
    "triton_clifford_interaction_forward",
    "has_triton_support",
]
