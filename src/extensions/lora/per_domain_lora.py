"""Per-domain LoRA adapters for HDIM domain expert specialization.

Each domain expert (math/language/code/science) gets a small LoRA delta
that specializes the shared representation for that domain.

Standard LoRA: output = x + B @ A @ x
  - A initialized with kaiming (non-zero at init for gradient flow)
  - B initialized with zeros (delta starts at zero, preserving pretrained)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class PerDomainLoRA(nn.Module):
    """Low-rank adaptation per domain expert.
    Each domain gets its own (A_d, B_d) LoRA pair.
    At domain d: output = x + B_d @ A_d @ x

    Args:
        dim: input/output dimension
        rank: LoRA rank (default 4)
        num_domains: number of domains (default 4)
    """

    def __init__(self, dim: int, rank: int = 4, num_domains: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.num_domains = num_domains

        self.A_stack = nn.Parameter(torch.zeros(num_domains, rank, dim))
        self.B_stack = nn.Parameter(torch.zeros(num_domains, dim, rank))

        # Standard LoRA init: A with kaiming, B with zeros
        for d in range(num_domains):
            init.kaiming_uniform_(self.A_stack.data[d])
        self.B_stack.data.zero_()

    def forward(self, x: Tensor, domain_idx: int) -> Tensor:
        """Apply LoRA for the given domain.

        Args:
            x: input tensor of shape (..., dim)
            domain_idx: integer domain index in [0, num_domains)

        Returns:
            Tensor of same shape as x with domain LoRA delta applied.
        """
        A = self.A_stack[domain_idx]  # (rank, dim)
        B = self.B_stack[domain_idx]  # (dim, rank)
        # x: (..., dim) -> x @ A^T -> (..., rank) -> (..., rank) @ B^T -> (..., dim)
        delta = (x @ A.T) @ B.T
        return x + delta

    def extra_repr(self) -> str:
        return f"dim={self.dim}, rank={self.rank}, num_domains={self.num_domains}"
