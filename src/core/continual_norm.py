#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Continual Normalization for Online Learning Stability.

Implements IL-ETransformer-style continual batch normalization that tracks
running statistics without task boundaries. Prevents distribution shift
in streaming scenarios where data distribution may evolve over time.

Reference:
    - IL-ETransformer: Incremental Learning without Forgetting (2024)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ContinualNorm(nn.Module):
    """Continual normalization for online learning stability.

    Unlike standard BatchNorm which resets statistics, ContinualNorm
    maintains EMA running statistics across task boundaries, enabling
    stable normalization in streaming/incremental learning scenarios.

    Args:
        num_features: Number of features/channels to normalize
        momentum: EMA momentum for running stats update (default: 0.1)
        eps: Small constant for numerical stability (default: 1e-5)
        affine: If True, learnable affine transform (default: True)

    Shape:
        - Input: (N, num_features) or (N, C, H, W) for 2D conv
        - Output: Same shape as input
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.affine = affine

        # Running statistics (no task reset)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches', torch.tensor(0, dtype=torch.long))

        # Learnable affine parameters
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_buffer('weight', None)
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply continual normalization.

        In training mode: updates running statistics via EMA without reset.
        In eval mode: uses accumulated running statistics for normalization.

        Args:
            x: Input tensor of shape (N, num_features) or (N, C, H, W)

        Returns:
            Normalized tensor with same shape as input
        """
        if self.training:
            # Compute batch statistics
            if x.dim() == 2:
                # (N, F) format
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False)
            elif x.dim() == 4:
                # (N, C, H, W) format - normalize over (N, H, W)
                batch_mean = x.mean(dim=(0, 2, 3))
                batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            else:
                raise ValueError(f"Expected 2D or 4D tensor, got {x.dim()}D")

            # EMA update (continual, no task reset)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(
                    batch_mean, alpha=self.momentum
                )
                self.running_var.mul_(1 - self.momentum).add_(
                    batch_var, alpha=self.momentum
                )
                self.num_batches.add_(1)
        else:
            # Use running stats in eval mode
            pass

        # Normalize using running statistics
        if x.dim() == 2:
            x_norm = (x - self.running_mean) / torch.sqrt(
                self.running_var + self.eps
            )
        else:
            # (N, C, H, W) - reshape for broadcast
            x_norm = (x - self.running_mean[None, :, None, None]) / torch.sqrt(
                self.running_var[None, :, None, None] + self.eps
            )

        # Apply affine transform if enabled
        if self.affine:
            if x.dim() == 2:
                x_norm = self.weight * x_norm + self.bias
            else:
                x_norm = (
                    self.weight[None, :, None, None] * x_norm +
                    self.bias[None, :, None, None]
                )

        return x_norm

    def reset_running_stats(self) -> None:
        """Reset running statistics to initial state.

        WARNING: Only use for testing or complete re-initialization.
        In continual learning, you typically NEVER call this.
        """
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches.zero_()

    def extra_repr(self) -> str:
        return (
            f'{self.num_features}, momentum={self.momentum}, '
            f'eps={self.eps}, affine={self.affine}'
        )


class ContinualNormLayer(nn.Module):
    """Layer normalization alternative with continual statistics.

    Uses layer-level statistics for normalization while maintaining
    continual EMA tracking for stability monitoring.

    Args:
        normalized_shape: Shape of normalized dimensions
        eps: Small constant for numerical stability (default: 1e-5)
        elementwise_affine: If True, learnable per-element affine (default: True)
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # Running layer statistics for monitoring
        self.register_buffer('running_mean', torch.zeros(normalized_shape))
        self.register_buffer('running_var', torch.ones(normalized_shape))
        self.register_buffer('num_batches', torch.tensor(0, dtype=torch.long))

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_buffer('weight', None)
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization with continual monitoring."""
        # Standard layer norm
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Track running statistics for monitoring (not for normalization)
        if self.training:
            with torch.no_grad():
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False)
                self.running_mean.lerp_(batch_mean, weight=0.01)
                self.running_var.lerp_(batch_var, weight=0.01)
                self.num_batches.add_(1)

        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias

        return x_norm

    def extra_repr(self) -> str:
        return (
            f'{self.normalized_shape}, eps={self.eps}, '
            f'elementwise_affine={self.elementwise_affine}'
        )
