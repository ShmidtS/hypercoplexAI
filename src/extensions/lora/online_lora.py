#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Online-LoRA for Task-Free Continual Learning.

Implements LoRA-style adapters with gradient-based importance weighting
for online adaptation without catastrophic forgetting.

Reference:
    Wei et al., WACV 2025
    "Online-LoRA: Task-Free Online Continual Learning via Low Rank Adaptation"

Features:
    - Low-rank adapters (rank 8-32 configurable)
    - Gradient-based importance tracking
    - EMA stabilization for plasticity-stability balance
    - No task boundaries required
    - Zero catastrophic forgetting via importance-weighted updates

Integration with OnlineLearner:
    OnlineLoRA provides per-layer adaptation, while OnlineLearner handles
    global coordination via surprise detection and replay buffer.
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class OnlineLoRAConfig:
    """Configuration for Online-LoRA adapter."""
    rank: int = 8
    alpha: float = 1.0
    dropout: float = 0.0
    importance_decay: float = 0.9
    importance_gain: float = 0.1
    ema_decay: float = 0.999
    gradient_clip: float = 1.0


class OnlineLoRA(nn.Module):
    """Online-LoRA for task-free continual learning.

    Wraps a base layer with low-rank adaptation and importance tracking.
    Uses gradient-based importance weighting to prevent catastrophic forgetting.

    Args:
        base_layer: The base nn.Linear or nn.Conv2d layer to adapt
        rank: Rank of low-rank matrices (default: 8)
        alpha: Scaling factor for LoRA output (default: 1.0)
        dropout: Dropout probability (default: 0.0)
        importance_decay: EMA decay for importance (default: 0.9)
        importance_gain: Learning rate for importance updates (default: 0.1)
        ema_decay: EMA decay for weight stabilization (default: 0.999)

    Example:
        >>> linear = nn.Linear(256, 128)
        >>> lora = OnlineLoRA(linear, rank=8)
        >>> output = lora(input)  # Forward pass with LoRA adaptation
    """

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        importance_decay: float = 0.9,
        importance_gain: float = 0.1,
        ema_decay: float = 0.999,
    ):
        super().__init__()

        # Validate base layer
        if not isinstance(base_layer, (nn.Linear, nn.Conv2d)):
            raise TypeError(f"Base layer must be nn.Linear or nn.Conv2d, got {type(base_layer)}")

        self.base = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout_p = dropout
        self.importance_decay = importance_decay
        self.importance_gain = importance_gain
        self.ema_decay = ema_decay

        # Freeze base layer parameters
        for param in self.base.parameters():
            param.requires_grad = False

        # Get dimensions
        if isinstance(base_layer, nn.Linear):
            in_features = base_layer.in_features
            out_features = base_layer.out_features
            self._is_conv = False
        else:  # nn.Conv2d
            in_features = base_layer.in_channels
            out_features = base_layer.out_channels
            self._is_conv = True

        # LoRA weights: A (down-projection), B (up-projection)
        # Initialize A with Kaiming, B with zeros (original LoRA pattern)
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.empty(rank, out_features))
        self._init_weights()

        # Importance tracking (gradient-based)
        # Higher importance = more critical for previous tasks = slower to change
        self.register_buffer('importance', torch.ones(in_features))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.long))

        # EMA of weights for stability
        self.register_buffer('lora_A_ema', torch.zeros_like(self.lora_A))
        self.register_buffer('lora_B_ema', torch.zeros_like(self.lora_B))
        self._ema_initialized = False

        # Optional dropout
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def _init_weights(self) -> None:
        """Initialize LoRA weights."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with LoRA adaptation.

        Args:
            x: Input tensor [batch, in_features] or [batch, C, H, W]

        Returns:
            Output tensor with LoRA contribution added
        """
        # Base output
        output = self.base(x)

        # LoRA contribution
        lora_out = self._compute_lora(x)

        # Update importance during training
        if self.training and x.requires_grad:
            self._schedule_importance_update(x)

        return output + lora_out

    def _compute_lora(self, x: Tensor) -> Tensor:
        """Compute LoRA contribution with importance weighting."""
        # Compute LoRA output
        if self._is_conv:
            # For Conv2d: reshape for linear operations
            # x: [batch, C_in, H, W] -> [batch*H*W, C_in]
            batch, C_in, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C_in)

            # Apply importance weighting to flattened input
            x_weighted = x_flat * self.importance

            # Apply dropout if configured
            if self.dropout is not None:
                x_weighted = self.dropout(x_weighted)

            # LoRA: [batch*H*W, C_in] @ [C_in, r] @ [r, C_out] -> [batch*H*W, C_out]
            lora_flat = x_weighted @ self.lora_A @ self.lora_B * self.scaling

            # Reshape back: [batch, C_out, H, W]
            lora_out = lora_flat.reshape(batch, H, W, -1).permute(0, 3, 1, 2)
        else:
            # Apply importance weighting to input
            x_weighted = x * self.importance

            # Apply dropout if configured
            if self.dropout is not None:
                x_weighted = self.dropout(x_weighted)

            # For Linear: [batch, in] @ [in, r] @ [r, out] -> [batch, out]
            lora_out = x_weighted @ self.lora_A @ self.lora_B * self.scaling

        return lora_out

    def _schedule_importance_update(self, x: Tensor) -> None:
        """Schedule importance update for after backward pass."""
        # Register hook to update importance after gradient computation
        if x.grad_fn is not None:
            x.register_hook(lambda grad: self._update_importance(x, grad))

    def _update_importance(self, x: Tensor, grad: Tensor) -> Tensor:
        """Update importance based on gradient magnitude.

        Higher gradient = more sensitive = higher importance.
        Uses EMA for stability.

        Args:
            x: Input tensor (unused, for shape reference)
            grad: Gradient tensor from backward pass

        Returns:
            Unmodified gradient (pass-through)
        """
        with torch.no_grad():
            # Compute gradient magnitude per feature
            if self._is_conv:
                # For Conv2d: average over batch, H, W dimensions
                grad_mag = grad.abs().mean(dim=(0, 2, 3))
            else:
                # For Linear: average over batch dimension
                grad_mag = grad.abs().mean(dim=0)

            # Ensure shape match
            if grad_mag.shape[0] != self.importance.shape[0]:
                # Skip if shapes don't match (shouldn't happen in normal use)
                return grad

            # EMA update: importance = decay * importance + gain * grad_mag
            self.importance.mul_(self.importance_decay).add_(
                grad_mag, alpha=self.importance_gain
            )

            # Update counter
            self.num_updates.add_(1)

        return grad

    def update_ema(self) -> None:
        """Update EMA of LoRA weights for stability.

        Should be called after each training step to maintain
        a stable reference for consolidation.
        """
        with torch.no_grad():
            if not self._ema_initialized:
                self.lora_A_ema.copy_(self.lora_A.data)
                self.lora_B_ema.copy_(self.lora_B.data)
                self._ema_initialized = True
            else:
                self.lora_A_ema.mul_(self.ema_decay).add_(
                    self.lora_A.data, alpha=1 - self.ema_decay
                )
                self.lora_B_ema.mul_(self.ema_decay).add_(
                    self.lora_B.data, alpha=1 - self.ema_decay
                )

    def consolidate(self, strength: float = 0.1) -> None:
        """Consolidate LoRA weights toward EMA to prevent drift.

        Called periodically to prevent catastrophic forgetting by
        anchoring weights to their historical average.

        Args:
            strength: How much to pull toward EMA (0 = none, 1 = full reset)
        """
        if not self._ema_initialized:
            return

        with torch.no_grad():
            # Importance-weighted consolidation
            # High importance features are pulled more toward EMA
            importance_weights = torch.sigmoid(self.importance - 1.0)

            # Update A weights: blend current with EMA (importance-weighted per feature)
            # lerp(t, a, b) = a + t * (b - a) = (1-t)*a + t*b
            blend_A = strength * importance_weights.unsqueeze(1)
            self.lora_A.data.lerp_(self.lora_A_ema, blend_A)

            # Update B weights: uniform consolidation
            self.lora_B.data.lerp_(self.lora_B_ema, strength)

    def get_importance(self) -> Tensor:
        """Get current importance weights.

        Returns:
            Importance tensor [in_features]
        """
        return self.importance.clone()

    def get_stats(self) -> Dict[str, Any]:
        """Get LoRA statistics for monitoring.

        Returns:
            Dictionary with rank, num_updates, importance stats, weight norms
        """
        return {
            'rank': self.rank,
            'num_updates': int(self.num_updates.item()),
            'importance_mean': float(self.importance.mean().item()),
            'importance_std': float(self.importance.std().item()),
            'lora_A_norm': float(self.lora_A.data.norm().item()),
            'lora_B_norm': float(self.lora_B.data.norm().item()),
            'ema_initialized': self._ema_initialized,
        }

    def reset_importance(self) -> None:
        """Reset importance weights to uniform.

        Useful when switching to a new domain or after consolidation.
        """
        self.importance.fill_(1.0)

    def extra_repr(self) -> str:
        return f'rank={self.rank}, scaling={self.scaling:.4f}'


class OnlineLoRALinear(OnlineLoRA):
    """Online-LoRA wrapper for nn.Linear layers.

    Convenience class that validates the base layer is Linear.

    Args:
        base_layer: nn.Linear layer to adapt
        **kwargs: Additional arguments passed to OnlineLoRA

    Example:
        >>> linear = nn.Linear(512, 256)
        >>> lora_linear = OnlineLoRALinear(linear, rank=16)
        >>> output = lora_linear(input)
    """

    def __init__(self, base_layer: nn.Linear, **kwargs):
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(base_layer)}")
        super().__init__(base_layer, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class OnlineLoRAConv(OnlineLoRA):
    """Online-LoRA wrapper for nn.Conv2d layers.

    Convenience class that validates the base layer is Conv2d.

    Args:
        base_layer: nn.Conv2d layer to adapt
        **kwargs: Additional arguments passed to OnlineLoRA

    Example:
        >>> conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        >>> lora_conv = OnlineLoRAConv(conv, rank=8)
        >>> output = lora_conv(input)
    """

    def __init__(self, base_layer: nn.Conv2d, **kwargs):
        if not isinstance(base_layer, nn.Conv2d):
            raise TypeError(f"Expected nn.Conv2d, got {type(base_layer)}")
        super().__init__(base_layer, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


def wrap_with_online_lora(
    module: nn.Module,
    config: Optional[OnlineLoRAConfig] = None,
    target_types: Tuple[type, ...] = (nn.Linear, nn.Conv2d),
) -> nn.Module:
    """Wrap all target layers in a module with Online-LoRA.

    Recursively finds all layers of target types and wraps them
    with OnlineLoRA adapters.

    Args:
        module: The module to wrap (e.g., a full model)
        config: LoRA configuration (uses defaults if None)
        target_types: Layer types to wrap (default: Linear, Conv2d)

    Returns:
        Module with wrapped layers (modified in-place)

    Example:
        >>> model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
        >>> model = wrap_with_online_lora(model, OnlineLoRAConfig(rank=16))
    """
    config = config or OnlineLoRAConfig()

    for name, child in list(module.named_children()):
        if isinstance(child, target_types):
            # Wrap the layer
            if isinstance(child, nn.Linear):
                wrapped = OnlineLoRALinear(
                    child,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                    importance_decay=config.importance_decay,
                    importance_gain=config.importance_gain,
                    ema_decay=config.ema_decay,
                )
            elif isinstance(child, nn.Conv2d):
                wrapped = OnlineLoRAConv(
                    child,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                    importance_decay=config.importance_decay,
                    importance_gain=config.importance_gain,
                    ema_decay=config.ema_decay,
                )
            else:
                continue

            setattr(module, name, wrapped)
        else:
            # Recurse into child modules
            wrap_with_online_lora(child, config, target_types)

    return module


class OnlineLoRAManager:
    """Manager for coordinating multiple Online-LoRA adapters.

    Provides centralized control for:
    - Batch EMA updates
    - Coordinated consolidation
    - Statistics aggregation
    - Integration with OnlineLearner

    Args:
        consolidation_interval: Steps between consolidations (default: 1000)
        consolidation_strength: How strongly to pull toward EMA (default: 0.1)

    Example:
        >>> model = wrap_with_online_lora(MyModel())
        >>> manager = OnlineLoRAManager()
        >>> manager.register_from_module(model)
        >>> manager.update_all_ema()  # After each training step
        >>> manager.consolidate_all()  # Periodically
    """

    def __init__(
        self,
        consolidation_interval: int = 1000,
        consolidation_strength: float = 0.1,
    ):
        self.adapters: List[OnlineLoRA] = []
        self.consolidation_interval = consolidation_interval
        self.consolidation_strength = consolidation_strength
        self._step_count = 0

    def register(self, adapter: OnlineLoRA) -> None:
        """Register an OnlineLoRA adapter for management."""
        if not isinstance(adapter, OnlineLoRA):
            raise TypeError(f"Expected OnlineLoRA, got {type(adapter)}")
        self.adapters.append(adapter)

    def register_from_module(self, module: nn.Module) -> int:
        """Find and register all OnlineLoRA adapters in a module.

        Returns:
            Number of adapters found and registered
        """
        count = 0
        for child in module.modules():
            if isinstance(child, OnlineLoRA) and child not in self.adapters:
                self.adapters.append(child)
                count += 1
        return count

    def update_all_ema(self) -> None:
        """Update EMA for all registered adapters."""
        for adapter in self.adapters:
            adapter.update_ema()
        self._step_count += 1

    def should_consolidate(self) -> bool:
        """Check if consolidation should be triggered."""
        return self._step_count > 0 and self._step_count % self.consolidation_interval == 0

    def consolidate_all(self, strength: Optional[float] = None) -> None:
        """Consolidate all registered adapters.

        Args:
            strength: Override consolidation strength (uses default if None)
        """
        strength = strength if strength is not None else self.consolidation_strength
        for adapter in self.adapters:
            adapter.consolidate(strength)

    def get_all_stats(self) -> Dict[str, Any]:
        """Aggregate statistics from all adapters."""
        if not self.adapters:
            return {'num_adapters': 0}

        stats = {
            'num_adapters': len(self.adapters),
            'step_count': self._step_count,
            'total_num_updates': sum(
                int(a.num_updates.item()) for a in self.adapters
            ),
            'adapters': [a.get_stats() for a in self.adapters],
        }

        # Aggregate importance stats
        all_importance = torch.cat([a.importance for a in self.adapters])
        stats['importance_mean_global'] = float(all_importance.mean().item())
        stats['importance_std_global'] = float(all_importance.std().item())

        return stats

    def reset_all_importance(self) -> None:
        """Reset importance for all adapters (e.g., new domain)."""
        for adapter in self.adapters:
            adapter.reset_importance()

    def step(self) -> None:
        """Perform one management step: update EMA and optionally consolidate.

        Call this after each training batch.
        """
        self.update_all_ema()
        if self.should_consolidate():
            self.consolidate_all()

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable LoRA parameters.

        Useful for creating optimizer with only LoRA params.
        """
        params = []
        for adapter in self.adapters:
            params.extend([adapter.lora_A, adapter.lora_B])
        return params
