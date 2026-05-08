"""
HDIM — Shared MoE utility functions.

Deduplicated loss functions and helpers used by both MoEKernel and SoftMoERouter.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_balance_loss(
    combine: torch.Tensor,
    num_experts: int,
    slots_per_expert: int,
) -> torch.Tensor:
    """Switch Transformer load balance loss.

    L_lb = E * sum_e f_e.detach() * mean_usage_e

    Normalized by log(num_experts) for scale-invariance across expert counts.

    Args:
        combine: (T, num_slots) combine weights from router
        num_experts: number of experts
        slots_per_expert: slots per expert

    Returns:
        Scalar load balance loss
    """
    T = combine.shape[0]
    expert_w = combine.reshape(T, num_experts, slots_per_expert).sum(-1)  # (T, E)
    routed_expert = expert_w.argmax(dim=-1)
    f_e = F.one_hot(routed_expert, num_classes=num_experts).to(expert_w.dtype).mean(0)
    p_e = expert_w.mean(0)
    raw_loss = num_experts * (f_e.detach() * p_e).sum()
    norm = math.log(num_experts) if num_experts > 1 else 1.0
    return raw_loss / norm


def entropy_load_balance_loss(
    combine: torch.Tensor,
    num_experts: int,
    slots_per_expert: int,
) -> torch.Tensor:
    """Dynamic Switch Transformer load balance loss.

    Uses runtime expert fraction f_e (not static weight-based p).
    Gradient flows only through mean_usage (not f_e).

    Normalized by log(num_experts) for scale-invariance across expert counts.

    Args:
        combine: (T, num_slots) combine weights from router
        num_experts: number of experts
        slots_per_expert: slots per expert

    Returns:
        Scalar load balance loss
    """
    T = combine.shape[0]
    expert_weights = combine.reshape(T, num_experts, slots_per_expert).mean(-1)
    f_e = expert_weights.mean(0).detach()
    mean_usage = expert_weights.mean(0)
    raw_loss = num_experts * (f_e * mean_usage).sum()
    norm = math.log(num_experts) if num_experts > 1 else 1.0
    return raw_loss / norm


def z_loss(
    logits: torch.Tensor,
    z_loss_weight: float = 0.01,
) -> torch.Tensor:
    """Router Z-loss (ST-MoE) for training stability.

    Penalizes large router logits to prevent numerical instability.

    Args:
        logits: (..., num_slots) router logits
        z_loss_weight: weight for z-loss (0 = disabled)

    Returns:
        Scalar z-loss (0 if z_loss_weight <= 0)
    """
    if z_loss_weight <= 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return torch.logsumexp(logits.float(), dim=-1).pow(2).mean()


def aux_loss_free_update(
    expert_bias: torch.Tensor,
    expert_load: torch.Tensor,
    target_load: torch.Tensor,
    aux_lr: float = 0.001,
    bias_update_frequency: int = 0,
    bias_step: int = 0,
) -> int:
    """Auxiliary-Loss-Free bias update (DeepSeek-V3, arXiv:2412.19437).

    Updates per-expert bias based on load imbalance:
    - Overloaded experts: decrease bias (reduce routing probability)
    - Underloaded experts: increase bias (increase routing probability)

    Uses sign-based update for stability:
        bias -= aux_lr * sign(load - target_load)

    Args:
        expert_bias: (num_experts,) bias tensor to update in-place
        expert_load: (num_experts,) current expert usage
        target_load: (num_experts,) target uniform load
        aux_lr: learning rate for bias updates
        bias_update_frequency: steps between updates (0 = every call)
        bias_step: current step counter

    Returns:
        Updated step counter
    """
    bias_step += 1

    if bias_update_frequency > 0 and bias_step % bias_update_frequency != 0:
        return bias_step

    delta = torch.sign(expert_load - target_load)
    expert_bias.data.sub_(delta, alpha=aux_lr)
    expert_bias.data.clamp_(-1.0, 1.0)

    return bias_step


def expert_orthogonalization_loss(
    experts: nn.ModuleList,
) -> torch.Tensor:
    """Expert Orthogonalization loss (arXiv:2505.22323).

    Penalizes expert weight matrices for being too similar.
    L_o = ||W_flat @ W_flat^T - I||_F^2 (averaged over experts)

    Truncates all weights to minimum flat dimension for compatibility
    with experts of different sizes (bottleneck architecture wider than standard).

    Args:
        experts: ModuleList of expert modules

    Returns:
        Scalar orthogonalization loss (0 if no Linear layers found)
    """
    raw_weights = []
    for expert in experts:
        for m in expert.modules():
            if isinstance(m, nn.Linear):
                raw_weights.append(m.weight.reshape(1, -1))
                break
    if not raw_weights:
        return torch.zeros((), device=next(experts.parameters()).device)

    min_len = min(w.shape[-1] for w in raw_weights)
    weights = [F.normalize(w[..., :min_len], dim=-1) for w in raw_weights]

    W = torch.cat(weights, dim=0)  # (E, min_len)
    E = W.shape[0]
    gram = W @ W.T  # (E, E)
    I = torch.eye(E, device=W.device, dtype=W.dtype)
    return ((gram - I) ** 2).mean()
