"""Core invariant losses for paired HDIM training."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=reference.device, dtype=reference.dtype)


def compute_iso_loss(
    u_inv_source: torch.Tensor, u_inv_target: torch.Tensor
) -> torch.Tensor:
    """Core invariant alignment loss between source and target invariants."""
    return F.mse_loss(u_inv_source, u_inv_target)


def compute_infonce_loss(
    source_inv: torch.Tensor,
    target_inv: torch.Tensor,
    pair_relation_label: torch.Tensor,
    pair_weight: torch.Tensor,
    temperature: float = 0.07,
    pair_group_id: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Core invariant InfoNCE loss for positive paired invariants."""
    B = source_inv.shape[0]
    if B < 2:
        return _zero_loss(source_inv)

    if torch.isnan(source_inv).any() or torch.isinf(source_inv).any():
        return _zero_loss(source_inv)
    if torch.isnan(target_inv).any() or torch.isinf(target_inv).any():
        return _zero_loss(source_inv)

    dev = source_inv.device
    positive_mask = pair_relation_label.to(dev) > 0.5
    if not positive_mask.any():
        return _zero_loss(source_inv)

    src = F.normalize(source_inv.float(), dim=-1)
    tgt = F.normalize(target_inv.float(), dim=-1)
    sim_matrix = (src @ tgt.T / temperature).float()

    if pair_group_id is not None:
        gid = pair_group_id.to(dev)
        group_match = gid.unsqueeze(0) == gid.unsqueeze(1)
        eye_mask = torch.eye(B, dtype=torch.bool, device=dev)
        sim_matrix = sim_matrix.masked_fill(
            group_match & ~eye_mask, -torch.finfo(torch.float32).max
        )
    elif positive_mask.sum() > 1:
        pos_mask_2d = positive_mask.unsqueeze(0).expand(B, -1)
        eye_mask = torch.eye(B, dtype=torch.bool, device=dev)
        sim_matrix = sim_matrix.masked_fill(
            pos_mask_2d & ~eye_mask, -torch.finfo(torch.float32).max
        )

    pos_indices = positive_mask.nonzero(as_tuple=True)[0]
    labels = torch.arange(B, device=dev)
    loss_per_sample = F.cross_entropy(
        sim_matrix[pos_indices], labels[pos_indices], reduction="none"
    )
    weights = pair_weight.to(dev)[pos_indices]
    return (loss_per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def compute_pair_iso_loss(
    training_invariant: torch.Tensor,
    iso_target: torch.Tensor,
    batch: Dict[str, Any],
    negative_margin: float = 1.0,
    device: torch.device | None = None,
    has_pairs: bool = True,
) -> torch.Tensor:
    """Core paired invariant loss with weighted positive and margin-negative pairs."""
    if not has_pairs:
        return compute_iso_loss(training_invariant, iso_target)

    dev = device or training_invariant.device
    pair_relation_label = batch.get("pair_relation_label")
    pair_weight = batch.get("pair_weight")
    if pair_relation_label is None or pair_weight is None:
        return compute_iso_loss(training_invariant, iso_target)

    pair_relation_label = pair_relation_label.to(dev, dtype=training_invariant.dtype)
    pair_weight = pair_weight.to(dev, dtype=training_invariant.dtype)
    per_sample_mse = F.mse_loss(
        training_invariant, iso_target, reduction="none"
    ).mean(dim=-1)
    positive_mask = pair_relation_label > 0.5
    negative_mask = ~positive_mask
    losses = []
    if positive_mask.any():
        positive_loss = (
            per_sample_mse[positive_mask] * pair_weight[positive_mask]
        ).sum() / pair_weight[positive_mask].sum().clamp_min(1e-8)
        losses.append(positive_loss)
    if negative_mask.any():
        negative_penalty = F.relu(negative_margin - per_sample_mse[negative_mask])
        weighted_negative = (
            negative_penalty * pair_weight[negative_mask]
        ).sum() / pair_weight[negative_mask].sum().clamp_min(1e-8)
        losses.append(weighted_negative)
    if not losses:
        return per_sample_mse.mean()
    return torch.stack(losses).mean()
