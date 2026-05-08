"""HDIM loss functions — extracted from HDIMTrainer for separation of concerns.

All loss computation logic lives here. The trainer delegates to
`compute_batch_losses()` which orchestrates forward-pass-agnostic loss
arithmetic. Individual loss functions are pure (no self references).

Math is EXACTLY preserved from the original trainer.py — no formula changes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hdim_model import HDIMAuxState
from src.training.temperature import cluster_scaled_temperature, effective_temperature

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LossConfig:
    """All loss-related hyperparameters (was scattered as HDIMTrainer attributes)."""

    lambda_routing: float = 0.01
    lambda_pair: float = 0.4
    lambda_memory: float = 0.05
    negative_margin: float = 1.0
    ranking_margin: float = 0.2
    use_infonce: bool = True
    infonce_temperature: float = 0.10
    lambda_z: float = 0.01
    lambda_expert_ortho: float = 0.01
    lambda_online: float = 0.01
    learnable_temperature: bool = False
    lambda_dcl: float = 0.05
    lambda_uniformity: float = 0.02
    use_sc_temperature: bool = False
    lambda_matryoshka: float = 0.15
    # Temperature scheduling
    temp_schedule: str = "none"
    tau_max: float = 0.1
    tau_min: float = 0.01
    temp_schedule_T_0: int = 20
    focal_gamma: float = 1.0
    # Hard negatives
    use_hard_negatives: bool = False


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=reference.device, dtype=reference.dtype)


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------

def compute_reconstruction_loss(
    output: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return nn.functional.mse_loss(output, target)


def compute_iso_loss(
    u_inv_source: torch.Tensor, u_inv_target: torch.Tensor
) -> torch.Tensor:
    return nn.functional.mse_loss(u_inv_source, u_inv_target)


def compute_routing_loss(routing_weights: torch.Tensor) -> torch.Tensor:
    mean_routing = routing_weights.mean(dim=0)
    eps = 1e-8
    entropy = -(mean_routing * (mean_routing + eps).log()).sum()
    return -entropy


def compute_infonce_loss(
    source_inv: torch.Tensor,
    target_inv: torch.Tensor,
    pair_relation_label: torch.Tensor,
    pair_weight: torch.Tensor,
    temperature: float = 0.07,
    pair_group_id: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """InfoNCE loss (NT-Xent) for positive/negative pair discrimination.

    Uses in-batch negatives (SimCLR, CLIP, sentence-transformers).

    Args:
        source_inv: (B, D) source exported_invariant, L2-normalized
        target_inv: (B, D) target exported_invariant, L2-normalized
        pair_relation_label: (B,) 1.0=positive, 0.0=negative
        pair_weight: (B,) per-sample weights
        temperature: softmax temperature
        pair_group_id: (B,) group/family IDs — same-group entries are false negatives
        device: torch device for tensor ops
    Returns:
        scalar InfoNCE loss
    """
    B = source_inv.shape[0]
    if B < 2:
        return _zero_loss(source_inv)

    # NaN/Inf protection during validation
    if torch.isnan(source_inv).any() or torch.isinf(source_inv).any():
        return _zero_loss(source_inv)
    if torch.isnan(target_inv).any() or torch.isinf(target_inv).any():
        return _zero_loss(source_inv)

    dev = source_inv.device
    positive_mask = pair_relation_label.to(dev) > 0.5
    if not positive_mask.any():
        return _zero_loss(source_inv)

    src = F.normalize(source_inv.float(), dim=-1)  # (B, D) — float32 for AMP safety
    tgt = F.normalize(target_inv.float(), dim=-1)  # (B, D)

    # Similarity matrix: (B, B) — force float32 to avoid fp16 overflow in masked_fill/logsumexp
    sim_matrix = (src @ tgt.T / temperature).float()

    # Mask out false negatives from denominator
    # 1) Same-family entries (same pair_group_id) — semantically similar but not true negatives
    if pair_group_id is not None:
        gid = pair_group_id.to(dev)
        group_match = (gid.unsqueeze(0) == gid.unsqueeze(1))  # (B, B)
        eye_mask = torch.eye(B, dtype=torch.bool, device=dev)
        sim_matrix = sim_matrix.masked_fill(group_match & ~eye_mask, -torch.finfo(torch.float32).max)
    # 2) Other positive pairs (label>0.5) — also false negatives
    elif positive_mask.sum() > 1:
        pos_mask_2d = positive_mask.unsqueeze(0).expand(B, -1)  # (B, B)
        eye_mask = torch.eye(B, dtype=torch.bool, device=dev)
        sim_matrix = sim_matrix.masked_fill(pos_mask_2d & ~eye_mask, -torch.finfo(torch.float32).max)

    # InfoNCE: for each positive pair (i,i), treat all other j!=i as negatives
    pos_indices = positive_mask.nonzero(as_tuple=True)[0]

    # Labels: diagonal = self-match for positives
    labels = torch.arange(B, device=dev)
    loss_per_sample = F.cross_entropy(sim_matrix[pos_indices], labels[pos_indices], reduction='none')

    weights = pair_weight.to(dev)[pos_indices]
    return (loss_per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def compute_focal_infonce_loss(
    source_inv: torch.Tensor,
    target_inv: torch.Tensor,
    pair_relation_label: torch.Tensor,
    pair_weight: torch.Tensor,
    temperature: float = 0.07,
    gamma: float = 0.5,
    pair_group_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Focal-InfoNCE loss (Hou & Li, EMNLP Findings 2023).

    Downweights easy negatives via focal exponent scaling.
    When gamma < 1.0, the denominator compresses contribution of
    well-separated negatives, focusing learning on hardest pairs.

    Args:
        source_inv: (B, D) source exported_invariant
        target_inv: (B, D) target exported_invariant
        pair_relation_label: (B,) 1.0=positive, 0.0=negative
        pair_weight: (B,) per-sample weights
        temperature: softmax temperature
        gamma: focal parameter (1.0=standard, 0.5=moderate focus on hard)
    Returns:
        scalar InfoNCE loss (always >= 0)
    """
    B = source_inv.shape[0]
    if B < 2:
        return _zero_loss(source_inv)

    # NaN/Inf protection during validation
    if torch.isnan(source_inv).any() or torch.isinf(source_inv).any():
        return _zero_loss(source_inv)
    if torch.isnan(target_inv).any() or torch.isinf(target_inv).any():
        return _zero_loss(source_inv)

    dev = source_inv.device
    positive_mask = pair_relation_label.to(dev) > 0.5
    if not positive_mask.any():
        return _zero_loss(source_inv)

    src = F.normalize(source_inv.float(), dim=-1)  # float32 for AMP safety
    tgt = F.normalize(target_inv.float(), dim=-1)

    pos_indices = positive_mask.nonzero(as_tuple=True)[0]
    labels = torch.arange(B, device=dev)

    if gamma >= 0.99:
        # Standard InfoNCE path (no focal modulation)
        sim_matrix = (src @ tgt.T / temperature).float()
        # Mask out false negatives from denominator
        if pair_group_id is not None:
            gid = pair_group_id.to(dev)
            group_match = (gid.unsqueeze(0) == gid.unsqueeze(1))
            sim_matrix = sim_matrix.masked_fill(group_match & ~torch.eye(B, dtype=torch.bool, device=dev), -torch.finfo(torch.float32).max)
        elif positive_mask.sum() > 1:
            pos_mask_2d = positive_mask.unsqueeze(0).expand(B, -1)
            sim_matrix = sim_matrix.masked_fill(pos_mask_2d & ~torch.eye(B, dtype=torch.bool, device=dev), -torch.finfo(torch.float32).max)
        loss_per_sample = F.cross_entropy(
            sim_matrix[pos_indices], labels[pos_indices], reduction='none'
        )
    else:
        # Focal-InfoNCE: scale logits by gamma before CE
        sim_matrix = (src @ tgt.T / temperature).float()
        if pair_group_id is not None:
            gid = pair_group_id.to(dev)
            group_match = (gid.unsqueeze(0) == gid.unsqueeze(1))
            sim_matrix = sim_matrix.masked_fill(group_match & ~torch.eye(B, dtype=torch.bool, device=dev), -torch.finfo(torch.float32).max)
        elif positive_mask.sum() > 1:
            pos_mask_2d = positive_mask.unsqueeze(0).expand(B, -1)
            sim_matrix = sim_matrix.masked_fill(pos_mask_2d & ~torch.eye(B, dtype=torch.bool, device=dev), -torch.finfo(torch.float32).max)
        pos_mask = (pair_relation_label > 0.5).to(dev)  # (B,)
        focal_matrix = torch.where(
            pos_mask.unsqueeze(1),
            sim_matrix,
            sim_matrix * gamma
        )
        loss_per_sample = F.cross_entropy(
            focal_matrix[pos_indices], labels[pos_indices], reduction='none'
        )

    weights = pair_weight.to(dev)[pos_indices]
    return (loss_per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def compute_dcl_loss(
    source_inv: torch.Tensor,
    target_inv: torch.Tensor,
    pair_relation_label: torch.Tensor,
    pair_weight: torch.Tensor,
    temperature: float = 0.07,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Decoupled Contrastive Loss (DCL) — Yeh et al., NeurIPS 2022.

    Key difference from InfoNCE: positive removed from denominator.
    L_DCL = -log( exp(s_pos/tau) / Sum_{j: j!=pos} exp(s_j/tau) )
    """
    _dev = device or source_inv.device
    B = source_inv.shape[0]
    if B < 2:
        return _zero_loss(source_inv)

    positive_mask = pair_relation_label > 0.5
    if not positive_mask.any():
        return _zero_loss(source_inv)

    src = F.normalize(source_inv.float(), dim=-1)   # (B, D) — fp32 to avoid AMP fp16 precision loss
    tgt = F.normalize(target_inv.float(), dim=-1)   # (B, D)

    sim_matrix = (src @ tgt.T / temperature).float()  # force fp32 for AMP safety

    diag_mask = torch.eye(B, dtype=torch.bool, device=_dev)

    # Vectorized: mask diagonal to -inf for all rows at once
    masked_sim = sim_matrix.clone()
    masked_sim.masked_fill_(diag_mask, -torch.finfo(torch.float32).max)
    log_denom = torch.logsumexp(masked_sim, dim=1)  # (B,)
    pos_mask = (pair_relation_label > 0.5).to(pair_relation_label.device).float()  # (B,)
    s_pos = (sim_matrix * pos_mask.unsqueeze(1)).sum(dim=1) / pos_mask.unsqueeze(1).sum(dim=1).clamp(min=1)
    loss_per_sample = -(s_pos - log_denom)  # (B,)
    # Apply weights only to positive samples
    weights = pair_weight * positive_mask.float()
    if weights.sum() == 0:
        return _zero_loss(source_inv)
    return ((loss_per_sample * weights).sum() / weights.sum().clamp_min(1e-8)).clamp(min=0)


def compute_uniformity_alignment_loss(
    source_inv: torch.Tensor,
    target_inv: torch.Tensor,
    pair_relation_label: torch.Tensor,
    t_uniform: float = 2.0,
    lambda_align: float = 1.0,
    lambda_uniform: float = 0.5,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Uniformity + Alignment loss (Wang & Isola, ICML 2020).

    L_align  = E[||f(x)-f(y)||^2]  — align positive pairs
    L_uniform = log E[exp(-t||f(x)-f(z)||^2)] — push all uniformly

    These two components are separated: alignment only for positives,
    uniformity for all. More stable than InfoNCE.
    """
    _dev = device or source_inv.device
    B = source_inv.shape[0]
    if B < 4:
        return _zero_loss(source_inv)

    positive_mask = pair_relation_label > 0.5
    src = F.normalize(source_inv.float(), dim=-1)  # (B, D)
    tgt = F.normalize(target_inv.float(), dim=-1)  # (B, D)

    # Alignment: only for positives
    if positive_mask.any():
        src_pos = src[positive_mask]  # (Np, D)
        tgt_pos = tgt[positive_mask]  # (Np, D)
        loss_align = (src_pos - tgt_pos).pow(2).sum(dim=-1).mean()
    else:
        loss_align = _zero_loss(source_inv)

    # Uniformity: across all pairs (src U tgt)
    all_vecs = torch.cat([src, tgt], dim=0)  # (2B, D)
    diffs = all_vecs.unsqueeze(0) - all_vecs.unsqueeze(1)  # (2B, 2B, D)
    sq_dists = diffs.pow(2).sum(dim=-1)  # (2B, 2B)
    # Mask diagonal (zero distance to self)
    mask = ~torch.eye(2 * B, dtype=torch.bool, device=_dev)
    sq_dists_off = sq_dists[mask].view(2 * B, 2 * B - 1)
    # Numerically stable log-mean-exp to avoid Inf overflow
    neg_sq = (-t_uniform * sq_dists_off).float()  # fp32 for logsumexp stability
    loss_uniform = torch.logsumexp(neg_sq, dim=-1).mean() - math.log(neg_sq.shape[-1])

    return (lambda_align * loss_align + lambda_uniform * loss_uniform).clamp(min=0)


def compute_matryoshka_loss(
    matryoshka_embs: Dict[int, torch.Tensor],
    batch: Dict[str, Any],
    effective_temp: float,
    matryoshka_target: Dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Matryoshka Representation Learning loss.

    Trains the encoder to produce meaningful embeddings at multiple dimensions.
    Each scale contributes equally to the total loss.

    Ref: Kusupati et al., NeurIPS 2022
    """
    if not matryoshka_embs:
        return _zero_loss(next(iter(batch.values())))

    # Need pair info for contrastive loss at each scale
    pair_relation_label = batch.get("pair_relation_label")
    if pair_relation_label is None:
        return _zero_loss(next(iter(matryoshka_embs.values())))

    pair_weight = batch.get("pair_weight")
    if pair_weight is None:
        pair_weight = torch.ones_like(pair_relation_label)

    # For Matryoshka, we need both source and target embeddings at each scale
    matryoshka_source = matryoshka_embs
    if matryoshka_target is None:
        matryoshka_target = batch.get("matryoshka_target", matryoshka_embs)

    if not matryoshka_source or not matryoshka_target:
        return _zero_loss(next(iter(matryoshka_embs.values())))

    total_loss = _zero_loss(next(iter(matryoshka_source.values())))
    n_scales = 0

    # Compute contrastive loss at each scale
    for dim in matryoshka_source.keys():
        if dim not in matryoshka_target:
            continue

        src_emb = matryoshka_source[dim]
        tgt_emb = matryoshka_target[dim]

        # Use InfoNCE at each scale with pair_group_id for SupCon
        scale_loss = compute_infonce_loss(
            src_emb,
            tgt_emb,
            pair_relation_label,
            pair_weight,
            temperature=effective_temp,
            pair_group_id=batch.get("pair_group_id"),
        )
        total_loss = total_loss + scale_loss
        n_scales += 1

    # Average across scales
    if n_scales > 0:
        total_loss = total_loss / n_scales

    return total_loss


def mine_hard_negatives(
    source_inv: torch.Tensor,
    target_inv: torch.Tensor,
    pair_relation_label: torch.Tensor,
    device: torch.device,
    top_k: int = 4,
) -> torch.Tensor:
    """Online hard negative mining: find hardest negatives in-batch.

    Returns indices (B,) — for each anchor, index of hardest negative.
    -1 means no hard negative found.
    """
    B = source_inv.shape[0]
    src = F.normalize(source_inv.detach().float(), dim=-1)  # fp32 for AMP safety
    tgt = F.normalize(target_inv.detach().float(), dim=-1)
    sim = src @ tgt.T  # (B, B)

    # Mask diagonal and true positives
    eye = torch.eye(B, dtype=torch.bool, device=device)
    pos_mask = pair_relation_label > 0.5
    # Use -1e4 instead of -1e9 to avoid fp16 overflow (max fp16 ~65504)
    _NEG_INF = -1e4
    sim = sim.masked_fill(eye, _NEG_INF)
    # Mask same-positive pairs
    for i in range(B):
        if pos_mask[i]:
            sim[i] = sim[i].masked_fill(pos_mask, _NEG_INF)
            sim[i, i] = _NEG_INF

    # Hardest negative = highest similarity among non-positives
    hard_neg_idx = sim.argmax(dim=-1)  # (B,)
    return hard_neg_idx


def compute_negative_pair_indices(
    pair_group_id: torch.Tensor,
    domain_id: torch.Tensor,
    pair_domain_id: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute indices of cross-group cross-domain negative pairs."""
    batch_size = pair_group_id.shape[0]
    # (B, B) mask: True where candidate j is from a different group than anchor i
    diff_group = pair_group_id.unsqueeze(0) != pair_group_id.unsqueeze(1)  # (B, B)
    # (B, B) mask: True where candidate j's pair_domain differs from anchor i's domain
    cross_domain = pair_domain_id.unsqueeze(0) != domain_id.unsqueeze(1)  # (B, B)
    # Prefer cross-domain negatives; fall back to any different-group negative
    valid_cross = diff_group & cross_domain  # (B, B)
    # For rows with no cross-domain candidate, fall back to diff_group only
    has_cross = valid_cross.any(dim=1)  # (B,)
    valid_mask = torch.where(
        has_cross.unsqueeze(1), valid_cross, diff_group
    )  # (B, B)
    col_indices = torch.arange(batch_size, device=device).unsqueeze(0).expand(batch_size, -1)  # (B, B)
    ranked = torch.where(valid_mask, col_indices.float(), float(batch_size))
    negative_indices = ranked.argmin(dim=1)  # (B,)
    # Mark rows with no valid candidate at all
    no_valid = ~valid_mask.any(dim=1)
    negative_indices[no_valid] = -1
    return negative_indices


def compute_pair_iso_loss(
    training_invariant: torch.Tensor,
    iso_target: torch.Tensor,
    batch: Dict[str, Any],
    negative_margin: float,
    device: torch.device,
    has_pairs: bool,
) -> torch.Tensor:
    """Pair ISO loss with optional weighted positive/negative handling."""
    if not has_pairs:
        return compute_iso_loss(training_invariant, iso_target)

    pair_relation_label = batch.get("pair_relation_label")
    pair_weight = batch.get("pair_weight")
    if pair_relation_label is None or pair_weight is None:
        return compute_iso_loss(training_invariant, iso_target)

    pair_relation_label = pair_relation_label.to(
        device, dtype=training_invariant.dtype
    )
    pair_weight = pair_weight.to(device, dtype=training_invariant.dtype)
    per_sample_mse = F.mse_loss(
        training_invariant, iso_target, reduction="none"
    ).mean(dim=-1)
    positive_mask = pair_relation_label > 0.5
    negative_mask = ~positive_mask
    losses = []
    if positive_mask.any():
        positive_loss = (
            per_sample_mse[positive_mask] * pair_weight[positive_mask]
        ).sum() / pair_weight[positive_mask].sum()
        losses.append(positive_loss)
    if negative_mask.any():
        negative_penalty = F.relu(
            negative_margin - per_sample_mse[negative_mask]
        )
        weighted_negative = (
            negative_penalty * pair_weight[negative_mask]
        ).sum() / pair_weight[negative_mask].sum()
        losses.append(weighted_negative)
    if not losses:
        return per_sample_mse.mean()
    return torch.stack(losses).mean()


def compute_diversity_loss(
    exported_invariant: torch.Tensor,
    lambda_diversity_var: float,
    lambda_diversity_ortho: float,
) -> torch.Tensor:
    """Anti-collapse: encourage variance in invariant vectors.

    Prevents the degenerate solution where all exported_invariants
    converge to the same constant vector.

    Uses two penalties:
    1. Variance penalty: -mean(||x_i - mean||^2) — push vectors apart
    2. Orthogonality bonus: encourage non-zero pairwise angles

    Returns zero loss when batch is too small (< 4) to be meaningful.
    """
    B = exported_invariant.shape[0]
    if B < 4:
        return _zero_loss(exported_invariant)

    x = F.normalize(exported_invariant.float(), dim=-1)

    mean_vec = x.mean(dim=0, keepdim=True)
    variance = ((x - mean_vec) ** 2).sum(dim=-1).mean()

    cos_matrix = x @ x.T
    eye = torch.eye(B, device=x.device, dtype=torch.bool)
    off_diag = cos_matrix.masked_fill(eye, 0.0)
    avg_cosine = off_diag.abs().sum() / max(2 * (B * B - B), 1)

    loss_div = -lambda_diversity_var * variance + lambda_diversity_ortho * avg_cosine
    return loss_div


def compute_pair_ranking_loss(
    exported_invariant: torch.Tensor,
    pair_exported_target: torch.Tensor | None,
    batch: Dict[str, Any],
    config: LossConfig,
    effective_temp: float,
    device: torch.device,
    has_pairs: bool,
) -> torch.Tensor:
    """Pair ranking loss — InfoNCE, Focal-InfoNCE, or legacy margin."""
    if not has_pairs or pair_exported_target is None:
        return _zero_loss(exported_invariant)

    pair_relation_label = batch.get("pair_relation_label")
    pair_weight = batch.get("pair_weight")
    pair_group_id = _resolve_pair_group_id(batch, device)
    if pair_relation_label is None or pair_weight is None:
        return _zero_loss(exported_invariant)

    pair_relation_label = pair_relation_label.to(
        device, dtype=exported_invariant.dtype
    )
    pair_weight = pair_weight.to(device, dtype=exported_invariant.dtype)

    # InfoNCE path (use_infonce=True by default via ranking_margin <= 0)
    if config.use_infonce:
        use_focal = config.focal_gamma < 1.0
        # Hard negative mining
        use_hard_neg = config.use_hard_negatives
        if use_hard_neg and exported_invariant.shape[0] >= 4:
            hard_neg_idx = mine_hard_negatives(
                exported_invariant, pair_exported_target, pair_relation_label, device
            )
            hard_pair_target = pair_exported_target[hard_neg_idx]
            hard_labels = torch.zeros_like(pair_relation_label)
            hard_weights = pair_weight * 0.5
            aug_src = torch.cat([exported_invariant, exported_invariant[pair_relation_label > 0.5]], dim=0)
            aug_tgt = torch.cat([pair_exported_target, hard_pair_target[pair_relation_label > 0.5]], dim=0)
            aug_lbl = torch.cat([pair_relation_label, hard_labels[pair_relation_label > 0.5]], dim=0)
            aug_w   = torch.cat([pair_weight, hard_weights[pair_relation_label > 0.5]], dim=0)
            if use_focal:
                infonce = compute_focal_infonce_loss(aug_src, aug_tgt, aug_lbl, aug_w, temperature=effective_temp, gamma=config.focal_gamma)
            else:
                infonce = compute_infonce_loss(aug_src, aug_tgt, aug_lbl, aug_w, temperature=effective_temp, pair_group_id=None)
        else:
            if use_focal:
                infonce = compute_focal_infonce_loss(
                    exported_invariant, pair_exported_target,
                    pair_relation_label, pair_weight,
                    temperature=effective_temp, gamma=config.focal_gamma,
                )
            else:
                infonce = compute_infonce_loss(
                    exported_invariant, pair_exported_target,
                    pair_relation_label, pair_weight,
                    temperature=effective_temp,
                    pair_group_id=pair_group_id,
                )
        return infonce

    # Legacy ranking margin fallback
    if pair_group_id is None:
        return _zero_loss(exported_invariant)

    pair_domain_id = batch["pair_domain_id"].to(device)
    domain_id = batch["domain_id"].to(device)

    positive_mask = pair_relation_label > 0.5
    if not positive_mask.any() or exported_invariant.shape[0] < 2:
        return _zero_loss(exported_invariant)

    source_normalized = F.normalize(exported_invariant.float(), dim=-1)  # fp32 for AMP safety
    target_normalized = F.normalize(pair_exported_target.float(), dim=-1)
    similarities = source_normalized @ target_normalized.transpose(0, 1)
    anchor_indices = torch.arange(exported_invariant.shape[0], device=device)
    negative_indices = compute_negative_pair_indices(
        pair_group_id,
        domain_id,
        pair_domain_id,
        device,
    )
    valid_mask = positive_mask & (negative_indices >= 0)
    if not valid_mask.any():
        return _zero_loss(exported_invariant)

    positive_scores = similarities[
        anchor_indices[valid_mask], anchor_indices[valid_mask]
    ]
    negative_scores = similarities[
        anchor_indices[valid_mask],
        negative_indices[valid_mask],
    ]
    margin_loss = F.relu(config.ranking_margin - positive_scores + negative_scores)
    weighted_loss = margin_loss * pair_weight[valid_mask]
    return weighted_loss.sum() / pair_weight[valid_mask].sum().clamp_min(1e-8)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _resolve_pair_group_id(batch: Dict[str, Any], device: torch.device) -> torch.Tensor | None:
    pair_group_id = batch.get("pair_group_id")
    if pair_group_id is None:
        pair_group_id = batch.get("pair_family_id")
    if pair_group_id is None:
        return None
    return pair_group_id.to(device)


def _compute_router_diagnostics(
    aux_state: HDIMAuxState,
) -> Dict[str, torch.Tensor]:
    return {
        "routing_entropy": aux_state.routing_entropy,
        "expert_usage": aux_state.expert_usage,
        "topk_idx": aux_state.topk_idx,
        "topk_gate_weights": aux_state.topk_gate_weights,
        "train_scores_snapshot": aux_state.train_scores_snapshot,
    }


# ---------------------------------------------------------------------------
# Main entry point — mirrors _compute_batch_losses exactly
# ---------------------------------------------------------------------------

def compute_batch_losses(
    batch: Dict[str, Any],
    output: torch.Tensor,
    recon_target: torch.Tensor,
    aux_state: HDIMAuxState,
    routing_weights: torch.Tensor,
    invariant: torch.Tensor,
    iso_target: torch.Tensor,
    pair_exported_target: torch.Tensor | None,
    config: LossConfig,
    device: torch.device,
    current_epoch: int,
    log_temp: nn.Parameter | None,
    model: Any,
) -> Dict[str, torch.Tensor]:
    """Compute all loss components from a forward pass result.

    This is the extracted version of HDIMTrainer._compute_batch_losses.
    The caller (trainer) handles the forward pass and passes results here.

    Args:
        batch: The input batch dict.
        output: Model output tensor.
        recon_target: Reconstruction target tensor.
        aux_state: HDIMAuxState from forward pass.
        routing_weights: Routing weights from forward pass.
        invariant: Invariant from forward pass.
        iso_target: ISO target tensor (already computed by trainer).
        pair_exported_target: Exported invariant for pair target (or None).
        config: Loss hyperparameters.
        device: torch device.
        current_epoch: Current training epoch.
        log_temp: Learnable temperature parameter (or None).
        model: The HDIMModel instance (for expert_ortho_loss).
    """
    training_invariant = aux_state.training_invariant
    # --- Effective temperature ---
    temp = effective_temperature(
        log_temp=log_temp,
        temp_schedule=config.temp_schedule,
        current_epoch=current_epoch,
        infonce_temperature=config.infonce_temperature,
        tau_max=config.tau_max,
        tau_min=config.tau_min,
        temp_schedule_T_0=config.temp_schedule_T_0,
    )
    if config.use_sc_temperature:
        temp = cluster_scaled_temperature(aux_state.exported_invariant, temp)

    # --- Pair loss ---
    has_pairs = (
        ("pair_encoding" in batch and "pair_domain_id" in batch) or
        (isinstance(batch.get("pair_text"), Sequence) and not isinstance(batch.get("pair_text"), (str, bytes)) and "pair_domain_id" in batch)
    )
    loss_pair = compute_pair_ranking_loss(
        aux_state.exported_invariant,
        pair_exported_target,
        batch,
        config,
        effective_temp=temp,
        device=device,
        has_pairs=has_pairs,
    )

    # --- Core losses ---
    loss_recon = compute_reconstruction_loss(output, recon_target)
    loss_routing = aux_state.router_loss
    pair_relation_label = batch.get("pair_relation_label")
    pair_weight = batch.get("pair_weight")
    loss_memory = aux_state.memory_loss

    # Router z-loss (ST-MoE stability)
    loss_z = aux_state.z_loss

    # Matryoshka multi-scale loss in exported_invariant space (not SBERT)
    loss_matryoshka = _zero_loss(training_invariant)
    matryoshka_embs = batch.get("matryoshka_embeddings")
    if matryoshka_embs is not None and isinstance(matryoshka_embs, dict):
        matryoshka_target = None
        _exported_src = aux_state.exported_invariant
        _exported_dim = _exported_src.shape[-1]
        _m_dims = [d for d in sorted(matryoshka_embs.keys()) if d < _exported_dim]
        if _m_dims and pair_exported_target is not None:
            matryoshka_embs = {d: _exported_src[..., :d].contiguous() for d in _m_dims}
            matryoshka_target = {d: pair_exported_target[..., :d].contiguous() for d in _m_dims}
        loss_matryoshka = compute_matryoshka_loss(
            matryoshka_embs,
            batch,
            effective_temp=temp,
            matryoshka_target=matryoshka_target,
        )

    # Phase 26: Expert orthogonalization loss
    loss_expert_ortho = _zero_loss(loss_recon)
    if config.lambda_expert_ortho > 0 and hasattr(model, 'compute_expert_ortho_loss'):
        loss_expert_ortho = model.compute_expert_ortho_loss()

    # --- Total loss assembly (ORDER MATTERS for gradient dynamics) ---
    loss_total = (
        loss_recon
        + config.lambda_pair * loss_pair
        + config.lambda_routing * loss_routing
        + config.lambda_memory * loss_memory
        + config.lambda_z * loss_z
        + config.lambda_matryoshka * loss_matryoshka
        + config.lambda_expert_ortho * loss_expert_ortho
        + config.lambda_online * aux_state.online_loss
    )

    # Pairwise auxiliary losses (added directly with their own lambdas)
    pair_group_id = _resolve_pair_group_id(batch, device)
    if pair_relation_label is not None and pair_weight is not None and pair_exported_target is not None:
        _prl_typed = pair_relation_label.to(device, dtype=training_invariant.dtype)
        _pw_typed = pair_weight.to(device, dtype=training_invariant.dtype)
        if config.lambda_dcl > 0:
            loss_dcl = compute_dcl_loss(
                aux_state.exported_invariant, pair_exported_target,
                _prl_typed, _pw_typed,
                temperature=temp,
                device=device,
            )
            loss_total = loss_total + config.lambda_dcl * loss_dcl
        if config.lambda_uniformity > 0:
            loss_uniformity = compute_uniformity_alignment_loss(
                aux_state.exported_invariant, pair_exported_target,
                _prl_typed,
                device=device,
            )
            loss_total = loss_total + config.lambda_uniformity * loss_uniformity

    # NaN protection: zero ALL loss components with correct dtype
    if torch.isnan(loss_total) or torch.isinf(loss_total):
        _components = {
            "loss_recon": loss_recon, "loss_pair": loss_pair,
            "loss_routing": loss_routing, "loss_memory": loss_memory,
            "loss_z": loss_z, "loss_matryoshka": loss_matryoshka,
            "loss_expert_ortho": loss_expert_ortho, "loss_online": aux_state.online_loss,
        }
        _nan_names = [k for k, v in _components.items() if torch.isnan(v) or torch.isinf(v)]
        logger.warning("[NaN guard] NaN/Inf in loss components: %s", _nan_names)
        batch_losses = {k: _zero_loss(loss_total) for k in _components}
        batch_losses["_nan_skip"] = True
        return batch_losses

    batch_losses = {
        "loss_total": loss_total,
        "loss_recon": loss_recon,
        "loss_pair": loss_pair,
        "loss_routing": loss_routing,
        "loss_memory": loss_memory,
        "loss_matryoshka": loss_matryoshka,
        "loss_expert_ortho": loss_expert_ortho,
        "loss_online": aux_state.online_loss,
        "loss_z": loss_z,
        "routing_weights": routing_weights,
        "invariant": invariant,
        "raw_invariant": aux_state.raw_invariant,
        "memory_augmented_invariant": aux_state.memory_augmented_invariant,
        "exported_invariant": aux_state.exported_invariant,
        "training_invariant": training_invariant,
    }
    if pair_relation_label is not None:
        batch_losses["pair_relation_label"] = pair_relation_label.to(
            device, dtype=training_invariant.dtype
        )
    if pair_weight is not None:
        batch_losses["pair_weight"] = pair_weight.to(
            device, dtype=training_invariant.dtype
        )
    if has_pairs:
        batch_losses["iso_target"] = iso_target
    batch_losses.update(_compute_router_diagnostics(aux_state))
    return batch_losses
