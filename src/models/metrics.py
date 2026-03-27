"""
HDIM Quality Metrics:
- STS_exported — косинусное сходство canonical transfer invariants
- STS_training — косинусное сходство training-facing invariants
- DRS (Domain Routing Stability) — стабильность роутера при повторных вызовах
- AFR (Analogy Feasibility Rate) — доля aligned pairs, проходящих margin-aware проверку
"""
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def _model_device(model) -> torch.device:
    """Возвращает устройство модели для согласованного metric evaluation."""
    return next(model.parameters()).device


def domain_routing_stability(
    routing_weights_list: List[torch.Tensor],
) -> torch.Tensor:
    """DRS: стабильность роутера — стандартное отклонение весов по нескольким прогонам."""
    stacked = torch.stack(routing_weights_list, dim=0)
    return stacked.std(dim=0).mean()


def _compute_negative_pair_indices(batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    pair_group_id = batch["pair_group_id"]
    pair_domain_id = batch["pair_domain_id"]
    source_domain_id = batch["domain_id"]
    batch_size = pair_group_id.shape[0]
    if batch_size < 2:
        return None
    batch_indices = torch.arange(batch_size, device=pair_group_id.device)
    negative_indices = torch.empty(batch_size, device=pair_group_id.device, dtype=torch.long)

    for idx in range(batch_size):
        valid_candidates = batch_indices[pair_group_id != pair_group_id[idx]]
        cross_domain_candidates = valid_candidates[
            pair_domain_id[valid_candidates] != source_domain_id[idx]
        ]
        if cross_domain_candidates.numel() > 0:
            negative_indices[idx] = cross_domain_candidates[0]
            continue
        if valid_candidates.numel() == 0:
            return None
        negative_indices[idx] = valid_candidates[0]

    return negative_indices


def _get_encodings(model, batch, device):
    """Extract or compute encodings from batch, supporting text-only batches."""
    enc = batch.get("encoding")
    pair_enc = batch.get("pair_encoding")
    pair_domain_ids = batch.get("pair_domain_id")
    domain_ids = batch["domain_id"].to(device)

    # Text-only batch (RealPairsDataset / TextHDIMModel)
    if enc is None and hasattr(model, 'encode_texts'):
        texts = batch.get("text")
        pair_texts = batch.get("pair_text")
        if texts is not None:
            enc = model.encode_texts(list(texts), device=device).detach()
        if pair_texts is not None:
            pair_enc = model.encode_texts(list(pair_texts), device=device).detach()
    elif enc is not None:
        enc = enc.to(device)
        if pair_enc is not None:
            pair_enc = pair_enc.to(device)

    if pair_domain_ids is not None:
        pair_domain_ids = pair_domain_ids.to(device)

    return enc, pair_enc, domain_ids, pair_domain_ids


def _paired_batch_metrics(model, batch) -> Dict[str, torch.Tensor]:
    device = _model_device(model)
    enc, pair_enc, domain_ids, pair_domain_ids = _get_encodings(model, batch, device)

    if enc is None:
        # Cannot compute metrics without encodings
        dummy = torch.zeros(1, device=device)
        return {"sts_exported": dummy, "sts_training": dummy, "pair_margin": dummy, "routing": dummy}

    if pair_enc is None or pair_domain_ids is None:
        _, routing, _, _, state = model(
            enc,
            domain_id=domain_ids,
            return_state=True,
            update_memory=False,
            memory_mode="retrieve",
        )
        src_norm = F.normalize(state.exported_invariant, dim=-1)
        tgt_norm = F.normalize(state.memory_augmented_invariant, dim=-1)
        sts_exported = (src_norm * tgt_norm).sum(dim=-1)
        sts_training = F.cosine_similarity(
            state.training_invariant,
            model.training_inv_head(state.memory_augmented_invariant),
            dim=-1,
        )
        return {
            "sts_exported": sts_exported,
            "sts_training": sts_training,
            "pair_margin": torch.zeros_like(sts_exported),
            "routing": routing,
        }

    _, routing, _, _, src_state = model.transfer_pairs(
        enc,
        domain_ids,
        pair_domain_ids,
        update_memory=False,
        memory_mode="retrieve",
    )
    _, _, _, _, tgt_state = model(
        pair_enc,
        domain_id=pair_domain_ids,
        return_state=True,
        update_memory=False,
        memory_mode="retrieve",
    )
    negative_pair_indices = _compute_negative_pair_indices(batch)

    src_norm = F.normalize(src_state.exported_invariant, dim=-1)
    tgt_norm = F.normalize(tgt_state.exported_invariant, dim=-1)
    sts_exported = (src_norm * tgt_norm).sum(dim=-1)
    sts_training = F.cosine_similarity(
        src_state.training_invariant,
        tgt_state.training_invariant,
        dim=-1,
    )

    if negative_pair_indices is None:
        pair_margin = torch.zeros_like(sts_exported)
    else:
        mismatched_pair_enc = pair_enc[negative_pair_indices]
        mismatched_pair_domain_ids = pair_domain_ids[negative_pair_indices]
        _, _, _, _, mismatched_state = model(
            mismatched_pair_enc,
            domain_id=mismatched_pair_domain_ids,
            return_state=True,
            update_memory=False,
            memory_mode="retrieve",
        )
        mismatched_scores = F.cosine_similarity(
            src_state.exported_invariant,
            mismatched_state.exported_invariant,
            dim=-1,
        )
        pair_margin = sts_exported - mismatched_scores
    return {
        "sts_exported": sts_exported,
        "sts_training": sts_training,
        "pair_margin": pair_margin,
        "routing": routing,
    }


def analogy_feasibility_rate(
    aligned_scores: torch.Tensor,
    pair_margins: torch.Tensor,
    *,
    similarity_threshold: float = 0.5,
    margin_threshold: float = 0.0,
) -> float:
    """AFR: доля aligned pairs, где similarity и pair margin проходят пороги."""
    if aligned_scores.numel() == 0:
        return 0.0
    feasible = (aligned_scores > similarity_threshold) & (pair_margins > margin_threshold)
    return feasible.float().mean().item()


def compute_all_metrics(
    model,
    dataloader,
    num_routing_runs: int = 3,
    *,
    afr_similarity_threshold: float = 0.3,
    afr_margin_threshold: float = 0.0,
) -> dict:
    """Compute STS_exported, STS_training, DRS, AFR, and pair margin metrics.

    pair_margin вычисляется глобально по всему датасету, чтобы избежать
    проблемы однородных батчей (все образцы одной группы).
    """
    model.eval()
    device = _model_device(model)

    # Собираем глобальные инварианты для pair_margin
    all_src_inv: List[torch.Tensor] = []
    all_tgt_inv: List[torch.Tensor] = []
    all_group_ids: List[torch.Tensor] = []
    all_domain_ids: List[torch.Tensor] = []
    all_pair_domain_ids: List[torch.Tensor] = []
    all_pair_labels: List[torch.Tensor] = []
    sts_exported_scores: List[torch.Tensor] = []
    sts_training_scores: List[torch.Tensor] = []
    routing_runs: List[List[torch.Tensor]] = [[] for _ in range(num_routing_runs)]

    with torch.no_grad():
        for batch in dataloader:
            for run_idx in range(num_routing_runs):
                metrics = _paired_batch_metrics(model, batch)
                routing_runs[run_idx].append(metrics["routing"].cpu())

            batch_metrics = _paired_batch_metrics(model, batch)
            sts_exported_scores.append(batch_metrics["sts_exported"].cpu())
            sts_training_scores.append(batch_metrics["sts_training"].cpu())

            # Собираем данные для глобального pair_margin
            enc_m, pair_enc_m, d_ids, pd_ids = _get_encodings(model, batch, device)
            if enc_m is not None and pair_enc_m is not None and pd_ids is not None:
                _, _, _, _, src_state = model.transfer_pairs(
                    enc_m, d_ids, pd_ids, update_memory=False, memory_mode="retrieve"
                )
                _, _, _, _, tgt_state = model(
                    pair_enc_m, domain_id=pd_ids, return_state=True, update_memory=False, memory_mode="retrieve"
                )
                all_src_inv.append(src_state.exported_invariant.cpu())
                all_tgt_inv.append(tgt_state.exported_invariant.cpu())
                if "pair_group_id" in batch:
                    all_group_ids.append(batch["pair_group_id"].cpu())
                if "pair_relation_label" in batch:
                    all_pair_labels.append(batch["pair_relation_label"].cpu())
                all_domain_ids.append(d_ids.cpu())
                all_pair_domain_ids.append(pd_ids.cpu())

    routing_weights_list = [torch.cat(run, dim=0) for run in routing_runs if run]
    all_sts_exported = torch.cat(sts_exported_scores) if sts_exported_scores else torch.empty(0)
    all_sts_training = torch.cat(sts_training_scores) if sts_training_scores else torch.empty(0)

    # Глобальный pair_margin — используем labeled positive/negative пары
    mean_pair_margin = 0.0
    if all_src_inv and all_tgt_inv:
        src = F.normalize(torch.cat(all_src_inv), dim=-1)
        tgt = F.normalize(torch.cat(all_tgt_inv), dim=-1)
        pair_labels_cat = torch.cat(all_pair_labels) if all_pair_labels else None
        if pair_labels_cat is not None and pair_labels_cat.shape[0] == src.shape[0]:
            pos_mask = pair_labels_cat > 0.5
            neg_mask = ~pos_mask
            diag_sim = (src * tgt).sum(dim=-1)  # per-sample similarity
            if pos_mask.any() and neg_mask.any():
                pos_mean = diag_sim[pos_mask].mean().item()
                neg_mean = diag_sim[neg_mask].mean().item()
                mean_pair_margin = pos_mean - neg_mean
            elif pos_mask.any():
                # Only positives: compare against global negatives (other groups)
                groups = torch.cat(all_group_ids) if all_group_ids else None
                if groups is not None:
                    pos_scores = diag_sim[pos_mask]
                    # Find cross-group negatives for each positive
                    neg_scores_list = []
                    pos_indices = pos_mask.nonzero(as_tuple=True)[0]
                    for pi in pos_indices:
                        neg_mask2 = groups != groups[pi]
                        if neg_mask2.any():
                            neg_sims = (src[pi:pi+1] * tgt[neg_mask2]).sum(dim=-1)
                            neg_scores_list.append(neg_sims.mean())
                    if neg_scores_list:
                        neg_mean = torch.stack(neg_scores_list).mean().item()
                        mean_pair_margin = pos_scores.mean().item() - neg_mean
        elif src.shape[0] > 0:
            mean_pair_margin = 0.0

    mean_sts_exported = all_sts_exported.mean().item() if all_sts_exported.numel() else 0.0
    mean_sts_training = all_sts_training.mean().item() if all_sts_training.numel() else 0.0
    if len(routing_weights_list) >= 2:
        drs = domain_routing_stability(routing_weights_list).item()
    else:
        drs = 0.0
    afr = analogy_feasibility_rate(
        all_sts_exported,
        torch.full_like(all_sts_exported, mean_pair_margin),
        similarity_threshold=afr_similarity_threshold,
        margin_threshold=afr_margin_threshold,
    )

    return {
        "STS_exported": mean_sts_exported,
        "STS_training": mean_sts_training,
        "DRS": drs,
        "AFR": afr,
        "pair_margin": mean_pair_margin,
    }

