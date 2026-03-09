"""
HDIM Quality Metrics:
- STS (Structural Transfer Score) — косинусное сходство инвариантов
- DRS (Domain Routing Stability) — стабильность роутера при повторных вызовах
- AFR (Analogy Feasibility Rate) — процент корректных аналогий
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple


def _model_device(model) -> torch.device:
    """Возвращает устройство модели для согласованного metric evaluation."""
    return next(model.parameters()).device


def structural_transfer_score(
    inv_source: torch.Tensor,
    inv_target: torch.Tensor,
) -> torch.Tensor:
    """STS: косинусное сходство между инвариантами двух доменов.
    Высокий STS = хорошее сохранение структуры при переносе.

    Args:
        inv_source: (B, D) инвариант исходного домена
        inv_target: (B, D) инвариант целевого домена
    Returns:
        scalar: среднее косинусное сходство
    """
    return F.cosine_similarity(inv_source, inv_target, dim=-1).mean()


def domain_routing_stability(
    routing_weights_list: List[torch.Tensor],
) -> torch.Tensor:
    """DRS: стабильность роутера — стандартное отклонение весов по нескольким прогонам.
    Низкий DRS = стабильная маршрутизация.

    Args:
        routing_weights_list: список тензоров (B, E) из нескольких прогонов
    Returns:
        scalar: среднее std по экспертам
    """
    stacked = torch.stack(routing_weights_list, dim=0)  # (T, B, E)
    return stacked.std(dim=0).mean()


def _paired_batch_metrics(model, batch) -> Tuple[List[float], torch.Tensor]:
    device = _model_device(model)
    enc: torch.Tensor = batch["encoding"].to(device)
    domain_ids: torch.Tensor = batch["domain_id"].to(device)
    pair_enc = batch.get("pair_encoding")
    pair_domain_ids = batch.get("pair_domain_id")

    if pair_enc is None or pair_domain_ids is None:
        _, routing, _, state = model(
            enc,
            domain_id=domain_ids,
            return_state=True,
            update_memory=False,
            memory_mode="retrieve",
        )
        sts_values = structural_transfer_score(
            state.exported_invariant,
            state.memory_augmented_invariant,
        ).repeat(len(enc))
        return sts_values.tolist(), routing.cpu()

    pair_enc = pair_enc.to(device)
    pair_domain_ids = pair_domain_ids.to(device)
    _, routing, _, src_state = model.transfer_pairs(
        enc,
        domain_ids,
        pair_domain_ids,
        update_memory=False,
        memory_mode="retrieve",
    )
    _, _, _, tgt_state = model(
        pair_enc,
        domain_id=pair_domain_ids,
        return_state=True,
        update_memory=False,
        memory_mode="retrieve",
    )
    sts_values = F.cosine_similarity(
        src_state.exported_invariant,
        tgt_state.exported_invariant,
        dim=-1,
    )
    return sts_values.tolist(), routing.cpu()


def analogy_feasibility_rate(
    model,
    test_pairs: List[Tuple[torch.Tensor, int, int]],
    threshold: float = 0.5,
) -> float:
    """AFR: процент пар где перенос даёт STS > threshold."""
    if not test_pairs:
        return 0.0

    model.eval()
    device = _model_device(model)
    correct = 0
    with torch.no_grad():
        for enc, src, tgt in test_pairs:
            enc = enc.to(device)
            if enc.dim() == 1:
                enc = enc.unsqueeze(0)

            src_domain_tensor = torch.tensor([src], dtype=torch.long, device=device)
            tgt_domain_tensor = torch.tensor([tgt], dtype=torch.long, device=device)
            _, _, inv_src = model(
                enc,
                domain_id=src_domain_tensor,
                update_memory=False,
                memory_mode="retrieve",
            )
            transferred = model.transfer(
                enc,
                source_domain=src,
                target_domain=tgt,
                update_memory=False,
                memory_mode="retrieve",
            )
            _, _, inv_tgt = model(
                transferred,
                domain_id=tgt_domain_tensor,
                update_memory=False,
                memory_mode="retrieve",
            )
            sts = structural_transfer_score(inv_src, inv_tgt)
            if sts.item() > threshold:
                correct += 1

    return correct / len(test_pairs)


def compute_all_metrics(
    model,
    dataloader,
    num_routing_runs: int = 3,
) -> dict:
    """Compute STS, DRS, AFR metrics on a full dataloader."""
    model.eval()
    sts_scores: List[float] = []
    routing_runs: List[List[torch.Tensor]] = [[] for _ in range(num_routing_runs)]

    with torch.no_grad():
        for batch in dataloader:
            for run_idx in range(num_routing_runs):
                _, routing = _paired_batch_metrics(model, batch)
                routing_runs[run_idx].append(routing)

            batch_sts_scores, _ = _paired_batch_metrics(model, batch)
            sts_scores.extend(batch_sts_scores)

    routing_weights_list = [torch.cat(run, dim=0) for run in routing_runs if run]
    mean_sts = sum(sts_scores) / len(sts_scores) if sts_scores else 0.0
    if len(routing_weights_list) >= 2:
        drs = domain_routing_stability(routing_weights_list).item()
    else:
        drs = 0.0
    afr = sum(1 for s in sts_scores if s > 0.3) / len(sts_scores) if sts_scores else 0.0

    return {
        "STS": mean_sts,
        "DRS": drs,
        "AFR": afr,
    }
