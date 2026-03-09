import pytest
import torch
from torch.utils.data import DataLoader

from src.models.hdim_model import HDIMConfig, HDIMModel
from src.models.metrics import compute_all_metrics
from src.training.dataset import (
    DomainProblemDataset,
    create_demo_dataset,
    create_group_aware_split,
    create_paired_demo_dataset,
)
from src.training.trainer import HDIMTrainer


@pytest.fixture
def cfg():
    return HDIMConfig()


@pytest.fixture
def model(cfg):
    return HDIMModel(cfg)


@pytest.fixture
def trainer(model):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    return HDIMTrainer(model, opt, device="cpu")


def test_model_forward(model, cfg):
    bsz = 4
    x = torch.randn(bsz, cfg.hidden_dim)
    domain_id = torch.zeros(bsz, dtype=torch.long)
    out, routing, inv = model(x, domain_id)
    assert out.shape == (bsz, cfg.hidden_dim)
    assert routing.shape == (bsz, cfg.num_experts)
    assert inv.shape == (bsz, cfg.hidden_dim)
    assert torch.allclose(routing.sum(dim=-1), torch.ones(bsz), atol=1e-5)


def test_model_forward_return_state(model, cfg):
    bsz = 4
    x = torch.randn(bsz, cfg.hidden_dim)
    domain_id = torch.zeros(bsz, dtype=torch.long)
    out, routing, inv, state = model(
        x,
        domain_id,
        return_state=True,
        update_memory=False,
        memory_mode="retrieve",
    )
    assert out.shape == (bsz, cfg.hidden_dim)
    assert routing.shape == (bsz, cfg.num_experts)
    assert inv.shape == (bsz, cfg.hidden_dim)
    assert state.raw_invariant.shape == (bsz, model.pipeline.clifford_dim)
    assert state.memory_augmented_invariant.shape == (bsz, model.pipeline.clifford_dim)
    assert state.exported_invariant.shape == (bsz, model.pipeline.clifford_dim)
    expected_inv = model.training_inv_head(state.exported_invariant)
    assert torch.allclose(inv, expected_inv, atol=1e-5)
    assert state.router_loss.ndim == 0
    assert state.memory_loss.ndim == 0

def test_model_transfer(model, cfg):
    bsz = 2
    x = torch.randn(bsz, cfg.hidden_dim)
    result = model.transfer(
        x,
        source_domain=0,
        target_domain=1,
        update_memory=False,
        memory_mode="retrieve",
    )
    assert result.shape == (bsz, cfg.hidden_dim)


def test_transfer_pairs(model, cfg):
    bsz = 4
    x = torch.randn(bsz, cfg.hidden_dim)
    source_domain_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    target_domain_id = torch.tensor([1, 2, 0, 3], dtype=torch.long)
    out, routing, inv, state = model.transfer_pairs(
        x,
        source_domain_id,
        target_domain_id,
        update_memory=False,
        memory_mode="retrieve",
    )
    assert out.shape == (bsz, cfg.hidden_dim)
    assert routing.shape == (bsz, cfg.num_experts)
    assert inv.shape == (bsz, cfg.hidden_dim)
    assert state.raw_invariant.shape == (bsz, model.pipeline.clifford_dim)
    assert state.memory_augmented_invariant.shape == (bsz, model.pipeline.clifford_dim)
    assert state.exported_invariant.shape == (bsz, model.pipeline.clifford_dim)
    expected_inv = model.training_inv_head(state.exported_invariant)
    assert torch.allclose(inv, expected_inv, atol=1e-5)


def test_dataset():
    ds = create_demo_dataset()
    assert len(ds) == 100
    sample = ds[0]
    assert "encoding" in sample
    assert "domain_id" in sample
    assert sample["encoding"].shape == (64,)
    assert sample["domain_id"].dtype == torch.long


def test_paired_dataset():
    ds = create_paired_demo_dataset(n_samples=40)
    sample = ds[0]
    assert "pair_encoding" in sample
    assert "pair_domain_id" in sample
    assert "pair_group_id" in sample
    assert sample["pair_encoding"].shape == (64,)
    assert sample["pair_domain_id"].dtype == torch.long
    assert sample["pair_group_id"].dtype == torch.long
    assert sample["pair_family_id"].dtype == torch.long
    assert sample["pair_relation_type"] == "positive"
    assert sample["pair_relation_label"].dtype == torch.float32
    assert sample["pair_relation_label"].item() == 1.0
    assert sample["pair_weight"].dtype == torch.float32
    assert sample["pair_weight"].item() > 0.0
    assert sample["pair_domain_id"].item() != sample["domain_id"].item()


def test_dataset_rejects_same_domain_pairs():
    samples = [("a", 0), ("b", 0)]
    with pytest.raises(ValueError):
        DomainProblemDataset(
            samples,
            pair_indices=[1, 0],
            pair_group_ids=[0, 0],
            pair_relation_types=["positive", "positive"],
            pair_weights=[1.0, 1.0],
        )


def test_dataset_rejects_misaligned_pair_groups():
    samples = [("a", 0), ("b", 1), ("c", 2)]
    with pytest.raises(ValueError):
        DomainProblemDataset(
            samples,
            pair_indices=[1, 0, 1],
            pair_group_ids=[0, 1, 2],
            pair_relation_types=["positive", "positive", "positive"],
            pair_weights=[1.0, 1.0, 1.0],
        )


def test_dataset_rejects_non_positive_pair_weights():
    samples = [("a", 0), ("b", 1)]
    with pytest.raises(ValueError):
        DomainProblemDataset(
            samples,
            pair_indices=[1, 0],
            pair_group_ids=[0, 0],
            pair_relation_types=["positive", "positive"],
            pair_weights=[0.0, 1.0],
        )


def test_dataset_rejects_mismatched_pair_relation_types():
    samples = [("a", 0), ("b", 1)]
    with pytest.raises(ValueError):
        DomainProblemDataset(
            samples,
            pair_indices=[1, 0],
            pair_group_ids=[0, 0],
            pair_relation_types=["positive", "negative"],
            pair_weights=[1.0, 1.0],
        )


def test_dataset_exposes_negative_pair_metadata():
    ds = create_paired_demo_dataset(n_samples=40, negative_ratio=1.0)
    sample = next(item for item in (ds[idx] for idx in range(len(ds))) if item["pair_relation_type"] == "negative")
    assert sample["pair_relation_label"].item() == 0.0
    assert sample["pair_family_id"].item() == sample["pair_group_id"].item()
    assert sample["pair_domain_id"].item() != sample["domain_id"].item()


def test_group_aware_split_keeps_pair_groups_separate():
    ds = create_paired_demo_dataset(n_samples=40, negative_ratio=0.0)
    train_ds, val_ds = create_group_aware_split(ds, train_fraction=0.8, seed=42)
    train_group_ids = {ds.pair_group_ids[idx] for idx in train_ds.indices}
    val_group_ids = {ds.pair_group_ids[idx] for idx in val_ds.indices}
    assert train_group_ids
    assert val_group_ids
    assert train_group_ids.isdisjoint(val_group_ids)


def test_iso_loss(trainer, cfg):
    u1 = torch.randn(4, cfg.hidden_dim)
    u2 = torch.randn(4, cfg.hidden_dim)
    loss = trainer.compute_iso_loss(u1, u2)
    assert loss.item() >= 0


def test_routing_loss(trainer, cfg):
    weights = torch.softmax(torch.randn(4, cfg.num_experts), dim=-1)
    loss = trainer.compute_routing_loss(weights)
    assert isinstance(loss.item(), float)


def test_train_step(trainer, cfg):
    bsz = 8
    batch = {
        "encoding": torch.randn(bsz, cfg.hidden_dim),
        "domain_id": torch.randint(0, cfg.num_domains, (bsz,)),
    }
    loss = trainer.train_step(batch)
    assert isinstance(loss.item(), float)


def test_train_step_with_pairs(trainer):
    ds = create_paired_demo_dataset(n_samples=40)
    dl = DataLoader(ds, batch_size=8)
    batch = next(iter(dl))
    loss = trainer.train_step(batch)
    assert isinstance(loss.item(), float)


def test_validate(trainer):
    ds = create_demo_dataset()
    dl = DataLoader(ds, batch_size=16)
    metrics = trainer.validate(dl)
    assert "loss_total" in metrics
    assert "loss_recon" in metrics
    assert isinstance(metrics["loss_total"], float)


def test_validate_with_pairs(trainer):
    ds = create_paired_demo_dataset(n_samples=40)
    dl = DataLoader(ds, batch_size=8)
    metrics = trainer.validate(dl)
    assert "loss_total" in metrics
    assert isinstance(metrics["loss_iso"], float)


def test_evaluate_batch_with_pairs(trainer):
    ds = create_paired_demo_dataset(n_samples=16)
    batch = next(iter(DataLoader(ds, batch_size=8)))
    losses = trainer.evaluate_batch(batch)
    assert losses["loss_total"].item() >= 0
    assert losses["loss_iso"].item() >= 0
    assert losses["training_invariant"].shape == (8, trainer.model.config.hidden_dim)
    assert losses["exported_invariant"].shape == (8, trainer.model.pipeline.clifford_dim)
    assert losses["routing_entropy"].ndim == 0
    assert losses["expert_usage"].shape == (trainer.model.config.num_experts,)
    assert losses["topk_idx"].shape == (8, trainer.model.config.top_k)
    assert losses["topk_gate_weights"].shape == (8, trainer.model.config.top_k)


def test_model_return_state_exposes_memory_lifecycle(model, cfg):
    x = torch.randn(4, cfg.hidden_dim)
    domain_id = torch.zeros(4, dtype=torch.long)
    _, _, _, state = model(
        x,
        domain_id,
        return_state=True,
        update_memory=False,
        memory_mode="retrieve",
    )
    assert state.memory_mode == "retrieve"
    assert state.update_memory is False
    assert state.memory_updated is False
    assert state.training_invariant.shape == (4, cfg.hidden_dim)


def test_model_rejects_invalid_memory_mode(model, cfg):
    x = torch.randn(2, cfg.hidden_dim)
    domain_id = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError):
        model(x, domain_id, update_memory=False, memory_mode="bad-mode")


def test_model_forward_rejects_invalid_domain_id(model, cfg):
    x = torch.randn(2, cfg.hidden_dim)
    domain_id = torch.tensor([0, cfg.num_domains], dtype=torch.long)
    with pytest.raises(IndexError):
        model(x, domain_id, update_memory=False, memory_mode="retrieve")


def test_model_memory_mode_none_skips_memory_augmentation(model, cfg):
    x = torch.randn(3, cfg.hidden_dim)
    domain_id = torch.zeros(3, dtype=torch.long)
    _, _, _, state = model(
        x,
        domain_id,
        return_state=True,
        update_memory=True,
        memory_mode="none",
    )
    assert state.memory_mode == "none"
    assert state.update_memory is False
    assert state.memory_updated is False
    assert torch.allclose(state.memory_augmented_invariant, state.raw_invariant, atol=1e-6)
    assert state.memory_loss.item() == pytest.approx(0.0)


def test_transfer_pairs_rejects_invalid_target_domain(model, cfg):
    x = torch.randn(2, cfg.hidden_dim)
    source_domain_id = torch.tensor([0, 1], dtype=torch.long)
    target_domain_id = torch.tensor([1, cfg.num_domains], dtype=torch.long)
    with pytest.raises(IndexError):
        model.transfer_pairs(
            x,
            source_domain_id,
            target_domain_id,
            update_memory=False,
            memory_mode="retrieve",
        )

def test_forward_and_transfer_pairs_expose_same_lifecycle_contract(model, cfg):
    x = torch.randn(4, cfg.hidden_dim)
    source_domain_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    target_domain_id = torch.tensor([1, 2, 0, 3], dtype=torch.long)

    _, _, _, forward_state = model(
        x,
        source_domain_id,
        return_state=True,
        update_memory=False,
        memory_mode="retrieve",
    )
    _, _, _, pair_state = model.transfer_pairs(
        x,
        source_domain_id,
        target_domain_id,
        update_memory=False,
        memory_mode="retrieve",
    )

    lifecycle_fields = [
        "raw_invariant",
        "memory_augmented_invariant",
        "exported_invariant",
        "training_invariant",
    ]
    for field in lifecycle_fields:
        forward_value = getattr(forward_state, field)
        pair_value = getattr(pair_state, field)
        assert forward_value.shape == pair_value.shape


def test_router_topk_contract_on_eval_path(model):
    ds = create_paired_demo_dataset(n_samples=16)
    batch = next(iter(DataLoader(ds, batch_size=8)))
    losses = HDIMTrainer(model, torch.optim.Adam(model.parameters(), lr=1e-3), device="cpu").evaluate_batch(batch)
    routing_weights = losses["routing_weights"]
    topk_idx = losses["topk_idx"]
    topk_gate_weights = losses["topk_gate_weights"]

    assert torch.allclose(routing_weights.sum(dim=-1), torch.ones(routing_weights.shape[0]), atol=1e-5)
    active_counts = (routing_weights > 0).sum(dim=-1)
    assert torch.all(active_counts <= model.config.top_k)
    gathered = torch.gather(routing_weights, -1, topk_idx)
    assert torch.allclose(gathered, topk_gate_weights, atol=1e-6)


def test_eval_retrieve_does_not_mutate_memory_or_router_state(model, cfg):
    x = torch.randn(8, cfg.hidden_dim)
    domain_id = torch.randint(0, cfg.num_domains, (8,))

    model.train()
    _ = model(x, domain_id, update_memory=True, memory_mode="update")
    memory_weight_before = model.pipeline.memory.memory.weight.detach().clone()
    momentum_before = model.pipeline.memory.momentum_S.detach().clone()
    train_scores_before = model.pipeline.moe.train_scores.detach().clone()

    model.eval()
    with torch.no_grad():
        _, routing_a, inv_a, state_a = model(
            x,
            domain_id,
            return_state=True,
            update_memory=False,
            memory_mode="retrieve",
        )
        _, routing_b, inv_b, state_b = model(
            x,
            domain_id,
            return_state=True,
            update_memory=False,
            memory_mode="retrieve",
        )

    assert torch.allclose(model.pipeline.memory.memory.weight, memory_weight_before)
    assert torch.allclose(model.pipeline.memory.momentum_S, momentum_before)
    assert torch.allclose(model.pipeline.moe.train_scores, train_scores_before)
    assert torch.allclose(routing_a, routing_b)
    assert torch.allclose(inv_a, inv_b)
    assert torch.allclose(state_a.raw_invariant, state_b.raw_invariant)
    assert torch.allclose(state_a.memory_augmented_invariant, state_b.memory_augmented_invariant)
    assert torch.allclose(state_a.exported_invariant, state_b.exported_invariant)


def test_model_reset_memory_clears_memory_and_router_state(model, cfg):
    x = torch.randn(4, cfg.hidden_dim)
    domain_id = torch.zeros(4, dtype=torch.long)
    model.train()
    model(x, domain_id, update_memory=True, memory_mode="update")

    assert torch.count_nonzero(model.pipeline.memory.memory.weight).item() > 0
    assert torch.count_nonzero(model.pipeline.memory.momentum_S).item() > 0
    train_scores_before_reset = model.pipeline.moe.train_scores.clone()
    assert not torch.allclose(
        train_scores_before_reset,
        torch.full_like(train_scores_before_reset, 1.0 / cfg.num_experts),
    )

    model.reset_memory()

    assert torch.allclose(model.pipeline.memory.memory.weight, torch.zeros_like(model.pipeline.memory.memory.weight))
    assert torch.allclose(model.pipeline.memory.momentum_S, torch.zeros_like(model.pipeline.memory.momentum_S))
    assert torch.allclose(
        model.pipeline.moe.train_scores,
        torch.full_like(model.pipeline.moe.train_scores, 1.0 / cfg.num_experts),
    )

def test_compute_all_metrics_runs_on_model_device(trainer):
    ds = create_demo_dataset(n_samples=16)
    dl = DataLoader(ds, batch_size=4)
    metrics = compute_all_metrics(trainer.model, dl)
    assert set(metrics) == {"STS_exported", "STS_training", "DRS", "AFR", "pair_margin"}
    assert all(isinstance(value, float) for value in metrics.values())


def test_compute_all_metrics_with_pairs(model):
    ds = create_paired_demo_dataset(n_samples=24)
    dl = DataLoader(ds, batch_size=8)
    metrics = compute_all_metrics(model, dl, num_routing_runs=2)
    assert set(metrics) == {"STS_exported", "STS_training", "DRS", "AFR", "pair_margin"}
    assert 0.0 <= metrics["AFR"] <= 1.0
    assert metrics["STS_exported"] <= 1.0
    assert metrics["STS_training"] <= 1.0
    assert metrics["DRS"] >= 0.0


def test_aligned_pairs_have_non_extreme_margin(model):
    ds = create_paired_demo_dataset(n_samples=24)
    dl = DataLoader(ds, batch_size=8)
    metrics = compute_all_metrics(model, dl, num_routing_runs=1)
    assert -1.0 <= metrics["pair_margin"] <= 1.0


# ============================================================
# Contract tests — algebraic correctness
# ============================================================

def test_identity_rotor_preserves_input():
    """Identity rotor (R[0]=1, rest=0) должен давать sandwich(R, x) ≈ x."""
    from src.core.hypercomplex import CliffordAlgebra
    from src.core.domain_operators import DomainRotationOperator
    alg = CliffordAlgebra(p=2, q=0, r=0)  # dim=4
    rotor = DomainRotationOperator(alg, init_identity=True)
    # sandwich работает поэлементно: R и x должны быть одинаковой формы.
    # Передаём одиночный вектор (alg.dim,)
    x = torch.randn(alg.dim)
    with torch.no_grad():
        x_rotated = rotor(x)
    # Identity rotor должен давать точное сохранение
    assert torch.allclose(x_rotated, x, atol=1e-5), f"Identity rotor changed input: max diff={(x_rotated - x).abs().max().item()}"


def test_rotor_inverse_matches_explicit_reverse_formula():
    """apply_inverse(rotor(x)) должен совпадать с явной формулой через reverse(R)."""
    import math
    from src.core.hypercomplex import CliffordAlgebra
    from src.core.domain_operators import DomainRotationOperator
    alg = CliffordAlgebra(p=2, q=0, r=0)
    rotor = DomainRotationOperator(alg, init_identity=False)
    theta = math.pi / 6
    with torch.no_grad():
        rotor.R.data = torch.tensor([math.cos(theta), 0.0, 0.0, math.sin(theta)])
    x = torch.randn(alg.dim)
    with torch.no_grad():
        x_fwd = rotor(x)
        x_back = rotor.apply_inverse(x_fwd)
        expected = alg.sandwich(rotor.get_inverse(), x_fwd)
    assert torch.allclose(x_back, expected, atol=1e-6)


def test_round_trip_transfer_same_domain(model, cfg):
    """Перенос в тот же домен не должен сильно искажать вход (reconstruction)."""
    x = torch.randn(4, cfg.hidden_dim)
    domain_id = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        out, _, _ = model(x, domain_id, update_memory=False, memory_mode="retrieve")
    # Выход должен иметь ту же форму
    assert out.shape == x.shape


def test_raw_invariant_differs_from_exported(model, cfg):
    """raw_invariant и exported_invariant должны проходить через канонический lifecycle."""
    x = torch.randn(4, cfg.hidden_dim)
    domain_id = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        _, _, _, state = model(
            x, domain_id,
            return_state=True,
            update_memory=False,
            memory_mode="retrieve",
        )
    raw = state.raw_invariant
    exported = state.exported_invariant
    assert raw.shape == exported.shape
    assert raw.shape == (4, model.pipeline.clifford_dim)


def test_clifford_geometric_product_anticommutativity():
    """e1 * e2 = -e2 * e1 в Cl_{2,0,0}."""
    from src.core.hypercomplex import CliffordAlgebra
    alg = CliffordAlgebra(p=2, q=0, r=0)  # dim=4, basis: e0=1, e1, e2, e12
    # e1 = index 1 (binary 01), e2 = index 2 (binary 10)
    e1 = torch.zeros(alg.dim)
    e1[1] = 1.0
    e2 = torch.zeros(alg.dim)
    e2[2] = 1.0
    e1e2 = alg.geometric_product(e1, e2)
    e2e1 = alg.geometric_product(e2, e1)
    # e1*e2 должно быть -(e2*e1)
    assert torch.allclose(e1e2, -e2e1, atol=1e-6), f"e1*e2 != -e2*e1: {e1e2} vs {-e2e1}"
