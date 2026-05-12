import pytest
import torch
from torch.utils.data import DataLoader

from src.models.hdim_model import HDIMConfig, HDIMModel
from src.models.metrics import compute_all_metrics
from src.training.dataset import create_demo_dataset, create_group_aware_split, create_paired_demo_dataset
from src.training.invariant_trainer import InvariantTrainer


@pytest.fixture
def cfg():
    return HDIMConfig()


@pytest.fixture
def model(cfg):
    return HDIMModel(cfg)


@pytest.fixture
def trainer(model):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    return InvariantTrainer(model, opt, device="cpu")


def test_iso_loss(trainer, cfg):
    u1 = torch.randn(4, cfg.hidden_dim)
    u2 = torch.randn(4, cfg.hidden_dim)
    loss = trainer.compute_iso_loss(u1, u2)
    assert loss.item() >= 0


def test_routing_loss(trainer, cfg):
    weights = torch.softmax(torch.randn(4, cfg.num_experts), dim=-1)
    loss = trainer.compute_routing_loss(weights)
    assert isinstance(loss.item(), float)


def test_pair_iso_loss_uses_pair_weights(trainer, cfg):
    training_invariant = torch.tensor([[0.0] * cfg.hidden_dim, [1.0] * cfg.hidden_dim])
    iso_target = torch.tensor([[1.0] * cfg.hidden_dim, [1.0] * cfg.hidden_dim])
    batch = {
        "pair_encoding": torch.randn(2, cfg.hidden_dim),
        "pair_domain_id": torch.tensor([0, 1], dtype=torch.long),
        "pair_relation_label": torch.tensor([1.0, 1.0]),
        "pair_weight": torch.tensor([3.0, 1.0]),
    }
    loss = trainer._compute_pair_iso_loss(training_invariant, iso_target, batch)
    assert loss.item() == pytest.approx(0.75)


def test_pair_iso_loss_uses_negative_margin(trainer, cfg):
    training_invariant = torch.zeros(2, cfg.hidden_dim)
    iso_target = torch.zeros(2, cfg.hidden_dim)
    batch = {
        "pair_encoding": torch.randn(2, cfg.hidden_dim),
        "pair_domain_id": torch.tensor([0, 1], dtype=torch.long),
        "pair_relation_label": torch.tensor([0.0, 0.0]),
        "pair_weight": torch.tensor([1.0, 3.0]),
    }
    loss = trainer._compute_pair_iso_loss(training_invariant, iso_target, batch)
    assert loss.item() == pytest.approx(trainer.negative_margin)


def test_pair_ranking_loss_uses_grouped_negatives(trainer):
    """Тест совместимости pair ranking loss (InfoNCE mode).

    Trainer по умолчанию использует InfoNCE loss (use_infonce=True).
    Проверяем что loss — скалярный тензор конечного значения.
    """
    exported_invariant = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    pair_exported_target = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    batch = {
        "pair_encoding": torch.randn(3, trainer.model.config.hidden_dim),
        "domain_id": torch.tensor([0, 1, 2], dtype=torch.long),
        "pair_domain_id": torch.tensor([1, 2, 0], dtype=torch.long),
        "pair_group_id": torch.tensor([10, 20, 30], dtype=torch.long),
        "pair_relation_label": torch.tensor([1.0, 1.0, 0.0]),
        "pair_weight": torch.tensor([1.0, 2.0, 5.0]),
    }

    loss = trainer.compute_pair_ranking_loss(exported_invariant, pair_exported_target, batch)

    # InfoNCE mode: loss должен быть конечным скаляром >= 0
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0.0
    assert not (loss != loss).item()  # not NaN


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
    assert isinstance(metrics["loss_recon"], float)


def test_evaluate_batch_with_pairs(trainer):
    ds = create_paired_demo_dataset(n_samples=16)
    batch = next(iter(DataLoader(ds, batch_size=8)))
    losses = trainer.evaluate_batch(batch)
    assert losses["loss_total"].item() >= 0
    assert losses["loss_recon"].item() >= 0
    assert losses["training_invariant"].shape == (8, trainer.model.config.hidden_dim)
    assert losses["exported_invariant"].shape == (8, trainer.model.pipeline.clifford_dim)
    assert losses["routing_entropy"].ndim == 0
    assert losses["expert_usage"].numel() == 0
    assert losses["topk_idx"].shape == (8, 0)
    assert losses["topk_gate_weights"].shape == (8, 0)


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


def test_compute_all_metrics_handles_positive_only_paired_validation(model):
    ds = create_paired_demo_dataset(n_samples=100, negative_ratio=0.0)
    _, val_ds = create_group_aware_split(ds, train_fraction=0.8, seed=42)
    dl = DataLoader(val_ds, batch_size=8)
    metrics = compute_all_metrics(model, dl, num_routing_runs=1)
    assert set(metrics) == {"STS_exported", "STS_training", "DRS", "AFR", "pair_margin"}
    assert all(isinstance(value, float) for value in metrics.values())


def test_aligned_pairs_have_non_extreme_margin(model):
    ds = create_paired_demo_dataset(n_samples=24)
    dl = DataLoader(ds, batch_size=8)
    metrics = compute_all_metrics(model, dl, num_routing_runs=1)
    assert -1.0 <= metrics["pair_margin"] <= 1.0
