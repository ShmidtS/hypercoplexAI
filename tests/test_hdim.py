import pytest
import torch
from torch.utils.data import DataLoader

from src.models.hdim_model import HDIMConfig, HDIMModel
from src.models.metrics import compute_all_metrics
from src.training.dataset import DomainProblemDataset, create_demo_dataset, create_paired_demo_dataset
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
    assert state["processed_invariant"].shape == (bsz, model.pipeline.clifford_dim)
    assert state["router_loss"].ndim == 0
    assert state["memory_loss"].ndim == 0

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
    assert state["processed_invariant"].shape == (bsz, model.pipeline.clifford_dim)


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
    assert sample["pair_encoding"].shape == (64,)
    assert sample["pair_domain_id"].dtype == torch.long
    assert sample["pair_domain_id"].item() != sample["domain_id"].item()


def test_dataset_rejects_same_domain_pairs():
    samples = [("a", 0), ("b", 0)]
    with pytest.raises(ValueError):
        DomainProblemDataset(samples, pair_indices=[1, 0])


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

def test_compute_all_metrics_runs_on_model_device(trainer):
    ds = create_demo_dataset(n_samples=16)
    dl = DataLoader(ds, batch_size=4)
    metrics = compute_all_metrics(trainer.model, dl)
    assert set(metrics) == {"STS", "DRS", "AFR"}
    assert all(isinstance(value, float) for value in metrics.values())


def test_compute_all_metrics_with_pairs(model):
    ds = create_paired_demo_dataset(n_samples=24)
    dl = DataLoader(ds, batch_size=8)
    metrics = compute_all_metrics(model, dl, num_routing_runs=2)
    assert set(metrics) == {"STS", "DRS", "AFR"}
    assert 0.0 <= metrics["AFR"] <= 1.0
    assert metrics["STS"] <= 1.0
    assert metrics["DRS"] >= 0.0
