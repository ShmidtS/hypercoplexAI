import pytest
import torch
from torch.utils.data import DataLoader

from src.core.engine import CoreEngineConfig, HDIMCoreEngine
from src.models.hdim_model import HDIMConfig, HDIMModel
from src.training.invariant_losses import compute_pair_iso_loss
from src.training.invariant_trainer import InvariantTrainer
from src.training.trainer import HDIMTrainer


def _batch(hidden_dim: int = 64, batch_size: int = 4) -> dict:
    return {
        "encoding": torch.randn(batch_size, hidden_dim),
        "domain_id": torch.zeros(batch_size, dtype=torch.long),
        "pair_encoding": torch.randn(batch_size, hidden_dim),
        "pair_domain_id": torch.ones(batch_size, dtype=torch.long),
        "pair_relation_label": torch.ones(batch_size),
        "pair_weight": torch.ones(batch_size),
    }


def test_trainer_accepts_hdim_model():
    model = HDIMModel(HDIMConfig(hidden_dim=64, num_domains=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = InvariantTrainer(model, optimizer, device="cpu")

    assert trainer.model is model


def test_trainer_accepts_core_engine():
    engine = HDIMCoreEngine(CoreEngineConfig(input_dim=64, domain_names=("source", "target"), dropout=0.0))
    optimizer = torch.optim.Adam(engine.parameters(), lr=1e-3)

    trainer = InvariantTrainer(engine, optimizer, device="cpu")

    assert trainer.model is engine


def test_training_step_returns_loss_dict():
    model = HDIMModel(HDIMConfig(hidden_dim=64, num_domains=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = InvariantTrainer(model, optimizer, device="cpu")

    losses = trainer.training_step(_batch())

    assert "loss" in losses
    assert losses["loss"].item() >= 0.0


def test_pair_iso_loss_computes_positive():
    source = torch.randn(4, 8)
    target = torch.randn(4, 8)
    batch = {
        "pair_relation_label": torch.ones(4),
        "pair_weight": torch.ones(4),
    }

    loss = compute_pair_iso_loss(source, target, batch, negative_margin=1.0, device=torch.device("cpu"), has_pairs=True)

    assert loss.item() > 0.0


def test_deprecated_trainer_warns():
    model = HDIMModel(HDIMConfig(hidden_dim=64, num_domains=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.warns(DeprecationWarning):
        HDIMTrainer(model, optimizer, device="cpu")
