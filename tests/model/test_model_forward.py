import pytest
import torch
from torch.utils.data import DataLoader

from src.models.hdim_model import HDIMConfig, HDIMModel
from src.training.dataset import create_paired_demo_dataset
from src.training.invariant_trainer import InvariantTrainer


@pytest.fixture
def cfg():
    return HDIMConfig()


@pytest.fixture
def model(cfg):
    return HDIMModel(cfg)


def test_model_forward(model, cfg):
    bsz = 4
    x = torch.randn(bsz, cfg.hidden_dim)
    domain_id = torch.zeros(bsz, dtype=torch.long)
    result = model(x, domain_id)
    out, inv = result.output, result.invariant
    assert out.shape == (bsz, cfg.hidden_dim)
    assert inv.shape == (bsz, cfg.hidden_dim)


def test_model_forward_return_state(model, cfg):
    bsz = 4
    x = torch.randn(bsz, cfg.hidden_dim)
    domain_id = torch.zeros(bsz, dtype=torch.long)
    result = model(
        x,
        domain_id,
        return_state=True,
        update_memory=False,
        memory_mode="retrieve",
    )
    out, inv, state = result.output, result.invariant, result.aux_state
    assert out.shape == (bsz, cfg.hidden_dim)
    assert state.matches == [[], [], [], []]
    assert inv.shape == (bsz, cfg.hidden_dim)
    assert state.raw_invariant.shape == (bsz, model.engine.algebra.dim)
    assert state.exported_invariant.shape == (bsz, model.engine.algebra.dim)
    expected_inv = model.training_inv_head(state.exported_invariant)
    assert torch.allclose(inv, expected_inv, atol=1e-5)


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
    result = model.transfer_pairs(
        x,
        source_domain_id,
        target_domain_id,
        update_memory=False,
        memory_mode="retrieve",
    )
    out, inv, state = result.output, result.invariant, result.aux_state
    assert out.shape == (bsz, cfg.hidden_dim)
    assert inv.shape == (bsz, cfg.hidden_dim)
    assert len(state.matches) == bsz
    assert state.raw_invariant.shape == (bsz, model.engine.algebra.dim)
    assert state.exported_invariant.shape == (bsz, model.engine.algebra.dim)
    expected_inv = model.training_inv_head(state.exported_invariant)
    assert torch.allclose(inv, expected_inv, atol=1e-5)


def test_model_return_state_exposes_memory_lifecycle(model, cfg):
    x = torch.randn(4, cfg.hidden_dim)
    domain_id = torch.zeros(4, dtype=torch.long)
    state = model(
        x,
        domain_id,
        return_state=True,
        update_memory=False,
        memory_mode="retrieve",
    ).aux_state
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
    state = model(
        x,
        domain_id,
        return_state=True,
        update_memory=True,
        memory_mode="none",
    ).aux_state
    assert state.memory_mode == "none"
    assert state.update_memory is False
    assert state.memory_updated is False
    assert torch.allclose(state.exported_invariant, model.engine.transfer(state.raw_invariant, model._domain_idx_to_name(0)), atol=1e-6)


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

    forward_state = model(
        x,
        source_domain_id,
        return_state=True,
        update_memory=False,
        memory_mode="retrieve",
    ).aux_state
    pair_state = model.transfer_pairs(
        x,
        source_domain_id,
        target_domain_id,
        update_memory=False,
        memory_mode="retrieve",
    ).aux_state

    lifecycle_fields = [
        "raw_invariant",
        "exported_invariant",
        "training_invariant",
    ]
    for field in lifecycle_fields:
        forward_value = getattr(forward_state, field)
        pair_value = getattr(pair_state, field)
        assert forward_value.shape == pair_value.shape


def test_core_adapter_exposes_empty_router_diagnostics_on_eval_path(model):
    ds = create_paired_demo_dataset(n_samples=16)
    batch = next(iter(DataLoader(ds, batch_size=8)))
    losses = InvariantTrainer(model, torch.optim.Adam(model.parameters(), lr=1e-3), device="cpu").evaluate_batch(batch)

    assert losses["routing_weights"].shape == (8, model.config.num_experts)
    assert torch.count_nonzero(losses["routing_weights"]).item() == 0
    assert losses["topk_idx"].shape == (8, 0)
    assert losses["topk_gate_weights"].shape == (8, 0)


def test_eval_retrieve_is_deterministic_without_memory_or_router_state(model, cfg):
    x = torch.randn(8, cfg.hidden_dim)
    domain_id = torch.randint(0, cfg.num_domains, (8,))

    model.eval()
    with torch.no_grad():
        res_a = model(
            x,
            domain_id,
            return_state=True,
            update_memory=False,
            memory_mode="retrieve",
        )
        res_b = model(
            x,
            domain_id,
            return_state=True,
            update_memory=False,
            memory_mode="retrieve",
        )

    assert torch.allclose(res_a.routing_weights, res_b.routing_weights)
    assert torch.allclose(res_a.invariant, res_b.invariant)
    assert torch.allclose(res_a.aux_state.raw_invariant, res_b.aux_state.raw_invariant)
    assert torch.allclose(res_a.aux_state.exported_invariant, res_b.aux_state.exported_invariant)


def test_model_reset_memory_is_noop_in_core_wrapper(model, cfg):
    x = torch.randn(4, cfg.hidden_dim)
    domain_id = torch.zeros(4, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        before = model(x, domain_id, return_state=True, update_memory=True, memory_mode="update").aux_state
        model.reset_memory(strategy='hard')
        after = model(x, domain_id, return_state=True, update_memory=True, memory_mode="update").aux_state

    assert torch.allclose(before.raw_invariant, after.raw_invariant)
    assert torch.allclose(before.exported_invariant, after.exported_invariant)
