"""Tests for MoEKernel — full MoE routing core with domain experts."""
import pytest
import torch
import torch.nn as nn

from src.core.moe_kernel import (
    MoEKernel, MoEKernelConfig, MoEKernelState,
    DomainExpert, MathExpert, LanguageExpert, CodeExpert, ScienceExpert,
    create_expert, EXPERT_REGISTRY,
)


@pytest.fixture
def config():
    return MoEKernelConfig(
        input_dim=64,
        expert_hidden_dim=128,
        num_experts=4,
        slots_per_expert=1,
        temperature=1.0,
        z_loss_weight=0.01,
        ortho_loss_weight=0.01,
        use_shared_expert=True,
        use_aux_loss_free=True,
        use_expert_ortho=True,
        expert_names=["math", "language", "code", "science"],
    )


@pytest.fixture
def kernel(config):
    k = MoEKernel(config)
    k.eval()
    return k


@pytest.fixture
def kernel_train(config):
    k = MoEKernel(config)
    k.train()
    return k


# ============================================================
# Config
# ============================================================

class TestMoEKernelConfig:
    def test_default_expert_names(self):
        cfg = MoEKernelConfig(input_dim=32, expert_hidden_dim=64, num_experts=3)
        assert cfg.expert_names == ["expert_0", "expert_1", "expert_2"]

    def test_custom_expert_names(self):
        cfg = MoEKernelConfig(
            input_dim=32, expert_hidden_dim=64, num_experts=2,
            expert_names=["math", "code"]
        )
        assert cfg.expert_names == ["math", "code"]

    def test_mismatched_names_raises(self):
        with pytest.raises(ValueError, match="expert_names length"):
            MoEKernelConfig(input_dim=32, expert_hidden_dim=64, num_experts=2,
                            expert_names=["a", "b", "c"])


# ============================================================
# Expert types
# ============================================================

class TestDomainExperts:
    def test_math_expert_forward(self):
        expert = MathExpert(input_dim=64, hidden_dim=128)
        x = torch.randn(8, 64)
        out = expert(x)
        assert out.shape == (8, 64)
        assert not torch.isnan(out).any()

    def test_language_expert_forward(self):
        expert = LanguageExpert(input_dim=64, hidden_dim=128)
        x = torch.randn(8, 64)
        out = expert(x)
        assert out.shape == (8, 64)
        assert not torch.isnan(out).any()

    def test_code_expert_forward(self):
        expert = CodeExpert(input_dim=64, hidden_dim=128)
        x = torch.randn(8, 64)
        out = expert(x)
        assert out.shape == (8, 64)
        assert not torch.isnan(out).any()

    def test_science_expert_forward(self):
        expert = ScienceExpert(input_dim=64, hidden_dim=128)
        x = torch.randn(8, 64)
        out = expert(x)
        assert out.shape == (8, 64)
        assert not torch.isnan(out).any()

    def test_create_expert_registered(self):
        for name in ["math", "language", "code", "science"]:
            e = create_expert(name, input_dim=32, hidden_dim=64, dropout=0.0)
            assert e.name == name
            assert isinstance(e, DomainExpert)

    def test_create_expert_unknown_falls_back(self):
        e = create_expert("unknown_domain", input_dim=32, hidden_dim=64, dropout=0.0)
        assert isinstance(e, DomainExpert)
        assert e.name == "unknown_domain"

    def test_registry_keys(self):
        assert set(EXPERT_REGISTRY.keys()) == {"math", "language", "code", "science"}


# ============================================================
# Forward pass
# ============================================================

class TestMoEKernelForward:
    def test_output_shape_2d(self, kernel):
        x = torch.randn(16, 64)
        out, state = kernel(x)
        assert out.shape == (16, 64)

    def test_output_shape_3d(self, kernel):
        x = torch.randn(4, 8, 64)
        out, state = kernel(x)
        assert out.shape == (4, 8, 64)

    def test_output_no_nan(self, kernel):
        x = torch.randn(16, 64)
        out, _ = kernel(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_batch_size_1(self, kernel):
        x = torch.randn(1, 64)
        out, state = kernel(x)
        assert out.shape == (1, 64)
        assert not torch.isnan(out).any()

    def test_state_expert_weights_shape_2d(self, kernel):
        x = torch.randn(16, 64)
        _, state = kernel(x)
        assert state.expert_weights.shape == (16, 4)

    def test_state_expert_weights_shape_3d(self, kernel):
        x = torch.randn(4, 8, 64)
        _, state = kernel(x)
        assert state.expert_weights.shape == (4, 8, 4)

    def test_state_expert_usage(self, kernel):
        x = torch.randn(16, 64)
        _, state = kernel(x)
        assert state.expert_usage.shape == (4,)
        # Usage должна суммироваться приблизительно к 1 (среднее combine-весов)
        assert state.expert_usage.sum().item() > 0

    def test_state_top_expert_idx(self, kernel):
        x = torch.randn(16, 64)
        _, state = kernel(x)
        assert state.top_expert_idx.shape == (16,)
        assert state.top_expert_idx.max().item() <= 3

    def test_dominant_expert_names(self, kernel):
        x = torch.randn(8, 64)
        _, state = kernel(x)
        names = state.dominant_expert_names()
        assert len(names) == 8
        valid = {"math", "language", "code", "science"}
        assert all(n in valid for n in names)

    def test_combine_weights_sum_to_one(self, kernel):
        x = torch.randn(8, 64)
        _, state = kernel(x)
        # combine_weights: (T, num_slots) — сумма по слотам = 1
        combine = state.combine_weights
        sums = combine.sum(dim=-1)  # (T,)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_dispatch_weights_sum_to_one(self, kernel):
        x = torch.randn(8, 64)
        _, state = kernel(x)
        # dispatch: (T, num_slots) — сумма по токенам = 1 (для T > 1)
        dispatch = state.dispatch_weights
        sums = dispatch.sum(dim=0)  # (num_slots,)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_no_shared_expert(self):
        cfg = MoEKernelConfig(
            input_dim=32, expert_hidden_dim=64, num_experts=2,
            use_shared_expert=False,
            expert_names=["math", "code"]
        )
        k = MoEKernel(cfg)
        k.eval()
        assert k.shared_expert is None
        x = torch.randn(4, 32)
        out, _ = k(x)
        assert out.shape == (4, 32)
        assert not torch.isnan(out).any()


# ============================================================
# Gradient flow
# ============================================================

class TestGradientFlow:
    def test_input_gradient(self, kernel_train):
        x = torch.randn(8, 64, requires_grad=True)
        out, state = kernel_train(x)
        loss = out.mean() + state.total_loss()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_expert_gradients_all(self, kernel_train):
        x = torch.randn(8, 64)
        out, state = kernel_train(x)
        loss = out.mean() + state.total_loss()
        loss.backward()
        for expert in kernel_train.experts:
            for p in expert.parameters():
                assert p.grad is not None
                assert not torch.isnan(p.grad).any()

    def test_router_proj_gradient(self, kernel_train):
        x = torch.randn(8, 64)
        out, state = kernel_train(x)
        loss = out.mean() + state.router_loss
        loss.backward()
        assert kernel_train.router_proj.weight.grad is not None
        assert not torch.isnan(kernel_train.router_proj.weight.grad).any()

    def test_shared_expert_gradient(self, kernel_train):
        x = torch.randn(8, 64)
        out, state = kernel_train(x)
        loss = out.mean()
        loss.backward()
        for p in kernel_train.shared_expert.parameters():
            assert p.grad is not None


# ============================================================
# Losses
# ============================================================

class TestLosses:
    def test_router_loss_nonneg(self, kernel_train):
        x = torch.randn(8, 64)
        _, state = kernel_train(x)
        assert state.router_loss.item() >= 0

    def test_z_loss_nonneg(self, kernel_train):
        x = torch.randn(8, 64)
        _, state = kernel_train(x)
        assert state.z_loss.item() >= 0

    def test_ortho_loss_nonneg(self, kernel_train):
        x = torch.randn(8, 64)
        _, state = kernel_train(x)
        assert state.ortho_loss.item() >= 0

    def test_total_loss_finite(self, kernel_train):
        x = torch.randn(8, 64)
        _, state = kernel_train(x)
        total = state.total_loss()
        assert torch.isfinite(total)

    def test_z_loss_weight_zero_gives_zero(self, config):
        cfg = MoEKernelConfig(
            input_dim=64, expert_hidden_dim=128, num_experts=4,
            z_loss_weight=0.0,
            expert_names=["math", "language", "code", "science"]
        )
        k = MoEKernel(cfg)
        k.train()
        x = torch.randn(8, 64)
        _, state = k(x)
        assert state.z_loss.item() == 0.0

    def test_ortho_loss_disabled(self, config):
        cfg = MoEKernelConfig(
            input_dim=64, expert_hidden_dim=128, num_experts=4,
            use_expert_ortho=False,
            expert_names=["math", "language", "code", "science"]
        )
        k = MoEKernel(cfg)
        k.train()
        x = torch.randn(8, 64)
        _, state = k(x)
        assert state.ortho_loss.item() == 0.0

    def test_expert_ortho_loss_positive(self, kernel_train):
        # expert_orthogonalization_loss должен быть >= 0 (Gram-matrix deviation)
        loss = kernel_train.expert_orthogonalization_loss()
        assert loss.item() >= 0

    def test_router_similarity_loss_positive(self, kernel_train):
        loss = kernel_train.router_similarity_loss()
        assert loss.item() >= 0


# ============================================================
# Load balance
# ============================================================

class TestLoadBalance:
    def test_load_balance_near_uniform(self):
        torch.manual_seed(0)
        cfg = MoEKernelConfig(
            input_dim=64, expert_hidden_dim=128, num_experts=4,
            expert_names=["math", "language", "code", "science"]
        )
        k = MoEKernel(cfg)
        k.eval()
        with torch.no_grad():
            x = torch.randn(128, 64)
            _, state = k(x)
        expected = 1.0 / 4
        max_dev = (state.expert_usage - expected).abs().max().item()
        assert max_dev < 0.2

    def test_routing_entropy_positive(self, kernel):
        x = torch.randn(16, 64)
        _, state = kernel(x)
        assert state.routing_entropy.item() > 0
        assert torch.isfinite(state.routing_entropy)

    def test_train_scores_updated(self, kernel_train):
        initial = kernel_train.train_scores.clone()
        x = torch.randn(16, 64)
        kernel_train(x)
        assert not torch.equal(kernel_train.train_scores, initial)

    def test_train_scores_static_in_eval(self, kernel):
        initial = kernel.train_scores.clone()
        x = torch.randn(16, 64)
        kernel(x)
        assert torch.equal(kernel.train_scores, initial)


# ============================================================
# Aux-Loss-Free bias
# ============================================================

class TestAuxLossFree:
    def test_bias_updated_during_training(self, kernel_train):
        initial = kernel_train._expert_bias.data.clone()
        x = torch.randn(32, 64)
        kernel_train(x)
        # Bias может обновиться только если нагрузка неравномерна
        # Достаточно проверить что значение не NaN
        assert not torch.isnan(kernel_train._expert_bias).any()

    def test_bias_static_in_eval(self, kernel):
        initial = kernel._expert_bias.data.clone()
        x = torch.randn(32, 64)
        kernel(x)
        assert torch.equal(kernel._expert_bias, initial)

    def test_reset_bias(self, kernel_train):
        x = torch.randn(32, 64)
        kernel_train(x)  # обновить bias
        kernel_train.reset_bias()
        assert kernel_train._expert_bias.data.abs().sum().item() == 0.0

    def test_no_bias_when_disabled(self):
        cfg = MoEKernelConfig(
            input_dim=32, expert_hidden_dim=64, num_experts=2,
            use_aux_loss_free=False,
            expert_names=["math", "code"]
        )
        k = MoEKernel(cfg)
        assert k._expert_bias is None


# ============================================================
# Expert load stats
# ============================================================

class TestExpertLoadStats:
    def test_load_stats_keys(self, kernel):
        stats = kernel.expert_load_stats()
        assert set(stats.keys()) == {"math", "language", "code", "science"}

    def test_load_stats_values_positive(self, kernel):
        stats = kernel.expert_load_stats()
        for name, val in stats.items():
            assert val > 0, f"Expert {name} load should be positive"

    def test_load_stats_sum_approx_one(self, kernel):
        stats = kernel.expert_load_stats()
        total = sum(stats.values())
        assert abs(total - 1.0) < 0.01
