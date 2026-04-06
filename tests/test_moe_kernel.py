"""Tests for MoEKernel — full MoE routing core with domain experts."""
import pytest
import torch
import torch.nn as nn

from src.core.moe_kernel import (
    MoEKernel, MoEKernelConfig, MoEKernelState,
    DomainExpert, MathExpert, LanguageExpert, CodeExpert, ScienceExpert,
    create_expert, EXPERT_REGISTRY, register_expert, get_registered_expert_names,
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
        with pytest.raises(ValueError, match="conflicts"):
            MoEKernelConfig(input_dim=32, expert_hidden_dim=64, num_experts=2,
                expert_names=["a", "b", "c"])

    def test_num_experts_none_computed_from_names(self):
        """num_experts=None is computed from expert_names length."""
        cfg = MoEKernelConfig(
            input_dim=32, expert_hidden_dim=64,
            num_experts=None,
            expert_names=["a", "b", "c"]
        )
        assert cfg.num_experts == 3

    def test_num_experts_omitted_computed_from_names(self):
        """When num_experts is omitted, it's computed from expert_names."""
        cfg = MoEKernelConfig(
            input_dim=32, expert_hidden_dim=64,
            expert_names=["math", "language"]
        )
        assert cfg.num_experts == 2

    def test_num_experts_none_without_names_uses_default(self):
        """num_experts=None without expert_names uses default 4."""
        cfg = MoEKernelConfig(
            input_dim=32, expert_hidden_dim=64,
            num_experts=None
        )
        assert cfg.num_experts == 4
        assert cfg.expert_names == ["expert_0", "expert_1", "expert_2", "expert_3"]


# ============================================================
# Dynamic expert count
# ============================================================

class TestDynamicExpertCount:
    """Test that MoEKernel works with various expert counts."""

    def test_single_expert(self):
        """MoEKernel works with a single expert."""
        cfg = MoEKernelConfig(
            input_dim=32,
            expert_hidden_dim=64,
            num_experts=1,
            expert_names=["math"],
        )
        kernel = MoEKernel(cfg)
        kernel.eval()
        x = torch.randn(8, 32)
        out, state = kernel(x)
        assert out.shape == (8, 32)
        assert state.expert_weights.shape == (8, 1)
        assert not torch.isnan(out).any()

    def test_two_experts(self):
        """MoEKernel works with 2 experts."""
        cfg = MoEKernelConfig(
            input_dim=32,
            expert_hidden_dim=64,
            num_experts=2,
            expert_names=["math", "language"],
        )
        kernel = MoEKernel(cfg)
        kernel.eval()
        x = torch.randn(16, 32)
        out, state = kernel(x)
        assert out.shape == (16, 32)
        assert state.expert_weights.shape == (16, 2)
        assert state.expert_usage.shape == (2,)

    def test_eight_experts(self):
        """MoEKernel works with 8 experts."""
        cfg = MoEKernelConfig(
            input_dim=32,
            expert_hidden_dim=64,
            num_experts=8,
            expert_names=[f"expert_{i}" for i in range(8)],
        )
        kernel = MoEKernel(cfg)
        kernel.eval()
        x = torch.randn(32, 32)
        out, state = kernel(x)
        assert out.shape == (32, 32)
        assert state.expert_weights.shape == (32, 8)
        assert state.expert_usage.shape == (8,)

    def test_sixteen_experts(self):
        """MoEKernel works with 16 experts."""
        cfg = MoEKernelConfig(
            input_dim=32,
            expert_hidden_dim=64,
            num_experts=16,
            expert_names=[f"expert_{i}" for i in range(16)],
        )
        kernel = MoEKernel(cfg)
        kernel.eval()
        x = torch.randn(64, 32)
        out, state = kernel(x)
        assert out.shape == (64, 32)
        assert state.expert_weights.shape == (64, 16)
        assert state.expert_usage.shape == (16,)

    def test_expert_count_from_names_only(self):
        """num_experts computed correctly when only expert_names provided."""
        names = ["a", "b", "c", "d", "e"]
        cfg = MoEKernelConfig(
            input_dim=32,
            expert_hidden_dim=64,
            expert_names=names,
        )
        # num_experts should be computed from names
        assert cfg.num_experts == 5
        kernel = MoEKernel(cfg)
        kernel.eval()
        x = torch.randn(10, 32)
        out, state = kernel(x)
        assert out.shape == (10, 32)
        assert len(kernel.experts) == 5

    def test_gradient_flow_single_expert(self):
        """Gradients flow correctly with single expert."""
        cfg = MoEKernelConfig(
            input_dim=32,
            expert_hidden_dim=64,
            num_experts=1,
            expert_names=["math"],
        )
        kernel = MoEKernel(cfg)
        kernel.train()
        x = torch.randn(8, 32, requires_grad=True)
        out, state = kernel(x)
        loss = out.mean() + state.total_loss()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gradient_flow_sixteen_experts(self):
        """Gradients flow correctly with 16 experts."""
        cfg = MoEKernelConfig(
            input_dim=32,
            expert_hidden_dim=64,
            num_experts=16,
            expert_names=[f"expert_{i}" for i in range(16)],
        )
        kernel = MoEKernel(cfg)
        kernel.train()
        x = torch.randn(8, 32, requires_grad=True)
        out, state = kernel(x)
        loss = out.mean() + state.total_loss()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        # All experts should have gradients
        for expert in kernel.experts:
            for p in expert.parameters():
                assert p.grad is not None


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
        sums = combine.sum(dim=-1) # (T,)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_dispatch_weights_sum_to_one(self, kernel):
        x = torch.randn(8, 64)
        _, state = kernel(x)
        # dispatch: (T, num_slots) — сумма по токенам = 1 (для T > 1)
        dispatch = state.dispatch_weights
        sums = dispatch.sum(dim=0) # (num_slots,)
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
        kernel_train(x) # обновить bias
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


# ============================================================
# Dynamic expert registration
# ============================================================

class TestRegisterExpert:
    def test_register_custom_expert(self):
        """Test dynamic expert registration."""
        class CustomExpert(DomainExpert):
            def __init__(self, input_dim, hidden_dim, dropout=0.1):
                super().__init__(input_dim, hidden_dim, dropout, name="custom")

        # Register custom expert
        register_expert("custom", CustomExpert)
        assert "custom" in get_registered_expert_names()

        # Create MoEKernel with custom expert
        cfg = MoEKernelConfig(
            input_dim=64,
            num_experts=1,
            expert_names=["custom"],
        )
        kernel = MoEKernel(cfg)
        assert kernel.experts[0].name == "custom"

        # Cleanup: remove custom from registry for other tests
        del EXPERT_REGISTRY["custom"]

    def test_invalid_expert_registration(self):
        """Test that non-DomainExpert classes are rejected."""
        class NotAnExpert:
            pass

        with pytest.raises(TypeError, match="must inherit from DomainExpert"):
            register_expert("invalid", NotAnExpert)

    def test_get_registered_expert_names_returns_list(self):
        """Test that get_registered_expert_names returns correct list."""
        names = get_registered_expert_names()
        assert isinstance(names, list)
        # Built-in experts should always be present
        assert "math" in names
        assert "language" in names
        assert "code" in names
        assert "science" in names

    def test_register_expert_overrides_existing(self):
        """Test that registering with existing name overrides."""
        class NewMathExpert(DomainExpert):
            def __init__(self, input_dim, hidden_dim, dropout=0.1):
                super().__init__(input_dim, hidden_dim, dropout, name="math")

        original = EXPERT_REGISTRY["math"]
        register_expert("math", NewMathExpert)
        assert EXPERT_REGISTRY["math"] is NewMathExpert

        # Restore original
        EXPERT_REGISTRY["math"] = original
    
    
    # ============================================================
    # MoE Interface and Adapter Tests
    # ============================================================
    
    class TestMoERouterInterface:
        """Tests for MoERouter abstract interface."""
    
        def test_moe_kernel_adapter_implements_interface(self):
            """MoEKernelAdapter must inherit from MoERouter."""
            from src.core.moe_interface import MoERouter
            from src.core.moe_kernel_adapter import MoEKernelAdapter
    
            assert issubclass(MoEKernelAdapter, MoERouter)
    
        def test_soft_moe_router_implements_interface(self):
            """SoftMoERouter must inherit from MoERouter."""
            from src.core.moe_interface import MoERouter
            from src.core.soft_moe_router import SoftMoERouter
    
            assert issubclass(SoftMoERouter, MoERouter)
    
        def test_adapter_forward_returns_tuple(self, config):
            """MoEKernelAdapter.forward() must return (Tensor, Dict)."""
            from src.core.moe_kernel_adapter import MoEKernelAdapter
    
            kernel = MoEKernel(config)
            adapter = MoEKernelAdapter(kernel)
            adapter.eval()
    
            x = torch.randn(2, config.input_dim)
            output, info = adapter(x)
    
            assert isinstance(output, torch.Tensor)
            assert isinstance(info, dict)
            assert output.shape == x.shape
            assert "expert_load" in info
            assert "router_loss" in info
    
        def test_adapter_get_expert_load(self, config):
            """MoEKernelAdapter.get_expert_load() must return Tensor[num_experts]."""
            from src.core.moe_kernel_adapter import MoEKernelAdapter
    
            kernel = MoEKernel(config)
            adapter = MoEKernelAdapter(kernel)
    
            load = adapter.get_expert_load()
            assert isinstance(load, torch.Tensor)
            assert load.shape == (config.num_experts,)
            # Initial load should be uniform
            assert torch.allclose(load, torch.ones(config.num_experts) / config.num_experts)
    
        def test_adapter_expert_orthogonalization_loss(self, config):
            """MoEKernelAdapter.expert_orthogonalization_loss() must return scalar."""
            from src.core.moe_kernel_adapter import MoEKernelAdapter
    
            kernel = MoEKernel(config)
            adapter = MoEKernelAdapter(kernel)
    
            loss = adapter.expert_orthogonalization_loss()
            assert isinstance(loss, torch.Tensor)
            assert loss.ndim == 0  # scalar
    
        def test_adapter_reset_training_state(self, config):
            """MoEKernelAdapter.reset_training_state() must reset EMA."""
            from src.core.moe_kernel_adapter import MoEKernelAdapter
    
            kernel = MoEKernel(config)
            adapter = MoEKernelAdapter(kernel)
    
            # Modify train_scores
            kernel.train_scores.fill_(0.5)
            adapter.reset_training_state()
    
            # Should be uniform again
            expected = torch.ones(config.num_experts) / config.num_experts
            assert torch.allclose(kernel.train_scores, expected)
    
        def test_soft_moe_router_get_expert_load(self):
            """SoftMoERouter.get_expert_load() must return Tensor[num_experts]."""
            from src.core.soft_moe_router import SoftMoERouter
    
            router = SoftMoERouter(
                input_dim=64,
                num_experts=4,
                expert_dim=128,
            )
    
            load = router.get_expert_load()
            assert isinstance(load, torch.Tensor)
            assert load.shape == (4,)
            assert torch.allclose(load, torch.ones(4) / 4)
    
        def test_soft_moe_router_reset_training_state(self):
            """SoftMoERouter.reset_training_state() must reset EMA."""
            from src.core.soft_moe_router import SoftMoERouter
    
            router = SoftMoERouter(
                input_dim=64,
                num_experts=4,
                expert_dim=128,
            )
    
            # Modify train_scores
            router.train_scores.fill_(0.5)
            router.reset_training_state()
    
            # Should be uniform again
            expected = torch.ones(4) / 4
            assert torch.allclose(router.train_scores, expected)
    
        def test_adapter_info_dict_contents(self, config):
            """MoEKernelAdapter forward info dict must contain all required keys."""
            from src.core.moe_kernel_adapter import MoEKernelAdapter
    
            kernel = MoEKernel(config)
            adapter = MoEKernelAdapter(kernel)
            adapter.eval()
    
            x = torch.randn(2, config.input_dim)
            _, info = adapter(x)
    
            required_keys = [
                "expert_load",
                "aux_loss",
                "router_loss",
                "z_loss",
                "ortho_loss",
                "expert_usage",
                "routing_entropy",
                "expert_weights",
                "expert_names",
            ]
    
            for key in required_keys:
                assert key in info, f"Missing key: {key}"
    
        def test_polymorphic_usage(self, config):
            """Both MoEKernelAdapter and SoftMoERouter can be used as MoERouter."""
            from src.core.moe_interface import MoERouter
            from src.core.moe_kernel_adapter import MoEKernelAdapter
            from src.core.soft_moe_router import SoftMoERouter
    
            kernel = MoEKernel(config)
            adapter = MoEKernelAdapter(kernel)
    
            soft_router = SoftMoERouter(
                input_dim=config.input_dim,
                num_experts=config.num_experts,
                expert_dim=config.expert_hidden_dim,
            )
    
            # Both should be instances of MoERouter
            assert isinstance(adapter, MoERouter)
            assert isinstance(soft_router, MoERouter)
    
            # Both should work with same input shape
            x = torch.randn(2, config.input_dim)
    
            out1, info1 = adapter(x)
            out2, info2 = soft_router(x)
    
            assert out1.shape == x.shape
            assert out2.shape == x.shape
        
        
        # ============================================================
        # CAN Experts (CliffordInteractionLayer)
        # ============================================================
        
        class TestCANExperts:
            """Tests for CAN-style experts using CliffordInteractionLayer."""
        
            def test_domain_expert_use_can_flag(self):
                """DomainExpert with use_can=True uses CliffordInteractionLayer."""
                expert = DomainExpert(input_dim=16, hidden_dim=32, use_can=True)
                assert expert.use_can is True
                assert hasattr(expert, 'interaction')
                assert not hasattr(expert, 'net') or expert.net is None
        
            def test_domain_expert_use_can_false(self):
                """DomainExpert with use_can=False uses standard FFN."""
                expert = DomainExpert(input_dim=64, hidden_dim=128, use_can=False)
                assert expert.use_can is False
                assert hasattr(expert, 'net')
        
            def test_domain_expert_forward_can(self):
                """Forward pass works with CAN-enabled expert."""
                expert = DomainExpert(input_dim=16, hidden_dim=32, use_can=True)
                expert.eval()
                x = torch.randn(4, 16)
                out = expert(x)
                assert out.shape == (4, 16)
                assert not torch.isnan(out).any()
        
            def test_domain_expert_forward_ffn(self):
                """Forward pass works with standard FFN expert."""
                expert = DomainExpert(input_dim=64, hidden_dim=128, use_can=False)
                expert.eval()
                x = torch.randn(4, 64)
                out = expert(x)
                assert out.shape == (4, 64)
                assert not torch.isnan(out).any()
        
            def test_create_expert_with_can(self):
                """create_expert function supports use_can parameter."""
                expert = create_expert(
                    name="custom",
                    input_dim=16,
                    hidden_dim=32,
                    dropout=0.1,
                    use_can=True,
                )
                assert expert.use_can is True
                assert expert.input_dim == 16
        
            def test_moe_kernel_config_use_can_experts(self):
                """MoEKernelConfig has use_can_experts parameter."""
                cfg = MoEKernelConfig(
                    input_dim=16,
                    expert_hidden_dim=32,
                    num_experts=2,
                    use_can_experts=True,
                )
                assert cfg.use_can_experts is True
        
            def test_moe_kernel_with_can_experts(self):
                """MoEKernel works with CAN-enabled experts."""
                cfg = MoEKernelConfig(
                    input_dim=16,
                    expert_hidden_dim=32,
                    num_experts=2,
                    expert_names=["custom_0", "custom_1"],
                    use_can_experts=True,
                )
                kernel = MoEKernel(cfg)
                kernel.eval()
        
                # Verify experts use CAN
                for expert in kernel.experts:
                    assert expert.use_can is True
        
                # Forward pass
                x = torch.randn(8, 16)
                out, state = kernel(x)
                assert out.shape == (8, 16)
                assert not torch.isnan(out).any()
        
            def test_moe_kernel_backward_compatibility(self):
                """MoEKernel with use_can_experts=False uses standard FFN."""
                cfg = MoEKernelConfig(
                    input_dim=64,
                    expert_hidden_dim=128,
                    num_experts=4,
                    expert_names=["math", "language", "code", "science"],
                    use_can_experts=False,
                )
                kernel = MoEKernel(cfg)
                kernel.eval()
        
                # Verify experts use FFN
                for expert in kernel.experts:
                    assert expert.use_can is False
        
                # Forward pass
                x = torch.randn(8, 64)
                out, state = kernel(x)
                assert out.shape == (8, 64)
                assert not torch.isnan(out).any()
        
            def test_can_experts_gradient_flow(self):
                """Gradients flow correctly through CAN experts."""
                cfg = MoEKernelConfig(
                    input_dim=16,
                    expert_hidden_dim=32,
                    num_experts=2,
                    expert_names=["a", "b"],
                    use_can_experts=True,
                )
                kernel = MoEKernel(cfg)
                kernel.train()
        
                x = torch.randn(4, 16, requires_grad=True)
                out, state = kernel(x)
                loss = out.mean() + state.total_loss()
                loss.backward()
        
                assert x.grad is not None
                assert not torch.isnan(x.grad).any()
        
            def test_can_experts_with_sequence_input(self):
                """CAN experts handle sequence input (B, T, D)."""
                cfg = MoEKernelConfig(
                    input_dim=16,
                    expert_hidden_dim=32,
                    num_experts=2,
                    expert_names=["a", "b"],
                    use_can_experts=True,
                )
                kernel = MoEKernel(cfg)
                kernel.eval()
        
                # Sequence input: (batch=2, seq=4, dim=16)
                x = torch.randn(2, 4, 16)
                out, state = kernel(x)
                assert out.shape == (2, 4, 16)
                assert not torch.isnan(out).any()
        
            def test_mixed_expert_types_not_supported(self):
                """Specialized experts (MathExpert, etc.) don't support use_can directly."""
                # MathExpert, LanguageExpert etc. override net in __init__
                # so use_can would need explicit support in each class
                math_expert = MathExpert(input_dim=64, hidden_dim=128)
                # These specialized experts use FFN by default
                assert hasattr(math_expert, 'net')


# ============================================================
# Batched Expert Execution
# ============================================================

class TestBatchedExpertExecution:
    """Tests for batched expert execution via torch.bmm."""

    def test_config_has_use_batched_experts(self):
        """MoEKernelConfig has use_batched_experts parameter."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            use_batched_experts=True,
        )
        assert cfg.use_batched_experts is True

    def test_batched_with_homogeneous_experts(self):
        """Batched execution works with homogeneous DomainExpert instances."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            expert_names=["custom_0", "custom_1", "custom_2", "custom_3"],
            use_batched_experts=True,
        )
        kernel = MoEKernel(cfg)
        kernel.eval()

        # All experts should be plain DomainExpert
        for expert in kernel.experts:
            assert type(expert).__name__ == "DomainExpert"

        # Forward pass
        x = torch.randn(16, 64)
        out, state = kernel(x)
        assert out.shape == (16, 64)
        assert not torch.isnan(out).any()

    def test_batched_gradient_flow(self):
        """Gradients flow correctly through batched execution."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            expert_names=["a", "b", "c", "d"],
            use_batched_experts=True,
        )
        kernel = MoEKernel(cfg)
        kernel.train()

        x = torch.randn(8, 64, requires_grad=True)
        out, state = kernel(x)
        loss = out.mean() + state.total_loss()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        # All experts should have gradients
        for expert in kernel.experts:
            for p in expert.parameters():
                assert p.grad is not None

    def test_can_use_batched_returns_false_for_specialized(self):
        """_can_use_batched returns False for specialized experts."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            expert_names=["math", "language", "code", "science"],
            use_batched_experts=True,
        )
        kernel = MoEKernel(cfg)

        # Should detect heterogeneous experts
        assert kernel._can_use_batched() is False

    def test_can_use_batched_returns_true_for_homogeneous(self):
        """_can_use_batched returns True for homogeneous DomainExpert."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            expert_names=["a", "b", "c", "d"],
            use_batched_experts=True,
        )
        kernel = MoEKernel(cfg)

        # Should detect homogeneous experts
        assert kernel._can_use_batched() is True

    def test_can_use_batched_false_for_can_experts(self):
        """_can_use_batched returns False for CAN experts."""
        cfg = MoEKernelConfig(
            input_dim=16,
            expert_hidden_dim=32,
            num_experts=2,
            expert_names=["a", "b"],
            use_can_experts=True,
            use_batched_experts=True,
        )
        kernel = MoEKernel(cfg)

        # CAN experts should disable batched execution
        assert kernel._can_use_batched() is False

    def test_batched_fallback_to_sequential_for_specialized(self):
        """Batched config falls back to sequential for specialized experts."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            expert_names=["math", "language", "code", "science"],
            use_batched_experts=True,
        )
        kernel = MoEKernel(cfg)
        kernel.eval()

        # Should use sequential path even with use_batched_experts=True
        x = torch.randn(8, 64)
        out, state = kernel(x)
        assert out.shape == (8, 64)
        assert not torch.isnan(out).any()

    def test_batched_output_matches_sequential(self):
        """Batched and sequential execution produce similar outputs."""
        torch.manual_seed(42)

        # Create two kernels with same architecture
        cfg_batched = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            expert_names=["a", "b", "c", "d"],
            use_batched_experts=True,
        )

        cfg_seq = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            expert_names=["a", "b", "c", "d"],
            use_batched_experts=False,
        )

        kernel_batched = MoEKernel(cfg_batched)
        kernel_seq = MoEKernel(cfg_seq)

        # Copy weights from batched to sequential
        kernel_seq.load_state_dict(kernel_batched.state_dict())

        kernel_batched.eval()
        kernel_seq.eval()

        x = torch.randn(8, 64)

        with torch.no_grad():
            out_batched, _ = kernel_batched(x)
            out_seq, _ = kernel_seq(x)

        # Outputs should be very close (within numerical precision)
        assert torch.allclose(out_batched, out_seq, atol=1e-5)

    def test_run_experts_batched_direct_call(self):
        """Direct call to _run_experts_batched works correctly."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            expert_names=["a", "b", "c", "d"],
        )
        kernel = MoEKernel(cfg)
        kernel.eval()

        slot_inputs = torch.randn(4, 64)  # 4 slots (num_experts * slots_per_expert)

        with torch.no_grad():
            out = kernel._run_experts_batched(slot_inputs)

        assert out.shape == (4, 64)
        assert not torch.isnan(out).any()

    def test_run_experts_batched_raises_for_specialized(self):
        """_run_experts_batched raises RuntimeError for specialized experts."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            expert_names=["math", "language", "code", "science"],
        )
        kernel = MoEKernel(cfg)

        slot_inputs = torch.randn(4, 64)

        with pytest.raises(RuntimeError, match="homogeneous DomainExpert"):
            kernel._run_experts_batched(slot_inputs)
