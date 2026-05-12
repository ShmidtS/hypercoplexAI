"""MoE forward pass and batched execution tests."""
import pytest
import torch

from src.extensions.moe import MoEKernel, MoEKernelConfig, MLPExpert, MoERouter, SoftMoERouter


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
        assert state["expert_weights"].shape == (16, 4)

    def test_state_expert_weights_shape_3d(self, kernel):
        x = torch.randn(4, 8, 64)
        _, state = kernel(x)
        assert state["expert_weights"].shape == (4, 8, 4)

    def test_state_expert_usage(self, kernel):
        x = torch.randn(16, 64)
        _, state = kernel(x)
        assert state["expert_usage"].shape == (4,)
        # Usage должна суммироваться приблизительно к 1 (среднее combine-весов)
        assert state["expert_usage"].sum().item() > 0

    def test_state_top_expert_idx(self, kernel):
        x = torch.randn(16, 64)
        _, state = kernel(x)
        assert state["top_expert_idx"].shape == (16, 2)
        assert state["top_expert_idx"].max().item() <= 3

    def test_dominant_expert_names(self, kernel):
        x = torch.randn(8, 64)
        _, state = kernel(x)
        expert_names = state["expert_names"]
        top_idx = state["top_expert_idx"][:, 0].tolist()
        names = [expert_names[i] for i in top_idx]
        assert len(names) == 8
        valid = {"math", "language", "code", "science"}
        assert all(n in valid for n in names)

    def test_combine_weights_sum_to_one(self, kernel):
        x = torch.randn(8, 64)
        _, state = kernel(x)
        # combine_weights: (T, num_slots) — сумма по слотам = 1
        combine = state["combine_weights"]
        sums = combine.sum(dim=-1) # (T,)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_dispatch_weights_sum_to_one(self, kernel):
        x = torch.randn(8, 64)
        _, state = kernel(x)
        # dispatch: (T, num_slots) — сумма по токенам = 1 (для T > 1)
        dispatch = state["dispatch_weights"]
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
            max_dev = (state["expert_usage"] - expected).abs().max().item()
            assert max_dev < 0.2

    def test_routing_entropy_positive(self, kernel):
        x = torch.randn(16, 64)
        _, state = kernel(x)
        assert state["routing_entropy"].item() > 0
        assert torch.isfinite(state["routing_entropy"])

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


class TestMoERouterInterface:
    """Tests for MoERouter abstract interface."""

    def test_moe_kernel_implements_interface(self):
        """MoEKernel must inherit from MoERouter."""
        assert issubclass(MoEKernel, MoERouter)

    def test_soft_moe_router_implements_interface(self):
        """SoftMoERouter must inherit from MoERouter."""
        assert issubclass(SoftMoERouter, MoERouter)

    def test_moe_kernel_forward_returns_tuple(self, config):
        """MoEKernel.forward() must return (Tensor, Dict)."""
        kernel = MoEKernel(config)
        kernel.eval()

        x = torch.randn(2, config.input_dim)
        output, info = kernel(x)

        assert isinstance(output, torch.Tensor)
        assert isinstance(info, dict)
        assert output.shape == x.shape
        assert "expert_load" in info
        assert "router_loss" in info

    def test_moe_kernel_get_expert_load(self, config):
        """MoEKernel.get_expert_load() must return Tensor[num_experts]."""
        kernel = MoEKernel(config)

        load = kernel.get_expert_load()
        assert isinstance(load, torch.Tensor)
        assert load.shape == (config.num_experts,)
        # Initial load should be uniform
        assert torch.allclose(load, torch.ones(config.num_experts) / config.num_experts)

    def test_moe_kernel_expert_orthogonalization_loss(self, config):
        """MoEKernel.expert_orthogonalization_loss() must return scalar."""
        kernel = MoEKernel(config)

        loss = kernel.expert_orthogonalization_loss()
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar

    def test_moe_kernel_reset_training_state(self, config):
        """MoEKernel.reset_training_state() must reset EMA."""
        kernel = MoEKernel(config)

        # Modify train_scores
        kernel.train_scores.fill_(0.5)
        kernel.reset_training_state()

        # Should be uniform again
        expected = torch.ones(config.num_experts) / config.num_experts
        assert torch.allclose(kernel.train_scores, expected)

    def test_soft_moe_router_get_expert_load(self):
        """SoftMoERouter.get_expert_load() must return Tensor[num_experts]."""
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

    def test_moe_kernel_info_dict_contents(self, config):
        """MoEKernel forward info dict must contain all required keys."""
        kernel = MoEKernel(config)
        kernel.eval()

        x = torch.randn(2, config.input_dim)
        _, info = kernel(x)

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
        """Both MoEKernel and SoftMoERouter can be used as MoERouter."""
        kernel = MoEKernel(config)

        soft_router = SoftMoERouter(
            input_dim=config.input_dim,
            num_experts=config.num_experts,
            expert_dim=config.expert_hidden_dim,
        )

        # Both should be instances of MoERouter
        assert isinstance(kernel, MoERouter)
        assert isinstance(soft_router, MoERouter)

        # Both should work with same input shape
        x = torch.randn(2, config.input_dim)

        out1, info1 = kernel(x)
        out2, info2 = soft_router(x)

        assert out1.shape == x.shape
        assert out2.shape == x.shape


class TestBatchedExpertExecution:
    """Tests for batched expert execution via torch.bmm."""

    def test_config_has_use_batched_experts(self):
        """MoEKernelConfig has use_batched_experts parameter with default True."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
        )
        assert cfg.use_batched_experts is True  # Default is now True

    def test_batched_with_homogeneous_experts(self):
        """Batched execution works with homogeneous MLPExpert instances."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            expert_names=["custom_0", "custom_1", "custom_2", "custom_3"],
            use_batched_experts=True,
        )
        kernel = MoEKernel(cfg)
        kernel.eval()

        # All experts should be plain MLPExpert (MLPExpert alias)
        for expert in kernel.experts:
            assert isinstance(expert, MLPExpert)

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
        loss = out.mean() + state["total_loss"]
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
        )
        kernel = MoEKernel(cfg)

        # Should detect heterogeneous experts
        assert kernel._can_use_batched() is False

    def test_can_use_batched_returns_true_for_homogeneous(self):
        """_can_use_batched returns True for homogeneous MLPExpert."""
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

        with pytest.raises(RuntimeError, match="homogeneous architecture"):
            kernel._run_experts_batched(slot_inputs)
