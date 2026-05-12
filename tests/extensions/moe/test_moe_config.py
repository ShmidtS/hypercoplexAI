"""MoE config and dynamic expert count tests."""
import logging

import pytest
import torch

from src.extensions.moe import MoEKernel, MoEKernelConfig


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
        assert state["expert_weights"].shape == (8, 1)
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
        assert state["expert_weights"].shape == (16, 2)
        assert state["expert_usage"].shape == (2,)

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
        assert state["expert_weights"].shape == (32, 8)
        assert state["expert_usage"].shape == (8,)

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
        assert state["expert_weights"].shape == (64, 16)
        assert state["expert_usage"].shape == (16,)

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
        loss = out.mean() + state["total_loss"]
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
        loss = out.mean() + state["total_loss"]
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        # All experts should have gradients
        for expert in kernel.experts:
            for p in expert.parameters():
                assert p.grad is not None


class TestConfigAndWarnings:
    """Tests for config parameters and heterogeneous expert warnings."""

    def test_config_has_batched_fallback(self):
        """MoEKernelConfig has batched_fallback parameter."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=2,
        )
        assert cfg.batched_fallback is True

    def test_config_has_expert_homogeneity_check(self):
        """MoEKernelConfig has expert_homogeneity_check parameter."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=2,
        )
        assert cfg.expert_homogeneity_check is True

    def test_heterogeneous_fallback_warning(self, caplog):
        """Heterogeneous experts trigger warning when use_batched_experts=True."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            expert_names=["math", "language", "code", "science"],
            use_batched_experts=True,
            expert_homogeneity_check=True,
        )

        with caplog.at_level(logging.WARNING):
            kernel = MoEKernel(cfg)

        # Should have logged a warning about heterogeneous experts
        assert any("heterogeneous" in record.message.lower() for record in caplog.records)

    def test_heterogeneous_fallback_disabled(self, caplog):
        """No warning when expert_homogeneity_check=False."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            expert_names=["math", "language", "code", "science"],
            use_batched_experts=True,
            expert_homogeneity_check=False,
        )

        with caplog.at_level(logging.WARNING):
            kernel = MoEKernel(cfg)

        # Should NOT have logged warning
        assert not any("heterogeneous" in record.message.lower() for record in caplog.records)

    def test_batched_fallback_config_option(self):
        """batched_fallback config option exists."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            batched_fallback=False,
        )
        assert cfg.batched_fallback is False
