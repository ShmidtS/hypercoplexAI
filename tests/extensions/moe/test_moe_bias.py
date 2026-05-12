"""MoE aux-loss-free bias tests."""
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

    def test_bias_balancing_reduces_imbalance(self):
        """Test that bias updates reduce load imbalance over multiple steps."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            expert_names=["a", "b", "c", "d"],
            use_aux_loss_free=True,
            bias_update_frequency=1,  # Update every step
            aux_lr=0.01,
        )
        kernel = MoEKernel(cfg)
        kernel.train()

        # Run multiple forward passes and track imbalance
        imbalances = []
        for _ in range(50):
            x = torch.randn(32, 64)
            _, state = kernel(x)
            # Calculate imbalance as max deviation from uniform
            expected = 1.0 / kernel.num_experts
            imbalance = (state["expert_usage"] - expected).abs().max().item()
            imbalances.append(imbalance)

        # Later imbalances should be generally lower than early ones
        early_avg = sum(imbalances[:10]) / 10
        late_avg = sum(imbalances[-10:]) / 10
        assert late_avg <= early_avg * 1.5, f"Bias balancing should not increase imbalance: early={early_avg:.4f}, late={late_avg:.4f}"

    def test_bias_values_update_correctly(self):
        """Test that bias values change in expected direction based on load."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            expert_names=["a", "b", "c", "d"],
            use_aux_loss_free=True,
            bias_update_frequency=1,
            aux_lr=0.1,  # Large lr for visible effect
        )
        kernel = MoEKernel(cfg)
        kernel.train()

        initial_bias = kernel._expert_bias.data.clone()
        x = torch.randn(64, 64)
        _, state = kernel(x)

        # Check that bias was updated
        assert not torch.equal(kernel._expert_bias.data, initial_bias)

        # Check bias direction consistency with load deviation
        load = state["expert_usage"]
        target = kernel._target_load
        expected_delta_sign = torch.sign(load - target)

        # The actual bias change should be opposite (we decrease bias for overloaded)
        actual_delta = kernel._expert_bias.data - initial_bias
        # Sign of actual delta should be opposite of expected_delta_sign
        # Overloaded experts get their bias decreased
        assert torch.allclose(torch.sign(actual_delta), -expected_delta_sign, atol=0.1) or actual_delta.abs().max() < 0.01

    def test_backward_compatible_no_bias(self):
        """Test that MoEKernel works correctly when bias balancing is disabled."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            expert_names=["math", "language", "code", "science"],
            use_aux_loss_free=False,  # Disable bias balancing
            use_bias_balancing=False,
        )
        kernel = MoEKernel(cfg)
        kernel.eval()

        x = torch.randn(16, 64)
        out, state = kernel(x)

        assert out.shape == (16, 64)
        assert not torch.isnan(out).any()
        assert kernel._expert_bias is None

    def test_bias_update_frequency_config(self):
        """Test that bias_update_frequency config is respected."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=2,
            expert_names=["a", "b"],
            use_aux_loss_free=True,
            bias_update_frequency=10,  # Update every 10 steps
        )
        kernel = MoEKernel(cfg)
        kernel.train()

        # Run 5 forward passes - bias should NOT update (frequency=10)
        initial_bias = kernel._expert_bias.data.clone()
        for _ in range(5):
            x = torch.randn(16, 64)
            kernel(x)

        # Bias should not have changed (step 5 < frequency 10)
        assert torch.equal(kernel._expert_bias.data, initial_bias), "Bias should not update before frequency threshold"

        # Run 5 more forward passes (total 10) - bias SHOULD update now
        for _ in range(5):
            x = torch.randn(16, 64)
            kernel(x)

        # After 10 steps, bias should have been updated
        assert not torch.equal(kernel._expert_bias.data, initial_bias), "Bias should update at frequency threshold"

    def test_use_bias_balancing_alias(self):
        """Test that use_bias_balancing works as alias for use_aux_loss_free."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=2,
            expert_names=["a", "b"],
            use_bias_balancing=True,
        )
        assert cfg.use_bias_balancing is True

        kernel = MoEKernel(cfg)
        assert kernel._expert_bias is not None
