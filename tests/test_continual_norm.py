"""Tests for ContinualNorm — streaming normalization stability."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest
import torch
import torch.nn as nn

from src.core.continual_norm import ContinualNorm, ContinualNormLayer


class TestContinualNormInit:
    """Test initialization and configuration."""

    def test_basic_init(self):
        """Test basic initialization with default parameters."""
        norm = ContinualNorm(num_features=64)
        assert norm.num_features == 64
        assert norm.momentum == 0.1
        assert norm.eps == 1e-5
        assert norm.affine is True

    def test_custom_momentum(self):
        """Test custom momentum parameter."""
        norm = ContinualNorm(num_features=32, momentum=0.01)
        assert norm.momentum == 0.01

    def test_no_affine(self):
        """Test initialization without affine transform."""
        norm = ContinualNorm(num_features=64, affine=False)
        assert norm.affine is False
        assert norm.weight is None
        assert norm.bias is None

    def test_running_stats_initialized(self):
        """Test that running statistics are properly initialized."""
        norm = ContinualNorm(num_features=128)
        assert norm.running_mean.shape == (128,)
        assert norm.running_var.shape == (128,)
        assert torch.all(norm.running_mean == 0).item()
        assert torch.all(norm.running_var == 1).item()
        assert norm.num_batches.item() == 0


class TestContinualNormForward:
    """Test forward pass behavior."""

    def test_forward_2d_training(self):
        """Test forward pass with 2D input in training mode."""
        norm = ContinualNorm(num_features=16)
        x = torch.randn(32, 16) * 10 + 5  # Non-zero mean, high variance

        norm.train()
        out = norm(x)

        # Output shape preserved
        assert out.shape == x.shape

        # Running stats updated
        assert norm.num_batches.item() == 1

    def test_forward_4d_training(self):
        """Test forward pass with 4D input (conv) in training mode."""
        norm = ContinualNorm(num_features=8)
        x = torch.randn(4, 8, 16, 16) * 5 + 2

        norm.train()
        out = norm(x)

        assert out.shape == x.shape
        assert norm.num_batches.item() == 1

    def test_eval_mode_uses_running_stats(self):
        """Test that eval mode uses running statistics."""
        norm = ContinualNorm(num_features=16)

        # Train for several batches to accumulate stats
        norm.train()
        for _ in range(10):
            x = torch.randn(32, 16) * 10 + 5
            norm(x)

        running_mean = norm.running_mean.clone()
        running_var = norm.running_var.clone()

        # Switch to eval - stats should not change
        norm.eval()
        x = torch.randn(32, 16) * 100 + 50
        out = norm(x)

        assert torch.allclose(norm.running_mean, running_mean)
        assert torch.allclose(norm.running_var, running_var)

    def test_output_normalization_in_eval(self):
        """Test that output is properly normalized in eval mode."""
        norm = ContinualNorm(num_features=4, affine=False)

        # Manually set running stats
        norm.running_mean.data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        norm.running_var.data = torch.tensor([1.0, 4.0, 9.0, 16.0])

        norm.eval()
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # Matches running mean

        out = norm(x)

        # Should be approximately zero (input matches mean)
        expected = torch.zeros(1, 4)
        assert torch.allclose(out, expected, atol=1e-4)


class TestContinualNormContinualBehavior:
    """Test continual learning behavior - no reset across batches."""

    def test_no_reset_across_batches(self):
        """Test that stats accumulate across batches (no reset)."""
        norm = ContinualNorm(num_features=8, momentum=0.5)

        norm.train()

        # First batch: all ones
        x1 = torch.ones(16, 8)
        norm(x1)

        mean1 = norm.running_mean.clone()
        var1 = norm.running_var.clone()

        # Second batch: all twos
        x2 = torch.ones(16, 8) * 2
        norm(x2)

        mean2 = norm.running_mean.clone()
        var2 = norm.running_var.clone()

        # Stats should have changed (EMA update)
        assert not torch.allclose(mean1, mean2)
        assert not torch.allclose(var1, var2)

        # num_batches should be 2
        assert norm.num_batches.item() == 2

    def test_ema_convergence(self):
        """Test that EMA converges to true distribution."""
        norm = ContinualNorm(num_features=1, momentum=0.1)
        norm.train()

        # Generate samples from N(5, 2^2)
        true_mean = 5.0
        true_std = 2.0

        for _ in range(500):
            x = torch.randn(32, 1) * true_std + true_mean
            norm(x)

        # Running stats should approximate true distribution
        assert abs(norm.running_mean.item() - true_mean) < 0.5
        assert abs(norm.running_var.sqrt().item() - true_std) < 0.5

    def test_distribution_shift_handling(self):
        """Test handling of gradual distribution shift."""
        norm = ContinualNorm(num_features=4, momentum=0.1)
        norm.train()

        # Gradually shift mean from 0 to 10 over many batches
        for i in range(200):
            mean = i * 0.05
            x = torch.randn(16, 4) + mean
            norm(x)

        # Running mean should track the shift (EMA with moderate momentum)
        # With momentum=0.1 and gradual shift, expect partial tracking
        final_running_mean = norm.running_mean.mean().item()
        assert final_running_mean > 2.0  # Should have tracked some of the shift


class TestContinualNormAffine:
    """Test affine transformation behavior."""

    def test_affine_parameters(self):
        """Test that affine parameters are learnable."""
        norm = ContinualNorm(num_features=16)
        assert isinstance(norm.weight, nn.Parameter)
        assert isinstance(norm.bias, nn.Parameter)

    def test_affine_transform(self):
        """Test that affine transform is applied correctly."""
        norm = ContinualNorm(num_features=4)

        # Set specific values
        with torch.no_grad():
            norm.weight.data = torch.tensor([1.0, 2.0, 3.0, 4.0])
            norm.bias.data = torch.tensor([0.1, 0.2, 0.3, 0.4])
            norm.running_mean.data = torch.zeros(4)
            norm.running_var.data = torch.ones(4)

        norm.eval()
        x = torch.ones(1, 4)
        out = norm(x)

        # x_norm = 1 (since mean=0, var=1)
        # out = weight * 1 + bias
        expected = torch.tensor([[1.1, 2.2, 3.3, 4.4]])
        assert torch.allclose(out, expected, atol=1e-4)

    def test_affine_4d(self):
        """Test affine transform with 4D input."""
        norm = ContinualNorm(num_features=3)

        with torch.no_grad():
            norm.weight.data = torch.tensor([2.0, 2.0, 2.0])
            norm.bias.data = torch.tensor([1.0, 1.0, 1.0])
            norm.running_mean.data = torch.zeros(3)
            norm.running_var.data = torch.ones(3)

        norm.eval()
        x = torch.ones(2, 3, 4, 4)
        out = norm(x)

        assert out.shape == x.shape
        assert torch.allclose(out, torch.ones(2, 3, 4, 4) * 3)


class TestContinualNormReset:
    """Test reset functionality."""

    def test_reset_running_stats(self):
        """Test that reset properly clears running stats."""
        norm = ContinualNorm(num_features=8)

        # Accumulate some stats
        norm.train()
        for _ in range(10):
            x = torch.randn(16, 8) * 10 + 5
            norm(x)

        # Reset
        norm.reset_running_stats()

        assert torch.all(norm.running_mean == 0).item()
        assert torch.all(norm.running_var == 1).item()
        assert norm.num_batches.item() == 0


class TestContinualNormLayer:
    """Test ContinualNormLayer alternative."""

    def test_init(self):
        """Test basic initialization."""
        layer = ContinualNormLayer(normalized_shape=64)
        assert layer.normalized_shape == 64
        assert layer.eps == 1e-5
        assert layer.elementwise_affine is True

    def test_forward(self):
        """Test forward pass."""
        layer = ContinualNormLayer(normalized_shape=16)
        x = torch.randn(8, 16) * 10 + 5

        layer.train()
        out = layer(x)

        assert out.shape == x.shape

    def test_monitoring_stats(self):
        """Test that monitoring stats are tracked."""
        layer = ContinualNormLayer(normalized_shape=8)
        layer.train()

        for _ in range(10):
            x = torch.randn(16, 8)
            layer(x)

        # Should have accumulated some stats
        assert layer.num_batches.item() == 10


class TestContinualNormIntegration:
    """Integration tests with PyTorch ecosystem."""

    def test_module_registration(self):
        """Test that module is properly registered."""
        model = nn.Sequential(
            nn.Linear(32, 64),
            ContinualNorm(64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        # Should be able to access parameters
        params = list(model.parameters())
        assert len(params) > 0

        # Find ContinualNorm
        norm_layer = None
        for module in model.modules():
            if isinstance(module, ContinualNorm):
                norm_layer = module
                break

        assert norm_layer is not None

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        norm = ContinualNorm(num_features=16)
        x = torch.randn(8, 16, requires_grad=True)

        out = norm(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert norm.weight.grad is not None
        assert norm.bias.grad is not None

    def test_device_movement(self):
        """Test that module can be moved between devices."""
        if torch.cuda.is_available():
            norm = ContinualNorm(num_features=16)
            x = torch.randn(8, 16)

            # CPU
            out_cpu = norm(x)
            assert out_cpu.device.type == 'cpu'

            # CUDA
            norm_cuda = norm.cuda()
            x_cuda = x.cuda()
            out_cuda = norm_cuda(x_cuda)
            assert out_cuda.device.type == 'cuda'
        else:
            pytest.skip("CUDA not available")

    def test_state_dict(self):
        """Test state dict serialization."""
        norm = ContinualNorm(num_features=16)

        # Modify some values
        norm.running_mean.data = torch.randn(16)
        norm.running_var.data = torch.abs(torch.randn(16)) + 0.1
        norm.num_batches.data = torch.tensor(42)

        # Save and load
        state = norm.state_dict()
        norm2 = ContinualNorm(num_features=16)
        norm2.load_state_dict(state)

        assert torch.allclose(norm.running_mean, norm2.running_mean)
        assert torch.allclose(norm.running_var, norm2.running_var)
        assert norm.num_batches == norm2.num_batches


class TestContinualNormEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_dimension(self):
        """Test that invalid dimensions raise error."""
        norm = ContinualNorm(num_features=16)
        x = torch.randn(8)  # 1D tensor

        with pytest.raises(ValueError, match="Expected 2D, 3D or 4D"):
            norm(x)

    def test_single_batch(self):
        """Test behavior with single batch."""
        norm = ContinualNorm(num_features=8)
        x = torch.randn(1, 8)

        norm.train()
        out = norm(x)
        assert out.shape == x.shape

    def test_very_small_variance(self):
        """Test numerical stability with small variance."""
        norm = ContinualNorm(num_features=4, eps=1e-10)
        x = torch.ones(16, 4) + torch.randn(16, 4) * 1e-8

        norm.train()
        out = norm(x)
        assert torch.isfinite(out).all()

    def test_large_values(self):
        """Test numerical stability with large values."""
        norm = ContinualNorm(num_features=4)
        x = torch.randn(16, 4) * 1e6

        norm.train()
        out = norm(x)
        assert torch.isfinite(out).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
