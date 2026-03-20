"""Unit tests for CliffordInteractionLayer.

Tests verify:
1. Basic forward pass with correct shapes
2. Multi-scale shift interactions
3. Inner/wedge product computation via CliffordAlgebra
4. Numerical stability (NaN/Inf handling)
5. Gradient flow
"""

import pytest
import torch
import torch.nn as nn

from src.core.clifford_interaction import CliffordInteractionLayer, CliffordInteractionExpert, CliffordFFN


class TestCliffordInteractionLayer:
    """Tests for CliffordInteractionLayer."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        layer = CliffordInteractionLayer()
        
        assert layer.dim == 16
        assert layer.shifts == [1, 2, 4, 8, 16]
        assert layer.use_inner is True
        assert layer.use_wedge is True
        assert isinstance(layer.clifford, nn.Module)

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        layer = CliffordInteractionLayer(
            dim=16,
            shifts=[1, 2, 4],
            use_inner=False,
            use_wedge=True,
            dropout=0.2
        )
        
        assert layer.dim == 16
        assert layer.shifts == [1, 2, 4]
        assert layer.use_inner is False
        assert layer.use_wedge is True

    def test_forward_2d_input(self):
        """Test forward pass with 2D input (B, D)."""
        layer = CliffordInteractionLayer()
        x = torch.randn(8, 16)  # (B, D)
        
        output = layer(x)
        
        assert output.shape == (8, 16)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_3d_input(self):
        """Test forward pass with 3D input (B, T, D)."""
        layer = CliffordInteractionLayer()
        x = torch.randn(4, 32, 16)  # (B, T, D)
        
        output = layer(x)
        
        assert output.shape == (4, 32, 16)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_short_sequence(self):
        """Test forward pass with sequence shorter than shifts."""
        layer = CliffordInteractionLayer(shifts=[1, 2, 4, 8, 16])
        x = torch.randn(2, 3, 16)  # T=3 < all shifts
        
        output = layer(x)
        
        assert output.shape == (2, 3, 16)
        assert not torch.isnan(output).any()

    def test_forward_single_token(self):
        """Test forward pass with single token (T=1)."""
        layer = CliffordInteractionLayer()
        x = torch.randn(4, 1, 16)  # (B, 1, D)
        
        output = layer(x)
        
        assert output.shape == (4, 1, 16)
        assert not torch.isnan(output).any()

    def test_inner_product_mode(self):
        """Test layer with only inner product enabled."""
        layer = CliffordInteractionLayer(use_inner=True, use_wedge=False)
        x = torch.randn(4, 16, 16)
        
        output = layer(x)
        
        assert output.shape == (4, 16, 16)
        assert not torch.isnan(output).any()

    def test_wedge_product_mode(self):
        """Test layer with only wedge product enabled."""
        layer = CliffordInteractionLayer(use_inner=False, use_wedge=True)
        x = torch.randn(4, 16, 16)
        
        output = layer(x)
        
        assert output.shape == (4, 16, 16)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        layer = CliffordInteractionLayer()
        x = torch.randn(4, 8, 16, requires_grad=True)
        
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.isnan(x.grad).any()

    def test_numerical_stability_large_values(self):
        """Test layer handles large input values without NaN/Inf."""
        layer = CliffordInteractionLayer()
        x = torch.randn(4, 8, 16) * 1e3  # Large values
        
        output = layer(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_numerical_stability_small_values(self):
        """Test layer handles small input values without underflow."""
        layer = CliffordInteractionLayer()
        x = torch.randn(4, 8, 16) * 1e-6  # Small values
        
        output = layer(x)
        
        assert not torch.isnan(output).any()
        assert output.shape == (4, 8, 16)

    def test_deterministic_output(self):
        """Test that layer produces deterministic output in eval mode."""
        layer = CliffordInteractionLayer(dropout=0.0)
        layer.eval()
        x = torch.randn(4, 8, 16)
        
        output1 = layer(x)
        output2 = layer(x)
        
        assert torch.allclose(output1, output2)

    def test_train_eval_mode_difference(self):
        """Test that dropout creates difference between train/eval modes."""
        layer = CliffordInteractionLayer(dropout=0.5)
        x = torch.randn(4, 8, 16)
        
        layer.train()
        outputs_train = [layer(x) for _ in range(5)]
        
        layer.eval()
        outputs_eval = [layer(x) for _ in range(5)]
        
        # Train outputs should differ due to dropout
        train_diff = sum(not torch.allclose(outputs_train[0], o) for o in outputs_train[1:])
        assert train_diff > 0  # At least some differ
        
        # Eval outputs should be identical
        for o in outputs_eval[1:]:
            assert torch.allclose(outputs_eval[0], o)

    def test_batch_independence(self):
        """Test that samples in batch are processed independently."""
        layer = CliffordInteractionLayer()
        x1 = torch.randn(1, 8, 16)
        x2 = torch.randn(1, 8, 16)
        x_batch = torch.cat([x1, x2], dim=0)
        
        layer.eval()
        output_batch = layer(x_batch)
        output1 = layer(x1)
        output2 = layer(x2)
        
        assert torch.allclose(output_batch[0:1], output1, atol=1e-5)
        assert torch.allclose(output_batch[1:2], output2, atol=1e-5)

    def test_device_consistency(self):
        """Test layer works on different devices."""
        layer = CliffordInteractionLayer()
        x_cpu = torch.randn(4, 8, 16)
        
        # CPU
        output_cpu = layer(x_cpu)
        assert output_cpu.device == x_cpu.device
        
        # CUDA if available - just verify device placement works
        if torch.cuda.is_available():
            layer_cuda = layer.cuda()
            x_cuda = x_cpu.cuda()
            output_cuda = layer_cuda(x_cuda)
            assert output_cuda.device == x_cuda.device
            # Note: outputs may differ due to dropout randomness


class TestCliffordInteractionExpert:
    """Tests for existing CliffordInteractionExpert class."""

    def test_forward_2d(self):
        """Test expert with 2D input."""
        expert = CliffordInteractionExpert(input_dim=16)
        x = torch.randn(8, 16)
        
        output = expert(x)
        
        assert output.shape == (8, 16)

    def test_forward_3d(self):
        """Test expert with 3D input."""
        expert = CliffordInteractionExpert(input_dim=16)
        x = torch.randn(4, 8, 16)
        
        output = expert(x)
        
        assert output.shape == (4, 8, 16)

    def test_no_gate_mode(self):
        """Test expert without gating."""
        expert = CliffordInteractionExpert(input_dim=16, use_gate=False)
        x = torch.randn(4, 8, 16)
        
        output = expert(x)
        
        assert output.shape == (4, 8, 16)
        assert not torch.isnan(output).any()


class TestCliffordFFN:
    """Tests for CliffordFFN class."""

    def test_forward_2d(self):
        """Test FFN with 2D input."""
        ffn = CliffordFFN(input_dim=16)
        x = torch.randn(8, 16)
        
        output = ffn(x)
        
        assert output.shape == (8, 16)

    def test_forward_3d(self):
        """Test FFN with 3D input."""
        ffn = CliffordFFN(input_dim=16)
        x = torch.randn(4, 8, 16)
        
        output = ffn(x)
        
        assert output.shape == (4, 8, 16)

    def test_custom_hidden_dim(self):
        """Test FFN with custom hidden dimension."""
        ffn = CliffordFFN(input_dim=16, hidden_dim=32)
        x = torch.randn(4, 16)
        
        output = ffn(x)
        
        assert output.shape == (4, 16)

    def test_gradient_flow(self):
        """Test gradient flow through FFN."""
        ffn = CliffordFFN(input_dim=16)
        x = torch.randn(4, 8, 16, requires_grad=True)
        
        output = ffn(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestIntegration:
    """Integration tests for CliffordInteractionLayer with HDIM components."""

    def test_with_clifford_algebra(self):
        """Test that layer correctly uses CliffordAlgebra for operations."""
        from src.core.hypercomplex import CliffordAlgebra
        
        layer = CliffordInteractionLayer()
        clifford = CliffordAlgebra(p=3, q=1, r=0)
        
        # Verify same algebra configuration
        assert layer.clifford.p == clifford.p
        assert layer.clifford.q == clifford.q
        assert layer.clifford.r == clifford.r
        assert layer.clifford.dim == 16

    def test_multivector_preservation(self):
        """Test that output maintains multivector structure."""
        layer = CliffordInteractionLayer()
        x = torch.randn(4, 8, 16)
        
        output = layer(x)
        
        # Output should have same dimensionality (16D multivector)
        assert output.shape[-1] == 16

    def test_cascade_layers(self):
        """Test stacking multiple layers."""
        layers = nn.Sequential(
            CliffordInteractionLayer(),
            CliffordInteractionLayer(),
            CliffordInteractionLayer()
        )
        x = torch.randn(4, 8, 16)
        
        output = layers(x)
        
        assert output.shape == (4, 8, 16)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
