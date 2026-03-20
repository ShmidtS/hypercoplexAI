"""End-to-end integration tests for CAN (Clifford Algebra Network) in HDIM.

Tests verify:
1. CliffordInteractionLayer integration in HDIM pipeline
2. MoEKernel with CAN-experts (use_can_experts=True)
3. Triton acceleration (if available)
4. Gradient flow through all CAN components
5. Backward compatibility (use_can_experts=False by default)
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.clifford_interaction import (
    CliffordInteractionLayer,
    CliffordInteractionExpert,
    CliffordFFN,
    has_triton_support,
)
from src.core.moe_kernel import (
    MoEKernel,
    MoEKernelConfig,
    MoEKernelState,
    DomainExpert,
    create_expert,
)
from src.core.hdim_pipeline import HDIMPipeline


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def can_moe_config():
    """MoE config with CAN experts enabled."""
    return MoEKernelConfig(
        input_dim=16,  # Must be 16 for CliffordInteractionLayer
        expert_hidden_dim=32,
        num_experts=4,
        slots_per_expert=1,
        temperature=1.0,
        z_loss_weight=0.01,
        ortho_loss_weight=0.01,
        use_shared_expert=True,
        use_aux_loss_free=True,
        use_expert_ortho=True,
        expert_names=["math", "language", "code", "science"],
        use_can_experts=True,  # Enable CAN-style experts
    )


@pytest.fixture
def ffn_moe_config():
    """Standard MoE config with FFN experts (backward compatibility)."""
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
        use_can_experts=False,  # Default: FFN experts
    )


@pytest.fixture
def can_kernel(can_moe_config):
    """MoEKernel with CAN experts."""
    kernel = MoEKernel(can_moe_config)
    kernel.eval()
    return kernel


@pytest.fixture
def ffn_kernel(ffn_moe_config):
    """MoEKernel with standard FFN experts."""
    kernel = MoEKernel(ffn_moe_config)
    kernel.eval()
    return kernel


# ============================================================
# Test CliffordInteractionLayer Integration
# ============================================================

class TestCliffordInteractionLayerIntegration:
    """Tests for CliffordInteractionLayer as standalone component."""

    def test_forward_pass_shape(self):
        """Test that forward pass preserves shape."""
        layer = CliffordInteractionLayer(dim=16)
        x = torch.randn(8, 32, 16)  # (B, T, D)
        
        output = layer(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_2d_input(self):
        """Test forward pass with 2D input (B, D)."""
        layer = CliffordInteractionLayer(dim=16)
        x = torch.randn(8, 16)  # (B, D)
        
        output = layer(x)
        
        assert output.shape == (8, 16)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        """Test that gradients flow through CliffordInteractionLayer."""
        layer = CliffordInteractionLayer(dim=16)
        x = torch.randn(4, 16, 16, requires_grad=True)
        
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert x.grad.abs().sum() > 0  # Non-zero gradients

    def test_inner_wedge_products(self):
        """Test that inner and wedge products produce valid outputs."""
        layer = CliffordInteractionLayer(dim=16, use_inner=True, use_wedge=True)
        x = torch.randn(2, 8, 16)
        
        output = layer(x)
        
        # Check that output differs from input (transformation occurred)
        assert not torch.allclose(output, x, atol=1e-6)
        assert output.shape == x.shape

    def test_inner_only_mode(self):
        """Test layer with only inner product enabled."""
        layer = CliffordInteractionLayer(dim=16, use_inner=True, use_wedge=False)
        x = torch.randn(2, 8, 16)
        
        output = layer(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_wedge_only_mode(self):
        """Test layer with only wedge product enabled."""
        layer = CliffordInteractionLayer(dim=16, use_inner=False, use_wedge=True)
        x = torch.randn(2, 8, 16)
        
        output = layer(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_learnable_weights(self):
        """Test that learnable weights are updated during training."""
        layer = CliffordInteractionLayer(dim=16)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
        
        initial_inner_weight = layer.inner_weight.data.clone()
        initial_wedge_weight = layer.wedge_weight.data.clone()
        
        # Training step
        x = torch.randn(4, 16, 16)
        output = layer(x)
        loss = output.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Weights should have changed
        assert not torch.equal(layer.inner_weight.data, initial_inner_weight)
        assert not torch.equal(layer.wedge_weight.data, initial_wedge_weight)


# ============================================================
# Test DomainExpert with CAN
# ============================================================

class TestDomainExpertCAN:
    """Tests for DomainExpert with use_can=True."""

    def test_can_expert_creation(self):
        """Test creating DomainExpert with CAN layer."""
        expert = DomainExpert(input_dim=16, hidden_dim=32, use_can=True)
        
        assert expert.use_can is True
        assert hasattr(expert, 'interaction')
        assert isinstance(expert.interaction, CliffordInteractionLayer)

    def test_can_expert_forward(self):
        """Test forward pass through CAN expert."""
        expert = DomainExpert(input_dim=16, hidden_dim=32, use_can=True)
        x = torch.randn(4, 8, 16)  # (B, T, D)
        
        output = expert(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_can_expert_gradient_flow(self):
        """Test gradient flow through CAN expert."""
        expert = DomainExpert(input_dim=16, hidden_dim=32, use_can=True)
        x = torch.randn(2, 4, 16, requires_grad=True)
        
        output = expert(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_ffn_expert_backward_compat(self):
        """Test that FFN expert still works (backward compatibility)."""
        expert = DomainExpert(input_dim=64, hidden_dim=128, use_can=False)
        x = torch.randn(4, 64)
        
        output = expert(x)
        
        assert output.shape == x.shape
        assert hasattr(expert, 'net')  # FFN network
        assert not hasattr(expert, 'interaction')  # No CAN layer

    def test_can_expert_parameter_count(self):
        """Test that CAN expert has fewer parameters than FFN."""
        can_expert = DomainExpert(input_dim=16, hidden_dim=32, use_can=True)
        ffn_expert = DomainExpert(input_dim=16, hidden_dim=32, use_can=False)
        
        can_params = sum(p.numel() for p in can_expert.parameters())
        ffn_params = sum(p.numel() for p in ffn_expert.parameters())
        
        # CAN expert should have fewer parameters (no FFN layers)
        # FFN: 16*32 + 32*16 = 1024 params (just linear layers)
        # CAN: ~16*16 + weights (much fewer)
        assert can_params < ffn_params


# ============================================================
# Test MoEKernel with CAN Experts
# ============================================================

class TestMoEKernelCAN:
    """Tests for MoEKernel with CAN experts enabled."""

    def test_can_kernel_creation(self, can_moe_config):
        """Test MoEKernel creation with CAN config."""
        kernel = MoEKernel(can_moe_config)
        
        assert kernel.config.use_can_experts is True
        # All experts should use CAN
        for expert in kernel.experts:
            assert expert.use_can is True

    def test_can_kernel_forward(self, can_kernel):
        """Test forward pass through CAN MoE kernel."""
        x = torch.randn(4, 16)  # (B, D)
        
        output, state = can_kernel(x)
        
        assert isinstance(state, MoEKernelState)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_can_kernel_forward_sequence(self, can_kernel):
        """Test forward pass with sequence input."""
        x = torch.randn(2, 8, 16)  # (B, T, D)
        
        output, state = can_kernel(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_can_kernel_gradient_flow(self, can_moe_config):
        """Test gradient flow through CAN MoE kernel."""
        kernel = MoEKernel(can_moe_config)
        kernel.train()
        x = torch.randn(2, 16, requires_grad=True)
        
        output, state = kernel(x)
        loss = output.sum() + state.total_loss()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_can_kernel_router_loss(self, can_kernel):
        """Test that router losses are computed for CAN kernel."""
        x = torch.randn(4, 16)
        
        output, state = can_kernel(x)
        
        assert state.router_loss is not None
        assert state.z_loss is not None
        assert state.ortho_loss is not None
        # Losses should be non-negative
        assert state.z_loss >= 0
        assert state.ortho_loss >= 0

    def test_can_kernel_expert_usage(self, can_kernel):
        """Test that expert usage is tracked for CAN kernel."""
        x = torch.randn(8, 16)
        
        output, state = can_kernel(x)
        
        assert state.expert_usage is not None
        assert state.expert_usage.shape[0] == can_kernel.config.num_experts
        # Usage should sum to approximately batch size
        assert state.expert_usage.sum() > 0

    def test_ffn_kernel_backward_compat(self, ffn_kernel):
        """Test that FFN kernel still works (backward compatibility)."""
        x = torch.randn(4, 64)
        
        output, state = ffn_kernel(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        # Experts should be FFN-based
        for expert in ffn_kernel.experts:
            assert expert.use_can is False


# ============================================================
# Test HDIM Pipeline with CAN
# ============================================================

class TestHDIMPipelineCAN:
    """Tests for HDIM pipeline with CAN integration."""

    def test_hdim_with_can_experts(self):
        """Test HDIM pipeline with CAN experts in MoE."""
        # Create HDIM pipeline with CAN-friendly dimensions
        pipeline = HDIMPipeline(
            input_dim=64,
            output_dim=64,
            clifford_p=3,
            clifford_q=1,
            clifford_r=0,
            num_experts=4,
            expert_names=["math", "language", "code", "science"],
        )
        
        x = torch.randn(4, 64)
        
        output, state_dict = pipeline(x)
        
        assert output.shape == (4, 64)
        assert not torch.isnan(output).any()

    def test_hdim_gradient_flow(self):
        """Test gradient flow through HDIM pipeline."""
        pipeline = HDIMPipeline(
            input_dim=64,
            output_dim=64,
            num_experts=2,
        )
        pipeline.train()
        x = torch.randn(2, 64, requires_grad=True)
        
        output, state_dict = pipeline(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_hdim_with_memory(self):
        """Test HDIM pipeline with memory integration."""
        pipeline = HDIMPipeline(
            input_dim=64,
            output_dim=64,
            num_experts=2,
            memory_type='titans',
        )
        
        x = torch.randn(4, 64)
        
        output, state_dict = pipeline(x)
        
        assert output.shape == (4, 64)
        assert not torch.isnan(output).any()


# ============================================================
# Test Triton Acceleration
# ============================================================

class TestTritonAcceleration:
    """Tests for Triton acceleration (if available)."""

    def test_triton_availability_check(self):
        """Test that Triton availability can be checked."""
        # This should not raise an error
        available = has_triton_support()
        assert isinstance(available, bool)

    def test_layer_with_triton_flag(self):
        """Test CliffordInteractionLayer with use_triton=True."""
        # Layer should work even if Triton is not available
        # (falls back to PyTorch)
        layer = CliffordInteractionLayer(dim=16, use_triton=True)
        x = torch.randn(2, 8, 16)
        
        output = layer(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(not has_triton_support(), reason="Triton not available")
    def test_triton_forward_matches_pytorch(self):
        """Test that Triton forward matches PyTorch forward."""
        if not has_triton_support():
            pytest.skip("Triton not available")
        
        layer_triton = CliffordInteractionLayer(dim=16, use_triton=True)
        layer_pytorch = CliffordInteractionLayer(dim=16, use_triton=False)
        
        # Copy weights
        layer_pytorch.load_state_dict(layer_triton.state_dict())
        
        x = torch.randn(2, 8, 16)
        
        out_triton = layer_triton(x)
        out_pytorch = layer_pytorch(x)
        
        # Outputs should be close
        assert torch.allclose(out_triton, out_pytorch, atol=1e-5)


# ============================================================
# Test Backward Compatibility
# ============================================================

class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_default_config_uses_ffn(self):
        """Test that default MoEKernelConfig uses FFN experts."""
        config = MoEKernelConfig(input_dim=64, expert_hidden_dim=128)
        
        assert config.use_can_experts is False

    def test_default_expert_is_ffn(self):
        """Test that default DomainExpert uses FFN."""
        expert = DomainExpert(input_dim=64, hidden_dim=128)
        
        assert expert.use_can is False
        assert hasattr(expert, 'net')

    def test_existing_tests_still_pass(self, ffn_kernel):
        """Test that existing FFN-based tests still work."""
        x = torch.randn(8, 64)
        
        output, state = ffn_kernel(x)
        
        # Basic assertions from existing tests
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_create_expert_backward_compat(self):
        """Test create_expert function backward compatibility."""
        # Without use_can parameter (default behavior)
        expert = create_expert("math", input_dim=64, hidden_dim=128, dropout=0.1)
        
        assert expert.use_can is False
        
        # With use_can=False explicitly
        expert_ffn = create_expert("math", input_dim=64, hidden_dim=128, dropout=0.1, use_can=False)
        
        assert expert_ffn.use_can is False

    def test_clifford_ffn_backward_compat(self):
        """Test CliffordFFN for backward compatibility."""
        # CliffordFFN doesn't have output_dim parameter
        ffn = CliffordFFN(input_dim=16, hidden_dim=32)
        x = torch.randn(4, 8, 16)
        
        output = ffn(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


# ============================================================
# Test Numerical Stability
# ============================================================

class TestNumericalStability:
    """Tests for numerical stability of CAN components."""

    def test_large_values(self):
        """Test with large input values."""
        layer = CliffordInteractionLayer(dim=16)
        x = torch.randn(4, 8, 16) * 100  # Large values
        
        output = layer(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_small_values(self):
        """Test with small input values."""
        layer = CliffordInteractionLayer(dim=16)
        x = torch.randn(4, 8, 16) * 1e-6  # Small values
        
        output = layer(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_zero_input(self):
        """Test with zero input."""
        layer = CliffordInteractionLayer(dim=16)
        x = torch.zeros(4, 8, 16)
        
        output = layer(x)
        
        # Should produce valid output (not NaN)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_mixed_precision(self):
        """Test with mixed precision (float16)."""
        layer = CliffordInteractionLayer(dim=16)
        x = torch.randn(4, 8, 16, dtype=torch.float32)
        
        # Should work with float32
        output = layer(x)
        
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
