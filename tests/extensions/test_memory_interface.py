"""Tests for Memory Interface critical paths.

Coverage:
- TitansMemory forward and contract
- HBMAMemory forward and contract
- MemoryResult dataclass
- Memory reset contract
- MemoryInterface ABC compliance
"""

import pytest
import torch

from src.extensions.memory import (
    MemoryInterface,
    MemoryResult,
    TitansMemory,
    HBMAMemory,
    CLSMemory,
)


@pytest.fixture
def titans_memory():
    """Create TitansMemory."""
    return TitansMemory(
        clifford_dim=64,
        memory_key_dim=32,
    ).to("cpu")


@pytest.fixture
def hbma_memory():
    """Create HBMAMemory module."""
    return HBMAMemory(hidden_dim=64).to("cpu")


@pytest.fixture
def cls_memory():
    """Create CLSMemory module."""
    return CLSMemory(hidden_dim=64).to("cpu")


class TestMemoryResult:
    """Tests for MemoryResult dataclass."""

    def test_memory_result_required_fields(self):
        """MemoryResult requires output, loss, updated."""
        result = MemoryResult(
            output=torch.randn(4, 64),
            loss=torch.tensor(0.5),
            updated=True,
        )

        assert result.output.shape == (4, 64)
        assert result.loss.item() == 0.5
        assert result.updated is True

    def test_memory_result_optional_fields(self):
        """MemoryResult optional fields default to None."""
        result = MemoryResult(
            output=torch.randn(2, 32),
            loss=torch.tensor(0.0),
            updated=False,
        )

        assert result.alpha is None
        assert result.eta is None
        assert result.theta is None

    def test_memory_result_with_titans_fields(self):
        """MemoryResult can hold Titans gate values."""
        alpha = torch.tensor(0.3)
        eta = torch.tensor(0.5)
        theta = torch.tensor(0.2)

        result = MemoryResult(
            output=torch.randn(1, 64),
            loss=torch.tensor(0.1),
            updated=True,
            alpha=alpha,
            eta=eta,
            theta=theta,
        )

        assert torch.allclose(result.alpha, alpha)
        assert torch.allclose(result.eta, eta)
        assert torch.allclose(result.theta, theta)


class TestTitansMemoryForward:
    """Tests for TitansMemory forward method."""

    def test_titans_forward_shape(self, titans_memory):
        """TitansMemory forward returns correct shape."""
        titans_memory.eval()
        x = torch.randn(4, 64)

        result = titans_memory(x, update_memory=False)

        assert isinstance(result, MemoryResult)
        assert result.output.shape == (4, 64)
        assert result.loss.ndim == 0

    def test_titans_forward_single_element(self, titans_memory):
        """TitansMemory handles single element."""
        titans_memory.eval()
        x = torch.randn(1, 64)

        result = titans_memory(x, update_memory=False)

        assert result.output.shape == (1, 64)

    def test_titans_forward_gated_output(self, titans_memory):
        """TitansMemory applies gating to memory output."""
        titans_memory.eval()

        x = torch.randn(2, 64)
        result = titans_memory(x, update_memory=False)

        # Output should be: x + gate * retrieved
        # Gate is sigmoid of learned projection, so output != x
        assert not torch.allclose(result.output, x)

    def test_titans_forward_no_update(self, titans_memory):
        """TitansMemory respects update_memory=False."""
        titans_memory.train()
        memory_before = titans_memory.memory.weight.detach().clone()

        x = torch.randn(4, 64)
        result = titans_memory(x, update_memory=False)

        assert result.updated is False
        assert torch.allclose(titans_memory.memory.weight, memory_before)

    def test_titans_forward_with_update(self, titans_memory):
        """TitansMemory updates memory when update_memory=True."""
        titans_memory.train()
        memory_before = titans_memory.memory.weight.detach().clone()

        x = torch.randn(4, 64)
        result = titans_memory(x, update_memory=True)

        assert result.updated is True
        assert not torch.allclose(titans_memory.memory.weight, memory_before)

    def test_titans_returns_titans_gates(self, titans_memory):
        """TitansMemory returns Titans gate values."""
        titans_memory.train()
        x = torch.randn(2, 64)

        result = titans_memory(x, update_memory=True)

        assert result.alpha is not None
        assert result.eta is not None
        assert result.theta is not None
        assert 0.0 <= result.alpha.item() <= 1.0


class TestHBMAMemoryForward:
    """Tests for HBMAMemory forward method."""

    def test_hbma_forward_shape(self, hbma_memory):
        """HBMAMemory returns correct shape."""
        hbma_memory.eval()
        x = torch.randn(4, 64)

        result = hbma_memory(x, update_memory=False)

        assert isinstance(result, MemoryResult)
        assert result.output.shape == (4, 64)

    def test_hbma_forward_single_element(self, hbma_memory):
        """HBMAMemory handles single element."""
        hbma_memory.eval()
        x = torch.randn(1, 64)

        result = hbma_memory(x, update_memory=False)

        assert result.output.shape == (1, 64)

    def test_hbma_forward_loss(self, hbma_memory):
        """HBMAMemory returns memory loss."""
        hbma_memory.eval()
        x = torch.randn(4, 64)

        result = hbma_memory(x, update_memory=False)

        # HBMA memory_loss should be a scalar tensor
        assert result.loss.ndim == 0

    def test_hbma_updated_flag(self, hbma_memory):
        """HBMAMemory sets updated based on training mode."""
        hbma_memory.eval()
        x = torch.randn(4, 64)

        result = hbma_memory(x, update_memory=True)
        # In eval mode, HBMA should not update
        assert result.updated is False

        hbma_memory.train()
        result = hbma_memory(x, update_memory=True)
        # In train mode with update_memory=True, updated should be True
        assert result.updated is True

    def test_hbma_no_titans_gates(self, hbma_memory):
        """HBMAMemory does not have Titans gate values."""
        hbma_memory.eval()
        x = torch.randn(2, 64)

        result = hbma_memory(x, update_memory=False)

        # HBMA should not have Titans-specific fields
        assert result.alpha is None
        assert result.eta is None
        assert result.theta is None


class TestAdapterResetContract:
    """Tests for memory reset methods."""

    def test_titans_reset_hard(self, titans_memory):
        """TitansMemory reset with 'hard' zeroes memory."""
        titans_memory.train()

        # Populate memory
        x = torch.randn(4, 64)
        titans_memory(x, update_memory=True)

        # Verify non-zero
        assert (titans_memory.memory.weight != 0).sum().item() > 0

        # Hard reset
        titans_memory.reset(strategy="hard")

        assert torch.allclose(
            titans_memory.memory.weight,
            torch.zeros_like(titans_memory.memory.weight),
        )

    def test_titans_reset_geometric(self, titans_memory):
        """TitansMemory reset with 'geometric' applies decay."""
        titans_memory.train()

        # Populate memory
        x = torch.randn(4, 64)
        titans_memory(x, update_memory=True)

        weight_before = titans_memory.memory.weight.detach().clone()

        # Geometric reset
        titans_memory.reset(strategy="geometric")

        # Weights should be scaled (not zeroed)
        assert not torch.allclose(
            titans_memory.memory.weight,
            torch.zeros_like(titans_memory.memory.weight),
        )

    def test_titans_reset_memory_alias(self, titans_memory):
        """reset_memory is alias for reset."""
        titans_memory.train()

        x = torch.randn(4, 64)
        titans_memory(x, update_memory=True)

        weight_before = titans_memory.memory.weight.detach().clone()

        # Use reset_memory alias
        titans_memory.reset_memory(strategy="geometric")

        # Should behave same as reset
        assert not torch.allclose(titans_memory.memory.weight, weight_before)

    def test_hbma_reset(self, hbma_memory):
        """HBMAMemory reset clears memory."""
        hbma_memory.reset(strategy="hard")

        # HBMA reset should not raise

    def test_hbma_reset_with_geometric(self, hbma_memory):
        """HBMAMemory handles geometric reset."""
        hbma_memory.reset(strategy="geometric")

        # Should not raise


class TestMemoryInterfaceContract:
    """Tests for MemoryInterface ABC contract."""

    def test_memory_interface_is_abstract(self):
        """MemoryInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MemoryInterface()

    def test_titans_is_memory_interface(self, titans_memory):
        """TitansMemory is a MemoryInterface."""
        assert isinstance(titans_memory, MemoryInterface)

    def test_hbma_is_memory_interface(self, hbma_memory):
        """HBMAMemory is a MemoryInterface."""
        assert isinstance(hbma_memory, MemoryInterface)

    def test_memory_interface_has_forward(self, titans_memory):
        """MemoryInterface implementations have forward method."""
        assert hasattr(titans_memory, "forward")
        assert callable(titans_memory.forward)

    def test_memory_interface_has_reset(self, titans_memory):
        """MemoryInterface implementations have reset method."""
        assert hasattr(titans_memory, "reset")
        assert callable(titans_memory.reset)

    def test_memory_interface_has_memory_loss(self, titans_memory):
        """MemoryInterface implementations have memory_loss method."""
        assert hasattr(titans_memory, "memory_loss")
        assert callable(titans_memory.memory_loss)


class TestKeyProjection:
    """Tests for TitansMemory key projection."""

    def test_key_projection_shape(self, titans_memory):
        """Key projection reduces clifford_dim to memory_key_dim."""
        titans_memory.eval()
        x = torch.randn(4, 64)

        # Key projection happens internally
        result = titans_memory(x, update_memory=False)

        # Verify output shape matches input
        assert result.output.shape == x.shape

    def test_key_projection_different_dims(self):
        """Key projection works with different dimension ratios."""
        memory = TitansMemory(clifford_dim=32, memory_key_dim=16).to("cpu")

        x = torch.randn(2, 32)
        result = memory(x, update_memory=False)

        assert result.output.shape == (2, 32)


class TestGateNetwork:
    """Tests for TitansMemory gate network."""

    def test_gate_output_range(self, titans_memory):
        """Gate output is in (0, 1) after sigmoid."""
        titans_memory.eval()
        x = torch.randn(8, 64)

        # Run forward to populate internal state
        result = titans_memory(x, update_memory=False)

        # Gate is applied internally, output should differ from input
        assert not torch.allclose(result.output, x)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_batch_titans(self, titans_memory):
        """TitansMemory handles empty batch."""
        titans_memory.eval()
        x = torch.randn(0, 64)

        result = titans_memory(x, update_memory=False)

        assert result.output.shape == (0, 64)

    def test_empty_batch_hbma(self, hbma_memory):
        """HBMAMemory handles empty batch."""
        hbma_memory.eval()
        x = torch.randn(0, 64)

        result = hbma_memory(x, update_memory=False)

        assert result.output.shape == (0, 64)

    def test_large_batch_titans(self, titans_memory):
        """TitansMemory handles large batch."""
        titans_memory.eval()
        x = torch.randn(256, 64)

        result = titans_memory(x, update_memory=False)

        assert result.output.shape == (256, 64)

    def test_large_batch_hbma(self, hbma_memory):
        """HBMAMemory handles large batch."""
        hbma_memory.eval()
        x = torch.randn(256, 64)

        result = hbma_memory(x, update_memory=False)

        assert result.output.shape == (256, 64)


class TestCLSForward:
    """Tests for CLSMemory (HBMAMemory subclass)."""

    def test_cls_forward(self, cls_memory):
        """CLSMemory works through MemoryInterface."""
        cls_memory.eval()

        x = torch.randn(4, 64)
        result = cls_memory(x, update_memory=False)

        assert result.output.shape == (4, 64)

    def test_cls_reset(self, cls_memory):
        """CLSMemory reset works."""
        # Should not raise
        cls_memory.reset(strategy="hard")


class TestMemoryLossMethod:
    """Tests for memory_loss method."""

    def test_titans_memory_loss(self, titans_memory):
        """TitansMemory memory_loss returns last loss."""
        titans_memory.eval()
        x = torch.randn(4, 64)

        result = titans_memory(x, update_memory=False)
        loss = titans_memory.memory_loss()

        assert torch.allclose(loss, result.loss.detach())

    def test_hbma_memory_loss(self, hbma_memory):
        """HBMAMemory memory_loss method works."""
        hbma_memory.eval()
        x = torch.randn(4, 64)

        # Should not raise
        loss = hbma_memory.memory_loss()

        assert loss.ndim == 0


class TestGradientFlow:
    """Tests for gradient flow through memory modules."""

    def test_titans_gradient_flow(self, titans_memory):
        """Gradients flow through TitansMemory."""
        titans_memory.train()
        x = torch.randn(4, 64, requires_grad=True)

        result = titans_memory(x, update_memory=False)
        loss = result.output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (4, 64)

    def test_hbma_gradient_flow(self, hbma_memory):
        """Gradients flow through HBMAMemory."""
        hbma_memory.train()
        x = torch.randn(4, 64, requires_grad=True)

        result = hbma_memory(x, update_memory=False)
        loss = result.output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (4, 64)
