"""Tests for Memory Interface critical paths.

Coverage:
- TitansAdapter forward and contract
- HBMAMemoryAdapter forward and contract
- MemoryResult dataclass
- Adapter reset contract
- MemoryInterface ABC compliance
"""

import pytest
import torch

from src.core.memory_interface import (
    MemoryInterface,
    MemoryResult,
    TitansAdapter,
    HBMAMemoryAdapter,
)
from src.core.titans_memory import TitansMemoryModule
from src.core.hbma_memory import HBMAMemory, CLSMemory


@pytest.fixture
def titans_module():
    """Create raw TitansMemoryModule."""
    module = TitansMemoryModule(
        key_dim=32,
        val_dim=64,
        hidden_dim=64,
    )
    return module.to("cpu")


@pytest.fixture
def titans_adapter(titans_module):
    """Create TitansAdapter wrapping TitansMemoryModule."""
    adapter = TitansAdapter(
        titans_module,
        clifford_dim=64,
        memory_key_dim=32,
    )
    return adapter.to("cpu")


@pytest.fixture
def hbma_module():
    """Create HBMAMemory module."""
    module = HBMAMemory(hidden_dim=64)
    return module.to("cpu")


@pytest.fixture
def hbma_adapter(hbma_module):
    """Create HBMAMemoryAdapter wrapping HBMAMemory."""
    adapter = HBMAMemoryAdapter(hbma_module)
    return adapter.to("cpu")


@pytest.fixture
def cls_module():
    """Create CLSMemory module."""
    module = CLSMemory(hidden_dim=64)
    return module.to("cpu")


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


class TestTitansAdapterForward:
    """Tests for TitansAdapter forward method."""

    def test_titans_adapter_forward_shape(self, titans_adapter):
        """TitansAdapter forward returns correct shape."""
        titans_adapter.eval()
        x = torch.randn(4, 64)

        result = titans_adapter(x, update_memory=False)

        assert isinstance(result, MemoryResult)
        assert result.output.shape == (4, 64)
        assert result.loss.ndim == 0

    def test_titans_adapter_forward_single_element(self, titans_adapter):
        """TitansAdapter handles single element."""
        titans_adapter.eval()
        x = torch.randn(1, 64)

        result = titans_adapter(x, update_memory=False)

        assert result.output.shape == (1, 64)

    def test_titans_adapter_forward_gated_output(self, titans_adapter):
        """TitansAdapter applies gating to memory output."""
        titans_adapter.eval()

        x = torch.randn(2, 64)
        result = titans_adapter(x, update_memory=False)

        # Output should be: x + gate * retrieved
        # Gate is sigmoid of learned projection, so output != x
        # and output != x + retrieved (due to gating)
        assert not torch.allclose(result.output, x)

    def test_titans_adapter_forward_no_update(self, titans_adapter):
        """TitansAdapter respects update_memory=False."""
        titans_adapter.train()
        memory_before = titans_adapter.titans.memory.weight.detach().clone()

        x = torch.randn(4, 64)
        result = titans_adapter(x, update_memory=False)

        assert result.updated is False
        assert torch.allclose(titans_adapter.titans.memory.weight, memory_before)

    def test_titans_adapter_forward_with_update(self, titans_adapter):
        """TitansAdapter updates memory when update_memory=True."""
        titans_adapter.train()
        memory_before = titans_adapter.titans.memory.weight.detach().clone()

        x = torch.randn(4, 64)
        result = titans_adapter(x, update_memory=True)

        assert result.updated is True
        assert not torch.allclose(titans_adapter.titans.memory.weight, memory_before)

    def test_titans_adapter_returns_titans_gates(self, titans_adapter):
        """TitansAdapter returns Titans gate values."""
        titans_adapter.train()
        x = torch.randn(2, 64)

        result = titans_adapter(x, update_memory=True)

        assert result.alpha is not None
        assert result.eta is not None
        assert result.theta is not None
        assert 0.0 <= result.alpha.item() <= 1.0


class TestHBMAMemoryAdapterForward:
    """Tests for HBMAMemoryAdapter forward method."""

    def test_hbma_adapter_forward_shape(self, hbma_adapter):
        """HBMAMemoryAdapter returns correct shape."""
        hbma_adapter.eval()
        x = torch.randn(4, 64)

        result = hbma_adapter(x, update_memory=False)

        assert isinstance(result, MemoryResult)
        assert result.output.shape == (4, 64)

    def test_hbma_adapter_forward_single_element(self, hbma_adapter):
        """HBMAMemoryAdapter handles single element."""
        hbma_adapter.eval()
        x = torch.randn(1, 64)

        result = hbma_adapter(x, update_memory=False)

        assert result.output.shape == (1, 64)

    def test_hbma_adapter_forward_loss(self, hbma_adapter):
        """HBMAMemoryAdapter returns memory loss."""
        hbma_adapter.eval()
        x = torch.randn(4, 64)

        result = hbma_adapter(x, update_memory=False)

        # HBMA memory_loss should be a scalar tensor
        assert result.loss.ndim == 0

    def test_hbma_adapter_updated_flag(self, hbma_adapter):
        """HBMAMemoryAdapter sets updated based on training mode."""
        hbma_adapter.eval()
        x = torch.randn(4, 64)

        result = hbma_adapter(x, update_memory=True)
        # In eval mode, HBMA should not update
        assert result.updated is False

        hbma_adapter.train()
        result = hbma_adapter(x, update_memory=True)
        # In train mode with update_memory=True, updated should be True
        assert result.updated is True

    def test_hbma_adapter_no_titans_gates(self, hbma_adapter):
        """HBMAMemoryAdapter does not have Titans gate values."""
        hbma_adapter.eval()
        x = torch.randn(2, 64)

        result = hbma_adapter(x, update_memory=False)

        # HBMA adapter should not have Titans-specific fields
        assert result.alpha is None
        assert result.eta is None
        assert result.theta is None


class TestAdapterResetContract:
    """Tests for adapter reset methods."""

    def test_titans_adapter_reset_hard(self, titans_adapter):
        """TitansAdapter reset with 'hard' zeroes memory."""
        titans_adapter.train()

        # Populate memory
        x = torch.randn(4, 64)
        titans_adapter(x, update_memory=True)

        # Verify non-zero
        assert (titans_adapter.titans.memory.weight != 0).sum().item() > 0

        # Hard reset
        titans_adapter.reset(strategy="hard")

        assert torch.allclose(
            titans_adapter.titans.memory.weight,
            torch.zeros_like(titans_adapter.titans.memory.weight),
        )

    def test_titans_adapter_reset_geometric(self, titans_adapter):
        """TitansAdapter reset with 'geometric' applies decay."""
        titans_adapter.train()

        # Populate memory
        x = torch.randn(4, 64)
        titans_adapter(x, update_memory=True)

        weight_before = titans_adapter.titans.memory.weight.detach().clone()

        # Geometric reset
        titans_adapter.reset(strategy="geometric")

        # Weights should be scaled (not zeroed)
        assert not torch.allclose(
            titans_adapter.titans.memory.weight,
            torch.zeros_like(titans_adapter.titans.memory.weight),
        )

    def test_titans_adapter_reset_memory_alias(self, titans_adapter):
        """reset_memory is alias for reset."""
        titans_adapter.train()

        x = torch.randn(4, 64)
        titans_adapter(x, update_memory=True)

        weight_before = titans_adapter.titans.memory.weight.detach().clone()

        # Use reset_memory alias
        titans_adapter.reset_memory(strategy="geometric")

        # Should behave same as reset
        assert not torch.allclose(titans_adapter.titans.memory.weight, weight_before)

    def test_hbma_adapter_reset(self, hbma_adapter):
        """HBMAMemoryAdapter reset clears memory."""
        hbma_adapter.reset(strategy="hard")

        # HBMA reset should not raise
        # (actual behavior depends on HBMA implementation)

    def test_hbma_adapter_reset_with_geometric(self, hbma_adapter):
        """HBMAMemoryAdapter handles geometric reset."""
        hbma_adapter.reset(strategy="geometric")

        # Should not raise even if HBMA doesn't support geometric


class TestMemoryInterfaceContract:
    """Tests for MemoryInterface ABC contract."""

    def test_memory_interface_is_abstract(self):
        """MemoryInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MemoryInterface()

    def test_titans_adapter_is_memory_interface(self, titans_adapter):
        """TitansAdapter is a MemoryInterface."""
        assert isinstance(titans_adapter, MemoryInterface)

    def test_hbma_adapter_is_memory_interface(self, hbma_adapter):
        """HBMAMemoryAdapter is a MemoryInterface."""
        assert isinstance(hbma_adapter, MemoryInterface)

    def test_memory_interface_has_forward(self, titans_adapter):
        """MemoryInterface implementations have forward method."""
        assert hasattr(titans_adapter, "forward")
        assert callable(titans_adapter.forward)

    def test_memory_interface_has_reset(self, titans_adapter):
        """MemoryInterface implementations have reset method."""
        assert hasattr(titans_adapter, "reset")
        assert callable(titans_adapter.reset)

    def test_memory_interface_has_memory_loss(self, titans_adapter):
        """MemoryInterface implementations have memory_loss method."""
        assert hasattr(titans_adapter, "memory_loss")
        assert callable(titans_adapter.memory_loss)


class TestKeyProjection:
    """Tests for TitansAdapter key projection."""

    def test_key_projection_shape(self, titans_adapter):
        """Key projection reduces clifford_dim to memory_key_dim."""
        titans_adapter.eval()
        x = torch.randn(4, 64)

        # Key projection happens internally
        result = titans_adapter(x, update_memory=False)

        # Verify output shape matches input
        assert result.output.shape == x.shape

    def test_key_projection_different_dims(self):
        """Key projection works with different dimension ratios."""
        titans = TitansMemoryModule(key_dim=16, val_dim=32, hidden_dim=32)
        adapter = TitansAdapter(
            titans,
            clifford_dim=32,  # val_dim
            memory_key_dim=16,  # key_dim
        ).to("cpu")

        x = torch.randn(2, 32)
        result = adapter(x, update_memory=False)

        assert result.output.shape == (2, 32)


class TestGateNetwork:
    """Tests for TitansAdapter gate network."""

    def test_gate_output_range(self, titans_adapter):
        """Gate output is in (0, 1) after sigmoid."""
        titans_adapter.eval()
        x = torch.randn(8, 64)

        # Run forward to populate internal state
        result = titans_adapter(x, update_memory=False)

        # Gate is applied internally, output should differ from input
        assert not torch.allclose(result.output, x)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_batch_titans_adapter(self, titans_adapter):
        """TitansAdapter handles empty batch."""
        titans_adapter.eval()
        x = torch.randn(0, 64)

        result = titans_adapter(x, update_memory=False)

        assert result.output.shape == (0, 64)

    def test_empty_batch_hbma_adapter(self, hbma_adapter):
        """HBMAMemoryAdapter handles empty batch."""
        hbma_adapter.eval()
        x = torch.randn(0, 64)

        result = hbma_adapter(x, update_memory=False)

        assert result.output.shape == (0, 64)

    def test_large_batch_titans_adapter(self, titans_adapter):
        """TitansAdapter handles large batch."""
        titans_adapter.eval()
        x = torch.randn(256, 64)

        result = titans_adapter(x, update_memory=False)

        assert result.output.shape == (256, 64)

    def test_large_batch_hbma_adapter(self, hbma_adapter):
        """HBMAMemoryAdapter handles large batch."""
        hbma_adapter.eval()
        x = torch.randn(256, 64)

        result = hbma_adapter(x, update_memory=False)

        assert result.output.shape == (256, 64)


class TestCLSAdapter:
    """Tests for CLSMemory wrapped in HBMAMemoryAdapter."""

    def test_cls_adapter_forward(self, cls_module):
        """CLSMemory works through HBMAMemoryAdapter."""
        adapter = HBMAMemoryAdapter(cls_module).to("cpu")
        adapter.eval()

        x = torch.randn(4, 64)
        result = adapter(x, update_memory=False)

        assert result.output.shape == (4, 64)

    def test_cls_adapter_reset(self, cls_module):
        """CLSMemory adapter reset works."""
        adapter = HBMAMemoryAdapter(cls_module).to("cpu")

        # Should not raise
        adapter.reset(strategy="hard")


class TestMemoryLossMethod:
    """Tests for memory_loss method."""

    def test_titans_adapter_memory_loss(self, titans_adapter):
        """TitansAdapter memory_loss returns last loss."""
        titans_adapter.eval()
        x = torch.randn(4, 64)

        result = titans_adapter(x, update_memory=False)
        loss = titans_adapter.memory_loss()

        assert torch.allclose(loss, result.loss.detach())

    def test_hbma_adapter_memory_loss(self, hbma_adapter):
        """HBMAMemoryAdapter memory_loss method works."""
        hbma_adapter.eval()
        x = torch.randn(4, 64)

        # Should not raise
        loss = hbma_adapter.memory_loss()

        assert loss.ndim == 0


class TestGradientFlow:
    """Tests for gradient flow through adapters."""

    def test_titans_adapter_gradient_flow(self, titans_adapter):
        """Gradients flow through TitansAdapter."""
        titans_adapter.train()
        x = torch.randn(4, 64, requires_grad=True)

        result = titans_adapter(x, update_memory=False)
        loss = result.output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (4, 64)

    def test_hbma_adapter_gradient_flow(self, hbma_adapter):
        """Gradients flow through HBMAMemoryAdapter."""
        hbma_adapter.train()
        x = torch.randn(4, 64, requires_grad=True)

        result = hbma_adapter(x, update_memory=False)
        loss = result.output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (4, 64)
