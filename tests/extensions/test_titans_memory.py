"""Tests for TitansMemory -- neural memory with TTT updates.

Coverage:
- Module initialization
- Basic memory update via MemoryInterface API
- Memory retrieval
- Gradient flow through forward
- Memory decay (reset)
- Numerical stability (no NaN/Inf)
"""

import pytest
import torch

from src.extensions.memory import TitansMemory, MemoryResult


@pytest.fixture
def memory_module():
    """Fixture: memory module with standard parameters."""
    return TitansMemory(clifford_dim=64, memory_key_dim=32)


class TestTitansInit:
    """Tests for TitansMemory initialization."""

    def test_titans_init(self, memory_module):
        """Verify correct module initialization."""
        assert memory_module.clifford_dim == 64
        assert memory_module.memory_key_dim == 32

        # Check components exist
        assert hasattr(memory_module, 'memory')
        assert hasattr(memory_module, 'gate_proj')
        assert hasattr(memory_module, 'momentum_S')

        # Check weight dimensions
        assert memory_module.memory.weight.shape == (64, 32)
        assert memory_module.gate_proj[2].weight.shape == (3, 64)  # 3 gates, hidden_dim

        # Check initial values
        assert torch.all(memory_module.momentum_S == 0)

    def test_titans_init_custom_dims(self):
        """Verify initialization with custom dimensions."""
        module = TitansMemory(clifford_dim=256, memory_key_dim=128)
        assert module.clifford_dim == 256
        assert module.memory_key_dim == 128
        assert module.memory.weight.shape == (256, 128)


class TestTitansUpdateBasic:
    """Tests for basic memory update."""

    def test_titans_update_basic(self, memory_module):
        """Basic memory update in training mode."""
        memory_module.train()
        x = torch.randn(2, 64)

        # Save initial weights
        initial_weight = memory_module.memory.weight.clone().detach()

        # Perform update
        result = memory_module(x, update_memory=True)

        # Check weights changed
        assert not torch.allclose(memory_module.memory.weight, initial_weight)

        # Check result type
        assert isinstance(result, MemoryResult)
        assert result.updated is True

        # Check Titans gate values are valid (sigmoid output)
        if result.alpha is not None:
            assert 0 <= result.alpha.item() <= 1

    def test_titans_update_frozen(self, memory_module):
        """Memory doesn't update when frozen."""
        memory_module.freeze_memory()
        memory_module.train()

        x = torch.randn(2, 64)
        initial_weight = memory_module.memory.weight.clone().detach()

        result = memory_module(x, update_memory=True)

        # Weights should not change
        assert torch.allclose(memory_module.memory.weight, initial_weight)
        assert result.updated is False


class TestTitansRetrieval:
    """Tests for memory retrieval."""

    def test_titans_retrieval(self, memory_module):
        """Retrieval from memory without update."""
        memory_module.eval()
        x = torch.randn(2, 64)

        result = memory_module(x, update_memory=False)

        # Check result type
        assert isinstance(result, MemoryResult)

        # Check dimensions
        assert result.output.shape == (2, 64)
        assert result.loss.ndim == 0  # scalar loss

        # Check update flag
        assert result.updated is False

    def test_titans_retrieve_and_update(self, memory_module):
        """Combined retrieve + update."""
        memory_module.train()
        x = torch.randn(2, 64)

        result = memory_module(x, update_memory=True)

        assert result.output.shape == (2, 64)
        assert result.updated is True
        assert result.alpha is not None
        assert result.eta is not None
        assert result.theta is not None


class TestTitansGradientFlow:
    """Tests for gradient flow."""

    def test_titans_gradient_flow(self, memory_module):
        """Gradients flow through forward pass."""
        memory_module.train()
        x = torch.randn(2, 64, requires_grad=True)

        result = memory_module(x, update_memory=False)
        loss = result.output.sum()
        loss.backward()

        # Check gradients computed for input
        assert x.grad is not None
        # Check gradients not NaN
        assert not torch.isnan(x.grad).any()

    def test_titans_memory_gradient(self, memory_module):
        """Gradients flow through memory weights."""
        memory_module.train()
        x = torch.randn(2, 64)

        result = memory_module(x, update_memory=False)
        result.loss.backward()

        # Memory weights should have gradients
        assert memory_module.memory.weight.grad is not None or not memory_module.memory.weight.requires_grad


class TestTitansMemoryDecay:
    """Tests for memory decay."""

    def test_titans_memory_decay_hard(self, memory_module):
        """Hard reset completely zeroes memory."""
        memory_module.train()
        x = torch.randn(2, 64)
        memory_module(x, update_memory=True)

        # Hard reset
        memory_module.reset(strategy='hard')

        # Check everything zeroed
        assert torch.all(memory_module.memory.weight == 0)
        assert torch.all(memory_module.momentum_S == 0)

    def test_titans_memory_decay_geometric(self, memory_module):
        """Geometric decay preserves patterns."""
        memory_module.train()
        x = torch.randn(2, 64)
        memory_module(x, update_memory=True)

        weight_before = memory_module.memory.weight.clone().detach()
        momentum_before = memory_module.momentum_S.clone().detach()

        # Geometric reset
        memory_module.reset(strategy='geometric', decay_window=50.0)

        # Weights should decrease but not zero out
        assert torch.all(memory_module.memory.weight.abs() <= weight_before.abs() + 1e-6)
        assert torch.all(memory_module.momentum_S.abs() <= momentum_before.abs() + 1e-6)

        # Check not everything zeroed
        assert memory_module.memory.weight.abs().sum() > 0 or weight_before.abs().sum() == 0

    def test_titans_memory_decay_stabilize(self, memory_module):
        """Stabilize only normalizes momentum."""
        memory_module.train()
        x = torch.randn(2, 64)
        memory_module(x, update_memory=True)

        weight_before = memory_module.memory.weight.clone().detach()

        memory_module.reset(strategy='stabilize')

        # Memory weights should not change
        assert torch.allclose(memory_module.memory.weight, weight_before)


class TestTitansNumericalStability:
    """Tests for numerical stability."""

    def test_titans_numerical_stability_normal(self, memory_module):
        """No NaN/Inf with normal inputs."""
        memory_module.train()
        x = torch.randn(4, 64)

        result = memory_module(x, update_memory=True)

        assert not torch.isnan(result.output).any()
        assert not torch.isinf(result.output).any()
        assert not torch.isnan(result.loss)
        assert not torch.isinf(result.loss)

    def test_titans_numerical_stability_large_values(self, memory_module):
        """Stability with large input values."""
        memory_module.train()
        x = torch.randn(4, 64) * 100

        result = memory_module(x, update_memory=True)

        # Thanks to clamp in update, weights shouldn't explode
        assert not torch.isnan(result.output).any()
        assert not torch.isinf(result.output).any()

        # Memory weights in reasonable range
        assert memory_module.memory.weight.abs().max() < 100

    def test_titans_numerical_stability_small_values(self, memory_module):
        """Stability with very small input values."""
        memory_module.train()
        x = torch.randn(4, 64) * 1e-6

        result = memory_module(x, update_memory=True)

        assert not torch.isnan(result.output).any()
        assert not torch.isinf(result.output).any()
