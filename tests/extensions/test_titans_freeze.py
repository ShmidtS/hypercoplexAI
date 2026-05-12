"""Tests for freeze API TitansMemory -- RAG compatibility.

Coverage:
- retrieve_only() -- RAG-compatible retrieval without update
- Gradients don't flow through retrieve_only
- Works in both frozen and unfrozen mode
"""

import pytest
import torch

from src.extensions.memory import TitansMemory


@pytest.fixture
def memory_module():
    """Fixture: memory module with standard parameters."""
    return TitansMemory(clifford_dim=64, memory_key_dim=32)


class TestRetrieveOnly:
    """Tests for retrieve_only method for RAG compatibility."""

    def test_retrieve_only_returns_tensor(self, memory_module):
        """retrieve_only returns tensor of correct shape."""
        key = torch.randn(4, 32)
        retrieved = memory_module.retrieve_only(key)

        assert isinstance(retrieved, torch.Tensor)
        assert retrieved.shape == (4, 64)

    def test_retrieve_only_no_grad(self, memory_module):
        """retrieve_only doesn't require gradients."""
        key = torch.randn(2, 32, requires_grad=True)
        retrieved = memory_module.retrieve_only(key)

        # Output doesn't require gradients (torch.no_grad context)
        assert not retrieved.requires_grad

    def test_retrieve_only_memory_unchanged(self, memory_module):
        """retrieve_only doesn't change memory weights."""
        memory_module.train()
        key = torch.randn(2, 32)
        initial_weight = memory_module.memory.weight.clone().detach()

        _ = memory_module.retrieve_only(key)

        # Weights unchanged
        assert torch.allclose(memory_module.memory.weight, initial_weight)

    def test_retrieve_only_works_frozen(self, memory_module):
        """retrieve_only works in frozen mode."""
        memory_module.freeze_memory()
        key = torch.randn(2, 32)
        initial_weight = memory_module.memory.weight.clone().detach()

        retrieved = memory_module.retrieve_only(key)

        assert retrieved.shape == (2, 64)
        assert torch.allclose(memory_module.memory.weight, initial_weight)

    def test_retrieve_only_works_unfrozen(self, memory_module):
        """retrieve_only works in unfrozen mode."""
        memory_module.unfreeze_memory()
        key = torch.randn(2, 32)
        initial_weight = memory_module.memory.weight.clone().detach()

        retrieved = memory_module.retrieve_only(key)

        assert retrieved.shape == (2, 64)
        assert torch.allclose(memory_module.memory.weight, initial_weight)

    def test_retrieve_only_batch_processing(self, memory_module):
        """retrieve_only correctly handles batches of different sizes."""
        for batch_size in [1, 4, 16, 32]:
            key = torch.randn(batch_size, 32)
            retrieved = memory_module.retrieve_only(key)
            assert retrieved.shape == (batch_size, 64)


class TestFreezeMemory:
    """Tests for freeze/unfreeze API."""

    def test_freeze_memory_disables_grad(self, memory_module):
        """freeze_memory disables requires_grad for memory weights."""
        assert memory_module.memory.weight.requires_grad is True
        assert memory_module.is_frozen() is False

        memory_module.freeze_memory()

        assert memory_module.memory.weight.requires_grad is False
        assert memory_module.is_frozen() is True

    def test_unfreeze_memory_enables_grad(self, memory_module):
        """unfreeze_memory restores requires_grad."""
        memory_module.freeze_memory()
        assert memory_module.memory.weight.requires_grad is False

        memory_module.unfreeze_memory()

        assert memory_module.memory.weight.requires_grad is True
        assert memory_module.is_frozen() is False

    def test_freeze_prevents_memory_update(self, memory_module):
        """In frozen mode, forward doesn't update memory."""
        memory_module.freeze_memory()
        memory_module.train()
        x = torch.randn(2, 64)
        initial_weight = memory_module.memory.weight.clone().detach()

        # forward with update_memory=True should be ignored
        result = memory_module(x, update_memory=True)

        assert torch.allclose(memory_module.memory.weight, initial_weight)
        assert result.updated is False

    def test_freeze_memory_idempotent(self, memory_module):
        """Repeated freeze doesn't break state."""
        memory_module.freeze_memory()
        memory_module.freeze_memory()  # repeated call

        assert memory_module.is_frozen() is True
        assert memory_module.memory.weight.requires_grad is False


class TestDeterminismAfterFreeze:
    """Tests for determinism after freeze."""

    def test_retrieve_only_deterministic(self, memory_module):
        """retrieve_only gives same result for same inputs."""
        memory_module.freeze_memory()
        key = torch.randn(4, 32)
        torch.manual_seed(42)

        result1 = memory_module.retrieve_only(key)
        result2 = memory_module.retrieve_only(key)

        assert torch.allclose(result1, result2)

    def test_deterministic_across_multiple_calls(self, memory_module):
        """Determinism preserved across multiple calls."""
        memory_module.freeze_memory()
        key = torch.randn(8, 32)

        results = [memory_module.retrieve_only(key) for _ in range(5)]

        for r in results[1:]:
            assert torch.allclose(results[0], r)

    def test_forward_frozen_deterministic(self, memory_module):
        """forward in frozen mode is deterministic."""
        memory_module.freeze_memory()
        memory_module.eval()
        x = torch.randn(4, 64)

        result1 = memory_module(x, update_memory=False)
        result2 = memory_module(x, update_memory=False)

        assert torch.allclose(result1.loss, result2.loss)
