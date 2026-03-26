"""Тесты для freeze API TitansMemory — RAG compatibility.

Покрытие:
- retrieve_only() — RAG-compatible retrieval без обновления
- Градиенты не проходят через retrieve_only
- Работает как в frozen, так и unfrozen режиме
"""

import pytest
import torch

from src.core.titans_memory import TitansMemoryModule


@pytest.fixture
def memory_module():
    """Фикстура: модуль памяти с стандартными параметрами."""
    return TitansMemoryModule(key_dim=32, val_dim=64, hidden_dim=64)


class TestRetrieveOnly:
    """Тесты retrieve_only метода для RAG compatibility."""

    def test_retrieve_only_returns_tensor(self, memory_module):
        """retrieve_only возвращает тензор правильной формы."""
        key = torch.randn(4, 32)
        retrieved = memory_module.retrieve_only(key)

        assert isinstance(retrieved, torch.Tensor)
        assert retrieved.shape == (4, 64)

    def test_retrieve_only_no_grad(self, memory_module):
        """retrieve_only не требует градиентов."""
        key = torch.randn(2, 32, requires_grad=True)
        retrieved = memory_module.retrieve_only(key)

        # Выход не требует градиентов (torch.no_grad context)
        assert not retrieved.requires_grad

    def test_retrieve_only_memory_unchanged(self, memory_module):
        """retrieve_only не изменяет веса памяти."""
        memory_module.train()
        key = torch.randn(2, 32)
        initial_weight = memory_module.memory.weight.clone().detach()

        _ = memory_module.retrieve_only(key)

        # Веса не изменились
        assert torch.allclose(memory_module.memory.weight, initial_weight)

    def test_retrieve_only_equivalent_to_forward_no_update(self, memory_module):
        """retrieve_only эквивалентен forward(update_memory=False)."""
        memory_module.eval()
        key = torch.randn(3, 32)
        value = torch.randn(3, 64)  # не используется в retrieve_only

        retrieved_only = memory_module.retrieve_only(key)
        retrieved_forward, _ = memory_module(key, value, update_memory=False)

        assert torch.allclose(retrieved_only, retrieved_forward)

    def test_retrieve_only_works_frozen(self, memory_module):
        """retrieve_only работает в frozen режиме."""
        memory_module.freeze_memory()
        key = torch.randn(2, 32)
        initial_weight = memory_module.memory.weight.clone().detach()

        retrieved = memory_module.retrieve_only(key)

        assert retrieved.shape == (2, 64)
        assert torch.allclose(memory_module.memory.weight, initial_weight)

    def test_retrieve_only_works_unfrozen(self, memory_module):
        """retrieve_only работает в unfrozen режиме."""
        memory_module.unfreeze_memory()
        key = torch.randn(2, 32)
        initial_weight = memory_module.memory.weight.clone().detach()

        retrieved = memory_module.retrieve_only(key)

        assert retrieved.shape == (2, 64)
        assert torch.allclose(memory_module.memory.weight, initial_weight)

    def test_retrieve_only_batch_processing(self, memory_module):
        """retrieve_only корректно обрабатывает батчи разного размера."""
        for batch_size in [1, 4, 16, 32]:
            key = torch.randn(batch_size, 32)
            retrieved = memory_module.retrieve_only(key)
            assert retrieved.shape == (batch_size, 64)
