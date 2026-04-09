"""Тесты для TitansMemoryModule — нейронной памяти с TTT обновлением.

Покрытие:
- Инициализация модуля
- Базовое обновление памяти
- Извлечение из памяти
- Градиенты проходят через forward
- Затухание памяти (reset)
- Числовая стабильность (нет NaN/Inf)
"""

import pytest
import torch

from src.core.titans_memory import TitansMemoryModule, MemoryState


@pytest.fixture
def memory_module():
    """Фикстура: модуль памяти с стандартными параметрами."""
    return TitansMemoryModule(key_dim=32, val_dim=64, hidden_dim=64)


@pytest.fixture
def sample_inputs():
    """Фикстура: тестовые ключи и значения."""
    batch_size = 4
    key = torch.randn(batch_size, 32)
    value = torch.randn(batch_size, 64)
    return key, value


class TestTitansInit:
    """Тесты инициализации TitansMemoryModule."""

    def test_titans_init(self, memory_module):
        """Проверка корректной инициализации модуля."""
        # Проверка размерностей
        assert memory_module.key_dim == 32
        assert memory_module.val_dim == 64

        # Проверка наличия компонентов
        assert hasattr(memory_module, 'memory')
        assert hasattr(memory_module, 'gate_proj')
        assert hasattr(memory_module, 'momentum_S')

        # Проверка размерностей весов
        assert memory_module.memory.weight.shape == (64, 32)
        assert memory_module.gate_proj[2].weight.shape == (3, 64)  # 3 gates, hidden_dim

        # Проверка начальных значений
        assert torch.all(memory_module.momentum_S == 0)

    def test_titans_init_custom_dims(self):
        """Проверка инициализации с кастомными размерностями."""
        module = TitansMemoryModule(key_dim=128, val_dim=256, hidden_dim=128)
        assert module.key_dim == 128
        assert module.val_dim == 256
        assert module.memory.weight.shape == (256, 128)


class TestTitansUpdateBasic:
    """Тесты базового обновления памяти."""

    def test_titans_update_basic(self, memory_module):
        """Базовое обновление памяти в training режиме."""
        memory_module.train()
        key = torch.randn(2, 32)
        value = torch.randn(2, 64)

        # Сохраняем начальное состояние весов
        initial_weight = memory_module.memory.weight.clone().detach()

        # Выполняем обновление
        alpha, eta, theta = memory_module.update(key, value)

        # Проверяем, что веса изменились
        assert not torch.allclose(memory_module.memory.weight, initial_weight)

        # Проверяем размерности возвращаемых значений
        assert alpha.shape == ()
        assert eta.shape == ()
        assert theta.shape == ()

        # Проверяем, что гейты валидные (sigmoid output)
        assert 0 <= alpha.item() <= 1
        assert 0 <= eta.item() <= 1
        assert 0 <= theta.item() <= 1

    def test_titans_update_frozen(self, memory_module):
        """Память не обновляется когда заморожена."""
        memory_module.freeze_memory()
        memory_module.train()

        key = torch.randn(2, 32)
        value = torch.randn(2, 64)
        initial_weight = memory_module.memory.weight.clone().detach()

        # retrieve_and_update с update_memory=True должен игнорироваться
        state = memory_module.retrieve_and_update(key, value, update_memory=True)

        # Веса не должны измениться
        assert torch.allclose(memory_module.memory.weight, initial_weight)
        assert state.updated is False


class TestTitansRetrieval:
    """Тесты извлечения из памяти."""

    def test_titans_retrieval(self, memory_module):
        """Извлечение из памяти без обновления."""
        memory_module.eval()
        key = torch.randn(2, 32)
        value = torch.randn(2, 64)

        state = memory_module.retrieve(key, value)

        # Проверка типа результата
        assert isinstance(state, MemoryState)

        # Проверка размерностей
        assert state.retrieved.shape == (2, 64)
        assert state.loss.ndim == 0  # scalar loss

        # Проверка флага обновления
        assert state.updated is False

    def test_titans_retrieve_and_update(self, memory_module):
        """Комбинированный retrieve + update."""
        memory_module.train()
        key = torch.randn(2, 32)
        value = torch.randn(2, 64)

        state = memory_module.retrieve_and_update(key, value, update_memory=True)

        assert state.retrieved.shape == (2, 64)
        assert state.updated is True
        assert state.alpha is not None
        assert state.eta is not None
        assert state.theta is not None


class TestTitansGradientFlow:
    """Тесты градиентного потока."""

    def test_titans_gradient_flow(self, memory_module):
        """Градиенты проходят через forward pass."""
        memory_module.train()
        key = torch.randn(2, 32, requires_grad=True)
        value = torch.randn(2, 64, requires_grad=True)

        retrieved, loss = memory_module(key, value, update_memory=False)

        # Backward pass
        loss.backward()

        # Проверяем, что градиенты вычислены для ключа
        assert key.grad is not None
        # Примечание: value.grad может быть None т.к. v используется только в loss
        # через detach() внутри retrieve() для стабильности TTT

        # Проверяем, что градиенты не NaN
        assert not torch.isnan(key.grad).any()

    def test_titans_memory_gradient(self, memory_module):
        """Градиенты проходят через веса памяти."""
        memory_module.train()
        key = torch.randn(2, 32)
        value = torch.randn(2, 64)

        retrieved, loss = memory_module(key, value, update_memory=False)
        loss.backward()

        # Градиенты должны быть у весов памяти
        assert memory_module.memory.weight.grad is not None or not memory_module.memory.weight.requires_grad


class TestTitansMemoryDecay:
    """Тесты затухания памяти."""

    def test_titans_memory_decay_hard(self, memory_module):
        """Hard reset полностью обнуляет память."""
        # Сначала обновляем память
        memory_module.train()
        key = torch.randn(2, 32)
        value = torch.randn(2, 64)
        memory_module.update(key, value)

        # Hard reset
        memory_module.reset_memory(strategy='hard')

        # Проверяем, что всё обнулено
        assert torch.all(memory_module.memory.weight == 0)
        assert torch.all(memory_module.momentum_S == 0)

    def test_titans_memory_decay_geometric(self, memory_module):
        """Geometric decay сохраняет паттерны."""
        # Обновляем память
        memory_module.train()
        key = torch.randn(2, 32)
        value = torch.randn(2, 64)
        memory_module.update(key, value)

        weight_before = memory_module.memory.weight.clone().detach()
        momentum_before = memory_module.momentum_S.clone().detach()

        # Geometric reset
        memory_module.reset_memory(strategy='geometric', decay_window=50.0)

        # Веса должны уменьшиться, но не обнулиться
        assert torch.all(memory_module.memory.weight.abs() <= weight_before.abs())
        assert torch.all(memory_module.momentum_S.abs() <= momentum_before.abs())

        # Проверяем, что не всё обнулилось
        assert memory_module.memory.weight.abs().sum() > 0 or weight_before.abs().sum() == 0

    def test_titans_memory_decay_stabilize(self, memory_module):
        """Stabilize только нормирует momentum."""
        memory_module.train()
        key = torch.randn(2, 32)
        value = torch.randn(2, 64)
        memory_module.update(key, value)

        weight_before = memory_module.memory.weight.clone().detach()

        memory_module.reset_memory(strategy='stabilize')

        # Веса памяти не должны измениться
        assert torch.allclose(memory_module.memory.weight, weight_before)


class TestTitansNumericalStability:
    """Тесты числовой стабильности."""

    def test_titans_numerical_stability_normal(self, memory_module):
        """Нет NaN/Inf при нормальных входах."""
        memory_module.train()
        key = torch.randn(4, 32)
        value = torch.randn(4, 64)

        retrieved, loss = memory_module(key, value, update_memory=True)

        assert not torch.isnan(retrieved).any()
        assert not torch.isinf(retrieved).any()
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_titans_numerical_stability_large_values(self, memory_module):
        """Стабильность при больших значениях входов."""
        memory_module.train()
        key = torch.randn(4, 32) * 100
        value = torch.randn(4, 64) * 100

        retrieved, loss = memory_module(key, value, update_memory=True)

        # Благодаря clamp в update, веса не должны взорваться
        assert not torch.isnan(retrieved).any()
        assert not torch.isinf(retrieved).any()

        # Проверяем, что веса памяти в разумных пределах
        assert memory_module.memory.weight.abs().max() < 100

    def test_titans_numerical_stability_small_values(self, memory_module):
        """Стабильность при очень маленьких значениях."""
        memory_module.train()
        key = torch.randn(4, 32) * 1e-6
        value = torch.randn(4, 64) * 1e-6

        retrieved, loss = memory_module(key, value, update_memory=True)

        assert not torch.isnan(retrieved).any()
        assert not torch.isinf(retrieved).any()


class TestTitansFreezeUnfreeze:
    """Тесты заморозки/разморозки памяти."""

    def test_freeze_memory(self, memory_module):
        """Заморозка памяти для RAG inference."""
        memory_module.freeze_memory()

        assert memory_module.is_frozen() is True
        assert memory_module.memory.weight.requires_grad is False

    def test_unfreeze_memory(self, memory_module):
        """Разморозка для обучения."""
        memory_module.freeze_memory()
        memory_module.unfreeze_memory()

        assert memory_module.is_frozen() is False
        assert memory_module.memory.weight.requires_grad is True

    def test_frozen_memory_no_update(self, memory_module):
        """Замороженная память не обновляется."""
        memory_module.freeze_memory()
        memory_module.train()

        key = torch.randn(2, 32)
        value = torch.randn(2, 64)
        initial_weight = memory_module.memory.weight.clone().detach()

        # Попытка обновления
        state = memory_module.retrieve_and_update(key, value, update_memory=True)

        # Веса не изменились
        assert torch.allclose(memory_module.memory.weight, initial_weight)
        assert state.updated is False
