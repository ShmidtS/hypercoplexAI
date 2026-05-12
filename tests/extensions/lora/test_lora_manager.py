"""Tests for Online-LoRA manager."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import math

import torch
import torch.nn as nn
import pytest

from src.extensions.lora.online_lora import (
    OnlineLoRALinear,
    OnlineLoRAManager,
    wrap_with_online_lora,
)


class TestOnlineLoRAManager:
    """Test centralized LoRA management."""

    def test_register_adapter(self):
        linear = nn.Linear(128, 64)
        adapter = OnlineLoRALinear(linear)

        manager = OnlineLoRAManager()
        manager.register(adapter)

        assert len(manager.adapters) == 1

    def test_register_from_module(self):
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.Linear(32, 16),
        )
        wrapped = wrap_with_online_lora(model)

        manager = OnlineLoRAManager()
        count = manager.register_from_module(wrapped)

        assert count == 2
        assert len(manager.adapters) == 2

    def test_update_all_ema(self):
        linear1 = nn.Linear(64, 32)
        linear2 = nn.Linear(32, 16)
        adapter1 = OnlineLoRALinear(linear1)
        adapter2 = OnlineLoRALinear(linear2)

        manager = OnlineLoRAManager()
        manager.register(adapter1)
        manager.register(adapter2)

        manager.update_all_ema()

        assert adapter1._ema_initialized
        assert adapter2._ema_initialized

    def test_consolidate_all(self):
        linear1 = nn.Linear(64, 32)
        linear2 = nn.Linear(32, 16)
        adapter1 = OnlineLoRALinear(linear1)
        adapter2 = OnlineLoRALinear(linear2)

        manager = OnlineLoRAManager(consolidation_strength=0.5)
        manager.register(adapter1)
        manager.register(adapter2)

        # Initialize and modify
        adapter1.update_ema()
        adapter2.update_ema()
        with torch.no_grad():
            adapter1.lora_A.fill_(1.0)
            adapter2.lora_A.fill_(1.0)

        manager.consolidate_all()
        # Both should have been consolidated
        assert adapter1.lora_A.data.max() < 1.0
        assert adapter2.lora_A.data.max() < 1.0

    def test_should_consolidate(self):
        manager = OnlineLoRAManager(consolidation_interval=5)
        linear = nn.Linear(64, 32)
        manager.register(OnlineLoRALinear(linear))

        for _ in range(4):
            manager.update_all_ema()
            assert not manager.should_consolidate()

        manager.update_all_ema()
        assert manager.should_consolidate()

    def test_get_all_stats(self):
        linear1 = nn.Linear(64, 32)
        linear2 = nn.Linear(32, 16)
        adapter1 = OnlineLoRALinear(linear1)
        adapter2 = OnlineLoRALinear(linear2)

        manager = OnlineLoRAManager()
        manager.register(adapter1)
        manager.register(adapter2)

        stats = manager.get_all_stats()

        assert stats['num_adapters'] == 2
        assert 'importance_mean_global' in stats
        assert 'adapters' in stats
        assert len(stats['adapters']) == 2

    def test_step(self):
        linear = nn.Linear(64, 32)
        adapter = OnlineLoRALinear(linear)

        manager = OnlineLoRAManager(consolidation_interval=3)
        manager.register(adapter)

        # Step 1-2: should not consolidate
        manager.step()
        manager.step()
        # EMA is initialized after first update_all_ema call
        assert manager._step_count == 2

        # Step 3: should trigger consolidation
        manager.step()
        assert manager._step_count == 3
        # After 3 updates, EMA should be initialized
        assert adapter._ema_initialized

    def test_get_trainable_parameters(self):
        linear1 = nn.Linear(64, 32)
        linear2 = nn.Linear(32, 16)
        adapter1 = OnlineLoRALinear(linear1)
        adapter2 = OnlineLoRALinear(linear2)

        manager = OnlineLoRAManager()
        manager.register(adapter1)
        manager.register(adapter2)

        params = manager.get_trainable_parameters()

        assert len(params) == 4  # 2 adapters x 2 params each
        assert all(isinstance(p, nn.Parameter) for p in params)

    def test_reset_all_importance(self):
        linear1 = nn.Linear(64, 32)
        linear2 = nn.Linear(32, 16)
        adapter1 = OnlineLoRALinear(linear1)
        adapter2 = OnlineLoRALinear(linear2)

        # Modify importance
        adapter1.importance.fill_(2.0)
        adapter2.importance.fill_(3.0)

        manager = OnlineLoRAManager()
        manager.register(adapter1)
        manager.register(adapter2)

        manager.reset_all_importance()

        assert torch.allclose(adapter1.importance, torch.ones(64))
        assert torch.allclose(adapter2.importance, torch.ones(32))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
