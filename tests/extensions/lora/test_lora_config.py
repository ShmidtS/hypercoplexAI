"""Tests for Online-LoRA configuration."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pytest

from src.extensions.lora.online_lora import OnlineLoRAConfig


class TestOnlineLoRAConfig:
    """Test configuration dataclass."""

    def test_default_values(self):
        config = OnlineLoRAConfig()
        assert config.rank == 8
        assert config.alpha == 1.0
        assert config.dropout == 0.0
        assert config.importance_decay == 0.9
        assert config.importance_gain == 0.1
        assert config.ema_decay == 0.999

    def test_custom_values(self):
        config = OnlineLoRAConfig(
            rank=16,
            alpha=2.0,
            dropout=0.1,
            importance_decay=0.95,
            importance_gain=0.05,
        )
        assert config.rank == 16
        assert config.alpha == 2.0
        assert config.dropout == 0.1
        assert config.importance_decay == 0.95
        assert config.importance_gain == 0.05


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
