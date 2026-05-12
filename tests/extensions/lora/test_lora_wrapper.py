"""Tests for Online-LoRA wrapping utilities."""

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
    OnlineLoRAConv,
    OnlineLoRAConfig,
    wrap_with_online_lora,
)


class TestWrapWithOnlineLora:
    """Test module wrapping utility."""

    def test_wrap_simple_model(self):
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        wrapped = wrap_with_online_lora(model, OnlineLoRAConfig(rank=8))

        # Linear layers should be wrapped
        assert isinstance(wrapped[0], OnlineLoRALinear)
        assert isinstance(wrapped[2], OnlineLoRALinear)
        # ReLU should remain unchanged
        assert isinstance(wrapped[1], nn.ReLU)

    def test_wrap_with_config(self):
        config = OnlineLoRAConfig(rank=16, dropout=0.1)
        model = nn.Sequential(nn.Linear(128, 64))
        wrapped = wrap_with_online_lora(model, config)

        assert wrapped[0].rank == 16
        assert wrapped[0].dropout_p == 0.1

    def test_wrap_nested_model(self):
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.Linear(32, 16),
                )

        model = NestedModel()
        wrapped = wrap_with_online_lora(model)

        # Both linear layers should be wrapped
        assert isinstance(wrapped.block[0], OnlineLoRALinear)
        assert isinstance(wrapped.block[1], OnlineLoRALinear)

    def test_wrap_conv_model(self):
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        wrapped = wrap_with_online_lora(model)

        assert isinstance(wrapped[0], OnlineLoRAConv)
        assert isinstance(wrapped[2], OnlineLoRALinear)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
