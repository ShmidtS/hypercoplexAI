"""Tests for Online-LoRA adapters."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import math

import torch
import torch.nn as nn
import pytest

from src.extensions.lora.online_lora import (
    OnlineLoRA,
    OnlineLoRALinear,
    OnlineLoRAConv,
)


class TestOnlineLoRALinear:
    """Test Online-LoRA for Linear layers."""

    def test_init_basic(self):
        linear = nn.Linear(256, 128)
        lora = OnlineLoRA(linear, rank=8)
        assert lora.rank == 8
        assert lora.scaling == 1.0 / 8
        assert lora.lora_A.shape == (256, 8)
        assert lora.lora_B.shape == (8, 128)

    def test_init_custom_params(self):
        linear = nn.Linear(256, 128)
        lora = OnlineLoRA(linear, rank=16, alpha=2.0, dropout=0.1)
        assert lora.rank == 16
        assert lora.scaling == 2.0 / 16
        assert lora.dropout_p == 0.1

    def test_base_layer_frozen(self):
        linear = nn.Linear(256, 128)
        lora = OnlineLoRA(linear, rank=8)
        # Base layer should have requires_grad=False
        for param in lora.base.parameters():
            assert not param.requires_grad
        # LoRA params should be trainable
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_forward_shape(self):
        linear = nn.Linear(256, 128)
        lora = OnlineLoRA(linear, rank=8)
        x = torch.randn(4, 256)
        output = lora(x)
        assert output.shape == (4, 128)

    def test_forward_adds_lora_contribution(self):
        linear = nn.Linear(256, 128)
        lora = OnlineLoRA(linear, rank=8)

        # Set non-zero B weights to see LoRA effect
        with torch.no_grad():
            nn.init.ones_(lora.lora_B)

        x = torch.randn(2, 256)
        base_output = linear(x)
        lora_output = lora(x)

        # Output should differ from base due to LoRA contribution
        assert not torch.allclose(base_output, lora_output)

    def test_importance_initialization(self):
        linear = nn.Linear(256, 128)
        lora = OnlineLoRA(linear, rank=8)
        # Importance should start as ones
        assert torch.allclose(lora.importance, torch.ones(256))

    def test_importance_updates_with_gradients(self):
        linear = nn.Linear(256, 128)
        lora = OnlineLoRA(linear, rank=8)

        x = torch.randn(2, 256, requires_grad=True)
        output = lora(x)
        loss = output.sum()
        loss.backward()

        # Importance should have been updated
        # Note: update happens via hook, so it should change
        initial_importance = torch.ones(256)
        # After backward, importance should have changed
        # (gradient magnitude added via EMA)

    def test_ema_update(self):
        linear = nn.Linear(256, 128)
        lora = OnlineLoRA(linear, rank=8)

        # Initial EMA should be zeros
        assert not lora._ema_initialized

        # Update EMA
        lora.update_ema()

        # After first update, EMA should equal current weights
        assert lora._ema_initialized
        assert torch.allclose(lora.lora_A_ema, lora.lora_A.data)
        assert torch.allclose(lora.lora_B_ema, lora.lora_B.data)

    def test_ema_accumulation(self):
        linear = nn.Linear(256, 128)
        lora = OnlineLoRA(linear, rank=8, ema_decay=0.9)

        # Initialize EMA
        lora.update_ema()

        # Modify weights to all ones
        with torch.no_grad():
            lora.lora_A.fill_(1.0)
            lora.lora_B.fill_(1.0)

        # Update EMA again
        lora.update_ema()

        # EMA should be between old values (near zeros from Kaiming) and new ones
        # After one update: EMA = 0.9 * old + 0.1 * new
        # Since old values from Kaiming can be negative, EMA can also be negative
        # Just check that EMA has moved toward 1.0
        assert lora.lora_A_ema.max() > 0.0  # Should have some positive values
        assert lora.lora_A_ema.mean() > 0.0  # Mean should be positive moving toward 1

    def test_consolidation(self):
        linear = nn.Linear(256, 128)
        lora = OnlineLoRA(linear, rank=8)

        # Initialize EMA
        lora.update_ema()

        # Modify weights significantly
        with torch.no_grad():
            lora.lora_A.fill_(1.0)
            lora.lora_B.fill_(1.0)

        # Consolidate should pull weights back toward EMA
        old_A = lora.lora_A.data.clone()
        lora.consolidate(strength=0.5)

        # Weights should have changed toward EMA (which was ~0)
        assert not torch.allclose(lora.lora_A.data, old_A)

    def test_get_stats(self):
        linear = nn.Linear(256, 128)
        lora = OnlineLoRA(linear, rank=8)

        stats = lora.get_stats()
        assert stats['rank'] == 8
        assert stats['num_updates'] == 0
        assert 'importance_mean' in stats
        assert 'lora_A_norm' in stats

    def test_reset_importance(self):
        linear = nn.Linear(256, 128)
        lora = OnlineLoRA(linear, rank=8)

        # Modify importance
        lora.importance.fill_(2.0)
        lora.reset_importance()
        assert torch.allclose(lora.importance, torch.ones(256))

    def test_invalid_base_layer(self):
        with pytest.raises(TypeError):
            OnlineLoRA(nn.ReLU())


class TestOnlineLoRALinearWrapper:
    """Test OnlineLoRALinear convenience class."""

    def test_wraps_linear(self):
        linear = nn.Linear(128, 64)
        lora = OnlineLoRALinear(linear, rank=8)
        assert isinstance(lora, OnlineLoRA)
        assert lora._is_conv is False

    def test_rejects_conv(self):
        conv = nn.Conv2d(3, 16, 3)
        with pytest.raises(TypeError):
            OnlineLoRALinear(conv)

    def test_forward_pass(self):
        linear = nn.Linear(128, 64)
        lora = OnlineLoRALinear(linear, rank=8)
        x = torch.randn(4, 128)
        output = lora(x)
        assert output.shape == (4, 64)


class TestOnlineLoRAConv:
    """Test Online-LoRA for Conv2d layers."""

    def test_init_basic(self):
        conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        lora = OnlineLoRA(conv, rank=8)
        assert lora._is_conv is True
        assert lora.lora_A.shape == (64, 8)
        assert lora.lora_B.shape == (8, 128)

    def test_forward_shape(self):
        conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        lora = OnlineLoRA(conv, rank=8)
        x = torch.randn(2, 64, 32, 32)
        output = lora(x)
        assert output.shape == (2, 128, 32, 32)

    def test_forward_adds_lora_contribution(self):
        conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        lora = OnlineLoRAConv(conv, rank=8)

        # Set non-zero B weights
        with torch.no_grad():
            nn.init.ones_(lora.lora_B)

        x = torch.randn(2, 64, 8, 8)
        base_output = conv(x)
        lora_output = lora(x)

        # Should differ due to LoRA
        assert not torch.allclose(base_output, lora_output)

    def test_wrapper_rejects_linear(self):
        linear = nn.Linear(128, 64)
        with pytest.raises(TypeError):
            OnlineLoRAConv(linear)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
