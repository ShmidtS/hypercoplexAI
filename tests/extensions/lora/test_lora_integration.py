"""Integration tests for Online-LoRA."""

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
    OnlineLoRAConfig,
    OnlineLoRAManager,
    wrap_with_online_lora,
)


class TestOnlineLoraIntegration:
    """Integration tests for Online-LoRA."""

    def test_training_loop(self):
        """Test that LoRA params update during training."""
        model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
        model = wrap_with_online_lora(model, OnlineLoRAConfig(rank=8))

        # Collect LoRA params
        manager = OnlineLoRAManager()
        manager.register_from_module(model)

        optimizer = torch.optim.SGD(manager.get_trainable_parameters(), lr=0.01)

        # Initial weights
        initial_A = model[0].lora_A.data.clone()

        # Training loop
        for _ in range(5):
            x = torch.randn(4, 128)
            output = model(x)
            loss = output.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            manager.update_all_ema()

        # Weights should have changed
        assert not torch.allclose(model[0].lora_A.data, initial_A)

    def test_base_weights_unchanged(self):
        """Verify base layer weights don't change during training."""
        base_linear = nn.Linear(64, 32)
        initial_weight = base_linear.weight.data.clone()

        lora = OnlineLoRALinear(base_linear, rank=8)
        optimizer = torch.optim.SGD([lora.lora_A, lora.lora_B], lr=0.01)

        for _ in range(5):
            x = torch.randn(2, 64)
            output = lora(x)
            loss = output.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Base weights should be unchanged
        assert torch.allclose(base_linear.weight.data, initial_weight)

    def test_dropout_effect(self):
        """Test that dropout is applied when configured."""
        linear = nn.Linear(64, 32)
        lora = OnlineLoRALinear(linear, rank=8, dropout=0.5)

        # Set non-zero B weights so LoRA contribution is non-zero
        # (B is initialized to zeros by default for LoRA)
        with torch.no_grad():
            nn.init.ones_(lora.lora_B)

        lora.train()
        # Use larger input for more stable statistics
        x = torch.randn(100, 64)

        # Multiple forward passes should give different results due to dropout
        outputs = [lora(x) for _ in range(5)]

        # Check that outputs differ (dropout should cause variation)
        # Compare first output with others
        different_count = 0
        for i in range(1, len(outputs)):
            if not torch.allclose(outputs[0], outputs[i], rtol=1e-3, atol=1e-3):
                different_count += 1

        # At least 2 out of 4 comparisons should differ
        assert different_count >= 2

    def test_eval_mode(self):
        """Test that eval mode works correctly."""
        linear = nn.Linear(64, 32)
        lora = OnlineLoRALinear(linear, rank=8, dropout=0.5)

        lora.eval()
        x = torch.randn(10, 64)

        # Multiple forward passes should give same results in eval mode
        out1 = lora(x)
        out2 = lora(x)
        assert torch.allclose(out1, out2)


class TestOnlineLoraEdgeCases:
    """Edge case tests."""

    def test_rank_1(self):
        """Test minimum rank."""
        linear = nn.Linear(128, 64)
        lora = OnlineLoRA(linear, rank=1)
        assert lora.lora_A.shape == (128, 1)
        assert lora.lora_B.shape == (1, 64)

        x = torch.randn(2, 128)
        output = lora(x)
        assert output.shape == (2, 64)

    def test_large_batch(self):
        """Test with large batch size."""
        linear = nn.Linear(256, 128)
        lora = OnlineLoRA(linear, rank=8)

        x = torch.randn(128, 256)
        output = lora(x)
        assert output.shape == (128, 128)

    def test_very_small_input(self):
        """Test with 1-element batch."""
        linear = nn.Linear(32, 16)
        lora = OnlineLoRA(linear, rank=4)

        x = torch.randn(1, 32)
        output = lora(x)
        assert output.shape == (1, 16)

    def test_alpha_scaling(self):
        """Test that alpha affects output scale."""
        linear = nn.Linear(64, 32)

        lora1 = OnlineLoRA(linear, rank=8, alpha=1.0)
        lora2 = OnlineLoRA(linear, rank=8, alpha=2.0)

        # Set same weights for comparison
        with torch.no_grad():
            lora2.lora_A.copy_(lora1.lora_A)
            lora2.lora_B.copy_(lora1.lora_B)

        x = torch.randn(2, 64)

        # Set to eval to avoid dropout randomness
        lora1.eval()
        lora2.eval()

        out1 = lora1(x)
        out2 = lora2(x)

        # alpha=2.0 should give 2x LoRA contribution
        # Base is same, so difference is in LoRA part
        base_out = linear(x)
        lora_contrib1 = out1 - base_out
        lora_contrib2 = out2 - base_out

        assert torch.allclose(lora_contrib2, lora_contrib1 * 2.0, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
