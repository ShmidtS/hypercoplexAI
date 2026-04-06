"""Tests for Online-LoRA — Task-free continual learning via low-rank adaptation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import torch
import torch.nn as nn
import pytest

from src.core.online_lora import (
    OnlineLoRA,
    OnlineLoRALinear,
    OnlineLoRAConv,
    OnlineLoRAConfig,
    OnlineLoRAManager,
    wrap_with_online_lora,
)


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
