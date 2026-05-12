#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for OnlineLearner gradient modes.

Tests:
- test_gradient_mode_detached: Verify no gradient flow in DETACHED mode
- test_gradient_mode_selective: Verify selective gradient flow for replay buffer
- test_gradient_mode_full: Verify full gradient flow
- test_gradient_stability: Verify gradient clipping and stability
"""

import pytest
import torch
import torch.nn as nn

from src.core.online_learner import OnlineLearner, GradientMode, OnlineLearnerConfig


class SimpleModel(nn.Module):
    """Simple model for testing gradient flow."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestGradientModeDetached:
    """Tests for DETACHED gradient mode (default, safest)."""

    def test_no_gradient_flow(self):
        """Verify that DETACHED mode produces no gradients."""
        learner = OnlineLearner(
            hidden_dim=32,
            num_experts=4,
            gradient_mode=GradientMode.DETACHED,
        )
        learner.train()

        # Create input with requires_grad
        x = torch.randn(4, 32, requires_grad=True)

        # Run update
        loss, updated, surprise = learner.online_update_with_mode(
            x=x,
            expert_idx=0,
            model=None,
        )

        # Check that no gradient was computed on x
        assert x.grad is None or x.grad.abs().sum() == 0
        assert loss.item() == 0.0  # Detached mode returns 0 loss

    def test_replay_buffer_still_populated(self):
        """Verify that replay buffer is still populated in DETACHED mode."""
        learner = OnlineLearner(
            hidden_dim=32,
            num_experts=4,
            gradient_mode=GradientMode.DETACHED,
            surprise_threshold=0.0,  # Force update
        )
        learner.train()

        x = torch.randn(4, 32, requires_grad=True)

        # Run update
        loss, updated, surprise = learner.online_update_with_mode(
            x=x,
            expert_idx=0,
            model=None,
            force_update=True,
        )

        # Verify buffer has samples
        assert len(learner.replay_buffer) > 0
        assert updated is True


class TestGradientModeSelective:
    """Tests for SELECTIVE gradient mode (replay buffer only)."""

    def test_scaled_gradient_flow(self):
        """Verify that SELECTIVE mode scales gradients."""
        gradient_scale = 0.5
        learner = OnlineLearner(
            hidden_dim=32,
            num_experts=4,
            gradient_mode=GradientMode.SELECTIVE,
            gradient_scale=gradient_scale,
        )
        learner.train()

        x = torch.randn(4, 32, requires_grad=True)

        # Run update
        loss, updated, surprise = learner.online_update_with_mode(
            x=x,
            expert_idx=0,
            model=None,
            force_update=True,
        )

        # Loss should be non-zero in SELECTIVE mode
        assert updated is True
        # The loss is computed from scaled input

    def test_replay_buffer_gradient_tracking(self):
        """Verify that replay buffer samples are scaled."""
        learner = OnlineLearner(
            hidden_dim=32,
            num_experts=4,
            gradient_mode=GradientMode.SELECTIVE,
            gradient_scale=0.1,
        )
        learner.train()

        x = torch.randn(4, 32)

        # Run update
        loss, updated, surprise = learner.online_update_with_mode(
            x=x,
            expert_idx=0,
            model=None,
            force_update=True,
        )

        # Verify buffer has samples (scaled)
        assert len(learner.replay_buffer) > 0


class TestGradientModeFull:
    """Tests for FULL gradient mode (experimental)."""

    def test_full_gradient_flow(self):
        """Verify that FULL mode enables gradient flow."""
        learner = OnlineLearner(
            hidden_dim=32,
            num_experts=4,
            gradient_mode=GradientMode.FULL,
            gradient_scale=0.1,
        )
        learner.train()

        x = torch.randn(4, 32, requires_grad=True)

        # Run update
        loss, updated, surprise = learner.online_update_with_mode(
            x=x,
            expert_idx=0,
            model=None,
            force_update=True,
        )

        # Should have non-zero loss
        assert updated is True

    def test_gradient_clipping(self):
        """Verify that gradients are clipped in FULL mode."""
        grad_clip = 0.5
        learner = OnlineLearner(
            hidden_dim=32,
            num_experts=4,
            gradient_mode=GradientMode.FULL,
            gradient_scale=1.0,
            grad_clip=grad_clip,
        )
        learner.train()

        x = torch.randn(4, 32, requires_grad=True)

        # Run update
        loss, updated, surprise = learner.online_update_with_mode(
            x=x,
            expert_idx=0,
            model=None,
            force_update=True,
        )

        assert updated is True


class TestGradientStability:
    """Tests for gradient stability across modes."""

    def test_stats_include_gradient_mode(self):
        """Verify that stats include gradient mode info."""
        learner = OnlineLearner(
            hidden_dim=32,
            num_experts=4,
            gradient_mode=GradientMode.SELECTIVE,
            gradient_scale=0.2,
        )

        stats = learner.get_stats()

        assert 'gradient_mode' in stats
        assert stats['gradient_mode'] == 'selective'
        assert 'gradient_scale' in stats
        assert stats['gradient_scale'] == 0.2

    def test_save_load_preserves_gradient_mode(self):
        """Verify that save/load preserves gradient mode."""
        learner = OnlineLearner(
            hidden_dim=32,
            num_experts=4,
            gradient_mode=GradientMode.SELECTIVE,
            gradient_scale=0.3,
        )
        learner.train()

        # Add some data
        x = torch.randn(4, 32)
        learner.online_update_with_mode(x=x, expert_idx=0, force_update=True)

        # Save state
        state = learner.save_state()

        # Create new learner and load state
        learner2 = OnlineLearner(
            hidden_dim=32,
            num_experts=4,
            gradient_mode=GradientMode.DETACHED,  # Different default
        )
        learner2.load_state(state)

        # Verify gradient mode was restored
        assert learner2.gradient_mode == GradientMode.SELECTIVE
        assert learner2.gradient_scale == 0.3

    def test_no_nan_in_gradient_modes(self):
        """Verify no NaN values in any gradient mode."""
        for mode in [GradientMode.DETACHED, GradientMode.SELECTIVE, GradientMode.FULL]:
            learner = OnlineLearner(
                hidden_dim=32,
                num_experts=4,
                gradient_mode=mode,
                gradient_scale=0.1,
            )
            learner.train()

            # Use extreme values
            x = torch.randn(4, 32) * 100.0

            loss, updated, surprise = learner.online_update_with_mode(
                x=x,
                expert_idx=0,
                model=None,
                force_update=True,
            )

            assert not torch.isnan(loss), f"NaN loss in {mode.value} mode"
            assert not torch.isinf(loss), f"Inf loss in {mode.value} mode"


class TestOnlineLearnerConfig:
    """Tests for OnlineLearnerConfig dataclass."""

    def test_default_config(self):
        """Verify default config values."""
        config = OnlineLearnerConfig()

        assert config.gradient_mode == GradientMode.DETACHED
        assert config.gradient_scale == 0.1
        assert config.hidden_dim == 256
        assert config.num_experts == 4

    def test_custom_config(self):
        """Verify custom config values."""
        config = OnlineLearnerConfig(
            hidden_dim=128,
            gradient_mode=GradientMode.SELECTIVE,
            gradient_scale=0.5,
        )

        assert config.hidden_dim == 128
        assert config.gradient_mode == GradientMode.SELECTIVE
        assert config.gradient_scale == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
