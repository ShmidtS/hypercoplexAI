"""Tests for SemanticEntropyProbe — linear probe for uncertainty quantification."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from src.core.semantic_entropy_probe import (
    SemanticEntropyProbe,
    SemanticEntropyResult,
)


class TestSemanticEntropyProbeInit:
    """Test probe initialization."""

    def test_default_hidden_dim(self):
        probe = SemanticEntropyProbe()
        assert probe.hidden_dim == 256
        assert probe.probe_layer == -1

    def test_custom_hidden_dim(self):
        probe = SemanticEntropyProbe(hidden_dim=512, probe_layer=-2)
        assert probe.hidden_dim == 512
        assert probe.probe_layer == -2

    def test_zero_init_weights(self):
        """Probe starts with zeros for stable training when init_scale=0."""
        probe = SemanticEntropyProbe(hidden_dim=128, init_scale=0.0)
        assert torch.allclose(probe.probe.weight, torch.zeros_like(probe.probe.weight))
        assert torch.allclose(probe.probe.bias, torch.zeros_like(probe.probe.bias))

    def test_init_scale_creates_nonzero_weights(self):
        """Positive init_scale creates small random weights."""
        probe = SemanticEntropyProbe(hidden_dim=128, init_scale=0.01)
        assert not torch.allclose(probe.probe.weight, torch.zeros_like(probe.probe.weight))


class TestSemanticEntropyProbeForward:
    """Test probe forward pass."""

    def test_forward_3d_input(self):
        """Standard input: (batch, seq_len, hidden_dim)."""
        probe = SemanticEntropyProbe(hidden_dim=64)
        hidden = torch.randn(4, 32, 64)

        entropy = probe(hidden)

        assert entropy.shape == (4,)
        assert (entropy >= 0).all() and (entropy <= 1).all()

    def test_forward_2d_input(self):
        """Pre-pooled input: (batch, hidden_dim)."""
        probe = SemanticEntropyProbe(hidden_dim=64)
        hidden = torch.randn(8, 64)

        entropy = probe(hidden)

        assert entropy.shape == (8,)
        assert (entropy >= 0).all() and (entropy <= 1).all()

    def test_forward_4d_input(self):
        """Multi-layer input: (batch, num_layers, seq_len, hidden_dim)."""
        probe = SemanticEntropyProbe(hidden_dim=64, probe_layer=-1)
        hidden = torch.randn(2, 12, 16, 64)

        entropy = probe(hidden)

        assert entropy.shape == (2,)
        assert (entropy >= 0).all() and (entropy <= 1).all()

    def test_forward_4d_with_layer_override(self):
        """Override probe_layer via layer_index."""
        probe = SemanticEntropyProbe(hidden_dim=64, probe_layer=-1)
        hidden = torch.randn(2, 12, 16, 64)

        entropy_layer_0 = probe(hidden, layer_index=0)
        entropy_layer_5 = probe(hidden, layer_index=5)
        entropy_last = probe(hidden, layer_index=-1)

        # Different layers should give different predictions (usually)
        assert entropy_layer_0.shape == (2,)
        assert entropy_layer_5.shape == (2,)

    def test_zero_input_gives_midpoint(self):
        """Zero input -> sigmoid(0) = 0.5."""
        probe = SemanticEntropyProbe(hidden_dim=64, init_scale=0.0)
        hidden = torch.zeros(4, 32, 64)

        entropy = probe(hidden)

        assert torch.allclose(entropy, torch.full((4,), 0.5), atol=1e-5)


class TestSemanticEntropyProbeMetadata:
    """Test predict_with_metadata method."""

    def test_returns_result_with_metadata(self):
        probe = SemanticEntropyProbe(hidden_dim=64)
        hidden = torch.randn(2, 16, 64)

        result = probe.predict_with_metadata(hidden)

        assert isinstance(result, SemanticEntropyResult)
        assert 0 <= result.entropy_pred <= 1
        assert result.hidden_norm > 0
        assert result.probe_layer == -1

    def test_to_dict_contains_all_keys(self):
        result = SemanticEntropyResult(
            entropy_pred=0.42,
            hidden_norm=3.14,
            probe_layer=-1,
        )

        d = result.to_dict()
        assert set(d.keys()) == {"entropy_pred", "hidden_norm", "probe_layer"}
        assert d["entropy_pred"] == 0.42

    def test_metadata_with_layer_override(self):
        probe = SemanticEntropyProbe(hidden_dim=64, probe_layer=-1)
        hidden = torch.randn(2, 12, 16, 64)

        result = probe.predict_with_metadata(hidden, layer_index=5)

        assert result.probe_layer == 5


class TestSemanticEntropyProbeBatching:
    """Test batch processing."""

    def test_single_sample(self):
        probe = SemanticEntropyProbe(hidden_dim=64)
        hidden = torch.randn(1, 32, 64)

        entropy = probe(hidden)

        assert entropy.shape == (1,)
        assert 0 <= entropy.item() <= 1

    def test_large_batch(self):
        probe = SemanticEntropyProbe(hidden_dim=64)
        hidden = torch.randn(64, 128, 64)

        entropy = probe(hidden)

        assert entropy.shape == (64,)
        assert (entropy >= 0).all() and (entropy <= 1).all()


class TestSemanticEntropyProbeReset:
    """Test parameter reset."""

    def test_reset_restores_zeros(self):
        probe = SemanticEntropyProbe(hidden_dim=64, init_scale=0.01)

        # Weights should be non-zero after init
        assert not torch.allclose(probe.probe.weight, torch.zeros_like(probe.probe.weight))

        probe.reset_parameters()

        # After reset, weights should be zero
        assert torch.allclose(probe.probe.weight, torch.zeros_like(probe.probe.weight))
        assert torch.allclose(probe.probe.bias, torch.zeros_like(probe.probe.bias))


class TestSemanticEntropyProbeGradient:
    """Test gradient flow."""

    def test_gradients_flow_through_probe(self):
        probe = SemanticEntropyProbe(hidden_dim=64)
        hidden = torch.randn(4, 32, 64, requires_grad=True)

        entropy = probe(hidden)
        loss = entropy.sum()
        loss.backward()

        assert hidden.grad is not None
        assert hidden.grad.shape == hidden.shape

    def test_probe_parameters_have_grad(self):
        probe = SemanticEntropyProbe(hidden_dim=64)
        hidden = torch.randn(4, 32, 64)

        entropy = probe(hidden)
        loss = entropy.sum()
        loss.backward()

        assert probe.probe.weight.grad is not None
        assert probe.probe.bias.grad is not None
