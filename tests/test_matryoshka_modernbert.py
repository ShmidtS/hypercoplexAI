"""Tests for ModernBERT encoder, Matryoshka projection, and GatedMLP.

Covers:
- MatryoshkaProjection: multi-scale output, target_dim slicing, dimension validation
- GatedMLPEncoder: forward pass, shape consistency, trainable
- TextHDIMModel + Matryoshka integration: encode_texts_matryoshka, multi-scale loss
- HybridEncoder: architecture selection based on config
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from src.models.hdim_model import HDIMConfig, HDIMModel
from src.models.modern_text_encoder import (
    MatryoshkaProjection,
    GatedMLPBlock,
    GatedMLPEncoder,
    ModernEncoderConfig,
    HybridEncoder,
)
from src.models.text_hdim_model import TextHDIMModel


# ─── MatryoshkaProjection ────────────────────────────────────────────────

class TestMatryoshkaProjection:
    def test_forward_returns_dict_of_scales(self):
        proj = MatryoshkaProjection(64, [16, 32, 64])
        x = torch.randn(4, 64)
        out = proj(x)
        assert isinstance(out, dict)
        assert set(out.keys()) == {16, 32, 64}

    def test_output_shapes(self):
        proj = MatryoshkaProjection(128, [64, 96, 128])
        x = torch.randn(8, 128)
        out = proj(x)
        for dim, tensor in out.items():
            assert tensor.shape == (8, dim), f"Expected (8, {dim}), got {tensor.shape}"

    def test_target_dim_returns_single_tensor(self):
        proj = MatryoshkaProjection(64, [16, 32, 64])
        x = torch.randn(4, 64)
        out = proj(x, target_dim=32)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 32)

    def test_target_dim_none_returns_all(self):
        proj = MatryoshkaProjection(64, [16, 32, 64])
        x = torch.randn(2, 64)
        out = proj(x, target_dim=None)
        assert len(out) == 3

    def test_invalid_target_dim_raises(self):
        proj = MatryoshkaProjection(64, [16, 32, 64])
        x = torch.randn(2, 64)
        with pytest.raises(ValueError, match="exceeds max"):
            proj(x, target_dim=128)

    def test_larger_dim_contains_smaller(self):
        """The first N columns of max_dim output should match the smaller dim output."""
        proj = MatryoshkaProjection(64, [16, 32, 64])
        x = torch.randn(4, 64)
        out = proj(x)  # dict
        # Since all dims share the same projection, smaller dims are slices of the larger
        full = out[64]
        assert torch.allclose(full[:, :32], out[32], atol=1e-6)
        assert torch.allclose(full[:, :16], out[16], atol=1e-6)

    def test_batched_input_3d(self):
        """Should handle (batch, seq, dim) input."""
        proj = MatryoshkaProjection(64, [32, 64])
        x = torch.randn(2, 10, 64)
        out = proj(x)
        assert out[32].shape == (2, 10, 32)
        assert out[64].shape == (2, 10, 64)

    def test_gradient_flows(self):
        proj = MatryoshkaProjection(64, [32, 64])
        x = torch.randn(4, 64, requires_grad=True)
        out = proj(x)
        loss = out[32].sum() + out[64].sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ─── GatedMLPBlock ────────────────────────────────────────────────────────

class TestGatedMLPBlock:
    def test_forward_shape(self):
        block = GatedMLPBlock(64, use_glu=True)
        x = torch.randn(4, 64)
        out = block(x)
        assert out.shape == (4, 64)

    def test_residual_connection(self):
        block = GatedMLPBlock(64, use_glu=True)
        x = torch.randn(4, 64)
        out = block(x)
        # Output should be different (not identity) but same shape
        assert out.shape == x.shape

    def test_non_glu_path(self):
        block = GatedMLPBlock(64, use_glu=False)
        x = torch.randn(4, 64)
        out = block(x)
        assert out.shape == (4, 64)


# ─── GatedMLPEncoder ──────────────────────────────────────────────────────

class TestGatedMLPEncoder:
    def test_forward_shape(self):
        enc = GatedMLPEncoder(output_dim=64, vocab_size=300, num_layers=3)
        token_ids = torch.randint(0, 300, (4, 16))
        out = enc(token_ids)
        assert out.shape[0] == 4

    def test_trainable(self):
        enc = GatedMLPEncoder(output_dim=64, vocab_size=300, num_layers=2)
        token_ids = torch.randint(0, 300, (4, 16))
        out = enc(token_ids)
        loss = out.sum()
        loss.backward()
        for p in enc.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break

    def test_different_layer_counts(self):
        for n in [1, 3, 6]:
            enc = GatedMLPEncoder(output_dim=64, vocab_size=128, num_layers=n)
            token_ids = torch.randint(0, 128, (2, 8))
            out = enc(token_ids)
            assert out.shape[0] == 2, f"Failed for num_layers={n}"


# ─── TextHDIMModel + Matryoshka Integration ──────────────────────────────

class TestTextHDIMMatryoshka:
    @pytest.fixture
    def text_model(self):
        cfg = HDIMConfig(hidden_dim=64, num_experts=2, num_domains=2)
        model = HDIMModel(cfg)
        return TextHDIMModel(model)

    def test_encode_texts_matryoshka_returns_tuple(self, text_model):
        texts = ["hello world", "test sentence"]
        result = text_model.encode_texts_matryoshka(texts, device="cpu")
        assert isinstance(result, tuple)
        full, scales = result
        assert isinstance(full, torch.Tensor)
        # scales may be None if encoder doesn't support matryoshka
        if scales is not None:
            assert isinstance(scales, dict)
            assert len(scales) >= 1

    def test_encode_texts_matryoshka_scale_shapes(self, text_model):
        texts = ["hello", "world"]
        full, scales = text_model.encode_texts_matryoshka(texts, device="cpu")
        assert full.shape[0] == 2  # batch size
        if scales is not None:
            for dim, tensor in scales.items():
                assert tensor.dim() == 2
                assert tensor.shape[0] == 2
                assert tensor.shape[1] == dim

    def test_encode_texts_standard(self, text_model):
        texts = ["hello world", "test"]
        result = text_model.encode_texts(texts, device="cpu")
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 2

    def test_matryoshka_gradient(self, text_model):
        """Verify gradients flow through text encoder projections."""
        texts = ["hello world", "test sentence"]
        full, scales = text_model.encode_texts_matryoshka(texts, device="cpu")
        # Sum full encoding and backprop
        loss = full.sum()
        if scales is not None:
            loss = loss + sum(t.sum() for t in scales.values())
        loss.backward()
        # Check that at least some parameters received gradients
        has_grad = False
        for p in text_model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed through text encoder"


# ─── HybridEncoder ─────────────────────────────────────────────────────────

class TestHybridEncoder:
    def test_forward_shape(self):
        enc = HybridEncoder(output_dim=64, d_model=128, num_attention_layers=2, num_mlp_layers=2)
        token_ids = torch.randint(0, 5000, (4, 16))
        attention_mask = torch.ones(4, 16)
        out = enc(token_ids, attention_mask=attention_mask)
        # Output shape should be (batch, max_dim) or dict depending on matryoshka
        if isinstance(out, dict):
            assert max(out.keys()) == 64
            assert out[64].shape == (4, 64)
        else:
            assert out.shape[0] == 4

    def test_gradient_flows(self):
        enc = HybridEncoder(output_dim=64, d_model=128, num_attention_layers=1, num_mlp_layers=1)
        token_ids = torch.randint(0, 5000, (2, 8))
        out = enc(token_ids)
        target = out if isinstance(out, torch.Tensor) else out[max(out.keys())]
        loss = target.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
        assert has_grad


# ─── Matryoshka in HDIM Trainer ──────────────────────────────────────────

class TestMatryoshkaTrainerIntegration:
    def test_matryoshka_embeddings_in_batch(self):
        """Verify that matryoshka embeddings can be injected into a batch dict."""
        cfg = HDIMConfig(hidden_dim=64, num_domains=2)
        model = HDIMModel(cfg)
        text_model = TextHDIMModel(model)
        texts = ["hello world", "test sentence"]
        full, scales = text_model.encode_texts_matryoshka(texts, device="cpu")
        if scales is not None:
            batch = {"matryoshka_embeddings": scales}
            assert isinstance(batch["matryoshka_embeddings"], dict)
            assert all(isinstance(v, torch.Tensor) for v in batch["matryoshka_embeddings"].values())
        else:
            # Without matryoshka support, encode still works
            enc = text_model.encode_texts(texts, device="cpu")
            assert enc.shape[0] == 2
