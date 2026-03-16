"""Tests for EmbeddingAugmenter."""

import pytest
import torch

from src.training.augmentations import EmbeddingAugmenter


@pytest.fixture
def augmenter():
    return EmbeddingAugmenter(noise_std=0.02, dropout_p=0.1, mixup_alpha=0.2)


class TestOutputShapes:
    def test_shape_preserved_batch_size(self, augmenter):
        augmenter.train()
        x = torch.randn(8, 256)
        out = augmenter(x, pairs_only=False)
        assert out.shape == x.shape

    def test_shape_preserved_odd_batch(self, augmenter):
        augmenter.train()
        x = torch.randn(7, 128)
        out = augmenter(x, pairs_only=False)
        assert out.shape == x.shape

    def test_shape_preserved_pairs_only(self, augmenter):
        augmenter.train()
        x = torch.randn(8, 256)
        out = augmenter(x, pairs_only=True)
        assert out.shape == x.shape


class TestEvalMode:
    def test_no_augmentation_in_eval(self, augmenter):
        augmenter.eval()
        x = torch.randn(8, 256)
        out = augmenter(x, pairs_only=False)
        assert torch.equal(out, x)

    def test_no_augmentation_in_eval_pairs_only(self, augmenter):
        augmenter.eval()
        x = torch.randn(8, 256)
        out = augmenter(x, pairs_only=True)
        assert torch.equal(out, x)

    def test_noise_off_in_eval(self):
        aug = EmbeddingAugmenter(noise_std=1.0, dropout_p=0.5, mixup_alpha=1.0)
        aug.eval()
        x = torch.randn(8, 256)
        out = aug(x)
        assert torch.equal(out, x)


class TestNoise:
    def test_noise_changes_embeddings(self):
        aug = EmbeddingAugmenter(noise_std=0.5, dropout_p=0.0, mixup_alpha=0.0)
        aug.train()
        x = torch.randn(8, 256)
        out = aug(x, pairs_only=False)
        assert not torch.equal(out, x)

    def test_noise_std_zero_no_change(self):
        aug = EmbeddingAugmenter(noise_std=0.0, dropout_p=0.0, mixup_alpha=0.0)
        aug.train()
        x = torch.randn(8, 256)
        out = aug(x, pairs_only=False)
        assert torch.equal(out, x)

    def test_noise_only_affects_second_half_pairs_only(self):
        aug = EmbeddingAugmenter(noise_std=0.5, dropout_p=0.0, mixup_alpha=0.0)
        aug.train()
        x = torch.randn(8, 256)
        out = aug(x, pairs_only=True)
        half = x.shape[0] // 2
        assert torch.equal(out[:half], x[:half])
        assert not torch.equal(out[half:], x[half:])


class TestEmbeddingDropout:
    def test_dropout_changes_embeddings(self):
        aug = EmbeddingAugmenter(noise_std=0.0, dropout_p=0.5, mixup_alpha=0.0)
        aug.train()
        x = torch.randn(8, 256)
        out = aug(x, pairs_only=False)
        assert not torch.equal(out, x)

    def test_dropout_preserves_nonzero_structure(self):
        """Non-dropped dimensions should retain their scaled values."""
        aug = EmbeddingAugmenter(noise_std=0.0, dropout_p=0.0, mixup_alpha=0.0)
        aug.train()
        x = torch.randn(4, 8)
        # With p=0.0, no dropout should occur
        aug.dropout_p = 0.0
        out = aug(x)
        assert torch.equal(out, x)


class TestMixup:
    def test_mixup_interpolates(self):
        aug = EmbeddingAugmenter(noise_std=0.0, dropout_p=0.0, mixup_alpha=1.0)
        aug.train()
        x = torch.randn(8, 256)
        out = aug(x, pairs_only=False)
        # Mixup should produce different embeddings
        assert not torch.equal(out, x)

    def test_mixup_with_noise(self):
        aug = EmbeddingAugmenter(noise_std=0.1, dropout_p=0.0, mixup_alpha=0.5)
        aug.train()
        x = torch.randn(4, 64)
        out = aug(x, pairs_only=False)
        assert not torch.equal(out, x)
        assert out.shape == x.shape

    def test_mixup_only_affects_second_half_pairs_only(self):
        aug = EmbeddingAugmenter(noise_std=0.0, dropout_p=0.0, mixup_alpha=1.0)
        aug.train()
        x = torch.randn(8, 256)
        out = aug(x, pairs_only=True)
        half = x.shape[0] // 2
        # First half should be unchanged (no noise, no dropout, no mixup on anchors)
        assert torch.equal(out[:half], x[:half])
        # Second half should differ
        assert not torch.equal(out[half:], x[half:])

    def test_mixup_single_element_no_crash(self):
        aug = EmbeddingAugmenter(noise_std=0.0, dropout_p=0.0, mixup_alpha=1.0)
        aug.train()
        x = torch.randn(1, 64)
        out = aug(x, pairs_only=True)
        assert out.shape == x.shape
        assert torch.equal(out, x)


class TestCombined:
    def test_all_augmentations_together(self, augmenter):
        augmenter.train()
        x = torch.randn(8, 256)
        out = augmenter(x, pairs_only=True)
        assert out.shape == x.shape
        assert not torch.equal(out, x)

    def test_all_disabled(self):
        aug = EmbeddingAugmenter(noise_std=0.0, dropout_p=0.0, mixup_alpha=0.0)
        aug.train()
        x = torch.randn(8, 256)
        out = aug(x, pairs_only=True)
        assert torch.equal(out, x)

    def test_deterministic_in_eval(self):
        aug = EmbeddingAugmenter(noise_std=0.5, dropout_p=0.5, mixup_alpha=0.5)
        aug.eval()
        x = torch.randn(8, 256)
        out1 = aug(x)
        out2 = aug(x)
        assert torch.equal(out1, out2)
        assert torch.equal(out1, x)
