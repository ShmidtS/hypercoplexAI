"""Embedding-level data augmentations for contrastive learning.

Pure PyTorch augmentations applied to embeddings during training.
Only non-anchor (pair) embeddings are augmented to preserve query consistency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EmbeddingAugmenter(nn.Module):
    """Applies augmentations to embeddings during training.

    Only augment non-anchor (pair) embeddings by default, keeping the anchor
    representation stable for contrastive loss computation.

    Args:
        noise_std: Standard deviation of Gaussian noise (0.0 = disabled).
        dropout_p: Probability of zeroing each embedding dimension (0.0 = disabled).
        mixup_alpha: Beta distribution alpha for mixup blending (0.0 = disabled).
    """

    def __init__(
        self,
        noise_std: float = 0.02,
        dropout_p: float = 0.1,
        mixup_alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.noise_std = noise_std
        self.dropout_p = dropout_p
        self.mixup_alpha = mixup_alpha

    def forward(self, embeddings: Tensor, pairs_only: bool = True) -> Tensor:
        """Apply augmentations. Only augment during training.

        Args:
            embeddings: Tensor of shape (batch_size, embed_dim).
            pairs_only: If True, only augment non-anchor embeddings (first half).

        Returns:
            Augmented embeddings tensor, same shape as input.
        """
        if not self.training:
            return embeddings

        result = embeddings

        if self.noise_std > 0:
            result = self._add_noise(result, pairs_only)

        if self.dropout_p > 0:
            result = self._embed_dropout(result, pairs_only)

        if self.mixup_alpha > 0:
            result = self._mixup(result, pairs_only)

        return result

    def _add_noise(self, x: Tensor, pairs_only: bool) -> Tensor:
        """Add Gaussian noise N(0, noise_std) to embeddings."""
        noise = torch.randn_like(x) * self.noise_std
        if pairs_only and x.shape[0] >= 2:
            half = x.shape[0] // 2
            result = x.clone()
            result[half:] = x[half:] + noise[half:]
            return result
        return x + noise

    def _embed_dropout(self, x: Tensor, pairs_only: bool) -> Tensor:
        """Feature-level dropout: zero random dimensions scaled by 1/(1-p)."""
        mask = torch.bernoulli(
            torch.full_like(x, 1.0 - self.dropout_p)
        )
        scaled = mask / (1.0 - self.dropout_p)
        if pairs_only and x.shape[0] >= 2:
            half = x.shape[0] // 2
            result = x.clone()
            result[half:] = x[half:] * scaled[half:]
            return result
        return x * scaled

    def _mixup(self, x: Tensor, pairs_only: bool) -> Tensor:
        """Mixup: blend each pair element with a noisy copy of itself.

        For each non-anchor embedding b, compute:
            b' = alpha * b + (1 - alpha) * noise(b)
        where alpha ~ Beta(mixup_alpha, mixup_alpha).
        """
        if x.shape[0] < 2:
            return x

        half = x.shape[0] // 2
        target = x[half:] if pairs_only else x
        alpha = torch.distributions.Beta(
            self.mixup_alpha, self.mixup_alpha
        ).sample().item()
        noisy = target + torch.randn_like(target) * self.noise_std * 0.5
        blended = alpha * target + (1.0 - alpha) * noisy

        if pairs_only:
            result = x.clone()
            result[half:] = blended
            return result
        return blended
