"""Frozen SBERT encoder wrapper for HDIM.

Использует предобученную sentence-transformer модель как frozen encoder.
Градиенты не текут через SBERT — обучается только HDIM core.

Преимущества:
- Реальная семантика с нуля (110M параметров уже обучены)
- output_dim=768 (paraphrase-multilingual-mpnet-base-v2)
- Поддержка multilingual текстов
- Не требует обучения encoder части
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


class SBERTEncoder(nn.Module):
    """Frozen sentence-transformer encoder + trainable projection.

    Архитектура:
      SBERT (frozen) → projection (trainable) → output_dim

    SBERT выдаёт 768-мерные embeddings.
    Projection адаптирует их под HDIM hidden_dim.
    """

    MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
    SBERT_DIM = 768

    def __init__(
        self,
        output_dim: int,
        *,
        model_name: str = MODEL_NAME,
        projection_hidden: Optional[int] = None,
        dropout: float = 0.1,
        freeze: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.model_name = model_name
        self.freeze = freeze

        # Lazy import to avoid hard dependency at module load
        from sentence_transformers import SentenceTransformer

        # Load SBERT on CPU
        _sbert_model = SentenceTransformer(model_name, device="cpu")
        self._sbert_dim = _sbert_model.get_sentence_embedding_dimension()

        if freeze:
            for param in _sbert_model.parameters():
                param.requires_grad = False
            # Store SBERT outside nn.Module registry so .to(device) does NOT move it
            # This keeps 278M params on CPU, freeing GPU VRAM for HDIM core
            object.__setattr__(self, '_sbert', _sbert_model)
            self._sbert_on_cpu = True
            # Embedding cache for frozen SBERT (avoids repeated CPU inference)
            object.__setattr__(self, '_embedding_cache', {})
        else:
            # Trainable SBERT: register normally so it participates in .to()
            self._sbert = _sbert_model
            self._sbert_on_cpu = False
            object.__setattr__(self, '_embedding_cache', None)

        # Trainable projection SBERT_DIM → output_dim
        phidden = projection_hidden or max(output_dim, self._sbert_dim // 2)
        self.projection = nn.Sequential(
            nn.Linear(self._sbert_dim, phidden),
            nn.LayerNorm(phidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(phidden, output_dim),
        )
        # Small init for stable start
        nn.init.normal_(self.projection[0].weight, std=0.02)
        nn.init.zeros_(self.projection[0].bias)
        nn.init.normal_(self.projection[4].weight, std=0.02)
        nn.init.zeros_(self.projection[4].bias)

    def _encode_with_sbert(
        self, texts: Sequence[str], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Run SBERT encoding, return tensor on target device.

        Frozen SBERT always encodes on CPU to save GPU VRAM (278M params).
        Result is moved to target_device after encoding.
        """
        target_device = device or next(self.projection.parameters()).device
        proj_dtype = next(self.projection.parameters()).dtype

        # Frozen SBERT: CPU to conserve GPU VRAM
        sbert_device = "cpu" if self.freeze else str(target_device)

        if self.freeze:
            with torch.no_grad():
                embeddings = self._sbert.encode(
                    list(texts),
                    convert_to_tensor=True,
                    device=sbert_device,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
        else:
            embeddings = self._sbert.encode(
                list(texts),
                convert_to_tensor=True,
                device=sbert_device,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

        # Move to target device and clone to exit inference_mode context
        return embeddings.to(device=target_device, dtype=proj_dtype).clone()

    def forward(
        self,
        texts: Sequence[str],
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Args:
            texts: список строк
        Returns:
            (batch, output_dim) projected SBERT representations
        """
        if len(texts) == 0:
            target_device = device or next(self.projection.parameters()).device
            return torch.empty(0, self.output_dim, device=target_device)

        # Use cache for frozen SBERT (avoid repeated CPU inference)
        if self._sbert_on_cpu and self._embedding_cache is not None:
            target_device = device or next(self.projection.parameters()).device
            proj_dtype = next(self.projection.parameters()).dtype
            cached = []
            missing_indices = []
            missing_texts = []
            for i, text in enumerate(texts):
                if text in self._embedding_cache:
                    cached.append((i, self._embedding_cache[text]))
                else:
                    missing_indices.append(i)
                    missing_texts.append(text)

            if missing_texts:
                new_embs = self._encode_with_sbert(missing_texts, device=target_device)
                for j, (idx, text) in enumerate(zip(missing_indices, missing_texts)):
                    self._embedding_cache[text] = new_embs[j].cpu()
                    cached.append((idx, self._embedding_cache[text]))

            cached.sort(key=lambda x: x[0])
            sbert_emb = torch.stack(
                [emb.to(device=target_device, dtype=proj_dtype) for _, emb in cached]
            )
        else:
            sbert_emb = self._encode_with_sbert(texts, device=device)  # (B, SBERT_DIM)

        projected = self.projection(sbert_emb)  # (B, output_dim)

        if dtype is not None:
            projected = projected.to(dtype=dtype)
        return projected

    def precompute_cache(self, texts: Sequence[str], batch_size: int = 256) -> None:
        """Pre-encode all texts with SBERT and store in cache."""
        import sys
        if not self._sbert_on_cpu:
            return
        # Filter already cached
        unique_texts = [t for t in set(texts) if t not in self._embedding_cache]
        if not unique_texts:
            print(f"SBERT cache: {len(self._embedding_cache)} embeddings (all cached)")
            sys.stdout.flush()
            return
        print(f"SBERT cache: encoding {len(unique_texts)} unique texts...")
        sys.stdout.flush()
        with torch.no_grad():
            embs = self._sbert.encode(
                unique_texts,
                convert_to_tensor=True,
                device="cpu",
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        for text, emb in zip(unique_texts, embs):
            self._embedding_cache[text] = emb.cpu()
        print(f"SBERT cache: {len(self._embedding_cache)} embeddings precomputed")
        sys.stdout.flush()
