"""Trainable text-facing wrapper for HDIM retrieval and transfer workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hdim_model import HDIMAuxState, HDIMModel


@dataclass(frozen=True)
class TextPairScoreResult:
    """Structured text-pair scoring artifact for retrieval/ranking workflows."""

    scores: torch.Tensor
    source_state: HDIMAuxState
    target_state: HDIMAuxState

    def to_dict(self) -> dict[str, torch.Tensor | bool | str]:
        result: dict[str, torch.Tensor | bool | str] = {"scores": self.scores}
        for prefix, state in (("source", self.source_state), ("target", self.target_state)):
            for key, value in state.to_dict().items():
                result[f"{prefix}_{key}"] = value
        return result


class SimpleTextEncoder(nn.Module):
    """Compact trainable text encoder for HDIM text experiments."""

    def __init__(
        self,
        output_dim: int,
        *,
        vocab_size: int = 257,
        max_length: int = 128,
        embedding_dim: int | None = None,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        embedding_dim = output_dim if embedding_dim is None else embedding_dim
        hidden_dim = output_dim if hidden_dim is None else hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.padding_idx = 0
        self.output_dim = output_dim
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.padding_idx)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def tokenize(self, texts: Sequence[str], *, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if len(texts) == 0:
            empty_tokens = torch.empty(0, self.max_length, dtype=torch.long, device=device)
            empty_mask = torch.empty(0, self.max_length, dtype=torch.bool, device=device)
            return empty_tokens, empty_mask

        token_ids = torch.zeros(len(texts), self.max_length, dtype=torch.long)
        attention_mask = torch.zeros(len(texts), self.max_length, dtype=torch.bool)
        for row, text in enumerate(texts):
            truncated = text[: self.max_length]
            encoded = [min(ord(ch), self.vocab_size - 2) + 1 for ch in truncated]
            if encoded:
                length = len(encoded)
                token_ids[row, :length] = torch.tensor(encoded, dtype=torch.long)
                attention_mask[row, :length] = True
        if device is not None:
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
        return token_ids, attention_mask

    def forward(
        self,
        texts: Sequence[str],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        token_ids, attention_mask = self.tokenize(texts, device=device)
        if token_ids.numel() == 0:
            return torch.empty(0, self.output_dim, device=device, dtype=dtype or torch.float32)

        positions = torch.arange(self.max_length, device=token_ids.device).unsqueeze(0)
        embeddings = self.token_embedding(token_ids) + self.position_embedding(positions)
        embeddings = self.norm(embeddings)
        mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
        pooled = (embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        pooled = pooled / counts
        encoded = self.mlp(pooled)
        if dtype is not None:
            encoded = encoded.to(dtype=dtype)
        return encoded


class TextHDIMModel(nn.Module):
    """Trainable text-entry wrapper around HDIMModel.

    Raw texts are tokenized with a compact trainable encoder and projected into
    the HDIM hidden space. This wrapper targets retrieval/ranking and transfer
    experiments, not generative language modeling.
    """

    def __init__(
        self,
        core_model: HDIMModel,
        *,
        max_length: int = 128,
        text_embedding_dim: int | None = None,
        text_hidden_dim: int | None = None,
        text_dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.core_model = core_model
        self.text_encoder = SimpleTextEncoder(
            output_dim=core_model.config.hidden_dim,
            max_length=max_length,
            embedding_dim=text_embedding_dim,
            hidden_dim=text_hidden_dim,
            dropout=core_model.config.dropout if text_dropout is None else text_dropout,
        )

    @property
    def config(self):
        return self.core_model.config

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = next(self.core_model.parameters()).dtype
        if device is None:
            device = next(self.core_model.parameters()).device
        return self.text_encoder(texts, device=device, dtype=dtype)

    def forward_texts(
        self,
        texts: Sequence[str],
        domain_id: torch.Tensor,
        *,
        return_state: bool = False,
        update_memory: bool = True,
        memory_mode: str = "update",
    ):
        encodings = self.encode_texts(texts, device=domain_id.device)
        return self.core_model(
            encodings,
            domain_id,
            return_state=return_state,
            update_memory=update_memory,
            memory_mode=memory_mode,
        )

    def transfer_texts(
        self,
        texts: Sequence[str],
        *,
        source_domain: int,
        target_domain: int,
        return_state: bool = False,
        update_memory: bool = True,
        memory_mode: str = "update",
    ):
        encodings = self.encode_texts(texts)
        return self.core_model.transfer(
            encodings,
            source_domain=source_domain,
            target_domain=target_domain,
            return_state=return_state,
            update_memory=update_memory,
            memory_mode=memory_mode,
        )

    def transfer_text_pairs(
        self,
        source_texts: Sequence[str],
        source_domain_id: torch.Tensor,
        target_domain_id: torch.Tensor,
        *,
        update_memory: bool = True,
        memory_mode: str = "update",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, HDIMAuxState]:
        encodings = self.encode_texts(source_texts, device=source_domain_id.device)
        return self.core_model.transfer_pairs(
            encodings,
            source_domain_id,
            target_domain_id,
            update_memory=update_memory,
            memory_mode=memory_mode,
        )

    def score_text_pairs(
        self,
        source_texts: Sequence[str],
        target_texts: Sequence[str],
        source_domain_id: torch.Tensor,
        target_domain_id: torch.Tensor,
    ) -> torch.Tensor:
        return self.score_text_pairs_with_state(
            source_texts,
            target_texts,
            source_domain_id,
            target_domain_id,
        ).scores

    def score_text_pairs_with_state(
        self,
        source_texts: Sequence[str],
        target_texts: Sequence[str],
        source_domain_id: torch.Tensor,
        target_domain_id: torch.Tensor,
    ) -> TextPairScoreResult:
        _, _, _, src_state = self.transfer_text_pairs(
            source_texts,
            source_domain_id,
            target_domain_id,
            update_memory=False,
            memory_mode="retrieve",
        )
        _, _, _, tgt_state = self.forward_texts(
            target_texts,
            target_domain_id,
            return_state=True,
            update_memory=False,
            memory_mode="retrieve",
        )
        scores = F.cosine_similarity(
            src_state.exported_invariant,
            tgt_state.exported_invariant,
            dim=-1,
        )
        return TextPairScoreResult(
            scores=scores,
            source_state=src_state,
            target_state=tgt_state,
        )
