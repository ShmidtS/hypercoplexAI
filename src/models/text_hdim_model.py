"""Trainable text-facing wrapper for HDIM retrieval and transfer workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hdim_model import HDIMAuxState, HDIMModel, HDIMTextConfig


@dataclass(frozen=True)
class TextPairScoreResult:
    """Structured text-pair scoring artifact for retrieval/ranking workflows."""

    scores: torch.Tensor
    source_state: HDIMAuxState
    target_state: HDIMAuxState

    def to_dict(self) -> dict[str, torch.Tensor | bool | str]:
        result: dict[str, torch.Tensor | bool | str] = {"scores": self.scores}
        for prefix, state in (
            ("source", self.source_state),
            ("target", self.target_state),
        ):
            for key, value in state.to_dict().items():
                result[f"{prefix}_{key}"] = value
        return result


def _load_vocab(vocab_path: Optional[str]) -> Optional[dict[str, int]]:
    if not vocab_path:
        return None

    path = Path(vocab_path)
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary path does not exist: {vocab_path}")

    vocab: dict[str, int] = {}
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        token = line.strip()
        if not token or token in vocab:
            continue
        vocab[token] = index
    return vocab


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
        vocab: Optional[dict[str, int]] = None,
        tokenizer_name: Optional[str] = None,
        vocab_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        embedding_dim = output_dim if embedding_dim is None else embedding_dim
        hidden_dim = output_dim if hidden_dim is None else hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.padding_idx = 0
        self.output_dim = output_dim
        self.vocab = vocab
        self.tokenizer_name = tokenizer_name
        self.vocab_path = vocab_path
        self.unk_idx = vocab_size - 1
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=self.padding_idx,
        )
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    @classmethod
    def from_text_config(
        cls,
        output_dim: int,
        text_config: HDIMTextConfig,
        *,
        fallback_dropout: float,
    ) -> "SimpleTextEncoder":
        vocab = _load_vocab(text_config.vocab_path)
        resolved_vocab_size = text_config.vocab_size
        if vocab is not None:
            max_vocab_token = max(vocab.values(), default=0)
            resolved_vocab_size = max(resolved_vocab_size, max_vocab_token + 2)

        return cls(
            output_dim=output_dim,
            vocab_size=resolved_vocab_size,
            max_length=text_config.max_length,
            embedding_dim=text_config.embedding_dim,
            hidden_dim=text_config.hidden_dim,
            dropout=fallback_dropout
            if text_config.dropout is None
            else text_config.dropout,
            vocab=vocab,
            tokenizer_name=text_config.tokenizer_name,
            vocab_path=text_config.vocab_path,
        )

    def _encode_text(self, text: str) -> list[int]:
        if self.vocab is not None:
            tokens = list(text) if self.tokenizer_name == "char" else text.split()
            return [self.vocab.get(token, self.unk_idx) for token in tokens[: self.max_length]]
        return [min(ord(ch), self.vocab_size - 2) + 1 for ch in text[: self.max_length]]

    def tokenize(
        self,
        texts: Sequence[str],
        *,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(texts) == 0:
            empty_tokens = torch.empty(
                0,
                self.max_length,
                dtype=torch.long,
                device=device,
            )
            empty_mask = torch.empty(
                0,
                self.max_length,
                dtype=torch.bool,
                device=device,
            )
            return empty_tokens, empty_mask

        token_ids = torch.zeros(len(texts), self.max_length, dtype=torch.long)
        attention_mask = torch.zeros(len(texts), self.max_length, dtype=torch.bool)
        for row, text in enumerate(texts):
            encoded = self._encode_text(str(text))
            if encoded:
                length = min(len(encoded), self.max_length)
                token_ids[row, :length] = torch.tensor(encoded[:length], dtype=torch.long)
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
            return torch.empty(
                0,
                self.output_dim,
                device=device,
                dtype=dtype or torch.float32,
            )

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
    """Trainable text-entry wrapper around HDIMModel."""

    def __init__(
        self,
        core_model: HDIMModel,
        *,
        max_length: int | None = None,
        text_embedding_dim: int | None = None,
        text_hidden_dim: int | None = None,
        text_dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.core_model = core_model
        text_config = core_model.config.text
        if any(
            value is not None
            for value in (max_length, text_embedding_dim, text_hidden_dim, text_dropout)
        ):
            text_config = HDIMTextConfig(
                vocab_size=text_config.vocab_size,
                max_length=text_config.max_length if max_length is None else max_length,
                embedding_dim=text_config.embedding_dim
                if text_embedding_dim is None
                else text_embedding_dim,
                hidden_dim=text_config.hidden_dim
                if text_hidden_dim is None
                else text_hidden_dim,
                dropout=text_config.dropout if text_dropout is None else text_dropout,
                vocab_path=text_config.vocab_path,
                tokenizer_name=text_config.tokenizer_name,
            )

        self.text_config = text_config
        self.text_encoder = SimpleTextEncoder.from_text_config(
            output_dim=core_model.config.hidden_dim,
            text_config=text_config,
            fallback_dropout=core_model.config.dropout,
        )

    @property
    def config(self):
        return self.core_model.config

    @property
    def pipeline(self):
        return self.core_model.pipeline

    @property
    def training_inv_head(self):
        return self.core_model.training_inv_head

    def forward(self, *args, **kwargs):
        return self.core_model(*args, **kwargs)

    def transfer(self, *args, **kwargs):
        return self.core_model.transfer(*args, **kwargs)

    def transfer_pairs(self, *args, **kwargs):
        return self.core_model.transfer_pairs(*args, **kwargs)

    def reset_memory(self, strategy: str = 'geometric') -> None:
        self.core_model.reset_memory(strategy=strategy)

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
        return HDIMModel.forward(
            self.core_model,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, HDIMAuxState]:
        encodings = self.encode_texts(source_texts, device=source_domain_id.device)
        return HDIMModel.transfer_pairs(
            self.core_model,
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
