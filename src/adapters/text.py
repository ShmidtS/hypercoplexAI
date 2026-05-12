from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn

from src.core.engine import HDIMCoreEngine
from src.core.types import AnalogyMatch


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
        text_config: Any,
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

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

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

    def encode(
        self,
        texts: list,
    ) -> torch.Tensor:
        return self.forward(texts, device=self.device, dtype=self.dtype)

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


class TextAdapter:
    def __init__(self, encoder: nn.Module, engine: HDIMCoreEngine):
        self.encoder = encoder
        self.engine = engine

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        device = next(self.engine.parameters()).device
        dtype = next(self.engine.parameters()).dtype
        return self.encoder(texts, device=device, dtype=dtype)

    def encode(self, texts: list[str]) -> torch.Tensor:
        """texts -> G (multivector)"""
        embeddings = self.encode_texts(texts)
        return self.engine.encode(embeddings)

    def extract_texts(self, texts: list[str], domain: str) -> torch.Tensor:
        """texts -> U_inv"""
        G = self.encode(texts)
        return self.engine.extract(G, domain)

    def match_texts(self, texts: list[str], domain: str, top_k: int = 5) -> list[list[AnalogyMatch]]:
        """Encode texts and return ranked analogy matches.

        Args:
            texts: Input strings encoded to invariants with shape (batch, clifford_dim).
            domain: Domain name used for extraction before index search.
        """
        U = self.extract_texts(texts, domain)
        return self.engine.index.search(U, top_k=top_k)
