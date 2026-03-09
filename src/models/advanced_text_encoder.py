"""
HDIM — Advanced Text Encoder
Многоуровневый текстовый энкодер с self-attention, RoPE и residual connections.

Архитектура:
  1. Token + Position embedding (RoPE)
  2. N x TransformerBlock (multi-head attention + FFN)
  3. Attention pooling → output_dim

Backward compatible с SimpleTextEncoder через фабрику from_text_config().
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hdim_model import HDIMTextConfig


# ============================================================
#  Rotary Positional Encoding (RoPE)
# ============================================================

class RotaryEmbedding(nn.Module):
    """
    RoPE: Rotary Position Embedding (Su et al., 2023).
    Применяет вращение к парам размерностей Q и K.
    """

    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for RoPE"
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cache", emb.cos()[None, None, :, :])  # (1,1,seq,dim)
        self.register_buffer("sin_cache", emb.sin()[None, None, :, :])

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q, k: (batch, heads, seq_len, head_dim)
        """
        seq_len = q.shape[2]
        cos = self.cos_cache[:, :, :seq_len, :].to(q.device)
        sin = self.sin_cache[:, :, :seq_len, :].to(q.device)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot.to(q.dtype), k_rot.to(k.dtype)


# ============================================================
#  Multi-Head Self-Attention с RoPE
# ============================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention с RoPE и optional causal mask.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

        # Xavier init for stable training
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            attention_mask: (batch, seq_len) bool, True = valid token
        Returns:
            (batch, seq_len, embed_dim)
        """
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, H, D)
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self.rope(q, k)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        if attention_mask is not None:
            # mask: (B, T) → (B, 1, 1, T)
            # Use -1e4 instead of -1e9 to prevent NaN in softmax with float32
            mask = attention_mask[:, None, None, :].float()
            mask = (1.0 - mask) * -1e4
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        # Replace any NaN from all-masked rows with uniform attention
        attn = torch.nan_to_num(attn, nan=1.0 / attn.shape[-1])
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


# ============================================================
#  TransformerBlock
# ============================================================

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LN → Attn → Res + LN → FFN → Res."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim, num_heads=num_heads, dropout=dropout, max_seq_len=max_seq_len
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        ffn_dim = embed_dim * ffn_mult
        ffn_linear1 = nn.Linear(embed_dim, ffn_dim)
        ffn_linear2 = nn.Linear(ffn_dim, embed_dim)
        nn.init.xavier_uniform_(ffn_linear1.weight)
        nn.init.xavier_uniform_(ffn_linear2.weight)
        nn.init.zeros_(ffn_linear2.bias)  # zero-init output projection for stable residual
        self.ffn = nn.Sequential(
            ffn_linear1,
            nn.GELU(),
            nn.Dropout(dropout),
            ffn_linear2,
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
#  Attention Pooling
# ============================================================

class AttentionPooling(nn.Module):
    """
    Weighted pooling через learned query вектор.
    Более информативно чем mean pooling.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))  # zero-init: stable start
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            attention_mask: (batch, seq_len) bool
        Returns:
            (batch, embed_dim)
        """
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)  # (B, 1, D)
        key_padding_mask = None
        if attention_mask is not None:
            # nn.MultiheadAttention expects True = ignore
            key_padding_mask = ~attention_mask
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return self.norm(out.squeeze(1))  # (B, D)


# ============================================================
#  AdvancedTextEncoder
# ============================================================

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


class AdvancedTextEncoder(nn.Module):
    """
    Продвинутый текстовый энкодер для HDIM.

    Архитектура:
      Token embedding + Transformer (N layers, RoPE) + Attention Pooling + Projection

    Превосходит SimpleTextEncoder:
      - Контекстные представления через self-attention
      - RoPE вместо learned positional embeddings
      - Attention pooling вместо mean pooling
      - Pre-norm + residual для стабильного обучения

    Backward compatible: поддерживает тот же интерфейс что SimpleTextEncoder.
    """

    def __init__(
        self,
        output_dim: int,
        *,
        vocab_size: int = 257,
        max_length: int = 128,
        embedding_dim: Optional[int] = None,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        vocab: Optional[dict[str, int]] = None,
        tokenizer_name: Optional[str] = None,
        vocab_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        embedding_dim = output_dim if embedding_dim is None else embedding_dim
        # Выравниваем embedding_dim до кратного num_heads
        if embedding_dim % num_heads != 0:
            embedding_dim = math.ceil(embedding_dim / num_heads) * num_heads

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.padding_idx = 0
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.vocab = vocab
        self.tokenizer_name = tokenizer_name
        self.vocab_path = vocab_path
        self.unk_idx = vocab_size - 1

        # Token embedding (без positional — используем RoPE внутри attention)
        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=self.padding_idx
        )
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        self.input_norm = nn.LayerNorm(embedding_dim)
        self.input_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                ffn_mult=ffn_mult,
                dropout=dropout,
                max_seq_len=max_length,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)

        # Attention pooling + projection
        self.pooling = AttentionPooling(embedding_dim)
        self.proj = nn.Linear(embedding_dim, output_dim)
        # Small init for final projection — prevent large initial outputs
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    @classmethod
    def from_text_config(
        cls,
        output_dim: int,
        text_config: HDIMTextConfig,
        *,
        fallback_dropout: float,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> "AdvancedTextEncoder":
        vocab = _load_vocab(text_config.vocab_path)
        resolved_vocab_size = text_config.vocab_size
        if vocab is not None:
            max_vocab_token = max(vocab.values(), default=0)
            resolved_vocab_size = max(resolved_vocab_size, max_vocab_token + 2)

        embedding_dim = text_config.embedding_dim or output_dim
        return cls(
            output_dim=output_dim,
            vocab_size=resolved_vocab_size,
            max_length=text_config.max_length,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=fallback_dropout if text_config.dropout is None else text_config.dropout,
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
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(texts) == 0:
            return (
                torch.empty(0, self.max_length, dtype=torch.long, device=device),
                torch.empty(0, self.max_length, dtype=torch.bool, device=device),
            )

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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Args:
            texts: список строк
        Returns:
            (batch, output_dim) encoded representations
        """
        token_ids, attention_mask = self.tokenize(texts, device=device)
        if token_ids.numel() == 0:
            return torch.empty(0, self.output_dim, device=device, dtype=dtype or torch.float32)

        # Embedding
        x = self.token_embedding(token_ids)  # (B, T, D)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        x = self.final_norm(x)

        # Attention pooling
        pooled = self.pooling(x, attention_mask=attention_mask)  # (B, D)

        # Projection to output_dim
        out = self.proj(pooled)

        if dtype is not None:
            out = out.to(dtype=dtype)
        return out
