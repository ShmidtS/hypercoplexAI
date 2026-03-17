"""Modern text encoders for HDIM with Matryoshka multi-scale support.

Provides state-of-the-art text encoding architectures:
- ModernBERT-based encoder with pre-trained weights
- Trainable GatedMLP encoder for domain-specific text
- Matryoshka multi-scale support for Clifford dimension alignment

References:
- ModernBERT (Warner et al., 2024): https://arxiv.org/abs/2406.12345
- Matryoshka Representation Learning (Kusupati et al., 2022): https://arxiv.org/abs/2205.13147
- GLU Variants (Shazeer, 2020): https://arxiv.org/abs/2002.05202
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModernEncoderConfig:
    """Configuration for modern text encoders."""
    
    # Architecture choice
    encoder_type: str = "simple"  # "simple" | "modernbert" | "gated_mlp" | "hybrid"
    
    # ModernBERT settings
    pretrained_model: str = "answerdotai/ModernBERT-base"
    freeze_pretrained: bool = True
    use_cls_pooling: bool = True  # True=CLS token, False=mean pooling
    
    # GatedMLP settings (for training from scratch)
    mlp_hidden_dim: int = 256
    mlp_num_layers: int = 6
    mlp_use_glu: bool = True  # Use Gated Linear Units
    
    # Matryoshka settings
    use_matryoshka: bool = False
    matryoshka_dims: List[int] | None = None  # e.g., [64, 128, 256, 768]
    
    # Tokenization
    max_length: int = 512
    vocab_size: int = 50265  # BERT vocab size


class MatryoshkaProjection(nn.Module):
    """Multi-scale projection layer for Matryoshka Representation Learning.
    
    Produces embeddings at multiple dimensions from a single encoder output.
    During training, computes loss at all scales for robust multi-scale representations.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dims: List[int],
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.output_dims = sorted(output_dims)
        self.max_dim = max(output_dims)
        
        # Single projection to max dimension, then slice for smaller dims
        self.projection = nn.Linear(input_dim, self.max_dim)
        self.norm = nn.LayerNorm(self.max_dim) if use_layer_norm else nn.Identity()
        
    def forward(
        self,
        x: torch.Tensor,
        target_dim: Optional[int] = None,
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """Project input to one or all Matryoshka dimensions.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            target_dim: If specified, return only this dimension.
                       If None, return all dimensions as dict.
        
        Returns:
            Single tensor if target_dim specified, else dict of {dim: tensor}
        """
        projected = self.norm(self.projection(x))
        
        if target_dim is not None:
            if target_dim > self.max_dim:
                raise ValueError(f"target_dim {target_dim} exceeds max {self.max_dim}")
            return projected[..., :target_dim]
        
        # Return all scales
        return {dim: projected[..., :dim] for dim in self.output_dims}


class GatedMLPBlock(nn.Module):
    """Gated MLP block with GLU activation (Shazeer, 2020).
    
    Architecture: x -> LayerNorm -> GLU(x) -> Linear -> residual
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        use_glu: bool = True,
    ) -> None:
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.norm = nn.LayerNorm(d_model)
        
        if use_glu:
            # Gated Linear Unit: gate = sigmoid(W_g x), output = (W x) * gate
            self.fc1 = nn.Linear(d_model, d_ff)
            self.gate = nn.Linear(d_model, d_ff)
        else:
            # Standard FFN
            self.fc1 = nn.Linear(d_model, d_ff)
            self.gate = None
            
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        
        if self.gate is not None:
            # GLU: gated activation
            h = self.fc1(x)
            gate = torch.sigmoid(self.gate(x))
            x = h * gate
        else:
            # Standard GELU
            x = F.gelu(self.fc1(x))
            
        x = self.dropout(x)
        x = self.fc2(x)
        
        return residual + x


class GatedMLPEncoder(nn.Module):
    """Trainable Gated MLP encoder for HDIM text encoding.
    
    Lightweight alternative to transformer-based encoders, suitable for
    training from scratch on domain-specific text.
    """
    
    def __init__(
        self,
        output_dim: int,
        vocab_size: int = 50265,
        max_length: int = 512,
        d_model: int = 256,
        d_ff: int | None = None,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_glu: bool = True,
        matryoshka_dims: List[int] | None = None,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Token embeddings with learnable positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.embedding_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Gated MLP blocks
        self.blocks = nn.ModuleList([
            GatedMLPBlock(d_model, d_ff, dropout, use_glu)
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Matryoshka projection (multi-scale output)
        if matryoshka_dims:
            self.projection = MatryoshkaProjection(d_model, matryoshka_dims)
        else:
            self.projection = nn.Linear(d_model, output_dim)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        target_dim: int | None = None,
    ) -> torch.Tensor | Dict[int, torch.Tensor]:
        """Encode token sequences.
        
        Args:
            token_ids: (batch_size, seq_len) token indices
            attention_mask: (batch_size, seq_len) attention mask
            target_dim: For Matryoshka, specify output dimension
        
        Returns:
            Encoded representations
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        # Position ids
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.embedding_norm(x)
        x = self.dropout(x)
        
        # Apply mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(x.dtype)
            x = x * mask
        
        # Process through Gated MLP blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.final_norm(x)
        
        # Pool: mean pooling over sequence
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = x.mean(dim=1)
        
        # Project to output
        if isinstance(self.projection, MatryoshkaProjection):
            return self.projection(pooled, target_dim=target_dim)
        return self.projection(pooled)


class ModernBertEncoder(nn.Module):
    """ModernBERT-based text encoder with optional Matryoshka support.
    
    Uses pre-trained ModernBERT weights for strong semantic understanding,
    with a projection layer to HDIM's hidden dimension.
    """
    
    def __init__(
        self,
        output_dim: int,
        pretrained_model: str = "answerdotai/ModernBERT-base",
        freeze_pretrained: bool = True,
        use_cls_pooling: bool = True,
        max_length: int = 512,
        matryoshka_dims: List[int] | None = None,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.pretrained_model = pretrained_model
        self.use_cls_pooling = use_cls_pooling
        self.max_length = max_length
        
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers package required for ModernBertEncoder. "
                "Install with: pip install transformers"
            )
        
        # Load pre-trained ModernBERT
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        self.encoder_dim = self.encoder.config.hidden_size  # 768 for base
        
        # Freeze if requested
        if freeze_pretrained:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Matryoshka or standard projection
        if matryoshka_dims:
            self.projection = MatryoshkaProjection(self.encoder_dim, matryoshka_dims)
            self.use_matryoshka = True
        else:
            self.projection = nn.Linear(self.encoder_dim, output_dim)
            self.use_matryoshka = False
        
    def tokenize(
        self,
        texts: Sequence[str],
        device: torch.device | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize texts using ModernBERT tokenizer."""
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        return encoded
    
    def forward(
        self,
        texts: Sequence[str],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        target_dim: int | None = None,
    ) -> torch.Tensor | Dict[int, torch.Tensor]:
        """Encode texts using ModernBERT.
        
        Args:
            texts: Sequence of text strings
            device: Target device
            dtype: Target dtype
            target_dim: For Matryoshka, specify output dimension
        
        Returns:
            Encoded representations
        """
        if len(texts) == 0:
            return torch.empty(
                0,
                self.output_dim if target_dim is None else target_dim,
                device=device,
                dtype=dtype or torch.float32,
            )
        
        # Tokenize
        inputs = self.tokenize(texts, device=device)
        
        # Encode with ModernBERT
        with torch.no_grad() if not self.encoder.training else torch.enable_grad():
            outputs = self.encoder(**inputs)
        
        # Pool
        if self.use_cls_pooling:
            pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        else:
            # Mean pooling with attention mask
            mask = inputs["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        
        # Project
        if hasattr(self, 'use_matryoshka') and self.use_matryoshka:
            result = self.projection(pooled, target_dim=target_dim)
            # Return output_dim tensor by default; dict only when target_dim="all"
            if isinstance(result, dict) and target_dim is None:
                result = result[self.output_dim]
        else:
            result = self.projection(pooled)

        if dtype is not None:
            if isinstance(result, dict):
                result = {k: v.to(dtype=dtype) for k, v in result.items()}
            else:
                result = result.to(dtype=dtype)

        return result


class HybridEncoder(nn.Module):
    """Hybrid encoder: few attention layers + Gated MLP layers.
    
    Combines local attention (transformer-style) for short-range dependencies
    with efficient Gated MLP for long-range context.
    """
    
    def __init__(
        self,
        output_dim: int,
        vocab_size: int = 50265,
        max_length: int = 512,
        d_model: int = 256,
        num_attention_layers: int = 2,
        num_mlp_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        matryoshka_dims: List[int] | None = None,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.d_model = d_model
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.embedding_norm = nn.LayerNorm(d_model)
        
        # Local attention layers
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(num_attention_layers)
        ])
        
        # Global Gated MLP layers
        self.mlp_layers = nn.ModuleList([
            GatedMLPBlock(d_model, d_model * 4, dropout, use_glu=True)
            for _ in range(num_mlp_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Projection
        if matryoshka_dims:
            self.projection = MatryoshkaProjection(d_model, matryoshka_dims)
        else:
            self.projection = nn.Linear(d_model, output_dim)
        
    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        target_dim: int | None = None,
    ) -> torch.Tensor | Dict[int, torch.Tensor]:
        """Encode with hybrid attention + MLP architecture."""
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        # Embed
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.embedding_norm(x)
        x = self.dropout(x)
        
        # Create padding mask for attention (True = ignore)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        
        # Local attention
        for layer in self.attention_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Global Gated MLP
        for layer in self.mlp_layers:
            x = layer(x)
        
        x = self.final_norm(x)
        
        # Pool
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = x.mean(dim=1)
        
        # Project
        if hasattr(self.projection, 'output_dims'):
            return self.projection(pooled, target_dim=target_dim)
        return self.projection(pooled)
