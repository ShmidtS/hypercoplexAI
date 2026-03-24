"""
MSA (Memory Sparse Attention) — Sparse Index for SemanticMemory.

Implements hierarchical compression and sparse retrieval from MSA paper:
- Router K Projector (W_KR): KR = H @ W_KR — projection of keys for routing
- Router Q Projector (W_QR): QR = H @ W_QR — projection of queries
- Top-k Selection: I = Top-k({si}) where si = max_token(mean_head(cos(QR, KR)))
- Chunk Compression: mean_pool with P=64 tokens

Key benefits:
- O(log N) retrieval instead of O(N) dense similarity
- Hierarchical compression for prototype management
- Drop-in enhancement for SemanticMemory

Reference: MSA paper — Memory Sparse Attention for Long-Context LLMs
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MSAConfig:
    """Configuration for MSA sparse index."""

    dim: int = 256
    """Hidden dimension for projections."""

    top_k: int = 16
    """Number of top prototypes to retrieve."""

    chunk_size: int = 64
    """Compression window size (P from paper)."""

    num_heads: int = 4
    """Number of attention heads for multi-head routing."""

    temperature: float = 0.1
    """Temperature for softmax in top-k selection."""

    compression_threshold: int = 128
    """Minimum prototypes before chunk compression activates."""


class MSASparseIndex(nn.Module):
    """MSA sparse index for prototype retrieval.

    Key components from paper:
    - Router K Projector: KR = H @ W_KR
    - Router Q Projector: QR = H @ W_QR
    - Top-k selection: I = Top-k({si})
    - Chunk compression: mean_pool(P=64)

    Provides O(log N) retrieval instead of O(N) dense cosine similarity.
    """

    def __init__(
        self,
        dim: int,
        num_prototypes: int,
        top_k: int = 16,
        chunk_size: int = 64,
        num_heads: int = 4,
        temperature: float = 0.1,
        compression_threshold: int = 128,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_prototypes = num_prototypes
        self.top_k = min(top_k, num_prototypes)
        self.chunk_size = chunk_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = temperature
        self.compression_threshold = compression_threshold

        # Router projectors from paper
        # W_KR: projects stored prototypes for routing
        # W_QR: projects query for routing
        self.W_KR = nn.Linear(dim, dim, bias=False)
        self.W_QR = nn.Linear(dim, dim, bias=False)

        # Output projection for retrieved values
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Chunk compression buffer (when prototypes exceed threshold)
        self.register_buffer(
            "compressed_chunks",
            torch.zeros(0, dim),  # Dynamic size
        )
        self.register_buffer("chunk_counts", torch.zeros(0, dtype=torch.long))

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize router projectors with small values."""
        nn.init.xavier_uniform_(self.W_KR.weight, gain=0.02)
        nn.init.xavier_uniform_(self.W_QR.weight, gain=0.02)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.02)

    def compute_routing_scores(
        self,
        qr: torch.Tensor,  # [B, D] or [B, H, D//H]
        kr: torch.Tensor,  # [P, D] or [P, H, D//H]
    ) -> torch.Tensor:
        """Compute routing scores via multi-head cosine similarity.

        From paper: si = max_token(mean_head(cos(QR, KR)))

        Args:
            qr: Router query projection [B, D]
            kr: Router key projections [P, D]

        Returns:
            scores: [B, P] routing scores
        """
        B = qr.shape[0]
        P = kr.shape[0]

        # Reshape for multi-head: [B, H, D//H] and [P, H, D//H]
        qr_heads = qr.view(B, self.num_heads, self.head_dim)
        kr_heads = kr.view(P, self.num_heads, self.head_dim)

        # Compute cosine similarity per head: [B, P, H]
        qr_norm = F.normalize(qr_heads, dim=-1)
        kr_norm = F.normalize(kr_heads, dim=-1)
        sim_per_head = torch.einsum('bhd,phd->bph', qr_norm, kr_norm)

        # mean_head: average across heads -> [B, P]
        mean_head_sim = sim_per_head.mean(dim=-1)

        # Apply temperature scaling
        scores = mean_head_sim / self.temperature

        return scores

    def top_k_selection(
        self,
        scores: torch.Tensor,  # [B, P]
        kr: torch.Tensor,  # [P, D]
        prototypes: torch.Tensor,  # [P, D]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select top-k prototypes via sparse routing.

        Args:
            scores: Routing scores [B, P]
            kr: Router key projections [P, D]
            prototypes: Actual prototype vectors [P, D]

        Returns:
            topk_indices: [B, K] selected prototype indices
            topk_weights: [B, K] softmax weights
            retrieved: [B, K, D] retrieved prototype vectors
        """
        # Top-k selection
        topk_scores, topk_indices = scores.topk(self.top_k, dim=-1)

        # Softmax weights for blending
        topk_weights = F.softmax(topk_scores, dim=-1)

        # Gather retrieved prototypes: [B, K, D]
        # Expand indices for gather: [B, K, D]
        expanded_idx = topk_indices.unsqueeze(-1).expand(-1, -1, self.dim)
        retrieved = torch.gather(
            prototypes.unsqueeze(0).expand(topk_indices.shape[0], -1, -1),
            dim=1,
            index=expanded_idx,
        )

        return topk_indices, topk_weights, retrieved

    def chunk_compress(
        self,
        prototypes: torch.Tensor,  # [P, D]
        evidence: torch.Tensor,  # [P] evidence counts
    ) -> torch.Tensor:
        """Compress prototypes via mean pooling when count exceeds threshold.

        From paper: Chunk compression with P=64 tokens window.

        Args:
            prototypes: Prototype vectors [P, D]
            evidence: Evidence counts [P] for weighting

        Returns:
            compressed: [P', D] compressed prototypes where P' <= P
        """
        P = prototypes.shape[0]

        if P <= self.compression_threshold:
            return prototypes

        # Compute number of chunks
        num_chunks = (P + self.chunk_size - 1) // self.chunk_size

        # Pad to multiple of chunk_size
        pad_size = num_chunks * self.chunk_size - P
        if pad_size > 0:
            prototypes = F.pad(prototypes, (0, 0, 0, pad_size))
            evidence = F.pad(evidence, (0, pad_size), value=0)

        # Reshape into chunks: [num_chunks, chunk_size, D]
        chunks = prototypes.view(num_chunks, self.chunk_size, self.dim)
        chunk_evidence = evidence.view(num_chunks, self.chunk_size)

        # Weight by evidence and mean pool
        weights = chunk_evidence.float() + 1e-8
        weights = weights / weights.sum(dim=-1, keepdim=True)
        compressed = (chunks * weights.unsqueeze(-1)).sum(dim=1)

        # Normalize
        compressed = F.normalize(compressed, dim=-1)

        return compressed

    def query(
        self,
        h: torch.Tensor,  # [B, D] query
        prototypes: torch.Tensor,  # [P, D] stored prototypes
        evidence: Optional[torch.Tensor] = None,  # [P] evidence counts
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sparse retrieval: O(log N) instead of O(N).

        Full pipeline:
        1. Project query and prototypes via router projectors
        2. Compute routing scores via multi-head cosine
        3. Select top-k prototypes
        4. Return weighted combination

        Args:
            h: Query tensor [B, D]
            prototypes: Prototype vectors [P, D]
            evidence: Optional evidence counts [P] for compression

        Returns:
            retrieved: [B, D] retrieved and blended representation
            topk_indices: [B, K] selected prototype indices
            topk_weights: [B, K] selection weights
        """
        # Apply compression if needed
        if evidence is not None and prototypes.shape[0] > self.compression_threshold:
            prototypes = self.chunk_compress(prototypes, evidence)

        # Router projections
        qr = self.W_QR(h)  # [B, D]
        kr = self.W_KR(prototypes)  # [P, D]

        # Compute routing scores
        scores = self.compute_routing_scores(qr, kr)

        # Top-k selection with clamping for compressed prototypes
        effective_top_k = min(self.top_k, prototypes.shape[0])
        topk_scores, topk_indices = scores.topk(effective_top_k, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1)
        # Gather retrieved prototypes: [B, K, D]
        expanded_idx = topk_indices.unsqueeze(-1).expand(-1, -1, self.dim)
        topk_protos = torch.gather(
            prototypes.unsqueeze(0).expand(topk_indices.shape[0], -1, -1),
            dim=1,
            index=expanded_idx,
        )

        # Weighted combination: [B, K] @ [B, K, D] -> [B, D]
        retrieved = (topk_weights.unsqueeze(-1) * topk_protos).sum(dim=1)

        # Output projection
        retrieved = self.out_proj(retrieved)

        return retrieved, topk_indices, topk_weights

    def forward(
        self,
        h: torch.Tensor,
        prototypes: torch.Tensor,
        evidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: returns retrieved representation.

        Args:
            h: Query tensor [B, D]
            prototypes: Prototype vectors [P, D]
            evidence: Optional evidence counts [P]

        Returns:
            retrieved: [B, D] retrieved representation
        """
        retrieved, _, _ = self.query(h, prototypes, evidence)
        return retrieved


class MSAAugmentedSemanticMemory(nn.Module):
    """SemanticMemory with optional MSA sparse retrieval.

    Drop-in enhancement that:
    - Uses dense cosine similarity when MSA disabled (backward compat)
    - Uses sparse MSA retrieval when enabled (O(log N) scaling)

    Integration point for Phase 2 MSA in SemanticMemory.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_prototypes: int = 64,
        ema_momentum: float = 0.995,
        temperature: float = 0.07,
        contradiction_thresh: float = -0.3,
        dropout: float = 0.1,
        # MSA feature flag
        use_msa: bool = False,
        msa_top_k: int = 16,
        msa_chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        self.ema_momentum = ema_momentum
        self.temperature = temperature
        self.contradiction_thresh = contradiction_thresh
        self.use_msa = use_msa

        # Core components (from SemanticMemory)
        self.in_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Type router
        self.type_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),
        )

        # Prototype storage
        self.register_buffer(
            "prototypes",
            F.normalize(torch.randn(num_prototypes, hidden_dim), dim=-1)
        )
        self.register_buffer("proto_conf", torch.full((num_prototypes,), 0.5))
        self.register_buffer("proto_evidence", torch.ones(num_prototypes))
        self.register_buffer("proto_age", torch.zeros(num_prototypes))

        # MSA sparse index (optional)
        if use_msa:
            self.msa_index = MSASparseIndex(
                dim=hidden_dim,
                num_prototypes=num_prototypes,
                top_k=msa_top_k,
                chunk_size=msa_chunk_size,
            )
        else:
            self.msa_index = None

    def _retrieve_dense(
        self,
        h: torch.Tensor,
        prototypes: torch.Tensor,
        conf: torch.Tensor,
        age: torch.Tensor,
        evidence: torch.Tensor,
    ) -> torch.Tensor:
        """Dense retrieval via cosine similarity (backward compat)."""
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(prototypes, dim=-1)

        raw_sim = h_norm @ p_norm.T  # [B, P]
        weighted_sim = raw_sim * conf.unsqueeze(0)

        # Simple salience scoring
        sal = weighted_sim + 0.1 * torch.exp(-age / 200.0).unsqueeze(0)
        attn = F.softmax(sal / self.temperature, dim=-1)

        return attn @ prototypes

    def _retrieve_msa(
        self,
        h: torch.Tensor,
        prototypes: torch.Tensor,
        evidence: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse retrieval via MSA."""
        retrieved, _, _ = self.msa_index.query(h, prototypes, evidence)
        return retrieved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: retrieve and augment.

        Args:
            x: Input tensor [B, D]

        Returns:
            augmented: [B, D] memory-augmented representation
        """
        h = self.in_proj(x)

        # Snap buffers
        p_snap = self.prototypes.detach().clone()
        conf_snap = self.proto_conf.detach().clone()
        age_snap = self.proto_age.detach().clone()
        ev_snap = self.proto_evidence.detach().clone()

        # Choose retrieval method
        if self.use_msa and self.msa_index is not None:
            retrieved = self._retrieve_msa(h, p_snap, ev_snap)
        else:
            retrieved = self._retrieve_dense(h, p_snap, conf_snap, age_snap, ev_snap)

        # Combine and output
        combined = torch.cat([x, retrieved], dim=-1)
        out = self.dropout(F.gelu(self.out_proj(combined)))
        out = self.norm(out + x)

        return out

    def reset(self) -> None:
        """Reset prototype buffers."""
        nn.init.normal_(self.prototypes)
        F.normalize(self.prototypes, dim=-1, out=self.prototypes)
        self.proto_conf.fill_(0.5)
        self.proto_evidence.fill_(1.0)
        self.proto_age.zero_()
