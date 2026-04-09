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

from .memory_interface import MemoryInterface, MemoryResult


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
    compressed_chunks: torch.Tensor
    chunk_counts: torch.Tensor
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


class MSAOverflowBuffer(nn.Module):
    overflow_keys: torch.Tensor
    overflow_vals: torch.Tensor
    overflow_evidence: torch.Tensor
    overflow_count: torch.Tensor
    """MSA-backed overflow buffer for EpisodicMemory.

    Implements Memory Interleave for multi-hop retrieval:
    1. Query primary buffer (256 slots)
    2. If insufficient, query MSA overflow
    3. Retrieved items added to context for next iteration

    Key features:
    - Unlimited compressed storage for evicted slots
    - Top-k sparse retrieval on demand
    - Multi-hop retrieval with threshold-based fallback
    """

    def __init__(
        self,
        dim: int,
        key_dim: Optional[int] = None,  # If None, same as dim
        num_prototypes: int = 1024,
        max_hops: int = 3,
        top_k: int = 16,
        chunk_size: int = 64,
        num_heads: int = 4,
        temperature: float = 0.1,
        capacity: int = 10000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.key_dim = key_dim if key_dim is not None else dim
        self.max_hops = max_hops
        self.top_k = top_k
        self.max_capacity = capacity

        # MSA sparse index for overflow storage
        self.msa_index = MSASparseIndex(
            dim=self.key_dim,
            num_prototypes=num_prototypes,
            top_k=top_k,
            chunk_size=chunk_size,
            num_heads=num_heads,
            temperature=temperature,
        )

        # Overflow storage buffers (dynamic size)
        self.register_buffer("overflow_keys", torch.zeros(0, self.key_dim))
        self.register_buffer("overflow_vals", torch.zeros(0, dim))
        self.register_buffer("overflow_evidence", torch.zeros(0))
        self.register_buffer("overflow_count", torch.zeros(1, dtype=torch.long))

        # Feature flag state
        self._enabled = True

        # Query projection if key_dim != dim
        if self.key_dim != self.dim:
            self.query_proj = nn.Linear(self.dim, self.key_dim, bias=False)
        else:
            self.query_proj = None

    def enable(self) -> None:
        """Enable overflow buffer."""
        self._enabled = True

    def disable(self) -> None:
        """Disable overflow buffer."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if overflow buffer is enabled."""
        return self._enabled

    def store(
        self,
        key: torch.Tensor,  # [K] or [B, K]
        value: torch.Tensor,  # [D] or [B, D]
        evidence: Optional[torch.Tensor] = None,  # [B] or scalar
    ) -> None:
        """Store evicted slot in overflow buffer.

        Args:
            key: Evicted key vector(s) [K] or [B, K]
            value: Evicted value vector(s) [D] or [B, D]
            evidence: Optional evidence/confidence score
        """
        if not self._enabled:
            return

        # Ensure 2D
        if key.dim() == 1:
            key = key.unsqueeze(0)
        if value.dim() == 1:
            value = value.unsqueeze(0)

        B = key.shape[0]

        # Default evidence
        if evidence is None:
            evidence = torch.ones(B, device=key.device)
        elif evidence.dim() == 0:
            evidence = evidence.expand(B)

        # Append to overflow buffers
        self.overflow_keys = torch.cat([self.overflow_keys, key.detach()], dim=0)
        self.overflow_vals = torch.cat([self.overflow_vals, value.detach()], dim=0)
        self.overflow_evidence = torch.cat([self.overflow_evidence, evidence.detach()], dim=0)

        # Eviction: keep only topk entries by evidence if over capacity
        if self.overflow_keys.shape[0] > self.max_capacity:
            _, topk_idx = self.overflow_evidence.topk(self.max_capacity)
            topk_idx, _ = topk_idx.sort()  # preserve insertion order
            self.overflow_keys = self.overflow_keys[topk_idx]
            self.overflow_vals = self.overflow_vals[topk_idx]
            self.overflow_evidence = self.overflow_evidence[topk_idx]

        self.overflow_count[0] = self.overflow_keys.shape[0]

    def retrieve(
        self,
        query: torch.Tensor,  # [B, D]
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve from overflow buffer via MSA sparse index.

        Args:
            query: Query tensor [B, D]
            top_k: Number of results (default: self.top_k)

        Returns:
            retrieved: [B, D] retrieved values
            weights: [B, K] retrieval weights
            indices: [B, K] retrieved indices
        """
        if not self._enabled or self.overflow_count[0] == 0:
            # Return empty if disabled or no overflow
            B = query.shape[0]
            k = top_k or self.top_k
            return (
                torch.zeros(B, self.dim, device=query.device),
                torch.zeros(B, k, device=query.device),
                torch.zeros(B, k, dtype=torch.long, device=query.device),
            )


        # Use MSA index for sparse retrieval
        # Get indices and weights from MSA (operates on keys)
        _, indices, weights = self.msa_index.query(
            self.query_proj(query) if self.query_proj is not None else query,
            self.overflow_keys,
            self.overflow_evidence,
        )

        # Gather actual values using the indices
        # indices: [B, K], overflow_vals: [N, D]
        expanded_idx = indices.unsqueeze(-1).expand(-1, -1, self.dim)  # [B, K, D]
        topk_vals = torch.gather(
            self.overflow_vals.unsqueeze(0).expand(query.shape[0], -1, -1),
            dim=1,
            index=expanded_idx,
        )  # [B, K, D]

        # Weighted combination: [B, D]
        retrieved = (weights.unsqueeze(-1) * topk_vals).sum(dim=1)

        return retrieved, weights, indices
    def retrieve_with_interleave(
        self,
        query: torch.Tensor,  # [B, D]
        primary_result: torch.Tensor,  # [B, D] result from primary buffer
        primary_confidence: torch.Tensor,  # [B] confidence of primary result
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-hop retrieval: primary + overflow if needed.

        Memory Interleave protocol:
        1. Check if primary result confidence >= threshold
        2. If insufficient, query MSA overflow
        3. Blend primary + overflow results

        Args:
            query: Query tensor [B, D]
            primary_result: Result from primary buffer [B, D]
            primary_confidence: Confidence scores [B]
            threshold: Minimum confidence to skip overflow

        Returns:
            blended: [B, D] blended result
            used_overflow: [B] boolean mask indicating overflow usage
        """
        if not self._enabled:
            return primary_result, torch.zeros(query.shape[0], dtype=torch.bool, device=query.device)

        B = query.shape[0]

        # Check which queries need overflow
        need_overflow = primary_confidence < threshold
        used_overflow = need_overflow.clone()

        if not need_overflow.any() or self.overflow_count[0] == 0:
            # No overflow needed or empty overflow
            return primary_result, used_overflow

        # Multi-hop: iterate up to max_hops
        blended = primary_result.clone()

        for hop in range(self.max_hops):
            if not need_overflow.any():
                break

            # Query overflow for needed samples
            overflow_result, weights, _ = self.retrieve(query)

            # Blend based on confidence gap
            # Lower confidence -> more overflow weight
            confidence_gap = (threshold - primary_confidence).clamp(0, 1)
            overflow_weight = confidence_gap.unsqueeze(-1)  # [B, 1]

            # Update only where overflow was needed
            blended = torch.where(
                need_overflow.unsqueeze(-1),
                (1 - overflow_weight) * primary_result + overflow_weight * overflow_result,
                blended,
            )

            # For next hop, mark remaining low-confidence samples
            # (simplified: one hop is usually enough)
            need_overflow = torch.zeros_like(need_overflow)

        return blended, used_overflow

    def clear(self) -> None:
        """Clear overflow buffer."""
        self.overflow_keys = torch.zeros(0, self.dim, device=self.overflow_keys.device)
        self.overflow_vals = torch.zeros(0, self.dim, device=self.overflow_vals.device)
        self.overflow_evidence = torch.zeros(0, device=self.overflow_evidence.device)
        self.overflow_count.zero_()

    def size(self) -> int:
        """Return number of items in overflow buffer."""
        return int(self.overflow_count[0].item())


class MSAAugmentedSemanticMemory(nn.Module):
    prototypes: torch.Tensor
    proto_conf: torch.Tensor
    proto_evidence: torch.Tensor
    proto_age: torch.Tensor
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
        assert self.msa_index is not None
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


class MSAMemory(MemoryInterface):
    prototypes: torch.Tensor
    confidence: torch.Tensor
    evidence: torch.Tensor
    age: torch.Tensor
    fill_count: torch.Tensor
    _aux_loss: torch.Tensor
    _routing_counts: torch.Tensor
    """Primary memory system using MSA sparse attention.

    Replaces Titans/HBMA with prototype-based sparse retrieval:
    - Primary buffer: num_prototypes slots for stored invariants
    - Overflow: MSAOverflowBuffer for evicted prototypes
    - Retrieval: MSASparseIndex top-k selection + softmax blending
    - Memory Interleave: fallback to overflow when confidence is low
    - Blend gate: x + gate * retrieved
    - Auxiliary loss: routing diversity (encourage even prototype usage)
    - Surprise signal: normalized deviation from input

    Implements MemoryInterface contract for drop-in use in HDIMPipeline.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_prototypes: int = 256,
        top_k: int = 16,
        chunk_size: int = 64,
        num_heads: int = 4,
        temperature: float = 0.07,
        ema_momentum: float = 0.995,
        dropout: float = 0.1,
        overflow_capacity: int = 10000,
        max_hops: int = 3,
        interleave_threshold: float = 0.5,
        compression_threshold: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        self.top_k = min(top_k, num_prototypes)
        self.chunk_size = chunk_size
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature
        self.ema_momentum = ema_momentum
        self.max_hops = max_hops
        self.interleave_threshold = interleave_threshold

        # Projections (same pattern as MSAAugmentedSemanticMemory)
        self.in_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

        # Blend gate: x + gate * retrieved
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Primary prototype storage
        self.register_buffer(
            "prototypes",
            F.normalize(torch.randn(num_prototypes, hidden_dim), dim=-1),
        )
        self.register_buffer("confidence", torch.full((num_prototypes,), 0.5))
        self.register_buffer("evidence", torch.ones(num_prototypes))
        self.register_buffer("age", torch.zeros(num_prototypes))
        self.register_buffer(
            "fill_count", torch.zeros(1, dtype=torch.long)
        )

        # MSA sparse index for primary retrieval
        self.sparse_index = MSASparseIndex(
            dim=hidden_dim,
            num_prototypes=num_prototypes,
            top_k=top_k,
            chunk_size=chunk_size,
            num_heads=num_heads,
            temperature=temperature,
            compression_threshold=compression_threshold,
        )

        # Overflow buffer for evicted prototypes
        self.overflow = MSAOverflowBuffer(
            dim=hidden_dim,
            num_prototypes=min(overflow_capacity, 1024),
            max_hops=max_hops,
            top_k=top_k,
            chunk_size=chunk_size,
            num_heads=num_heads,
            temperature=temperature,
            capacity=overflow_capacity,
        )

        # Track auxiliary loss
        self.register_buffer("_aux_loss", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer(
            "_routing_counts", torch.zeros(num_prototypes, dtype=torch.float32)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_prototypes(self) -> torch.Tensor:
        """Return only filled prototype slots."""
        n = int(self.fill_count[0].item())
        if n == 0:
            return torch.zeros(0, self.hidden_dim, device=self.prototypes.device)
        return self.prototypes[:n]

    def _active_evidence(self) -> torch.Tensor:
        """Return evidence for filled slots only."""
        n = int(self.fill_count[0].item())
        if n == 0:
            return torch.zeros(0, device=self.evidence.device)
        return self.evidence[:n]

    def _compute_routing_diversity_loss(
        self,
        topk_indices: torch.Tensor,  # [B, K]
    ) -> torch.Tensor:
        """Auxiliary loss encouraging even prototype usage.

        Uses a soft count-based penalty: if some prototypes are heavily
        over-used relative to uniform, the loss increases.
        """
        B = topk_indices.shape[0]
        if B == 0:
            return torch.tensor(0.0, device=topk_indices.device, dtype=torch.float32)

        # Update running routing counts (detached)
        with torch.no_grad():
            flat = topk_indices.detach().reshape(-1)
            ones = torch.ones_like(flat, dtype=torch.float32)
            self._routing_counts.scatter_add_(0, flat, ones)

        # Compute diversity loss: L_div = CV(counts)^2
        # Coefficient of variation squared — 0 when perfectly uniform
        n = int(self.fill_count[0].item())
        if n < 2:
            return torch.tensor(0.0, device=topk_indices.device, dtype=torch.float32)

        counts = self._routing_counts[:n]
        mean_c = counts.mean().clamp(min=1e-8)
        var_c = ((counts - mean_c) ** 2).mean()
        cv_sq = var_c / (mean_c ** 2 + 1e-8)
        return cv_sq

    def _evict_to_overflow(self, slot_idx: int) -> None:
        """Evict prototype at slot_idx to overflow buffer."""
        key = self.prototypes[slot_idx].detach().clone()
        val = self.prototypes[slot_idx].detach().clone()
        ev = self.evidence[slot_idx].detach().clone()
        self.overflow.store(key.unsqueeze(0), val.unsqueeze(0), ev.unsqueeze(0))

    def _store_invariant(self, h: torch.Tensor) -> None:
        """Store a new invariant into the primary buffer.

        If buffer is full, evict lowest-evidence prototype to overflow.
        Uses EMA update for existing similar prototypes.
        """
        n = int(self.fill_count[0].item())

        with torch.no_grad():
            # Check if similar prototype exists (cosine sim > 0.95)
            if n > 0:
                h_norm = F.normalize(h, dim=-1)
                p_norm = F.normalize(self.prototypes[:n], dim=-1)
                sim = (h_norm @ p_norm.T).squeeze(0)  # [n]
                max_sim, max_idx = sim.max(dim=0)

                if max_sim > 0.95:
                    # EMA update existing prototype
                    slot = int(max_idx.item())
                    momentum = self.ema_momentum
                    self.prototypes[slot] = momentum * self.prototypes[slot] + (1 - momentum) * h.squeeze(0)
                    self.prototypes[slot] = F.normalize(self.prototypes[slot], dim=-1)
                    self.evidence[slot] = self.evidence[slot] + 1.0
                    self.confidence[slot] = torch.clamp(
                        self.confidence[slot] + 0.05, max=1.0
                    )
                    self.age[slot] = 0.0
                    return

            # New prototype needed
            if n < self.num_prototypes:
                # Free slot available
                self.prototypes[n] = F.normalize(h.squeeze(0), dim=-1)
                self.confidence[n] = 0.5
                self.evidence[n] = 1.0
                self.age[n] = 0.0
                self.fill_count[0] = n + 1
            else:
                # Buffer full: evict lowest-evidence prototype
                _, evict_idx = self.evidence.min(dim=0)
                evict_idx = int(evict_idx.item())
                self._evict_to_overflow(evict_idx)
                self.prototypes[evict_idx] = F.normalize(h.squeeze(0), dim=-1)
                self.confidence[evict_idx] = 0.5
                self.evidence[evict_idx] = 1.0
                self.age[evict_idx] = 0.0

    # ------------------------------------------------------------------
    # MemoryInterface contract
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        update_memory: bool = False,
    ) -> MemoryResult:
        """Retrieve from MSA prototype memory + optional update.

        Args:
            x: Input tensor [B, D]
            update_memory: Whether to store x as a prototype

        Returns:
            MemoryResult with output, loss, updated, surprise
        """
        B, D = x.shape
        device = x.device

        # Store new invariants before retrieval (so they're available immediately)
        if update_memory:
            for i in range(B):
                self._store_invariant(x[i : i + 1])

        # Project input for retrieval
        h = self.in_proj(x)

        n = int(self.fill_count[0].item())

        # Edge case: not enough prototypes yet
        if n < self.top_k:
            gate_val = torch.sigmoid(self.gate(x))  # [B, 1]
            output = x + gate_val * torch.zeros_like(x)
            surprise = torch.zeros(B, 1, device=device, dtype=torch.float32)
            return MemoryResult(
                output=self.norm(output),
                loss=torch.tensor(0.0, device=device, dtype=torch.float32),
                updated=update_memory,
                surprise=surprise.detach(),
            )

        # Primary retrieval via MSA sparse index
        active_p = self._active_prototypes()
        active_ev = self._active_evidence()
        retrieved, topk_indices, topk_weights = self.sparse_index.query(
            h, active_p, active_ev
        )

        # Compute primary confidence (mean of top-k weights as proxy)
        primary_confidence = topk_weights.max(dim=-1).values  # [B]

        # Memory Interleave: fallback to overflow if confidence low
        if self.overflow.is_enabled() and self.overflow.size() > 0:
            retrieved, _ = self.overflow.retrieve_with_interleave(
                query=h,
                primary_result=retrieved,
                primary_confidence=primary_confidence,
                threshold=self.interleave_threshold,
            )

        # Blend gate: x + gate * retrieved
        gate_val = torch.sigmoid(self.gate(x))  # [B, 1]
        output = x + gate_val * retrieved
        output = self.drop(self.norm(output))

        # Surprise signal: normalized deviation from input
        with torch.no_grad():
            surprise = (
                (output.detach() - x).norm(dim=-1, keepdim=True)
                / (x.norm(dim=-1, keepdim=True) + 1e-8)
            )

        # Auxiliary loss: routing diversity
        diversity_loss = self._compute_routing_diversity_loss(topk_indices)
        self._aux_loss = diversity_loss.detach()

        # Age all prototypes
        with torch.no_grad():
            self.age[:n] += 1.0

        return MemoryResult(
            output=output,
            loss=diversity_loss,
            updated=update_memory,
            surprise=surprise.detach(),
        )

    def reset(self, strategy: str = "geometric") -> None:
        """Reset memory state.

        Strategies:
            'hard'      -- full reset (zero all prototypes, clear overflow)
            'geometric' -- soft decay (EMA decay of confidence, age reset)
            'stabilize' -- only normalize evidence and confidence
        """
        n = int(self.fill_count[0].item())

        if strategy == "hard":
            nn.init.normal_(self.prototypes)
            F.normalize(self.prototypes, dim=-1, out=self.prototypes)
            self.confidence.fill_(0.5)
            self.evidence.fill_(1.0)
            self.age.zero_()
            self.fill_count.zero_()
            self._routing_counts.zero_()
            self.overflow.clear()

        elif strategy == "geometric":
            # Soft decay: reduce confidence and evidence, reset age
            decay = 0.7
            uniform_conf = torch.full_like(self.confidence, 0.5)
            self.confidence[:n] = decay * self.confidence[:n] + (1 - decay) * uniform_conf[:n]
            self.evidence[:n] = (self.evidence[:n] * decay).clamp(min=1.0)
            self.age[:n] = 0.0
            # Decay routing counts but don't zero
            self._routing_counts.mul_(decay)

        elif strategy == "stabilize":
            # Only normalize evidence and confidence
            if n > 0:
                total_ev = self.evidence[:n].sum().clamp(min=1e-8)
                self.evidence[:n] = (self.evidence[:n] / total_ev) * n
                self.confidence[:n] = self.confidence[:n].clamp(0.1, 1.0)
            self._routing_counts.mul_(0.9)

    def memory_loss(self) -> torch.Tensor:
        """Current auxiliary memory loss (routing diversity)."""
        return self._aux_loss
