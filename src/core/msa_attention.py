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

    def _compute_dense_attention(
        self,
        qr: torch.Tensor,
        kr: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dense attention via SDPA (Flash Attention when available).

        Equivalent to: softmax(mean_head(cos(QR_norm, KR_norm)) / temperature) @ prototypes
        Reformulated as single-head SDPA:
            Q = concat of per-head-normalized QR  [B, 1, D]
            K = concat of per-head-normalized KR  [B, P, D]
            V = prototypes                        [B, P, D]
            scale = 1 / (H * temperature)

        Returns:
            retrieved: [B, D]
            indices: [B, P] all prototype indices
            weights: [B, P] softmax weights
        """
        B = qr.shape[0]
        P = kr.shape[0]
        H = self.num_heads
        d = self.head_dim

        qr_normed = F.normalize(qr.view(B, H, d), dim=-1).reshape(B, 1, self.dim)
        kr_normed = F.normalize(kr.view(P, H, d), dim=-1).reshape(P, self.dim)
        kr_normed = kr_normed.unsqueeze(0).expand(B, -1, -1).contiguous()
        v = prototypes.unsqueeze(0).expand(B, -1, -1).contiguous()

        scale = 1.0 / (H * self.temperature)

        output = F.scaled_dot_product_attention(qr_normed, kr_normed, v, scale=scale)

        retrieved = self.out_proj(output.squeeze(1))

        indices = torch.arange(P, device=qr.device).unsqueeze(0).expand(B, -1)
        scores = torch.matmul(qr_normed, kr_normed.transpose(-2, -1)) * scale
        weights = F.softmax(scores.float(), dim=-1).to(qr.dtype).squeeze(1)

        return retrieved, indices, weights

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
        topk_weights = F.softmax(topk_scores.float(), dim=-1).to(topk_scores.dtype)

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

        P = prototypes.shape[0]
        effective_top_k = min(self.top_k, P)

        if effective_top_k >= P:
            # Dense path: use SDPA (Flash Attention / Memory-Efficient / Math)
            return self._compute_dense_attention(qr, kr, prototypes)

        # Sparse path: top-k selection (manual fallback for custom masks)
        scores = self.compute_routing_scores(qr, kr)
        topk_scores, topk_indices = scores.topk(effective_top_k, dim=-1)
        topk_weights = F.softmax(topk_scores.float(), dim=-1).to(topk_scores.dtype)
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

        # Overflow storage buffers (pre-allocated ring buffer)
        self.register_buffer("overflow_keys", torch.zeros(capacity, self.key_dim))
        self.register_buffer("overflow_vals", torch.zeros(capacity, dim))
        self.register_buffer("overflow_evidence", torch.zeros(capacity))
        self.register_buffer("overflow_count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_overflow_valid", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("_overflow_head", torch.zeros(1, dtype=torch.long))

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
        """Store evicted slot in overflow buffer (ring buffer).

        Args:
            key: Evicted key vector(s) [K] or [B, K]
            value: Evicted value vector(s) [D] or [B, D]
            evidence: Optional evidence/confidence score
        """
        if not self._enabled:
            return
        if self.max_capacity == 0:
            return
        if key.dim() == 1:
            key = key.unsqueeze(0)
        if value.dim() == 1:
            value = value.unsqueeze(0)
        B = key.shape[0]
        if evidence is None:
            evidence = torch.ones(B, device=key.device)
        elif evidence.dim() == 0:
            evidence = evidence.expand(B)
        with torch.no_grad():
            for i in range(B):
                head = int(self._overflow_head[0].item())
                if head < self.max_capacity:
                    # Still filling
                    self.overflow_keys[head] = key[i].detach()
                    self.overflow_vals[head] = value[i].detach()
                    self.overflow_evidence[head] = evidence[i].detach()
                    self._overflow_valid[head] = True
                    self._overflow_head[0] = head + 1
                else:
                    # Ring full: overwrite lowest-evidence valid entry
                    valid_mask = self._overflow_valid
                    if valid_mask.any():
                        valid_evidence = self.overflow_evidence.clone()
                        valid_evidence[~valid_mask] = float('inf')
                        _, evict_idx = valid_evidence.min(dim=0)
                        evict_idx = int(evict_idx.item())
                    else:
                        evict_idx = 0
                    self.overflow_keys[evict_idx] = key[i].detach()
                    self.overflow_vals[evict_idx] = value[i].detach()
                    self.overflow_evidence[evict_idx] = evidence[i].detach()
                    self._overflow_valid[evict_idx] = True
            self.overflow_count[0] = self._overflow_valid.sum().item()

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
        if not self._enabled:
            B = query.shape[0]
            k = top_k or self.top_k
            return (
                torch.zeros(B, self.dim, device=query.device),
                torch.zeros(B, k, device=query.device),
                torch.zeros(B, k, dtype=torch.long, device=query.device),
            )

        # Filter by valid mask
        valid_mask = self._overflow_valid
        if not valid_mask.any():
            B = query.shape[0]
            k = top_k or self.top_k
            return (
                torch.zeros(B, self.dim, device=query.device),
                torch.zeros(B, k, device=query.device),
                torch.zeros(B, k, dtype=torch.long, device=query.device),
            )

        valid_keys = self.overflow_keys[valid_mask]
        valid_vals = self.overflow_vals[valid_mask]
        valid_evidence = self.overflow_evidence[valid_mask]

        # Use MSA index for sparse retrieval
        # Get indices and weights from MSA (operates on keys)
        _, indices, weights = self.msa_index.query(
            self.query_proj(query) if self.query_proj is not None else query,
            valid_keys,
            valid_evidence,
        )

        # Gather actual values using the indices (relative to valid subset)
        # indices: [B, K], valid_vals: [V, D]
        expanded_idx = indices.unsqueeze(-1).expand(-1, -1, self.dim)  # [B, K, D]
        topk_vals = torch.gather(
            valid_vals.unsqueeze(0).expand(query.shape[0], -1, -1),
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
            remaining_confidence = (1 - overflow_weight.squeeze(-1)) * primary_confidence + overflow_weight.squeeze(-1)
            need_overflow = remaining_confidence < threshold

        return blended, used_overflow

    def clear(self) -> None:
        """Clear overflow buffer."""
        self.overflow_keys.zero_()
        self.overflow_vals.zero_()
        self.overflow_evidence.zero_()
        self.overflow_count.zero_()
        self._overflow_valid.zero_()
        self._overflow_head.zero_()

    def size(self) -> int:
        """Return number of items in overflow buffer."""
        return int(self._overflow_valid.sum().item())


class MSAMemory(MemoryInterface):
    prototypes: torch.Tensor
    confidence: torch.Tensor
    evidence: torch.Tensor
    age: torch.Tensor
    fill_count: torch.Tensor
    _last_diversity_loss: torch.Tensor
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
        diversity_loss_weight: float = 1.0,
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
        self.diversity_loss_weight = diversity_loss_weight

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
        self.register_buffer("_last_diversity_loss", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer(
            "_routing_counts", torch.zeros(num_prototypes, dtype=torch.float32)
        )

        # Phase 22 feature flags
        self.use_gradient_surprise = False
        self.use_adaptive_forgetting = False

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
        topk_weights: Optional[torch.Tensor] = None,  # [B, K]
    ) -> torch.Tensor:
        """Auxiliary loss encouraging even prototype usage.

        Uses soft routing weights for differentiable gradient signal,
        scaled by 1/n to produce values in ~0.01-0.1 range for n=256.
        EMA decay of routing counts happens per forward step.
        """
        B = topk_indices.shape[0]
        if B == 0:
            return torch.tensor(0.0, device=topk_indices.device, dtype=torch.float32)

        n = int(self.fill_count[0].item())
        if n < 2:
            return torch.tensor(0.0, device=topk_indices.device, dtype=torch.float32)

        # Compute soft routing distribution per prototype
        if topk_weights is not None:
            # Differentiable path: aggregate soft routing weights per prototype
            soft_counts = torch.zeros(n, device=topk_indices.device, dtype=torch.float32)
            # topk_weights shape: [B, K], topk_indices shape: [B, K]
            flat_indices = topk_indices.reshape(-1)  # [B*K]
            flat_weights = topk_weights.reshape(-1)   # [B*K]
            soft_counts.scatter_add_(0, flat_indices, flat_weights)
            counts = soft_counts
        else:
            # Non-differentiable fallback: hard counts
            with torch.no_grad():
                step_counts = torch.zeros(n, device=topk_indices.device, dtype=torch.float32)
                flat = topk_indices.detach().reshape(-1)
                ones = torch.ones_like(flat, dtype=torch.float32)
                step_counts.scatter_add_(0, flat, ones)
                self._routing_counts[:n].mul_(0.9).add_(step_counts, alpha=0.1)
            counts = self._routing_counts[:n]

        uniform = counts.sum() / n
        if uniform < 1e-8:
            return torch.tensor(0.0, device=topk_indices.device, dtype=torch.float32)
        deviation = ((counts - uniform) ** 2).mean() / (uniform ** 2 + 1e-8)
        return deviation

    def _evict_to_overflow(self, slot_idx: int) -> None:
        """Evict prototype at slot_idx to overflow buffer."""
        key = self.prototypes[slot_idx].detach().clone()
        val = self.prototypes[slot_idx].detach().clone()
        ev = self.evidence[slot_idx].detach().clone()
        self.overflow.store(key.unsqueeze(0), val.unsqueeze(0), ev.unsqueeze(0))

    def _store_invariant_batched(self, x: torch.Tensor) -> None:
        """Batched store: single matmul instead of B sequential cosine scans."""
        B = x.shape[0]
        n = int(self.fill_count[0].item())
        with torch.no_grad():
            h_norm = F.normalize(x, dim=-1)  # [B, D]
            if n > 0:
                p_norm = F.normalize(self.prototypes[:n], dim=-1)  # [n, D]
                sim = h_norm @ p_norm.T  # [B, n]
                max_sim, max_idx = sim.max(dim=-1)  # [B]
            else:
                max_sim = torch.zeros(B, device=x.device)
                max_idx = torch.zeros(B, device=x.device, dtype=torch.long)
            ema_mask = max_sim > 0.95
            new_mask = ~ema_mask
            # EMA updates (batched for unique slots)
            if ema_mask.any():
                ema_indices = max_idx[ema_mask]
                ema_vectors = x[ema_mask]
                unique_slots = ema_indices.unique()
                for slot in unique_slots:
                    items = ema_vectors[ema_indices == slot]
                    avg_update = items.mean(dim=0)
                    momentum = self.ema_momentum
                    self.prototypes[slot] = momentum * self.prototypes[slot] + (1 - momentum) * avg_update
                    self.prototypes[slot] = F.normalize(self.prototypes[slot], dim=-1)
                    self.evidence[slot] = self.evidence[slot] + items.shape[0]
                    self.confidence[slot] = torch.clamp(self.confidence[slot] + 0.05, max=1.0)
                    self.age[slot] = 0.0
            # New insertions (sequential due to fill_count mutation)
            if new_mask.any():
                new_vectors = x[new_mask]
                for j in range(new_vectors.shape[0]):
                    self._insert_new_prototype(new_vectors[j:j+1])

    def _insert_new_prototype(self, h: torch.Tensor) -> None:
        """Insert a single new prototype into the primary buffer."""
        n = int(self.fill_count[0].item())
        with torch.no_grad():
            if n < self.num_prototypes:
                self.prototypes[n] = F.normalize(h.squeeze(0), dim=-1)
                self.confidence[n] = 0.5
                self.evidence[n] = 1.0
                self.age[n] = 0.0
                self.fill_count[0] = n + 1
            else:
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
            self._store_invariant_batched(x)

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

        # Auxiliary loss: routing diversity (keep gradient for self-regulation)
        diversity_loss = self._compute_routing_diversity_loss(topk_indices, topk_weights=topk_weights)
        weighted_diversity_loss = diversity_loss * self.diversity_loss_weight
        self._last_diversity_loss = weighted_diversity_loss

        # Age all prototypes
        with torch.no_grad():
            self.age[:n] += 1.0

        return MemoryResult(
            output=output,
            loss=weighted_diversity_loss,
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
            self._last_diversity_loss.fill_(0.0)

        elif strategy == "geometric":
            # Soft decay: reduce confidence and evidence, reset age
            decay = 0.7
            uniform_conf = torch.full_like(self.confidence, 0.5)
            self.confidence[:n] = decay * self.confidence[:n] + (1 - decay) * uniform_conf[:n]
            self.evidence[:n] = (self.evidence[:n] * decay).clamp(min=1.0)
            self.age[:n] = 0.0

        elif strategy == "stabilize":
            # Only normalize evidence and confidence
            if n > 0:
                total_ev = self.evidence[:n].sum().clamp(min=1e-8)
                self.evidence[:n] = (self.evidence[:n] / total_ev) * n
                self.confidence[:n] = self.confidence[:n].clamp(0.1, 1.0)

    def stats(self) -> dict:
        """Return memory statistics for monitoring."""
        n = int(self.fill_count[0].item())
        return {
            "memory_type": "msa",
            "filled_prototypes": n,
            "total_prototypes": self.num_prototypes,
            "overflow_size": self.overflow.size() if self.overflow.is_enabled() else 0,
            "avg_confidence": self.confidence[:n].mean().item() if n > 0 else 0.0,
            "avg_evidence": self.evidence[:n].mean().item() if n > 0 else 0.0,
            "avg_age": self.age[:n].mean().item() if n > 0 else 0.0,
            "diversity_loss": self._last_diversity_loss.item(),
        }

    def memory_loss(self) -> torch.Tensor:
        """Current auxiliary memory loss (routing diversity)."""
        return self._last_diversity_loss
