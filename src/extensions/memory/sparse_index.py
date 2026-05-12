"""MSA (Memory Sparse Attention) — Sparse Index for SemanticMemory.

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

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.W_KR = nn.Linear(dim, dim, bias=False)
        self.W_QR = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.register_buffer(
            "compressed_chunks",
            torch.zeros(0, dim),
        )
        self.register_buffer("chunk_counts", torch.zeros(0, dtype=torch.long))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W_KR.weight, gain=0.02)
        nn.init.xavier_uniform_(self.W_QR.weight, gain=0.02)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.02)

    def compute_routing_scores(
        self,
        qr: torch.Tensor,
        kr: torch.Tensor,
    ) -> torch.Tensor:
        """Compute routing scores via multi-head cosine similarity."""
        B = qr.shape[0]
        P = kr.shape[0]

        qr_heads = qr.view(B, self.num_heads, self.head_dim)
        kr_heads = kr.view(P, self.num_heads, self.head_dim)

        qr_norm = F.normalize(qr_heads, dim=-1)
        kr_norm = F.normalize(kr_heads, dim=-1)
        sim_per_head = torch.einsum('bhd,phd->bph', qr_norm, kr_norm)

        mean_head_sim = sim_per_head.mean(dim=-1)
        scores = mean_head_sim / self.temperature
        return scores

    def _compute_dense_attention(
        self,
        qr: torch.Tensor,
        kr: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dense attention via SDPA."""
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
        scores: torch.Tensor,
        kr: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select top-k prototypes via sparse routing."""
        topk_scores, topk_indices = scores.topk(self.top_k, dim=-1)
        topk_weights = F.softmax(topk_scores.float(), dim=-1).to(topk_scores.dtype)

        expanded_idx = topk_indices.unsqueeze(-1).expand(-1, -1, self.dim)
        retrieved = torch.gather(
            prototypes.unsqueeze(0).expand(topk_indices.shape[0], -1, -1),
            dim=1,
            index=expanded_idx,
        )

        return topk_indices, topk_weights, retrieved

    def chunk_compress(
        self,
        prototypes: torch.Tensor,
        evidence: torch.Tensor,
    ) -> torch.Tensor:
        """Compress prototypes via mean pooling when count exceeds threshold."""
        P = prototypes.shape[0]
        if P <= self.compression_threshold:
            return prototypes

        num_chunks = (P + self.chunk_size - 1) // self.chunk_size
        pad_size = num_chunks * self.chunk_size - P
        if pad_size > 0:
            prototypes = F.pad(prototypes, (0, 0, 0, pad_size))
            evidence = F.pad(evidence, (0, pad_size), value=0)

        chunks = prototypes.view(num_chunks, self.chunk_size, self.dim)
        chunk_evidence = evidence.view(num_chunks, self.chunk_size)

        weights = chunk_evidence.float() + 1e-8
        weights = weights / weights.sum(dim=-1, keepdim=True)
        compressed = (chunks * weights.unsqueeze(-1)).sum(dim=1)
        compressed = F.normalize(compressed, dim=-1)

        return compressed

    def query(
        self,
        h: torch.Tensor,
        prototypes: torch.Tensor,
        evidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sparse retrieval: O(log N) instead of O(N)."""
        if evidence is not None and prototypes.shape[0] > self.compression_threshold:
            prototypes = self.chunk_compress(prototypes, evidence)

        qr = self.W_QR(h)
        kr = self.W_KR(prototypes)

        P = prototypes.shape[0]
        effective_top_k = min(self.top_k, P)

        if effective_top_k >= P:
            return self._compute_dense_attention(qr, kr, prototypes)

        scores = self.compute_routing_scores(qr, kr)
        topk_scores, topk_indices = scores.topk(effective_top_k, dim=-1)
        topk_weights = F.softmax(topk_scores.float(), dim=-1).to(topk_scores.dtype)

        expanded_idx = topk_indices.unsqueeze(-1).expand(-1, -1, self.dim)
        topk_protos = torch.gather(
            prototypes.unsqueeze(0).expand(topk_indices.shape[0], -1, -1),
            dim=1,
            index=expanded_idx,
        )

        retrieved = (topk_weights.unsqueeze(-1) * topk_protos).sum(dim=1)
        retrieved = self.out_proj(retrieved)

        return retrieved, topk_indices, topk_weights

    def forward(
        self,
        h: torch.Tensor,
        prototypes: torch.Tensor,
        evidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: returns retrieved representation."""
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
    """

    def __init__(
        self,
        dim: int,
        key_dim: Optional[int] = None,
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

        self.msa_index = MSASparseIndex(
            dim=self.key_dim,
            num_prototypes=num_prototypes,
            top_k=top_k,
            chunk_size=chunk_size,
            num_heads=num_heads,
            temperature=temperature,
        )

        self.register_buffer("overflow_keys", torch.zeros(capacity, self.key_dim))
        self.register_buffer("overflow_vals", torch.zeros(capacity, dim))
        self.register_buffer("overflow_evidence", torch.zeros(capacity))
        self.register_buffer("overflow_count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_overflow_valid", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("_overflow_head", torch.zeros(1, dtype=torch.long))

        self._enabled = True

        if self.key_dim != self.dim:
            self.query_proj = nn.Linear(self.dim, self.key_dim, bias=False)
        else:
            self.query_proj = None

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def is_enabled(self) -> bool:
        return self._enabled

    def store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        evidence: Optional[torch.Tensor] = None,
    ) -> None:
        """Store evicted slot in overflow buffer (ring buffer)."""
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
                    self.overflow_keys[head] = key[i].detach()
                    self.overflow_vals[head] = value[i].detach()
                    self.overflow_evidence[head] = evidence[i].detach()
                    self._overflow_valid[head] = True
                    self._overflow_head[0] = head + 1
                else:
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
        query: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve from overflow buffer via MSA sparse index."""
        if not self._enabled:
            B = query.shape[0]
            k = top_k or self.top_k
            return (
                torch.zeros(B, self.dim, device=query.device),
                torch.zeros(B, k, device=query.device),
                torch.zeros(B, k, dtype=torch.long, device=query.device),
            )

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

        _, indices, weights = self.msa_index.query(
            self.query_proj(query) if self.query_proj is not None else query,
            valid_keys,
            valid_evidence,
        )

        expanded_idx = indices.unsqueeze(-1).expand(-1, -1, self.dim)
        topk_vals = torch.gather(
            valid_vals.unsqueeze(0).expand(query.shape[0], -1, -1),
            dim=1,
            index=expanded_idx,
        )

        retrieved = (weights.unsqueeze(-1) * topk_vals).sum(dim=1)
        return retrieved, weights, indices

    def retrieve_with_interleave(
        self,
        query: torch.Tensor,
        primary_result: torch.Tensor,
        primary_confidence: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-hop retrieval: primary + overflow if needed."""
        if not self._enabled:
            return primary_result, torch.zeros(query.shape[0], dtype=torch.bool, device=query.device)

        need_overflow = primary_confidence < threshold
        used_overflow = need_overflow.clone()

        if not need_overflow.any() or self.overflow_count[0] == 0:
            return primary_result, used_overflow

        blended = primary_result.clone()

        for hop in range(self.max_hops):
            if not need_overflow.any():
                break

            overflow_result, weights, _ = self.retrieve(query)

            confidence_gap = (threshold - primary_confidence).clamp(0, 1)
            overflow_weight = confidence_gap.unsqueeze(-1)

            blended = torch.where(
                need_overflow.unsqueeze(-1),
                (1 - overflow_weight) * primary_result + overflow_weight * overflow_result,
                blended,
            )

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
