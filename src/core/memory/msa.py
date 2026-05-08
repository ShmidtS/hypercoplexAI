"""MSA (Memory Sparse Attention) — Primary memory system using MSA sparse attention.

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

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interface import MemoryInterface, MemoryResult
from .sparse_index import MSASparseIndex, MSAOverflowBuffer


class MSAMemory(MemoryInterface):
    prototypes: torch.Tensor
    confidence: torch.Tensor
    evidence: torch.Tensor
    age: torch.Tensor
    fill_count: torch.Tensor
    _last_diversity_loss: torch.Tensor
    _routing_counts: torch.Tensor
    """Primary memory system using MSA sparse attention.

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

        self.in_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

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

        self.sparse_index = MSASparseIndex(
            dim=hidden_dim,
            num_prototypes=num_prototypes,
            top_k=top_k,
            chunk_size=chunk_size,
            num_heads=num_heads,
            temperature=temperature,
            compression_threshold=compression_threshold,
        )

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

        self.register_buffer("_last_diversity_loss", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer(
            "_routing_counts", torch.zeros(num_prototypes, dtype=torch.float32)
        )

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
        topk_indices: torch.Tensor,
        topk_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Auxiliary loss encouraging even prototype usage."""
        B = topk_indices.shape[0]
        if B == 0:
            return torch.tensor(0.0, device=topk_indices.device, dtype=torch.float32)

        n = int(self.fill_count[0].item())
        if n < 2:
            return torch.tensor(0.0, device=topk_indices.device, dtype=torch.float32)

        if topk_weights is not None:
            soft_counts = torch.zeros(n, device=topk_indices.device, dtype=torch.float32)
            flat_indices = topk_indices.reshape(-1)
            flat_weights = topk_weights.reshape(-1)
            soft_counts.scatter_add_(0, flat_indices, flat_weights)
            counts = soft_counts
        else:
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
            h_norm = F.normalize(x, dim=-1)
            if n > 0:
                p_norm = F.normalize(self.prototypes[:n], dim=-1)
                sim = h_norm @ p_norm.T
                max_sim, max_idx = sim.max(dim=-1)
            else:
                max_sim = torch.zeros(B, device=x.device)
                max_idx = torch.zeros(B, device=x.device, dtype=torch.long)
            ema_mask = max_sim > 0.95
            new_mask = ~ema_mask
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
        """Retrieve from MSA prototype memory + optional update."""
        B, D = x.shape
        device = x.device

        if update_memory:
            self._store_invariant_batched(x)

        h = self.in_proj(x)

        n = int(self.fill_count[0].item())

        if n < self.top_k:
            gate_val = torch.sigmoid(self.gate(x))
            output = x + gate_val * torch.zeros_like(x)
            surprise = torch.zeros(B, 1, device=device, dtype=torch.float32)
            return MemoryResult(
                output=self.norm(output),
                loss=torch.tensor(0.0, device=device, dtype=torch.float32),
                updated=update_memory,
                surprise=surprise.detach(),
            )

        active_p = self._active_prototypes()
        active_ev = self._active_evidence()
        retrieved, topk_indices, topk_weights = self.sparse_index.query(
            h, active_p, active_ev
        )

        primary_confidence = topk_weights.max(dim=-1).values

        if self.overflow.is_enabled() and self.overflow.size() > 0:
            retrieved, _ = self.overflow.retrieve_with_interleave(
                query=h,
                primary_result=retrieved,
                primary_confidence=primary_confidence,
                threshold=self.interleave_threshold,
            )

        gate_val = torch.sigmoid(self.gate(x))
        output = x + gate_val * retrieved
        output = self.drop(self.norm(output))

        with torch.no_grad():
            surprise = (
                (output.detach() - x).norm(dim=-1, keepdim=True)
                / (x.norm(dim=-1, keepdim=True) + 1e-8)
            )

        diversity_loss = self._compute_routing_diversity_loss(topk_indices, topk_weights=topk_weights)
        weighted_diversity_loss = diversity_loss * self.diversity_loss_weight
        self._last_diversity_loss = weighted_diversity_loss

        with torch.no_grad():
            self.age[:n] += 1.0

        return MemoryResult(
            output=output,
            loss=weighted_diversity_loss,
            updated=update_memory,
            surprise=surprise.detach(),
        )

    def reset(self, strategy: str = "geometric") -> None:
        """Reset memory state."""
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
            decay = 0.7
            uniform_conf = torch.full_like(self.confidence, 0.5)
            self.confidence[:n] = decay * self.confidence[:n] + (1 - decay) * uniform_conf[:n]
            self.evidence[:n] = (self.evidence[:n] * decay).clamp(min=1.0)
            self.age[:n] = 0.0

        elif strategy == "stabilize":
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
