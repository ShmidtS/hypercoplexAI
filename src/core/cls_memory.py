"""
CLS Memory — Complementary Learning Systems (McClelland et al., 1995).

HippocampusMemory : fast episodic binding via online gradient descent
NeocortexMemory  : slow semantic compression via momentum-based distillation
CLSMemory        : combined CLS system routing between hippocampus and neocortex
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Hippocampus — fast episodic memory (associative binding)
# ---------------------------------------------------------------------------

class HippocampusMemory(nn.Module):
    """
    Fast episodic memory inspired by CA3/CA1 hippocampal circuits.
    Uses online gradient-descent meta-learning on a small key-value store
    (similar to Titans "surprise" memory but with pattern-completion dynamics).

    Key properties:
    - High plasticity, rapid one-shot binding
    - Short retention horizon (controlled by `forgetting_rate`)
    - Surprise-gated writes (novel patterns update memory more)
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_slots: int = 64,
        key_dim: int = 32,
        forgetting_rate: float = 0.05,
        surprise_threshold: float = 0.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        self.key_dim = key_dim
        self.forgetting_rate = forgetting_rate
        self.surprise_threshold = surprise_threshold

        # Projection layers
        self.key_proj = nn.Linear(hidden_dim, key_dim, bias=False)
        self.val_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Persistent memory slots (keys + values)
        self.register_buffer("mem_keys", torch.randn(memory_slots, key_dim) * 0.02)
        self.register_buffer("mem_vals", torch.zeros(memory_slots, hidden_dim))
        self.register_buffer("mem_age", torch.zeros(memory_slots))
        # Learnable memory write rate
        self.write_gate = nn.Parameter(torch.tensor(0.5))

    def _surprise(self, query_k: torch.Tensor) -> torch.Tensor:
        """Compute surprise = 1 - max cosine similarity to stored keys."""
        keys_norm = F.normalize(self.mem_keys.detach(), dim=-1) # [S, D]
        query_norm = F.normalize(query_k, dim=-1)               # [B, D]
        sim = query_norm @ keys_norm.T                          # [B, S]
        max_sim, _ = sim.max(dim=-1)                            # [B]
        return (1.0 - max_sim).clamp(0, 1)

    @torch.no_grad()
    def _write(self, keys: torch.Tensor, vals: torch.Tensor, surprise: torch.Tensor) -> None:
        """Write to memory slots weighted by surprise."""
        # Only write if surprise > threshold
        write_mask = surprise > self.surprise_threshold          # [B]
        if not write_mask.any():
            return

        write_rate = torch.sigmoid(self.write_gate).detach()
        # Forgetting: exponential decay
        self.mem_age += 1
        decay = torch.exp(-self.forgetting_rate * self.mem_age)
        self.mem_vals.mul_(decay.unsqueeze(-1))

        # Write top-surprise item to least-used slot
        surprise_masked = surprise * write_mask.float()
        best_idx = surprise_masked.argmax().item()
        lru_slot = self.mem_age.argmax().item()

        self.mem_keys[lru_slot].copy_(
            (1 - write_rate) * self.mem_keys[lru_slot] + write_rate * keys[best_idx]
        )
        self.mem_vals[lru_slot].copy_(
            (1 - write_rate) * self.mem_vals[lru_slot] + write_rate * vals[best_idx]
        )
        self.mem_age[lru_slot] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D] — returns memory-augmented representation."""
        query_k = self.key_proj(x)                              # [B, K]
        query_v = self.val_proj(x)                              # [B, D]
        surprise = self._surprise(query_k)                      # [B]

        # Read: soft attention over memory slots
        # Snap buffers before potential inplace writes in _write()
        mem_keys_snap = self.mem_keys.detach().clone()          # [S, K]
        mem_vals_snap = self.mem_vals.detach().clone()          # [S, D]
        keys_norm = F.normalize(mem_keys_snap, dim=-1)          # [S, K]
        query_norm = F.normalize(query_k, dim=-1)               # [B, K]
        attn_logits = query_norm @ keys_norm.T / math.sqrt(self.key_dim)  # [B, S]
        attn = F.softmax(attn_logits, dim=-1)                   # [B, S]
        retrieved = attn @ mem_vals_snap                        # [B, D]

        # Gate: how much memory to blend
        combined = torch.cat([x, retrieved], dim=-1)            # [B, 2D]
        gate_val = torch.sigmoid(self.gate(combined))           # [B, D]
        blended = gate_val * retrieved + (1 - gate_val) * x    # [B, D]
        out = self.dropout(F.gelu(self.out_proj(combined)))
        out = self.norm(out + blended)

        # Write new episodic traces
        if self.training:
            self._write(query_k.detach(), query_v.detach(), surprise.detach())

        return out

    def reset(self) -> None:
        nn.init.normal_(self.mem_keys, std=0.02)
        self.mem_vals.zero_()
        self.mem_age.zero_()


# ---------------------------------------------------------------------------
# Neocortex — slow semantic memory (statistical compression)
# ---------------------------------------------------------------------------

class NeocortexMemory(nn.Module):
    """
    Slow semantic memory inspired by neocortical consolidation.
    Maintains a compressed prototype store updated via EMA (exponential
    moving average) — analogous to slow, interleaved replay in neocortex.

    Key properties:
    - Low plasticity, high capacity, slow updates
    - Stores statistical regularities / cluster centroids
    - Prototype-based retrieval (nearest prototype)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_prototypes: int = 32,
        ema_momentum: float = 0.99,
        temperature: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        self.ema_momentum = ema_momentum
        self.temperature = temperature

        self.in_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Prototype store — updated via EMA (no gradients through prototypes)
        self.register_buffer("prototypes", F.normalize(
            torch.randn(num_prototypes, hidden_dim), dim=-1
        ))
        self.register_buffer("proto_counts", torch.ones(num_prototypes))

    @torch.no_grad()
    def _update_prototypes(self, x: torch.Tensor) -> None:
        """EMA update of prototype centroids."""
        # Assign each sample to nearest prototype
        x_norm = F.normalize(x.detach(), dim=-1)               # [B, D]
        proto_norm = F.normalize(self.prototypes, dim=-1)       # [P, D]
        sim = x_norm @ proto_norm.T                             # [B, P]
        assignments = sim.argmax(dim=-1)                        # [B]

        for p_idx in range(self.num_prototypes):
            mask = assignments == p_idx
            if mask.any():
                centroid = x_norm[mask].mean(0)                 # [D]
                self.prototypes[p_idx] = F.normalize(
                    self.ema_momentum * self.prototypes[p_idx] +
                    (1 - self.ema_momentum) * centroid,
                    dim=-1,
                )
                self.proto_counts[p_idx] = (
                    self.ema_momentum * self.proto_counts[p_idx] + (1 - self.ema_momentum)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D] — returns prototype-augmented representation."""
        h = self.in_proj(x)                                     # [B, D]
        h_norm = F.normalize(h, dim=-1)
        # Detach prototypes for retrieval: they are buffers updated inplace,
        # which would corrupt autograd version tracking if not detached.
        proto_snap = self.prototypes.detach().clone()            # [P, D] — independent copy
        proto_norm = F.normalize(proto_snap, dim=-1)            # [P, D]

        # Soft prototype retrieval
        sim = h_norm @ proto_norm.T / self.temperature          # [B, P]
        attn = F.softmax(sim, dim=-1)                           # [B, P]
        retrieved = attn @ proto_snap                           # [B, D]

        combined = torch.cat([x, retrieved], dim=-1)            # [B, 2D]
        out = self.dropout(F.gelu(self.out_proj(combined)))
        out = self.norm(out + x)

        if self.training:
            self._update_prototypes(h)

        return out

    def reset(self) -> None:
        nn.init.normal_(self.prototypes)
        F.normalize(self.prototypes, dim=-1, out=self.prototypes)
        self.proto_counts.fill_(1.0)


# ---------------------------------------------------------------------------
# CLS Memory — combined Complementary Learning Systems
# ---------------------------------------------------------------------------

class CLSMemory(nn.Module):
    """
    Complementary Learning Systems memory combining Hippocampus (fast episodic)
    and Neocortex (slow semantic) with a learned routing gate.

    The gate decides how much of each system to blend based on input novelty:
    - High surprise → hippocampus dominates (new episodic encoding)
    - Low surprise / familiar pattern → neocortex dominates (prototype retrieval)
    """

    def __init__(
        self,
        hidden_dim: int,
        # Hippocampus params
        memory_slots: int = 64,
        key_dim: int = 32,
        forgetting_rate: float = 0.05,
        surprise_threshold: float = 0.5,
        # Neocortex params
        num_prototypes: int = 32,
        ema_momentum: float = 0.99,
        neo_temperature: float = 0.1,
        # Shared
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hippocampus = HippocampusMemory(
            hidden_dim=hidden_dim,
            memory_slots=memory_slots,
            key_dim=key_dim,
            forgetting_rate=forgetting_rate,
            surprise_threshold=surprise_threshold,
            dropout=dropout,
        )
        self.neocortex = NeocortexMemory(
            hidden_dim=hidden_dim,
            num_prototypes=num_prototypes,
            ema_momentum=ema_momentum,
            temperature=neo_temperature,
            dropout=dropout,
        )
        # Learned routing gate: predict blend weight from input
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, D]
        returns: [B, D] memory-augmented representation
        (optionally also returns gate value [B, 1])
        """
        h_out = self.hippocampus(x)    # [B, D]
        n_out = self.neocortex(x)      # [B, D]

        # gate ~ 1 → hippocampus, gate ~ 0 → neocortex
        gate = self.router(x)          # [B, 1]
        blended = gate * h_out + (1 - gate) * n_out
        out = self.norm(blended)

        if return_gate:
            return out, gate
        return out

    def reset(self) -> None:
        self.hippocampus.reset()
        self.neocortex.reset()

    def memory_loss(self) -> torch.Tensor:
        """
        Optional auxiliary loss: encourage neocortex prototypes to be diverse
        (prevents prototype collapse) via cosine similarity penalty.
        """
        proto = F.normalize(self.neocortex.prototypes, dim=-1)  # [P, D]
        sim = proto @ proto.T                                   # [P, P]
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        off_diag = sim[mask]
        return off_diag.pow(2).mean()
