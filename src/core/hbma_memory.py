"""
HBMA — Human-Brain-Inspired Memory Architecture (pure PyTorch).

Four memory systems inspired by McClelland et al. (1995) and HBMA paper:
  WorkingMemory   : sliding circular buffer, salience-filtered context
  EpisodicMemory  : fast episodic binding, surprise-gated, temporal ordering
  SemanticMemory  : slow semantic compression, EMA prototypes + confidence
  ProceduralMemory: learnable pattern store, trigger detection, step retrieval

All systems are pure PyTorch nn.Module — no external DBs, no Redis.
Drop-in replacement for CLSMemory in HDIMPipeline.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Salience Scorer (vectorised, differentiable where needed)
# ---------------------------------------------------------------------------

class SalienceScorer(nn.Module):
    """
    Multi-factor salience scoring.

    score = 0.45*similarity + 0.20*recency + 0.15*frequency
            + 0.10*importance + 0.10*type_weight

    All inputs are tensors; operates on [B, S] similarity matrices.
    """

    W_SIM   = 0.45
    W_REC   = 0.20
    W_FREQ  = 0.15
    W_IMP   = 0.10
    W_TYPE  = 0.10

    def score(
        self,
        similarity: torch.Tensor,   # [B, S]  cosine similarity to slots
        age: torch.Tensor,          # [S]     slot age in steps
        frequency: torch.Tensor,    # [S]     access count
        importance: torch.Tensor,   # [S]     slot importance in [0,1]
        type_weight: float = 0.8,
        decay_half_life: float = 200.0,
    ) -> torch.Tensor:              # [B, S]
        # recency: exponential decay by age
        recency = torch.exp(-age / decay_half_life).unsqueeze(0)          # [1,S]
        # frequency: log-normalised
        freq_norm = (torch.log(frequency + 1.0) /
                     (torch.log(frequency.max() + 2.0) + 1e-8)).unsqueeze(0)  # [1,S]
        imp = importance.unsqueeze(0)                                     # [1,S]
        tw  = torch.full_like(recency, type_weight)

        sal = (self.W_SIM  * similarity
             + self.W_REC  * recency
             + self.W_FREQ * freq_norm
             + self.W_IMP  * imp
             + self.W_TYPE * tw)
        return sal.clamp(0, 1)


# ---------------------------------------------------------------------------
# Working Memory — circular buffer with salience filtering
# ---------------------------------------------------------------------------

class WorkingMemory(nn.Module):
    """
    Sliding circular buffer of recent hidden states.

    Inspired by Miller's 7+-2 rule and Baddeley's working memory model.
    Stores the N most recent vectors with exponential recency weighting.
    Retrieval uses soft attention weighted by salience.

    Properties:
    - High temporal resolution (step-level)
    - Fast read/write (O(N) attention)
    - No gradient through buffer writes (stateful but detached)
    """

    def __init__(
        self,
        hidden_dim: int,
        capacity: int = 16,       # Miller 7+-2, using 16 for richer context
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.capacity   = capacity

        self.q_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate     = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm     = nn.LayerNorm(hidden_dim)
        self.dropout  = nn.Dropout(dropout)

        # Buffer: circular store
        self.register_buffer("buf",         torch.zeros(capacity, hidden_dim))
        self.register_buffer("buf_age",     torch.zeros(capacity))
        self.register_buffer("buf_freq",    torch.ones(capacity))
        self.register_buffer("buf_imp",     torch.full((capacity,), 0.5))
        self.register_buffer("write_ptr",   torch.zeros(1, dtype=torch.long))
        self.register_buffer("filled",      torch.zeros(1, dtype=torch.long))
        self._salience = SalienceScorer()

    @torch.no_grad()
    def _write(self, x: torch.Tensor, importance: float = 0.5) -> None:
        """Write mean of batch to next buffer slot."""
        slot = int(self.write_ptr.item()) % self.capacity
        self.buf[slot] = x.detach().mean(0)
        self.buf_age[slot] = 0
        self.buf_freq[slot] = 1
        self.buf_imp[slot] = importance
        self.buf_age += 1
        self.buf_age[slot] = 0
        self.write_ptr[0] = (slot + 1) % self.capacity
        self.filled[0] = min(self.filled.item() + 1, self.capacity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D] → memory-augmented [B, D]"""
        n = max(int(self.filled.item()), 1)
        buf_snap = self.buf[:n].detach().clone()     # [N, D]
        age_snap  = self.buf_age[:n].detach().clone()
        freq_snap = self.buf_freq[:n].detach().clone()
        imp_snap  = self.buf_imp[:n].detach().clone()

        q = self.q_proj(x)                           # [B, D]
        k = self.k_proj(buf_snap)                    # [N, D]
        v = self.v_proj(buf_snap)                    # [N, D]

        # Attention
        scale = math.sqrt(self.hidden_dim)
        sim   = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).T) / scale  # [B, N]
        sal   = self._salience.score(sim, age_snap, freq_snap, imp_snap,
                                     type_weight=1.0)
        attn  = F.softmax(sal, dim=-1)               # [B, N]
        retrieved = attn @ v                         # [B, D]

        combined = torch.cat([x, retrieved], dim=-1)
        gate_val = torch.sigmoid(self.gate(combined))
        blended  = gate_val * retrieved + (1 - gate_val) * x
        out = self.dropout(F.gelu(self.out_proj(combined)))
        out = self.norm(out + blended)

        if self.training:
            self._write(x)

        return out

    def reset(self) -> None:
        self.buf.zero_()
        self.buf_age.zero_()
        self.buf_freq.fill_(1)
        self.buf_imp.fill_(0.5)
        self.write_ptr.zero_()
        self.filled.zero_()


# ---------------------------------------------------------------------------
# Episodic Memory — fast surprise-gated binding with temporal ordering
# ---------------------------------------------------------------------------

class EpisodicMemory(nn.Module):
    """
    Fast episodic memory with temporal context.

    Extends HippocampusMemory with:
    - Temporal position encoding (sequence order preservation)
    - Importance-weighted writes (HBMA salience model)
    - Confidence per slot
    - Session-level consolidation signal

    Analogous to CA3/CA1 hippocampal circuits.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_slots: int = 64,
        key_dim: int = 32,
        forgetting_rate: float = 0.05,
        surprise_threshold: float = 0.4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim        = hidden_dim
        self.num_slots         = num_slots
        self.key_dim           = key_dim
        self.forgetting_rate   = forgetting_rate
        self.surprise_threshold = surprise_threshold

        self.key_proj  = nn.Linear(hidden_dim, key_dim, bias=False)
        self.val_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj  = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate      = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm      = nn.LayerNorm(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.write_rate = nn.Parameter(torch.tensor(0.3))

        # Temporal position encoding
        self.pos_enc = nn.Embedding(num_slots, key_dim)

        self.register_buffer("mem_keys",    torch.randn(num_slots, key_dim) * 0.02)
        self.register_buffer("mem_vals",    torch.zeros(num_slots, hidden_dim))
        self.register_buffer("mem_age",     torch.zeros(num_slots))
        self.register_buffer("mem_conf",    torch.full((num_slots,), 0.5))
        self.register_buffer("mem_imp",     torch.full((num_slots,), 0.5))
        self.register_buffer("slot_order",  torch.arange(num_slots))
        self.register_buffer("step",        torch.zeros(1, dtype=torch.long))
        self._salience = SalienceScorer()

    def _surprise(self, query_k: torch.Tensor) -> torch.Tensor:
        keys_norm  = F.normalize(self.mem_keys.detach(), dim=-1)
        query_norm = F.normalize(query_k, dim=-1)
        sim = query_norm @ keys_norm.T
        return (1.0 - sim.max(dim=-1).values).clamp(0, 1)

    @torch.no_grad()
    def _write(
        self,
        keys: torch.Tensor,
        vals: torch.Tensor,
        surprise: torch.Tensor,
        importance: float = 0.5,
    ) -> None:
        write_mask = surprise > self.surprise_threshold
        if not write_mask.any():
            return
        wr = torch.sigmoid(self.write_rate).item()
        # Forgetting: exponential decay of values
        self.mem_age += 1
        decay = torch.exp(-self.forgetting_rate * self.mem_age)
        self.mem_vals.mul_(decay.unsqueeze(-1))
        # Confidence decay
        self.mem_conf.mul_(0.99)

        # LRU write target
        lru_slot = self.mem_age.argmax().item()
        best_idx = (surprise * write_mask.float()).argmax().item()

        self.mem_keys[lru_slot].copy_((1 - wr) * self.mem_keys[lru_slot] + wr * keys[best_idx])
        self.mem_vals[lru_slot].copy_((1 - wr) * self.mem_vals[lru_slot] + wr * vals[best_idx])
        self.mem_age[lru_slot]  = 0
        self.mem_conf[lru_slot] = float(surprise[best_idx].item())
        self.mem_imp[lru_slot]  = importance
        # Temporal ordering: record step
        self.slot_order[lru_slot] = self.step[0]
        self.step[0] += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query_k  = self.key_proj(x)    # [B, K]
        query_v  = self.val_proj(x)    # [B, D]
        surprise = self._surprise(query_k)

        # Snap buffers
        keys_snap = self.mem_keys.detach().clone()   # [S, K]
        vals_snap = self.mem_vals.detach().clone()   # [S, D]
        age_snap  = self.mem_age.detach().clone()
        conf_snap = self.mem_conf.detach().clone()

        # Add temporal position encoding to keys
        pos_idx = (self.slot_order % self.num_slots).detach()
        pos_emb = self.pos_enc(pos_idx)             # [S, K]
        keys_with_pos = F.normalize(keys_snap + 0.1 * pos_emb, dim=-1)
        query_norm    = F.normalize(query_k, dim=-1)

        sim  = query_norm @ keys_with_pos.T / math.sqrt(self.key_dim)  # [B, S]
        sal  = self._salience.score(sim, age_snap, age_snap.clamp(min=1),
                                    conf_snap, type_weight=0.8)
        attn = F.softmax(sal, dim=-1)
        retrieved = attn @ vals_snap

        combined = torch.cat([x, retrieved], dim=-1)
        gate_val = torch.sigmoid(self.gate(combined))
        blended  = gate_val * retrieved + (1 - gate_val) * x
        out = self.dropout(F.gelu(self.out_proj(combined)))
        out = self.norm(out + blended)

        if self.training:
            self._write(query_k.detach(), query_v.detach(), surprise.detach())

        return out

    def reset(self) -> None:
        nn.init.normal_(self.mem_keys, std=0.02)
        self.mem_vals.zero_()
        self.mem_age.zero_()
        self.mem_conf.fill_(0.5)
        self.mem_imp.fill_(0.5)
        self.slot_order.copy_(torch.arange(self.num_slots))
        self.step.zero_()


# ---------------------------------------------------------------------------
# Semantic Memory — slow EMA prototypes with confidence scoring
# ---------------------------------------------------------------------------

class SemanticMemory(nn.Module):
    """
    Slow semantic memory — stable facts and preferences.

    Extends NeocortexMemory with:
    - Per-prototype confidence scores (HBMA evidence_count model)
    - Contradiction detection via cosine distance threshold
    - Slow EMA updates (high momentum = very slow plasticity)
    - Fact-type routing: prototypes partitioned into preference/skill/profile/context

    Analogous to neocortical consolidation via interleaved replay.
    """

    FACT_TYPES = ["preference", "skill", "profile", "context"]

    def __init__(
        self,
        hidden_dim: int,
        num_prototypes: int = 64,     # 4 types x 16 each
        ema_momentum: float = 0.995,  # very slow — HBMA "slow decay"
        temperature: float = 0.07,
        contradiction_thresh: float = -0.3,  # negative cosine = contradiction
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim           = hidden_dim
        self.num_prototypes       = num_prototypes
        self.ema_momentum         = ema_momentum
        self.temperature          = temperature
        self.contradiction_thresh = contradiction_thresh
        self.protos_per_type      = num_prototypes // 4

        self.in_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm     = nn.LayerNorm(hidden_dim)
        self.dropout  = nn.Dropout(dropout)

        # Type router: classify input to one of 4 fact types
        self.type_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),
        )

        self.register_buffer("prototypes",    F.normalize(torch.randn(num_prototypes, hidden_dim), dim=-1))
        self.register_buffer("proto_conf",    torch.full((num_prototypes,), 0.5))   # confidence [0,1]
        self.register_buffer("proto_evidence",torch.ones(num_prototypes))            # evidence count
        self.register_buffer("proto_age",     torch.zeros(num_prototypes))           # steps since update
        self._salience = SalienceScorer()

    @torch.no_grad()
    def _update_prototypes(self, h: torch.Tensor) -> None:
        """EMA update of prototype centroids with confidence tracking."""
        h_norm  = F.normalize(h.detach(), dim=-1)                    # [B, D]
        p_norm  = F.normalize(self.prototypes, dim=-1)               # [P, D]
        sim     = h_norm @ p_norm.T                                  # [B, P]
        assigns = sim.argmax(dim=-1)                                 # [B]

        self.proto_age += 1

        for p in range(self.num_prototypes):
            mask = assigns == p
            if not mask.any():
                continue
            centroid = h_norm[mask].mean(0)

            # Contradiction check: if very dissimilar to current prototype
            curr_sim = F.cosine_similarity(centroid.unsqueeze(0),
                                           self.prototypes[p].unsqueeze(0)).item()
            if curr_sim < self.contradiction_thresh:
                # Conflicting info — reduce confidence
                self.proto_conf[p] = max(0.1, self.proto_conf[p] - 0.1)
            else:
                # Consistent evidence — boost confidence, slow EMA update
                self.proto_conf[p]    = min(0.95, self.proto_conf[p] + 0.02)
                self.proto_evidence[p] = self.proto_evidence[p] + 1
                self.proto_age[p]     = 0

                # EMA update (very slow for semantic stability)
                updated = self.ema_momentum * self.prototypes[p] + (1 - self.ema_momentum) * centroid
                self.prototypes[p] = F.normalize(updated, dim=-1)

    def diversity_loss(self) -> torch.Tensor:
        """Prototype diversity loss — prevents semantic collapse."""
        p = F.normalize(self.prototypes.detach(), dim=-1)
        sim = p @ p.T
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        return sim[mask].pow(2).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D] → semantically-augmented [B, D]"""
        h = self.in_proj(x)                                          # [B, D]
        h_norm = F.normalize(h, dim=-1)

        # Type routing — soft routing to emphasise relevant prototype partition
        type_logits = self.type_router(x)                            # [B, 4]
        type_weights = F.softmax(type_logits, dim=-1)                # [B, 4]

        # Snap all buffers before potential inplace updates in _update_prototypes
        p_snap    = self.prototypes.detach().clone()                 # [P, D]
        conf_snap = self.proto_conf.detach().clone()
        age_snap  = self.proto_age.detach().clone()
        ev_snap   = self.proto_evidence.detach().clone()
        p_norm    = F.normalize(p_snap, dim=-1)

        # Similarity with confidence weighting
        raw_sim = h_norm @ p_norm.T                                  # [B, P]
        # Weight by confidence and type relevance
        type_idx = torch.arange(self.num_prototypes, device=x.device)
        type_assign = type_idx // self.protos_per_type               # [P] -> type 0-3
        type_mask   = type_weights[:, type_assign]                   # [B, P]
        weighted_sim = raw_sim * type_mask * conf_snap.unsqueeze(0)  # [B, P]

        sal  = self._salience.score(weighted_sim, age_snap, ev_snap,
                                    conf_snap, type_weight=0.9)
        attn = F.softmax(sal / self.temperature, dim=-1)             # [B, P]
        retrieved = attn @ p_snap                                    # [B, D]

        combined = torch.cat([x, retrieved], dim=-1)                 # [B, 2D]
        out = self.dropout(F.gelu(self.out_proj(combined)))
        out = self.norm(out + x)

        if self.training:
            self._update_prototypes(h)

        return out

    def reset(self) -> None:
        nn.init.normal_(self.prototypes)
        F.normalize(self.prototypes, dim=-1, out=self.prototypes)
        self.proto_conf.fill_(0.5)
        self.proto_evidence.fill_(1.0)
        self.proto_age.zero_()


# ---------------------------------------------------------------------------
# Procedural Memory — learnable pattern store with trigger detection
# ---------------------------------------------------------------------------

class ProceduralMemory(nn.Module):
    """
    Implicit procedural memory — skills and how-to patterns.

    Unlike episodic/semantic (explicit recall), procedural memory operates
    implicitly: patterns are gradually learned via gradient descent on a
    small content-addressable store.

    Design:
    - num_patterns learnable prototype vectors ("skill embeddings")
    - Trigger detector: binary gate — is this query procedural?
    - Step projector: maps retrieved pattern to actionable output
    - Success tracking: EMA success rate per pattern slot
    """

    def __init__(
        self,
        hidden_dim: int,
        num_patterns: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.num_patterns = num_patterns

        # Learnable pattern prototypes (trained by backprop)
        self.patterns = nn.Parameter(torch.randn(num_patterns, hidden_dim) * 0.02)

        # Trigger detector: is this a procedural query?
        self.trigger_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Step projector: patterns -> actionable representation
        self.step_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blend_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out_proj   = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm       = nn.LayerNorm(hidden_dim)
        self.dropout    = nn.Dropout(dropout)

        # Success rate per pattern (non-differentiable, EMA tracked)
        self.register_buffer("success_rate", torch.full((num_patterns,), 0.5))
        self.register_buffer("usage_count",  torch.zeros(num_patterns))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D] → procedurally-augmented [B, D]"""
        # Trigger gate: how procedural is this query?
        trigger = self.trigger_detector(x)                           # [B, 1]

        # Pattern matching via cosine similarity
        p_norm = F.normalize(self.patterns, dim=-1)                  # [P, D]
        x_norm = F.normalize(x, dim=-1)                              # [B, D]
        sim    = x_norm @ p_norm.T                                   # [B, P]

        # Weight by success rate
        weighted_sim = sim * self.success_rate.detach().clone().unsqueeze(0)  # [B, P]
        attn = F.softmax(weighted_sim / 0.1, dim=-1)                 # [B, P]

        # Retrieve and project patterns
        retrieved = attn @ self.patterns                             # [B, D]
        stepped   = self.step_proj(retrieved)                        # [B, D]

        # Blend: trigger gate controls how much procedural info to inject
        combined  = torch.cat([x, stepped], dim=-1)                  # [B, 2D]
        gate_val  = torch.sigmoid(self.blend_gate(combined)) * trigger  # [B, D]
        blended   = gate_val * stepped + (1 - gate_val) * x
        out = self.dropout(F.gelu(self.out_proj(combined)))
        out = self.norm(out + blended)

        # Update usage stats (no_grad)
        if self.training:
            with torch.no_grad():
                best_patterns = attn.argmax(dim=-1)                  # [B]
                for idx in best_patterns:
                    self.usage_count[idx] += 1

        return out

    @torch.no_grad()
    def update_success(self, pattern_idx: int, success: bool, ema: float = 0.9) -> None:
        """Update success rate for a specific pattern slot."""
        self.success_rate[pattern_idx] = (
            ema * self.success_rate[pattern_idx] + (1 - ema) * float(success)
        )

    def reset(self) -> None:
        nn.init.normal_(self.patterns, std=0.02)
        self.success_rate.fill_(0.5)
        self.usage_count.zero_()


# ---------------------------------------------------------------------------
# Consolidation Engine — Working -> Episodic -> Semantic pipeline
# ---------------------------------------------------------------------------

class ConsolidationEngine(nn.Module):
    """
    Memory consolidation: transfers patterns across memory hierarchy.

    Working -> Episodic : recent context encoded as episodic traces
    Episodic -> Semantic: repeated patterns promoted to semantic facts
    Episodic forgetting  : low-confidence slots decayed aggressively
    Semantic forgetting  : slow decay, conflict-aware

    HBMA paper: "sleep cycle" consolidation with salience scoring.
    In our pure-tensor model, consolidation runs at each forward step
    (lightweight) with a separate consolidation_step() for heavier ops.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Working -> Episodic projection (compress context)
        self.w2e_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Episodic -> Semantic extraction (fact distillation)
        self.e2s_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Importance estimator: scalar salience for incoming pattern
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def consolidate(
        self,
        x: torch.Tensor,
        working: WorkingMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
    ) -> torch.Tensor:
        """
        Run lightweight consolidation step.
        x: [B, D] current hidden state
        Returns consolidated representation [B, D].
        """
        # Estimate importance of current input
        importance = self.importance_head(x.detach()).mean().item()

        # Working -> Episodic: promote important patterns
        if importance > 0.5 and self.training:
            e_candidate = self.w2e_proj(x)               # [B, D]
            # Manually trigger episodic write for important inputs
            with torch.no_grad():
                ek = episodic.key_proj(e_candidate.detach())
                ev = episodic.val_proj(e_candidate.detach())
                surprise = episodic._surprise(ek)
                episodic._write(ek, ev, surprise, importance=importance)

        # Episodic -> Semantic: distill high-confidence episodic patterns
        if importance > 0.7 and self.training:
            s_candidate = self.e2s_proj(x)
            with torch.no_grad():
                semantic._update_prototypes(s_candidate.detach())

        return self.dropout(x)


# ---------------------------------------------------------------------------
# HBMAMemory — top-level drop-in replacement for CLSMemory
# ---------------------------------------------------------------------------

class HBMAMemory(nn.Module):
    """
    Human-Brain-Inspired Memory Architecture (pure PyTorch).

    4-system hierarchy:
      Working Memory    : sliding context buffer (immediate attention)
      Episodic Memory   : surprise-gated fast binding (hippocampus)
      Semantic Memory   : EMA prototype store (neocortex)
      Procedural Memory : learnable pattern store (implicit skills)

    Plus:
      ConsolidationEngine : Working->Episodic->Semantic pipeline
      SalienceScorer      : multi-factor retrieval weighting
      Learned routing gate: decides blend of all four systems

    Interface identical to CLSMemory:
      forward(x)          -> augmented_x
      memory_loss()       -> auxiliary diversity loss
      reset()             -> clear all buffers

    Usage in HDIMConfig: memory_type='hbma'
    """

    def __init__(
        self,
        hidden_dim: int,
        # Working memory
        wm_capacity: int = 16,
        # Episodic memory
        ep_slots: int = 64,
        ep_key_dim: int = 32,
        ep_forgetting_rate: float = 0.05,
        ep_surprise_threshold: float = 0.4,
        # Semantic memory
        sem_prototypes: int = 64,
        sem_ema_momentum: float = 0.995,
        sem_temperature: float = 0.07,
        # Procedural memory
        proc_patterns: int = 32,
        # Shared
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.working    = WorkingMemory(
            hidden_dim=hidden_dim,
            capacity=wm_capacity,
            dropout=dropout,
        )
        self.episodic   = EpisodicMemory(
            hidden_dim=hidden_dim,
            num_slots=ep_slots,
            key_dim=ep_key_dim,
            forgetting_rate=ep_forgetting_rate,
            surprise_threshold=ep_surprise_threshold,
            dropout=dropout,
        )
        self.semantic   = SemanticMemory(
            hidden_dim=hidden_dim,
            num_prototypes=sem_prototypes,
            ema_momentum=sem_ema_momentum,
            temperature=sem_temperature,
            dropout=dropout,
        )
        self.procedural = ProceduralMemory(
            hidden_dim=hidden_dim,
            num_patterns=proc_patterns,
            dropout=dropout,
        )
        self.consolidation = ConsolidationEngine(hidden_dim=hidden_dim, dropout=dropout)

        # Learned 4-way routing gate
        # Outputs [B, 4] softmax weights for [working, episodic, semantic, procedural]
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),
        )

        # Output fusion
        self.fusion_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.fusion_gate = nn.Linear(hidden_dim, hidden_dim)
        self.norm        = nn.LayerNorm(hidden_dim)
        self.dropout_    = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor:
        """
        x: [B, D]
        Returns: [B, D] HBMA-augmented representation.
        """
        # Run consolidation (promotes patterns across hierarchy)
        x = self.consolidation.consolidate(
            x, self.working, self.episodic, self.semantic
        )

        # Parallel retrieval from all four systems
        wm_out   = self.working(x)      # [B, D]
        ep_out   = self.episodic(x)     # [B, D]
        sem_out  = self.semantic(x)     # [B, D]
        proc_out = self.procedural(x)   # [B, D]

        # Learned routing: how much of each system to blend
        gate = F.softmax(self.router(x), dim=-1)  # [B, 4]
        wm_w, ep_w, sem_w, proc_w = gate.unbind(dim=-1)  # each [B]

        # Weighted blend
        blended = (
            wm_w.unsqueeze(-1)   * wm_out
          + ep_w.unsqueeze(-1)   * ep_out
          + sem_w.unsqueeze(-1)  * sem_out
          + proc_w.unsqueeze(-1) * proc_out
        )  # [B, D]

        # Concatenate all outputs for richer fusion
        concat = torch.cat([wm_out, ep_out, sem_out, proc_out], dim=-1)  # [B, 4D]
        fused  = self.dropout_(F.gelu(self.fusion_proj(concat)))          # [B, D]
        fg     = torch.sigmoid(self.fusion_gate(fused))                   # [B, D]
        out    = fg * blended + (1 - fg) * fused
        out    = self.norm(out)

        if return_gate:
            return out, gate
        return out

    def reset(self) -> None:
        """Reset all stateful buffers across all memory systems."""
        self.working.reset()
        self.episodic.reset()
        self.semantic.reset()
        self.procedural.reset()

    def memory_loss(self) -> torch.Tensor:
        """
        Combined auxiliary loss:
        - Semantic diversity (prevents prototype collapse)
        - Procedural pattern diversity
        """
        sem_loss  = self.semantic.diversity_loss()
        p_norm    = F.normalize(self.procedural.patterns, dim=-1)
        proc_sim  = p_norm @ p_norm.T
        mask      = ~torch.eye(proc_sim.shape[0], dtype=torch.bool,
                               device=proc_sim.device)
        proc_loss = proc_sim[mask].pow(2).mean()
        return 0.7 * sem_loss + 0.3 * proc_loss


# ---------------------------------------------------------------------------
# Legacy aliases (backward compat with CLSMemory interface)
# ---------------------------------------------------------------------------

# Keep old names available so existing code importing CLSMemory sub-modules works
HippocampusMemory = EpisodicMemory
NeocortexMemory   = SemanticMemory


class CLSMemory(HBMAMemory):
    """
    Backward-compatible alias: CLSMemory now delegates to HBMAMemory.
    Existing code using CLSMemory(hidden_dim=...) continues to work.
    """
    pass
