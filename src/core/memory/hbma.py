"""HBMA — Human-Brain-Inspired Memory Architecture (pure PyTorch).

Four memory systems inspired by McClelland et al. (1995) and HBMA paper:
  WorkingMemory   : sliding circular buffer, salience-filtered context
  EpisodicMemory  : fast episodic binding, surprise-gated, temporal ordering
  SemanticMemory  : slow semantic compression, EMA prototypes + confidence
  ProceduralMemory: learnable pattern store, trigger detection, step retrieval

All systems are pure PyTorch nn.Module — no external DBs, no Redis.
Implements MemoryInterface directly — adapter logic is inlined:
  - forward(x, update_memory=False) -> MemoryResult
  - memory_loss() returns combined auxiliary loss
  - reset() clears all stateful buffers

Extensible via MemorySubsystemPlugin for 5th+ subsystems.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar
if TYPE_CHECKING:
    from src.models.hdim_model import MSAConfig
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interface import MemoryInterface, MemoryResult
from .sparse_index import MSASparseIndex, MSAOverflowBuffer


# ---------------------------------------------------------------------------
# NARS Truth-Value (inlined from nars_truth.py)
# ---------------------------------------------------------------------------

@dataclass
class NarsTruth:
    """NARS truth-value: (frequency, confidence) pair."""

    freq: float = 0.5
    conf: float = 0.0

    EVIDENTIAL_HORIZON: ClassVar[float] = 1.0
    MAX_CONFIDENCE: ClassVar[float] = 0.99
    RELIANCE: ClassVar[float] = 0.9

    def __post_init__(self):
        self.freq = max(0.0, min(1.0, self.freq))
        self.conf = max(0.0, min(self.MAX_CONFIDENCE, self.conf))

    def evidential_weight(self) -> float:
        if self.conf <= 0.0:
            return 0.0
        if self.conf >= self.MAX_CONFIDENCE:
            return 1e6
        return self.EVIDENTIAL_HORIZON * self.conf / (1.0 - self.conf)

    def expectation(self) -> float:
        return self.conf * (self.freq - 0.5) + 0.5

    @staticmethod
    def w2c(w: float, horizon: float = 1.0) -> float:
        if w <= 0.0:
            return 0.0
        return min(NarsTruth.MAX_CONFIDENCE, w / (w + horizon))

    @staticmethod
    def c2w(c: float, horizon: float = 1.0) -> float:
        if c <= 0.0:
            return 0.0
        if c >= NarsTruth.MAX_CONFIDENCE:
            return 1e6
        return horizon * c / (1.0 - c)

    @staticmethod
    def revision(a: NarsTruth, b: NarsTruth, horizon: float = 1.0) -> NarsTruth:
        w1 = a.evidential_weight()
        w2 = b.evidential_weight()
        total_w = w1 + w2
        if total_w <= 0.0:
            return NarsTruth(freq=0.5, conf=0.0)
        freq = (w1 * a.freq + w2 * b.freq) / total_w
        conf = NarsTruth.w2c(total_w, horizon)
        conf = max(conf, a.conf, b.conf)
        return NarsTruth(freq=freq, conf=min(conf, NarsTruth.MAX_CONFIDENCE))

    @staticmethod
    def projection(conf: float, time_diff: float, decay: float = 0.8) -> float:
        return conf * (decay ** abs(time_diff))

    def __repr__(self) -> str:
        return f"NarsTruth(f={self.freq:.3f}, c={self.conf:.3f})"


# ---------------------------------------------------------------------------
# Plugin system — extensible 5th+ subsystem
# ---------------------------------------------------------------------------

class MemorySubsystemPlugin(nn.Module, ABC):
    """Base class for HBMA memory subsystem plugins."""

    name: str = "plugin"
    priority: int = 10

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def on_consolidate(self, ctx: ConsolidationContext) -> None:
        pass

    def reset(self) -> None:
        pass

    def auxiliary_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, dtype=torch.float32)


@dataclass
class ConsolidationContext:
    """Passed to plugins during consolidation."""
    hidden: torch.Tensor
    working: WorkingMemory
    episodic: EpisodicMemory
    semantic: SemanticMemory
    procedural: Optional[ProceduralMemory]
    is_training: bool
    step: int


# ---------------------------------------------------------------------------
# Salience Scorer (vectorised, differentiable where needed)
# ---------------------------------------------------------------------------

class SalienceScorer(nn.Module):
    """Multi-factor salience scoring."""

    W_SIM   = 0.45
    W_REC   = 0.20
    W_FREQ  = 0.15
    W_IMP   = 0.10
    W_TYPE  = 0.10

    def score(
        self,
        similarity: torch.Tensor,
        age: torch.Tensor,
        frequency: torch.Tensor,
        importance: torch.Tensor,
        type_weight: float = 0.8,
        decay_half_life: float = 200.0,
    ) -> torch.Tensor:
        recency = torch.exp((-age / decay_half_life).clamp(max=80)).unsqueeze(0)
        freq_norm = (torch.log(frequency + 1.0) /
                     (torch.log(frequency.max() + 2.0).clamp(min=1e-8) + 1e-8)).unsqueeze(0)
        imp = importance.unsqueeze(0)
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
    """Sliding circular buffer of recent hidden states."""

    def __init__(
        self,
        hidden_dim: int,
        capacity: int = 16,
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

        self.register_buffer("buf",         torch.zeros(capacity, hidden_dim))
        self.register_buffer("buf_age",     torch.zeros(capacity))
        self.register_buffer("buf_freq",    torch.ones(capacity))
        self.register_buffer("buf_imp",     torch.full((capacity,), 0.5))
        self.register_buffer("write_ptr",   torch.zeros(1, dtype=torch.long))
        self.register_buffer("filled",      torch.zeros(1, dtype=torch.long))
        self._salience = SalienceScorer()

    @torch.no_grad()
    def _write(self, x: torch.Tensor, importance: float = 0.5) -> None:
        slot = self.write_ptr[0] % self.capacity
        self.buf[slot] = x.detach().mean(0)
        self.buf_age[slot] = 0
        self.buf_freq[slot] = 1
        self.buf_imp[slot] = importance
        self.buf_age += 1
        self.buf_age[slot] = 0
        self.write_ptr[0] = (slot + 1) % self.capacity
        self.filled[0] = (self.filled[0] + 1).clamp(max=self.capacity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = max(int(self.filled[0]), 1)
        buf_snap = self.buf[:n].detach().clone()
        age_snap = self.buf_age[:n].detach()
        freq_snap = self.buf_freq[:n].detach()
        imp_snap = self.buf_imp[:n].detach()

        q = self.q_proj(x)
        k = self.k_proj(buf_snap)
        v = self.v_proj(buf_snap)

        sim   = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).T
        sal   = self._salience.score(sim, age_snap, freq_snap, imp_snap,
                                     type_weight=1.0)
        attn  = F.softmax(sal.float(), dim=-1).to(sal.dtype)
        with torch.no_grad():
            self.buf_freq[:n].add_(attn.detach().sum(dim=0))
        retrieved = attn @ v

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
    """Fast episodic memory with temporal context."""

    def __init__(
        self,
        hidden_dim: int,
        num_slots: int = 64,
        key_dim: int = 32,
        forgetting_rate: float = 0.05,
        surprise_threshold: float = 0.4,
        dropout: float = 0.1,
        use_per_slot_durability: bool = False,
        use_overflow: bool = True,
        overflow_num_prototypes: int = 1024,
        overflow_max_hops: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim        = hidden_dim
        self.num_slots         = num_slots
        self.key_dim           = key_dim
        self.forgetting_rate   = forgetting_rate
        self.surprise_threshold = surprise_threshold
        self.use_overflow = use_overflow
        self.use_per_slot_durability = use_per_slot_durability

        self.key_proj  = nn.Linear(hidden_dim, key_dim, bias=False)
        self.val_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj  = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate      = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm      = nn.LayerNorm(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.write_rate = nn.Parameter(torch.tensor(0.3))

        self.pos_enc = nn.Embedding(num_slots, key_dim)

        self.register_buffer("mem_keys",    torch.randn(num_slots, key_dim) * 0.02)
        self.register_buffer("mem_vals",    torch.zeros(num_slots, hidden_dim))
        self.register_buffer("mem_age",     torch.zeros(num_slots))
        self.register_buffer("mem_conf",    torch.full((num_slots,), 0.5))
        self.register_buffer("mem_imp",     torch.full((num_slots,), 0.5))
        self.register_buffer("mem_durability", torch.full((num_slots,), forgetting_rate))
        self.register_buffer("slot_order",  torch.arange(num_slots))
        self.register_buffer("step",        torch.zeros(1, dtype=torch.long))
        self._salience = SalienceScorer()

        if use_overflow:
            self.overflow = MSAOverflowBuffer(
                dim=hidden_dim,
                key_dim=key_dim,
                num_prototypes=overflow_num_prototypes,
                max_hops=overflow_max_hops,
            )
        else:
            self.overflow = None

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
        wr = torch.sigmoid(self.write_rate)
        self.mem_age += 1
        decay_rate = self.mem_durability if self.use_per_slot_durability else torch.as_tensor(self.forgetting_rate, device=self.mem_vals.device)
        decay = torch.exp(-decay_rate).clamp_min(torch.exp(torch.tensor(-80.0, device=self.mem_vals.device)))
        self.mem_vals.mul_(decay.unsqueeze(-1) if decay.ndim > 0 else decay)
        self.mem_conf.mul_(0.99)

        lru_slot = self.mem_age.argmax()
        if self.overflow is not None and self.overflow.is_enabled():
            self.overflow.store(
                self.mem_keys[lru_slot].detach(),
                self.mem_vals[lru_slot].detach(),
                self.mem_conf[lru_slot].detach(),
            )
        best_idx = (surprise * write_mask.float()).argmax()

        self.mem_keys[lru_slot].copy_((1 - wr) * self.mem_keys[lru_slot] + wr * keys[best_idx])
        self.mem_vals[lru_slot].copy_((1 - wr) * self.mem_vals[lru_slot] + wr * vals[best_idx])
        self.mem_age[lru_slot]  = 0
        self.mem_conf[lru_slot] = surprise[best_idx]
        self.mem_imp[lru_slot]  = importance
        if self.use_per_slot_durability:
            self.mem_durability[lru_slot].copy_((self.mem_durability[lru_slot] * 0.8).clamp(min=0.01))
        else:
            self.mem_durability[lru_slot].fill_(self.forgetting_rate)
        self.slot_order[lru_slot] = self.step[0].clone()
        self.step[0] += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query_k  = self.key_proj(x)
        query_v  = self.val_proj(x)
        surprise = self._surprise(query_k)

        keys_snap = self.mem_keys.detach().clone()
        vals_snap = self.mem_vals.detach().clone()
        age_snap  = self.mem_age.detach().clone()
        conf_snap = self.mem_conf.detach().clone()

        pos_idx = (self.slot_order % self.num_slots).detach()
        pos_emb = self.pos_enc(pos_idx)
        keys_with_pos = F.normalize(keys_snap + 0.1 * pos_emb, dim=-1)
        query_norm    = F.normalize(query_k, dim=-1)

        sim  = query_norm @ keys_with_pos.T
        sal  = self._salience.score(sim, age_snap, age_snap.clamp(min=1),
                                    conf_snap, type_weight=0.8)
        attn = F.softmax(sal.float(), dim=-1).to(sal.dtype)
        retrieved = attn @ vals_snap

        if self.overflow is not None and self.overflow.is_enabled():
            primary_conf = (attn * conf_snap.unsqueeze(0)).sum(dim=-1)
            retrieved, used_overflow = self.overflow.retrieve_with_interleave(
                query=x,
                primary_result=retrieved,
                primary_confidence=primary_conf,
                threshold=0.5,
            )

        combined = torch.cat([x, retrieved], dim=-1)
        gate_val = torch.sigmoid(self.gate(combined))
        blended  = gate_val * retrieved + (1 - gate_val) * x
        out = self.dropout(F.gelu(self.out_proj(combined)))
        out = self.norm(out + blended)

        if self.training:
            self._write(query_k.detach(), query_v.detach(), surprise.detach())
            with torch.no_grad():
                matched_slot = attn.detach().argmax(dim=-1)
                if self.use_per_slot_durability:
                    counts = torch.bincount(matched_slot, minlength=self.num_slots).to(self.mem_durability.dtype)
                    update = counts * 0.005
                    self.mem_durability.copy_((self.mem_durability - update).clamp(min=0.0, max=0.2))

        return out

    def reset(self) -> None:
        nn.init.normal_(self.mem_keys, std=0.02)
        self.mem_vals.zero_()
        self.mem_age.zero_()
        self.mem_durability.fill_(self.forgetting_rate)
        self.mem_conf.fill_(0.5)
        self.mem_imp.fill_(0.5)
        self.slot_order.copy_(torch.arange(self.num_slots))
        self.step.zero_()
        if self.use_overflow:
            self.overflow.clear()


# ---------------------------------------------------------------------------
# Semantic Memory — slow EMA prototypes with confidence scoring
# ---------------------------------------------------------------------------

class SemanticMemory(nn.Module):
    """Slow semantic memory — stable facts and preferences."""

    FACT_TYPES = ["preference", "skill", "profile", "context"]

    def __init__(
        self,
        hidden_dim: int,
        num_prototypes: int = 64,
        ema_momentum: float = 0.995,
        temperature: float = 0.07,
        contradiction_thresh: float = -0.3,
        dropout: float = 0.1,
        use_msa: bool = True,
        msa_config: Optional[MSAConfig] = None,
        use_nars_salience: bool = False,
        use_nars_revision: bool = False,
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

        self.type_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),
        )

        self.prototypes = nn.Parameter(F.normalize(torch.randn(num_prototypes, hidden_dim), dim=-1))
        self.prototypes.requires_grad = False
        self.register_buffer("proto_conf",    torch.full((num_prototypes,), 0.5))
        self.register_buffer("proto_evidence",torch.ones(num_prototypes))
        self.register_buffer("proto_age",     torch.zeros(num_prototypes))
        self._salience = SalienceScorer()
        self.use_activation_spreading = False
        self.spreading_decay = 0.3

        self.use_nars_salience = use_nars_salience
        self.use_nars_revision = use_nars_revision
        self.use_msa = use_msa
        if use_msa:
            if msa_config is not None:
                cfg = msa_config
            else:
                from src.models.hdim_model import MSAConfig as _MSAConfig
                cfg = _MSAConfig()
            self.msa_index = MSASparseIndex(
                dim=hidden_dim,
                num_prototypes=num_prototypes,
                top_k=cfg.top_k,
                chunk_size=cfg.chunk_size,
                temperature=cfg.temperature,
                compression_threshold=cfg.compression_threshold,
            )
        else:
            self.msa_index = None

    @torch.no_grad()
    def _update_prototypes(self, h: torch.Tensor) -> None:
        """EMA update of prototype centroids with confidence tracking."""
        h_norm = F.normalize(h.detach(), dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        sim    = h_norm @ p_norm.T
        assigns = sim.argmax(dim=-1)

        self.proto_age += 1

        D = h_norm.shape[1]
        proto_sum   = torch.zeros(self.num_prototypes, D, device=h.device, dtype=torch.float32)
        proto_count = torch.zeros(self.num_prototypes, device=h.device, dtype=torch.float32)
        h_norm_f32 = h_norm.float()

        proto_sum.scatter_add_(0, assigns.unsqueeze(-1).expand(-1, D), h_norm_f32)
        proto_count.scatter_add_(0, assigns, torch.ones_like(assigns, dtype=torch.float32))

        has_samples = proto_count > 0
        safe_count = proto_count.clamp(min=1).unsqueeze(-1)
        centroids = proto_sum / safe_count

        active_mask = has_samples
        if not active_mask.any():
            return

        active_idx = active_mask.nonzero(as_tuple=True)[0]
        active_centroids = centroids[active_idx]
        active_protos = self.prototypes[active_idx]
        curr_sims = F.cosine_similarity(active_centroids, active_protos, dim=-1)

        contradicted = curr_sims < self.contradiction_thresh
        aligned = ~contradicted

        if self.use_nars_revision:
            for i, p in enumerate(active_idx):
                p = p.item()
                if contradicted[i]:
                    existing = NarsTruth(freq=float(self.proto_conf[p]), conf=NarsTruth.w2c(float(self.proto_evidence[p])))
                    negative = NarsTruth(freq=0.0, conf=0.5)
                    revised = NarsTruth.revision(existing, negative)
                    self.proto_conf[p] = revised.freq
                    self.proto_evidence[p] = NarsTruth.c2w(revised.conf)
                else:
                    existing = NarsTruth(freq=float(self.proto_conf[p]), conf=NarsTruth.w2c(float(self.proto_evidence[p])))
                    observation = NarsTruth(freq=float(curr_sims[i]), conf=NarsTruth.RELIANCE)
                    revised = NarsTruth.revision(existing, observation)
                    self.proto_conf[p] = revised.freq
                    self.proto_evidence[p] = NarsTruth.c2w(revised.conf)
                    self.proto_age[p] = 0
                    updated = self.ema_momentum * self.prototypes[p] + (1 - self.ema_momentum) * centroids[p]
                    self.prototypes[p].copy_(F.normalize(updated, dim=-1))
        else:
            contra_idx = active_idx[contradicted]
            align_idx = active_idx[aligned]

            if contra_idx.numel() > 0:
                self.proto_conf[contra_idx] = (self.proto_conf[contra_idx] - 0.1).clamp(min=0.0)

            if align_idx.numel() > 0:
                self.proto_conf[align_idx] = (self.proto_conf[align_idx] + 0.05).clamp(max=1.0)
                self.proto_age[align_idx] = 0
                updated = (self.ema_momentum * self.prototypes[align_idx]
                           + (1 - self.ema_momentum) * centroids[align_idx])
                self.prototypes.index_copy_(0, align_idx, F.normalize(updated, dim=-1))

    def _activation_spreading(self, attn: torch.Tensor, p_norm: torch.Tensor) -> torch.Tensor:
        proto_sim = p_norm @ p_norm.T
        proto_sim = proto_sim * (1 - torch.eye(self.num_prototypes, device=proto_sim.device))
        spread = attn @ proto_sim
        spread_attn = attn + self.spreading_decay * spread
        return spread_attn

    def _dense_retrieval(self, h_norm, p_snap, p_norm, conf_snap, age_snap, ev_snap, type_weights, x):
        raw_sim = h_norm @ p_norm.T
        type_idx = torch.arange(self.num_prototypes, device=x.device)
        type_assign = type_idx // self.protos_per_type
        type_mask = type_weights[:, type_assign]
        nars_conf = ev_snap / (ev_snap + 1.0)
        weighted_sim = raw_sim * type_mask * nars_conf.unsqueeze(0)
        sal = self._salience.score(weighted_sim, age_snap, ev_snap, nars_conf, type_weight=0.9)
        attn = F.softmax((sal / self.temperature).float(), dim=-1).to(sal.dtype)
        if self.use_activation_spreading:
            attn = self._activation_spreading(attn, p_norm)
            attn = F.softmax(attn.float(), dim=-1).to(attn.dtype)
        return attn @ p_snap

    def diversity_loss(self) -> torch.Tensor:
        p = F.normalize(self.prototypes, dim=-1)
        sim = p @ p.T
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        return sim[mask].pow(2).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h_norm = F.normalize(h, dim=-1)

        type_logits = self.type_router(x)
        type_weights = F.softmax(type_logits.float(), dim=-1).to(type_logits.dtype)

        p_snap    = self.prototypes.detach().clone()
        conf_snap = self.proto_conf.detach().clone()
        age_snap  = self.proto_age.detach().clone()
        ev_snap   = self.proto_evidence.detach().clone()
        p_norm    = F.normalize(p_snap, dim=-1)

        raw_sim = h_norm @ p_norm.T
        type_idx = torch.arange(self.num_prototypes, device=x.device)
        type_assign = type_idx // self.protos_per_type
        type_mask   = type_weights[:, type_assign]

        if self.use_nars_salience:
            conf_weight = ev_snap / (ev_snap + 1.0)
        else:
            conf_weight = conf_snap
        weighted_sim = raw_sim * type_mask * conf_weight.unsqueeze(0)
        sal  = self._salience.score(weighted_sim, age_snap, ev_snap,
                                    conf_weight, type_weight=0.9)
        attn = F.softmax((sal / self.temperature).float(), dim=-1).to(sal.dtype)
        if self.use_activation_spreading:
            attn = self._activation_spreading(attn, p_norm)
            attn = F.softmax(attn.float(), dim=-1).to(attn.dtype)
        retrieved = attn @ p_snap

        combined = torch.cat([x, retrieved], dim=-1)
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
    """Implicit procedural memory — skills and how-to patterns."""

    def __init__(
        self,
        hidden_dim: int,
        num_patterns: int = 32,
        dropout: float = 0.1,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.num_patterns = num_patterns
        self.temperature  = temperature

        self.patterns = nn.Parameter(torch.randn(num_patterns, hidden_dim) * 0.02)

        self.trigger_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.step_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blend_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out_proj   = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm       = nn.LayerNorm(hidden_dim)
        self.dropout    = nn.Dropout(dropout)

        self.register_buffer("success_rate", torch.full((num_patterns,), 0.5))
        self.register_buffer("usage_count",  torch.zeros(num_patterns))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trigger = self.trigger_detector(x)

        p_norm = F.normalize(self.patterns, dim=-1)
        x_norm = F.normalize(x, dim=-1)
        sim    = x_norm @ p_norm.T

        weighted_sim = sim * self.success_rate.detach().clone().unsqueeze(0)
        attn = F.softmax((weighted_sim / self.temperature).float(), dim=-1).to(weighted_sim.dtype)

        retrieved = attn @ self.patterns
        stepped   = self.step_proj(retrieved)

        combined  = torch.cat([x, stepped], dim=-1)
        gate_val  = torch.sigmoid(self.blend_gate(combined)) * trigger
        blended   = gate_val * stepped + (1 - gate_val) * x
        out = self.dropout(F.gelu(self.out_proj(combined)))
        out = self.norm(out + blended)

        if self.training:
            with torch.no_grad():
                best_patterns = attn.argmax(dim=-1)
                counts = torch.bincount(best_patterns, minlength=self.num_patterns)
                self.usage_count += counts.to(self.usage_count.device)

        return out

    @torch.no_grad()
    def update_success(self, pattern_idx: int, success: bool, ema: float = 0.9) -> None:
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
    """Memory consolidation: transfers patterns across memory hierarchy."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.w2e_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.e2s_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

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
        importance_tensor = self.importance_head(x.detach()).mean()

        if self.training and importance_tensor > 0.5:
            e_candidate = self.w2e_proj(x)
            with torch.no_grad():
                ek = episodic.key_proj(e_candidate.detach())
                ev = episodic.val_proj(e_candidate.detach())
                surprise = episodic._surprise(ek)
                episodic._write(ek, ev, surprise, importance=float(importance_tensor))

        if importance_tensor > 0.7 and self.training:
            s_candidate = self.e2s_proj(x)
            with torch.no_grad():
                semantic._update_prototypes(s_candidate.detach())

        return self.dropout(x)


# ---------------------------------------------------------------------------
# HBMAMemory — implements MemoryInterface directly (adapter logic inlined)
# ---------------------------------------------------------------------------

class HBMAMemory(MemoryInterface):
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

    Implements MemoryInterface directly (adapter logic inlined):
      forward(x, update_memory=False) -> MemoryResult
      memory_loss() -> combined auxiliary loss
      reset() -> clear all buffers
    """

    def __init__(
        self,
        hidden_dim: int,
        wm_capacity: int = 64,
        ep_slots: int = 256,
        ep_key_dim: int = 32,
        ep_forgetting_rate: float = 0.05,
        ep_surprise_threshold: float = 0.4,
        sem_prototypes: int = 256,
        sem_ema_momentum: float = 0.995,
        sem_temperature: float = 0.07,
        proc_patterns: int = 128,
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

        self._plugins = nn.ModuleList()
        self._plugin_names: list[str] = []
        self._needs_rebuild = False
        self._global_step = 0

        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),
        )

        self.fusion_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.fusion_gate = nn.Linear(hidden_dim, hidden_dim)
        self.norm        = nn.LayerNorm(hidden_dim)
        self.dropout_    = nn.Dropout(dropout)

    def register_plugin(self, plugin: MemorySubsystemPlugin) -> None:
        """Register a plugin subsystem. Call before first forward()."""
        if plugin.name in self._plugin_names:
            raise ValueError(f"Plugin '{plugin.name}' already registered")
        self._plugins.append(plugin)
        self._plugin_names.append(plugin.name)
        self._needs_rebuild = True

    def _get_all_subsystems(self) -> list[tuple[str, nn.Module]]:
        """Returns ordered list of (name, module) for all subsystems."""
        result = [
            ("working", self.working),
            ("episodic", self.episodic),
            ("semantic", self.semantic),
            ("procedural", self.procedural),
        ]
        indexed = list(zip(self._plugin_names, list(self._plugins)))
        indexed.sort(key=lambda x: x[1].priority)
        for name, mod in indexed:
            result.append((name, mod))
        return result

    def _maybe_rebuild(self) -> None:
        """Rebuild router/fusion if plugins have been added."""
        if not self._needs_rebuild:
            return
        n = len(self._get_all_subsystems())
        old_router_out = self.router[-1].out_features
        if n == old_router_out:
            self._needs_rebuild = False
            return

        self.router = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, n),
        )
        self.fusion_proj = nn.Linear(self.hidden_dim * n, self.hidden_dim)
        self.fusion_gate = nn.Linear(self.hidden_dim, self.hidden_dim)
        self._needs_rebuild = False

    def _hbma_forward(
        self,
        x: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Internal HBMA forward: returns augmented representation.

        This is the original HBMAMemory.forward logic, preserved exactly.
        """
        self._maybe_rebuild()

        x = self.consolidation.consolidate(
            x, self.working, self.episodic, self.semantic
        )

        if self._plugins:
            ctx = ConsolidationContext(
                hidden=x, working=self.working, episodic=self.episodic,
                semantic=self.semantic, procedural=self.procedural,
                is_training=self.training, step=self._global_step,
            )
            for p in self._plugins:
                p.on_consolidate(ctx)

        all_subs = self._get_all_subsystems()
        outputs = [mod(x) for _, mod in all_subs]

        gate = F.softmax(self.router(x).float(), dim=-1).to(x.dtype)
        blended = sum(
            gate[:, i:i+1] * out for i, out in enumerate(outputs)
        )

        concat = torch.cat(outputs, dim=-1)
        fused  = self.dropout_(F.gelu(self.fusion_proj(concat)))
        fg     = torch.sigmoid(self.fusion_gate(fused))
        out    = fg * blended + (1 - fg) * fused
        out    = self.norm(out)

        self._global_step += 1

        if return_gate:
            return out, gate
        return out

    # ------------------------------------------------------------------
    # MemoryInterface contract
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        update_memory: bool = False,
    ) -> MemoryResult:
        """MemoryInterface: single-input forward returning MemoryResult.

        Inlined from HBMAMemoryAdapter:
        - Calls internal _hbma_forward for actual computation
        - Wraps output in MemoryResult
        - Computes surprise as normalized deviation from input
        - update_memory flag controls whether memory is actually updated
          (HBMA updates internally during training when self.training is True)
        """
        output = self._hbma_forward(x)

        loss = self._compute_memory_loss()
        actually_updated = update_memory and self.training

        surprise = (output - x).norm(dim=-1, keepdim=True) / (x.norm(dim=-1, keepdim=True) + 1e-8)

        return MemoryResult(
            output=output,
            loss=loss,
            updated=actually_updated,
            surprise=surprise.detach(),
        )

    def _compute_memory_loss(self) -> torch.Tensor:
        """Combined auxiliary loss from all subsystems."""
        sem_loss  = self.semantic.diversity_loss()
        p_norm    = F.normalize(self.procedural.patterns, dim=-1)
        proc_sim  = p_norm @ p_norm.T
        mask      = ~torch.eye(proc_sim.shape[0], dtype=torch.bool,
                               device=proc_sim.device)
        proc_loss = proc_sim[mask].pow(2).mean()

        base_loss = 0.7 * sem_loss + 0.3 * proc_loss

        if not self._plugins:
            return base_loss
        plugin_losses = torch.tensor(0.0, device=base_loss.device, dtype=base_loss.dtype)
        for p in self._plugins:
            pl = p.auxiliary_loss()
            if isinstance(pl, torch.Tensor):
                plugin_losses = plugin_losses + pl
        n = len(self._plugins) + 1
        return (base_loss + plugin_losses) / n

    def reset(self, strategy: str = 'geometric') -> None:
        """Reset all stateful buffers across all memory systems."""
        self.working.reset()
        self.episodic.reset()
        self.semantic.reset()
        self.procedural.reset()
        for p in self._plugins:
            p.reset()

    def memory_loss(self) -> torch.Tensor:
        """Current auxiliary memory loss."""
        return self._compute_memory_loss()


# ---------------------------------------------------------------------------
# Legacy aliases (backward compat with CLSMemory interface)
# ---------------------------------------------------------------------------

HippocampusMemory = EpisodicMemory
NeocortexMemory   = SemanticMemory


class CLSMemory(HBMAMemory):
    """Backward-compatible alias: CLSMemory now delegates to HBMAMemory."""
    pass
