"""Semantic memory subsystem for HBMA."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .salience import SalienceScorer
from .sparse_index import MSASparseIndex
from .truth import NarsTruth

if TYPE_CHECKING:
    from src.extensions.memory.config import MSAConfig

# Lean4 mapping: formalization/Extensions.lean memory sections.


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
        self.proto_conf: torch.Tensor
        self.proto_evidence: torch.Tensor
        self.proto_age: torch.Tensor
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
                from src.extensions.memory.config import MSAConfig
                cfg = MSAConfig()
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
