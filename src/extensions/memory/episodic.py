"""Episodic memory subsystem for HBMA."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .salience import SalienceScorer
from .sparse_index import MSAOverflowBuffer

# Lean4 mapping: formalization/Extensions.lean memory sections.


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

        self.mem_keys: torch.Tensor
        self.mem_vals: torch.Tensor
        self.mem_age: torch.Tensor
        self.mem_conf: torch.Tensor
        self.mem_imp: torch.Tensor
        self.mem_durability: torch.Tensor
        self.slot_order: torch.Tensor
        self.step: torch.Tensor
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
        if self.overflow is not None:
            self.overflow.clear()
