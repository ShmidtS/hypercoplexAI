"""Working memory subsystem for HBMA."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .salience import SalienceScorer

# Lean4 mapping: formalization/Extensions.lean memory sections.


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

        self.buf: torch.Tensor
        self.buf_age: torch.Tensor
        self.buf_freq: torch.Tensor
        self.buf_imp: torch.Tensor
        self.write_ptr: torch.Tensor
        self.filled: torch.Tensor
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
