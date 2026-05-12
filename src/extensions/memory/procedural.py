"""Procedural memory subsystem for HBMA."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lean4 mapping: formalization/Extensions.lean memory sections.


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

        self.success_rate: torch.Tensor
        self.usage_count: torch.Tensor
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
