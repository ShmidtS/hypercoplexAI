"""Consolidation engine for HBMA memory hierarchy."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .episodic import EpisodicMemory
    from .semantic import SemanticMemory
    from .working import WorkingMemory

# Lean4 mapping: formalization/Extensions.lean consolidation behavior.


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
