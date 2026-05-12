"""Salience scoring for HBMA memory retrieval."""
from __future__ import annotations

import torch
import torch.nn as nn


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
