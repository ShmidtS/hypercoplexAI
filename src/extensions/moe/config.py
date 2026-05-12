"""Configuration for optional Mixture-of-Experts extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class MoEConfig:
    """Mixture of Experts configuration."""

    num_experts: Optional[int] = None
    top_k: int = 2
    n_shared_experts: int = 0
    z_loss_weight: float = 0.0
    use_aux_loss_free: bool = False
    aux_lr: float = 0.001
    use_expert_ortho: bool = False
