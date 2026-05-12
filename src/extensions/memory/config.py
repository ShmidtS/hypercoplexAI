"""Configuration for optional memory extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class MSAConfig:
    """Configuration for MSA prototype memory subsystem."""

    dim: int = 256
    num_prototypes: int = 256
    top_k: int = 16
    chunk_size: int = 64
    num_heads: int = 4
    temperature: float = 0.1
    ema_momentum: float = 0.995
    overflow_capacity: int = 10000
    max_hops: int = 3
    interleave_threshold: float = 0.5
    compression_threshold: int = 128
    diversity_loss_weight: float = 1.0


@dataclass
class MemoryConfig:
    """Memory subsystem configuration."""

    memory_type: str = "titans"
    memory_key_dim: int = 32
    msa: Optional[MSAConfig] = None
    use_gradient_surprise: bool = False
    use_adaptive_forgetting: bool = False
