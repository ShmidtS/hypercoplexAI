"""Configuration for optional runtime extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RuntimeConfig:
    """Training and runtime configuration."""

    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 100
    online_learning: bool = False
    online_replay_size: int = 10000
    online_surprise_threshold: float = 0.3
    online_ttt_lr: float = 1e-5
    online_gradient_mode: str = "detached"
    online_gradient_scale: float = 0.1
    hallucination_detection: bool = False
    hallucination_risk_threshold: float = 0.5
    hallucination_feedback: bool = False
    hallucination_feedback_config: Optional[dict] = None
    gradient_checkpointing: bool = False
