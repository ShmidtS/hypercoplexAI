"""Consolidation context for HBMA plugins."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from .episodic import EpisodicMemory
    from .procedural import ProceduralMemory
    from .semantic import SemanticMemory
    from .working import WorkingMemory


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
