from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class InvariantRecord:
    key: str
    invariant: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalogyMatch:
    key: str
    score: float
    invariant: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
