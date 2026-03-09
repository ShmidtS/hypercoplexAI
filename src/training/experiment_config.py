from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    description: str = "baseline"
    epochs: int = 3
    batch_size: int = 16
    lr: float = 1e-3
    device: str = "cpu"
    num_samples: int = 100
    use_pairs: bool = False
    negative_ratio: float = 0.0
    train_fraction: float = 0.8
    seed: int = 42
    text_mode: bool = False
    output_dir: str | None = None
    results_json: str | None = None
    ledger_path: str | None = None
    status: str = "pending"
    model_overrides: dict[str, Any] = field(default_factory=dict)
    trainer_overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:12]
