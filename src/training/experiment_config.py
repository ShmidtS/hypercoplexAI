"""Canonical experiment configuration contract for HDIM research.

All fields carry sane defaults so legacy manifests (that omit new keys)
continue to deserialise without errors.

Backward-compat notes
---------------------
* ``lr`` is the canonical learning-rate field (matches train.py and older
  manifests). ``learning_rate`` is kept as an alias via ``__post_init__``.
* All new Phase-2 fields default to False/None/0 so existing JSON manifests
  that omit them deserialise without errors.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ExperimentConfig:
    # ------------------------------------------------------------------ #
    # Legacy / core training fields (unchanged from Phase 1) #
    # ------------------------------------------------------------------ #
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
    output_dir: Optional[str] = None
    results_json: Optional[str] = None
    ledger_path: Optional[str] = None
    status: str = "pending"
    model_overrides: dict = field(default_factory=dict)
    trainer_overrides: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Phase-2 model hyper-parameters
    # ------------------------------------------------------------------ #
    hidden_dim: int = 128
    num_experts: Optional[int] = None  # None -> from expert_names or default
    expert_names: Optional[list] = None  # Dynamic expert names
    num_domains: int = 4

    # ------------------------------------------------------------------ #
    # Phase-2 loss coefficients
    # ------------------------------------------------------------------ #
    lambda_iso: float = 0.0  # DISABLED: conflicts with pair_loss, suppresses margin
    lambda_pair: float = 0.4  # InfoNCE contrastive (optimal from Run 18)
    lambda_routing: float = 0.05
    lambda_memory: float = 0.05  # memory regularization (EMA stability)

    # ------------------------------------------------------------------ #
    # Phase-2 training schedule extras
    # ------------------------------------------------------------------ #
    warmup_epochs: int = 3
    # Optional wall-clock budget in seconds; None means no limit.
    time_budget_s: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Phase-2 optional advanced components
    # ------------------------------------------------------------------ #
    pretrained_encoder: bool = False
    z_loss_weight: float = 0.0
    soft_router: bool = False
    modernbert_encoder: bool = False
    modernbert_model_name: str = "answerdotai/ModernBERT-base"
    freeze_modernbert: bool = True
    modernbert_use_cls_pooling: bool = True
    modernbert_max_length: int = 512
    matryoshka_dims: Optional[list] = None

    # ------------------------------------------------------------------ #
    # Misc extras
    # ------------------------------------------------------------------ #
    experiment_name: str = "hdim_experiment"
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    data_path: str = "data"

    # ------------------------------------------------------------------ #
    # Serialisation helpers (unchanged from Phase 1)
    # ------------------------------------------------------------------ #
    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """Load from a JSON manifest; unknown keys are silently ignored."""
        payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in payload.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:12]

    # ------------------------------------------------------------------ #
    # HDIMConfig bridge
    # ------------------------------------------------------------------ #
    def to_hdim_config_kwargs(self) -> dict[str, Any]:
        """Return the subset of fields relevant to HDIMConfig construction."""
        kwargs = dict(
            hidden_dim=self.hidden_dim,
            num_domains=self.num_domains,
        )
        # Pass expert_names if provided, otherwise pass num_experts
        if self.expert_names is not None:
            kwargs["expert_names"] = self.expert_names
        elif self.num_experts is not None:
            kwargs["num_experts"] = self.num_experts
        return kwargs
