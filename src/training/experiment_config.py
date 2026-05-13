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
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from src.training.auto_config import TrainingDefaults

_TD = TrainingDefaults()

@dataclass
class ExperimentConfig:
    # ------------------------------------------------------------------ #
    # Legacy / core training fields (unchanged from Phase 1) #
    # ------------------------------------------------------------------ #
    description: str = _TD.description
    epochs: int = _TD.epochs
    batch_size: int = _TD.batch_size
    lr: float = _TD.learning_rate
    device: str = _TD.device
    num_samples: int = 100
    use_pairs: bool = False
    negative_ratio: float = _TD.negative_ratio
    train_fraction: float = _TD.train_fraction
    seed: int = _TD.seed
    text_mode: bool = False
    output_dir: str | None = None
    results_json: str | None = None
    ledger_path: str | None = None
    status: str = "pending"
    model_overrides: dict = field(default_factory=dict)
    trainer_overrides: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Phase-2 model hyper-parameters
    # ------------------------------------------------------------------ #
    hidden_dim: int = 64
    num_experts: int | None = None  # None -> from expert_names or default
    expert_names: list | None = None  # Dynamic expert names
    num_domains: int = 4

    # ------------------------------------------------------------------ #
    # Phase-2 loss coefficients
    # ------------------------------------------------------------------ #
    lambda_iso: float = 0.0  # DISABLED: conflicted with pair_loss. Try 0.005 max for A/B test.
    lambda_pair: float = 0.4  # InfoNCE contrastive (optimal from best-score run)
    lambda_sts: float = 0.0  # DISABLED: duplicated InfoNCE. Try 0.01-0.02 for A/B test.
    lambda_routing: float = 0.01  # MoE load balance (reduced: routing loss normalized by log(E))
    lambda_memory: float = 0.05  # memory regularization (EMA stability)
    lambda_dcl: float = 0.05  # DCL loss — decorrelation (Yeh et al. 2022)
    lambda_uniformity: float = 0.02  # uniformity on hypersphere (Wang & Isola 2020)
    lambda_matryoshka: float = 0.15  # Matryoshka multi-scale loss

    # ------------------------------------------------------------------ #
    # Phase-2 training schedule extras
    # ------------------------------------------------------------------ #
    warmup_epochs: int = 3
    early_stopping_patience: int = 8
    infonce_temperature: float = 0.10  # InfoNCE temperature (optimal from best-score run)
    scheduler: str = "onecycle"  # LR scheduler type
    # Optional wall-clock budget in seconds; None means no limit.
    time_budget_s: float | None = None

    # ------------------------------------------------------------------ #
    # Phase-2 optional advanced components
    # ------------------------------------------------------------------ #
    pretrained_encoder: bool = False
    z_loss_weight: float = 0.0
    soft_router: bool = False
    moe_kernel: bool = False
    moe_kernel_expert_names: list | None = None
    n_shared_experts: int = 0
    modernbert_encoder: bool = False
    modernbert_model_name: str = "answerdotai/ModernBERT-base"
    freeze_modernbert: bool = True
    modernbert_use_cls_pooling: bool = True
    modernbert_max_length: int = 512
    matryoshka_dims: list | None = None
    use_domain_embedding: bool = False
    use_domain_lora: bool = False
    domain_lora_rank: int = 4

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
    def from_json(cls, path: str | Path) -> ExperimentConfig:
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
        kwargs: dict[str, Any] = {
            "hidden_dim": self.hidden_dim,
            "num_domains": self.num_domains,
            "expert_names": self.expert_names,
        }
        moe: dict[str, Any] = {}
        if self.num_experts is not None:
            moe["num_experts"] = self.num_experts
        if self.n_shared_experts > 0:
            moe["n_shared_experts"] = self.n_shared_experts
        if self.z_loss_weight > 0:
            moe["z_loss_weight"] = self.z_loss_weight
        kwargs["moe"] = moe
        if self.use_domain_embedding:
            kwargs["use_domain_embedding"] = self.use_domain_embedding
        if self.use_domain_lora:
            kwargs["use_domain_lora"] = self.use_domain_lora
            kwargs["domain_lora_rank"] = self.domain_lora_rank
        return kwargs
