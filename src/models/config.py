"""HDIM model configuration split into focused sub-configs."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field
from typing import List, Literal, Optional


_LEGACY_MEMORY_MAP = {
    "hippocampus": "hbma",
    "neocortex": "hbma",
    "cls": "hbma",
    "prototype": "msa",
}
_VALID_MEMORY_TYPES = {"titans", "hbma", "msa"}


def _normalize_memory_type(value: str) -> str:
    normalized = _LEGACY_MEMORY_MAP.get(value, value)
    if normalized not in _VALID_MEMORY_TYPES:
        raise ValueError(f"Unknown memory_type: {value}")
    return normalized


@dataclass(frozen=True)
class HDIMTextConfig:
    """Configuration for the minimal HDIM text encoder path."""

    vocab_size: int = 257
    max_length: int = 128
    embedding_dim: Optional[int] = None
    hidden_dim: Optional[int] = None
    dropout: Optional[float] = None
    vocab_path: Optional[str] = None
    tokenizer_name: Optional[str] = None


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
class MoEConfig:
    """Mixture of Experts configuration."""

    num_experts: Optional[int] = None
    top_k: int = 2
    n_shared_experts: int = 0
    z_loss_weight: float = 0.0
    use_aux_loss_free: bool = False
    aux_lr: float = 0.001
    use_expert_ortho: bool = False


@dataclass
class MemoryConfig:
    """Memory subsystem configuration."""

    memory_type: str = "titans"
    memory_key_dim: int = 32
    msa: Optional[MSAConfig] = None
    use_gradient_surprise: bool = False
    use_adaptive_forgetting: bool = False


@dataclass
class LossConfig:
    """Loss function configuration."""

    lambda_recon: float = 1.0
    lambda_pair: float = 0.4
    lambda_routing: float = 0.01
    lambda_memory: float = 0.05
    lambda_z: float = 0.01
    lambda_expert_ortho: float = 0.01
    lambda_matryoshka: float = 0.15
    lambda_iso: float = 0.0
    lambda_sts: float = 0.0
    lambda_angle: float = 0.0
    lambda_supcon: float = 0.0


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


@dataclass
class HDIMRuntimeConfig:
    """Runtime controls for memory lifecycle during HDIM execution."""

    update_memory: bool = True
    memory_mode: Literal["none", "retrieve", "update"] = "update"


@dataclass(init=False)
class HDIMConfig:
    """Top-level HDIM configuration — delegates to sub-configs."""

    moe: MoEConfig = field(default_factory=MoEConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    hidden_dim: int = 64
    num_domains: int = 4
    dropout: float = 0.1
    clifford_p: int = 3
    clifford_q: int = 1
    clifford_r: int = 0
    domain_names: Optional[List[str]] = None
    expert_names: Optional[List[str]] = None
    text: HDIMTextConfig = field(default_factory=HDIMTextConfig)
    use_domain_embedding: bool = False
    use_domain_lora: bool = False
    domain_lora_rank: int = 4

    def __init__(self, **kwargs):
        subconfigs = {
            "moe": MoEConfig,
            "memory": MemoryConfig,
            "loss": LossConfig,
            "runtime": RuntimeConfig,
        }
        for name, cls in subconfigs.items():
            provided = kwargs.pop(name, None)
            values = {
                key: kwargs.pop(key)
                for key in list(kwargs)
                if key in cls.__dataclass_fields__
            }
            if provided is None:
                setattr(self, name, cls(**values))
            elif isinstance(provided, cls):
                setattr(self, name, cls(**{**provided.__dict__, **values}))
            elif isinstance(provided, dict):
                setattr(self, name, cls(**{**provided, **values}))
            else:
                raise TypeError(f"{name} must be {cls.__name__} or dict, got {type(provided).__name__}")

        msa_values = {
            key.removeprefix("msa_"): kwargs.pop(key)
            for key in list(kwargs)
            if key.startswith("msa_") and key.removeprefix("msa_") in MSAConfig.__dataclass_fields__
        }
        if self.memory.msa is None:
            self.memory.msa = MSAConfig(**msa_values)
        elif msa_values:
            self.memory.msa = MSAConfig(
                **{**self.memory.msa.__dict__, **msa_values}
            )

        for field_name in (
            "hidden_dim",
            "num_domains",
            "dropout",
            "clifford_p",
            "clifford_q",
            "clifford_r",
            "domain_names",
            "expert_names",
            "text",
            "use_domain_embedding",
            "use_domain_lora",
            "domain_lora_rank",
        ):
            if field_name in kwargs:
                setattr(self, field_name, kwargs.pop(field_name))
            else:
                field_def = self.__dataclass_fields__[field_name]
                if field_def.default_factory is not MISSING:
                    setattr(self, field_name, field_def.default_factory())
                else:
                    setattr(self, field_name, field_def.default)

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected HDIMConfig field(s): {unknown}")
        self.__post_init__()

    def __post_init__(self):
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if self.num_domains <= 0:
            raise ValueError(f"num_domains must be > 0, got {self.num_domains}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.memory.memory_key_dim <= 0:
            raise ValueError(f"memory_key_dim must be > 0, got {self.memory.memory_key_dim}")
        self.memory.memory_type = _normalize_memory_type(self.memory.memory_type)
        if self.domain_names is not None and len(self.domain_names) != self.num_domains:
            raise ValueError(
                f"len(domain_names)={len(self.domain_names)} must equal num_domains={self.num_domains}"
            )
        if self.expert_names is not None and len(set(self.expert_names)) != len(self.expert_names):
            raise ValueError("expert_names must be unique")
        if self.memory.msa is not None and self.memory.msa.top_k > self.memory.msa.num_prototypes:
            raise ValueError(
                f"msa.top_k={self.memory.msa.top_k} must be <= msa.num_prototypes={self.memory.msa.num_prototypes}"
            )
        if self.expert_names is not None:
            computed = len(self.expert_names)
            if self.moe.num_experts is not None and self.moe.num_experts != computed:
                raise ValueError(
                    f"num_experts={self.moe.num_experts} conflicts with "
                    f"len(expert_names)={computed}"
                )
            self.moe.num_experts = computed
        elif self.moe.num_experts is None:
            self.moe.num_experts = 4
        if self.moe.num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {self.moe.num_experts}")
        if self.moe.top_k > self.moe.num_experts:
            raise ValueError(f"top_k={self.moe.top_k} must be <= num_experts={self.moe.num_experts}")

    def __setattr__(self, name, value):
        if name in {"moe", "memory", "loss", "runtime"} or name in self.__dataclass_fields__:
            super().__setattr__(name, value)
            return
        for subconfig_name in ("moe", "memory", "loss", "runtime"):
            if subconfig_name not in self.__dict__:
                continue
            subconfig = getattr(self, subconfig_name)
            if hasattr(subconfig, name):
                if name == "memory_type":
                    value = _normalize_memory_type(value)
                setattr(subconfig, name, value)
                return
        super().__setattr__(name, value)

    def __getattr__(self, name):
        for subconfig in (self.moe, self.memory, self.loss, self.runtime):
            if hasattr(subconfig, name):
                return getattr(subconfig, name)
        if name == "msa":
            return self.memory.msa
        if name.startswith("msa_") and self.memory.msa is not None:
            msa_name = name.removeprefix("msa_")
            if hasattr(self.memory.msa, msa_name):
                return getattr(self.memory.msa, msa_name)
        raise AttributeError(f"'HDIMConfig' has no attribute '{name}'")

    def get_domain_names(self) -> List[str]:
        """Return the resolved list of domain names."""
        if self.domain_names is not None:
            return list(self.domain_names)
        return [f"domain_{i}" for i in range(self.num_domains)]
