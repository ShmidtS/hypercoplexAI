"""Core HDIM model configuration."""

from __future__ import annotations

import warnings
from dataclasses import MISSING
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal

from src.extensions.memory.config import MemoryConfig as _MemoryConfig
from src.extensions.memory.config import MSAConfig as _MSAConfig
from src.extensions.moe.config import MoEConfig as _MoEConfig
from src.extensions.runtime_config import RuntimeConfig as _RuntimeConfig


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


_LEGACY_MEMORY_MAP = {
    "hippocampus": "hbma",
    "neocortex": "hbma",
    "cls": "hbma",
    "prototype": "msa",
}
_VALID_MEMORY_TYPES = {"titans", "hbma", "msa"}


_MOE_FIELDS = set(_MoEConfig.__dataclass_fields__)
_MEMORY_FIELDS = set(_MemoryConfig.__dataclass_fields__) - {"msa"}
_MSA_FIELDS = set(_MSAConfig.__dataclass_fields__)
_RUNTIME_FIELDS = set(_RuntimeConfig.__dataclass_fields__)
_LOSS_FIELDS = set(LossConfig.__dataclass_fields__)


@dataclass(frozen=True)
class HDIMTextConfig:
    """Configuration for the minimal HDIM text encoder path."""

    vocab_size: int = 257
    max_length: int = 128
    embedding_dim: int | None = None
    hidden_dim: int | None = None
    dropout: float | None = None
    vocab_path: str | None = None
    tokenizer_name: str | None = None


@dataclass
class HDIMRuntimeConfig:
    """Runtime controls for memory lifecycle during HDIM execution."""

    update_memory: bool = True
    memory_mode: Literal["none", "retrieve", "update"] = "update"


def _normalize_memory_type(value: str) -> str:
    normalized = _LEGACY_MEMORY_MAP.get(value, value)
    if normalized not in _VALID_MEMORY_TYPES:
        raise ValueError(f"Unknown memory_type: {value}")
    return normalized


def _dataclass_values(config: object) -> dict[str, Any]:
    return {key: getattr(config, key) for key in getattr(config, "__dataclass_fields__", {})}


@dataclass(init=False)
class HDIMConfig:
    """Core HDIM configuration; optional systems live under extensions."""

    hidden_dim: int = 64
    num_domains: int = 4
    domain_names: tuple | None = None
    clifford_p: int = 3
    clifford_q: int = 1
    clifford_r: int = 0
    dropout: float = 0.1
    text_encoder: str | None = None
    extensions: dict = field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        extensions = dict(kwargs.pop("extensions", {}) or {})

        self._collect_extension_config(kwargs, extensions, "moe", _MoEConfig, _MOE_FIELDS)
        self._collect_extension_config(kwargs, extensions, "memory", _MemoryConfig, _MEMORY_FIELDS)
        self._collect_extension_config(kwargs, extensions, "runtime", _RuntimeConfig, _RUNTIME_FIELDS)
        self._collect_extension_config(kwargs, extensions, "loss", LossConfig, _LOSS_FIELDS)

        msa_values = {
            key.removeprefix("msa_"): kwargs.pop(key)
            for key in list(kwargs)
            if key.startswith("msa_") and key.removeprefix("msa_") in _MSA_FIELDS
        }
        if msa_values:
            warnings.warn(
                "msa_* fields are deprecated in core config; use extensions['memory']['msa']",
                DeprecationWarning,
                stacklevel=2,
            )
            memory_ext = self._ensure_extension_dict(extensions, "memory")
            memory_ext["msa"] = {**dict(memory_ext.get("msa", {}) or {}), **msa_values}

        for field_name in (
            "hidden_dim",
            "num_domains",
            "domain_names",
            "clifford_p",
            "clifford_q",
            "clifford_r",
            "dropout",
            "text_encoder",
            "extensions",
        ):
            if field_name == "extensions":
                setattr(self, field_name, extensions)
                continue
            if field_name in kwargs:
                setattr(self, field_name, kwargs.pop(field_name))
            else:
                field_def = self.__dataclass_fields__[field_name]
                if field_def.default_factory is not MISSING:
                    setattr(self, field_name, field_def.default_factory())
                else:
                    setattr(self, field_name, field_def.default)

        text = kwargs.pop("text", None)
        if text is not None:
            self.text = text if isinstance(text, HDIMTextConfig) else HDIMTextConfig(**text)
        else:
            self.text = HDIMTextConfig()

        self.expert_names = kwargs.pop("expert_names", None)
        self.use_domain_embedding = kwargs.pop("use_domain_embedding", False)
        self.use_domain_lora = kwargs.pop("use_domain_lora", False)
        self.domain_lora_rank = kwargs.pop("domain_lora_rank", 4)

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected HDIMConfig field(s): {unknown}")
        self.__post_init__()

    def _collect_extension_config(
        self,
        kwargs: dict[str, Any],
        extensions: dict,
        name: str,
        cls: type,
        fields: set[str],
    ) -> None:
        provided = kwargs.pop(name, None)
        values = {key: kwargs.pop(key) for key in list(kwargs) if key in fields}
        if provided is None and not values:
            return
        if values:
            first_key = next(iter(values))
            warnings.warn(
                f"{first_key} is deprecated in core config; use extensions['{name}']",
                DeprecationWarning,
                stacklevel=3,
            )
        merged = self._coerce_extension(provided, cls)
        merged.update(values)
        extensions[name] = {**dict(extensions.get(name, {}) or {}), **merged}

    def _coerce_extension(self, provided: Any, cls: type) -> dict[str, Any]:
        if provided is None:
            return {}
        if isinstance(provided, dict):
            return dict(provided)
        if isinstance(provided, cls):
            return _dataclass_values(provided)
        if hasattr(provided, "__dataclass_fields__"):
            return _dataclass_values(provided)
        raise TypeError(f"extension config must be {cls.__name__} or dict, got {type(provided).__name__}")

    def _ensure_extension_dict(self, extensions: dict, name: str) -> dict[str, Any]:
        current = extensions.get(name)
        if current is None:
            current = {}
            extensions[name] = current
        if hasattr(current, "__dataclass_fields__"):
            current = _dataclass_values(current)
            extensions[name] = current
        if not isinstance(current, dict):
            raise TypeError(f"extensions['{name}'] must be a dict")
        return current

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if self.num_domains <= 0:
            raise ValueError(f"num_domains must be > 0, got {self.num_domains}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.domain_names is not None and len(self.domain_names) != self.num_domains:
            raise ValueError(f"len(domain_names)={len(self.domain_names)} must equal num_domains={self.num_domains}")
        if self.expert_names is not None and len(set(self.expert_names)) != len(self.expert_names):
            raise ValueError("expert_names must be unique")

        if self.expert_names is not None:
            moe_ext = self._ensure_extension_dict(self.extensions, "moe")
            computed = len(self.expert_names)
            if moe_ext.get("num_experts") is not None and moe_ext["num_experts"] != computed:
                raise ValueError(f"num_experts={moe_ext['num_experts']} conflicts with len(expert_names)={computed}")
            moe_ext["num_experts"] = computed

        memory_ext = self.extensions.get("memory")
        if isinstance(memory_ext, dict) and "memory_type" in memory_ext:
            memory_ext["memory_type"] = _normalize_memory_type(memory_ext["memory_type"])
        if isinstance(memory_ext, dict) and isinstance(memory_ext.get("msa"), _MSAConfig):
            memory_ext["msa"] = _dataclass_values(memory_ext["msa"])
        msa_ext = memory_ext.get("msa") if isinstance(memory_ext, dict) else None
        if isinstance(msa_ext, dict) and msa_ext.get("top_k", 16) > msa_ext.get("num_prototypes", 256):
            raise ValueError(f"msa.top_k={msa_ext['top_k']} must be <= msa.num_prototypes={msa_ext['num_prototypes']}")

    @property
    def moe(self) -> _MoEConfig:
        values = dict(self.extensions.get("moe", {}) or {})
        if values.get("num_experts") is None:
            values["num_experts"] = 4
        return _MoEConfig(**values)

    @moe.setter
    def moe(self, value: _MoEConfig | dict) -> None:
        self.extensions["moe"] = self._coerce_extension(value, _MoEConfig)

    @property
    def memory(self) -> _MemoryConfig:
        values = dict(self.extensions.get("memory", {}) or {})
        msa = values.get("msa")
        if isinstance(msa, dict):
            values["msa"] = _MSAConfig(**msa)
        return _MemoryConfig(**values)

    @memory.setter
    def memory(self, value: _MemoryConfig | dict) -> None:
        values = self._coerce_extension(value, _MemoryConfig)
        msa = values.get("msa")
        if isinstance(msa, _MSAConfig):
            values["msa"] = _dataclass_values(msa)
        self.extensions["memory"] = values

    @property
    def loss(self) -> LossConfig:
        return LossConfig(**dict(self.extensions.get("loss", {}) or {}))

    @loss.setter
    def loss(self, value: LossConfig | dict) -> None:
        self.extensions["loss"] = self._coerce_extension(value, LossConfig)

    @property
    def runtime(self) -> _RuntimeConfig:
        return _RuntimeConfig(**dict(self.extensions.get("runtime", {}) or {}))

    @runtime.setter
    def runtime(self, value: _RuntimeConfig | dict) -> None:
        self.extensions["runtime"] = self._coerce_extension(value, _RuntimeConfig)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dataclass_fields__ or name in {
            "text",
            "expert_names",
            "use_domain_embedding",
            "use_domain_lora",
            "domain_lora_rank",
        }:
            super().__setattr__(name, value)
            return
        for extension_name, fields in (
            ("moe", _MOE_FIELDS),
            ("memory", _MEMORY_FIELDS),
            ("runtime", _RUNTIME_FIELDS),
            ("loss", _LOSS_FIELDS),
        ):
            if name in fields and "extensions" in self.__dict__:
                if name == "memory_type":
                    value = _normalize_memory_type(value)
                self._ensure_extension_dict(self.extensions, extension_name)[name] = value
                return
        if name.startswith("msa_") and "extensions" in self.__dict__:
            msa_name = name.removeprefix("msa_")
            if msa_name in _MSA_FIELDS:
                memory_ext = self._ensure_extension_dict(self.extensions, "memory")
                msa_ext = dict(memory_ext.get("msa", {}) or {})
                msa_ext[msa_name] = value
                memory_ext["msa"] = msa_ext
                return
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        if name == "msa":
            return self.memory.msa
        if name.startswith("msa_"):
            msa = self.memory.msa
            msa_name = name.removeprefix("msa_")
            if msa is not None and hasattr(msa, msa_name):
                return getattr(msa, msa_name)
        for extension_name, fields in (
            ("moe", _MOE_FIELDS),
            ("memory", _MEMORY_FIELDS),
            ("runtime", _RUNTIME_FIELDS),
            ("loss", _LOSS_FIELDS),
        ):
            if name in fields:
                value = getattr(getattr(self, extension_name), name)
                if name == "num_experts" and value is None:
                    return 4
                return value
        raise AttributeError(f"'HDIMConfig' has no attribute '{name}'")

    def get_domain_names(self) -> list[str]:
        """Return the resolved list of domain names."""
        if self.domain_names is not None:
            return list(self.domain_names)
        return [f"domain_{i}" for i in range(self.num_domains)]
