"""Streamlined HDIM core exports."""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import Any

from .algebra import CliffordAlgebra
from .engine import CoreEngineConfig, HDIMCoreEngine
from .invariant_index import InvariantIndex
from .invariants import InvariantExtractor, sandwich_transfer
from .rotors import DomainRotationOperator
from .types import AnalogyMatch, InvariantRecord

__all__ = [
    "CliffordAlgebra",
    "DomainRotationOperator",
    "InvariantExtractor",
    "sandwich_transfer",
    "AnalogyMatch",
    "InvariantRecord",
    "InvariantIndex",
    "HDIMCoreEngine",
    "CoreEngineConfig",
]

_EXTENSION_EXPORTS = {
    "MemoryInterface": "src.extensions.memory",
    "MemoryResult": "src.extensions.memory",
    "TitansMemory": "src.extensions.memory",
    "TitansMemoryModule": "src.extensions.memory",
    "HBMAMemory": "src.extensions.memory",
    "WorkingMemory": "src.extensions.memory",
    "EpisodicMemory": "src.extensions.memory",
    "SemanticMemory": "src.extensions.memory",
    "ProceduralMemory": "src.extensions.memory",
    "ConsolidationEngine": "src.extensions.memory",
    "MemorySubsystemPlugin": "src.extensions.memory",
    "ConsolidationContext": "src.extensions.memory",
    "SalienceScorer": "src.extensions.memory",
    "NarsTruth": "src.extensions.memory",
    "CLSMemory": "src.extensions.memory",
    "MSAMemory": "src.extensions.memory",
    "MSASparseIndex": "src.extensions.memory",
    "MSAOverflowBuffer": "src.extensions.memory",
    "MoERouter": "src.extensions.moe",
    "MoEKernel": "src.extensions.moe",
    "MoEKernelConfig": "src.extensions.moe",
    "SoftMoERouter": "src.extensions.moe",
}


def __getattr__(name: str) -> Any:
    module_name = _EXTENSION_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    warnings.warn(
        f"src.core.{name} is deprecated; import from {module_name} instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(import_module(module_name), name)
