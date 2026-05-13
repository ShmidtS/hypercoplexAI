"""Streamlined HDIM core exports."""

from __future__ import annotations

from .algebra import CliffordAlgebra
from .engine import CoreEngineConfig
from .engine import HDIMCoreEngine
from .invariant_index import InvariantIndex
from .invariants import InvariantExtractor
from .invariants import sandwich_transfer
from .rotors import DomainRotationOperator

__all__ = [  # noqa: RUF022 - public API order is covered by tests
    "CliffordAlgebra",
    "DomainRotationOperator",
    "InvariantExtractor",
    "sandwich_transfer",
    "InvariantIndex",
    "CoreEngineConfig",
    "HDIMCoreEngine",
]
