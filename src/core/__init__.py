"""Streamlined HDIM core exports."""

from __future__ import annotations

from .algebra import CliffordAlgebra
from .engine import CoreEngineConfig, HDIMCoreEngine
from .invariant_index import InvariantIndex
from .invariants import InvariantExtractor, sandwich_transfer
from .rotors import DomainRotationOperator

__all__ = [
    "CliffordAlgebra",
    "DomainRotationOperator",
    "InvariantExtractor",
    "sandwich_transfer",
    "InvariantIndex",
    "CoreEngineConfig",
    "HDIMCoreEngine",
]
