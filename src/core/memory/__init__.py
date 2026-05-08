"""Unified memory subpackage for HDIM pipeline.

Exports:
    MemoryInterface, MemoryResult — abstract contract
    TitansMemory — neural associative memory with TTT updates
    HBMAMemory — hippocampal-based memory architecture
    MSAMemory — prototype-based sparse retrieval
    MSASparseIndex, MSAOverflowBuffer — MSA utilities
    NarsTruth — NARS truth-value system
"""

from .interface import MemoryInterface, MemoryResult
from .titans import TitansMemory, TitansMemoryModule
from .hbma import (
    HBMAMemory,
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
    ConsolidationEngine,
    MemorySubsystemPlugin,
    ConsolidationContext,
    SalienceScorer,
    NarsTruth,
    CLSMemory,
)
from .msa import MSAMemory
from .sparse_index import MSASparseIndex, MSAOverflowBuffer

__all__ = [
    "MemoryInterface",
    "MemoryResult",
    "TitansMemory",
    "TitansMemoryModule",
    "HBMAMemory",
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "ConsolidationEngine",
    "MemorySubsystemPlugin",
    "ConsolidationContext",
    "SalienceScorer",
    "NarsTruth",
    "CLSMemory",
    "MSAMemory",
    "MSASparseIndex",
    "MSAOverflowBuffer",
]
