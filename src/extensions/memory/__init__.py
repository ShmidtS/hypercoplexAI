"""Optional memory backend extensions for HDIM."""

from .config import MemoryConfig, MSAConfig
from .interface import MemoryInterface, MemoryResult
from .invariant_processor import InvariantMemoryState, InvariantProcessor
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
    "MemoryConfig",
    "MSAConfig",
    "MemoryInterface",
    "MemoryResult",
    "InvariantMemoryState",
    "InvariantProcessor",
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
