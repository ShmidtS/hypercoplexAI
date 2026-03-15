"""
cls_memory.py — backward-compatible shim.

All implementations have been superseded by HBMAMemory in hbma_memory.py.
This module re-exports the classes under their original names for compatibility.
"""
from .hbma_memory import (
    HBMAMemory,
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
    ConsolidationEngine,
    SalienceScorer,
    CLSMemory,
    # Legacy aliases
    HippocampusMemory,
    NeocortexMemory,
)

__all__ = [
    "HBMAMemory",
    "CLSMemory",
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "ConsolidationEngine",
    "SalienceScorer",
    "HippocampusMemory",
    "NeocortexMemory",
]
