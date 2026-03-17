"""
HDIM src/core — ядро гиперкомплексной архитектуры.
"""
from .hypercomplex import (
    CliffordAlgebra,
    QuaternionLinear,
    QLayerNorm,
)
from .domain_operators import (
    DomainRotationOperator,
    InvariantExtractor,
    sandwich_transfer,
)
from .titans_memory import TitansMemoryModule
from .hbma_memory import HippocampusMemory, HBMAMemory, WorkingMemory, EpisodicMemory, SemanticMemory, ProceduralMemory
from .hdim_pipeline import HDIMPipeline, HDIMEncoder

__all__ = [
    'CliffordAlgebra', 'QuaternionLinear', 'QLayerNorm',
    'DomainRotationOperator', 'InvariantExtractor', 'sandwich_transfer',
    'TitansMemoryModule',
    'HippocampusMemory', 'HBMAMemory', 'WorkingMemory', 'EpisodicMemory', 'SemanticMemory', 'ProceduralMemory',
    'HDIMPipeline', 'HDIMEncoder',
]
