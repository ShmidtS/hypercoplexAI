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
from .cls_memory import HippocampusMemory, NeocortexMemory, CLSMemory, HBMAMemory, WorkingMemory, EpisodicMemory, SemanticMemory, ProceduralMemory
from .hdim_pipeline import HDIMPipeline, HDIMEncoder, HDIMDecoder

__all__ = [
    'CliffordAlgebra', 'QuaternionLinear', 'QLayerNorm',
    'DomainRotationOperator', 'InvariantExtractor', 'sandwich_transfer',
    'TitansMemoryModule',
    'HippocampusMemory', 'NeocortexMemory', 'CLSMemory', 'HBMAMemory', 'WorkingMemory', 'EpisodicMemory', 'SemanticMemory', 'ProceduralMemory',
    'HDIMPipeline', 'HDIMEncoder', 'HDIMDecoder',
]
