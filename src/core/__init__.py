"""
HDIM src/core — ядро гиперкомплексной архитектуры.
"""
from .hypercomplex import (
    CliffordAlgebra,
    QuaternionLinear,
    PHMLinear,
    QLayerNorm,
    hamilton_product,
)
from .domain_operators import (
    DomainRotationOperator,
    InvariantExtractor,
    DomainRegistry,
    sandwich_transfer,
)
from .titans_memory import TitansMemoryModule
from .moe_router import R3MoERouter
from .hdim_pipeline import HDIMPipeline, HDIMEncoder, HDIMDecoder

__all__ = [
    'CliffordAlgebra', 'QuaternionLinear', 'PHMLinear', 'QLayerNorm', 'hamilton_product',
    'DomainRotationOperator', 'InvariantExtractor', 'DomainRegistry', 'sandwich_transfer',
    'TitansMemoryModule', 'R3MoERouter',
    'HDIMPipeline', 'HDIMEncoder', 'HDIMDecoder',
]
