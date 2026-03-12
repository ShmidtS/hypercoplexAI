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
from .hdim_pipeline import HDIMPipeline, HDIMEncoder, HDIMDecoder
from .modular_moe import ModularMoERouter, ExpertConfig, ExpertModule, build_modular_moe

__all__ = [
    'CliffordAlgebra', 'QuaternionLinear', 'PHMLinear', 'QLayerNorm', 'hamilton_product',
    'DomainRotationOperator', 'InvariantExtractor', 'DomainRegistry', 'sandwich_transfer',
    'TitansMemoryModule',
    'HDIMPipeline', 'HDIMEncoder', 'HDIMDecoder',
    'ModularMoERouter', 'ExpertConfig', 'ExpertModule', 'build_modular_moe',
]
