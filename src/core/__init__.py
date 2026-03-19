"""HDIM src/core — ядро гиперкомплексной архитектуры."""

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
from .hbma_memory import HBMAMemory, WorkingMemory, EpisodicMemory, SemanticMemory, ProceduralMemory
from .hdim_pipeline import HDIMPipeline, HDIMEncoder, HDIMDecoder
from .transfer_state import TransferState
from .moe_kernel import (
    MoEKernel,
    MoEKernelConfig,
    MoEKernelState,
    DomainExpert,
    MathExpert,
    LanguageExpert,
    CodeExpert,
    ScienceExpert,
    create_expert,
    EXPERT_REGISTRY,
)
from .moe_interface import MoERouter
from .moe_kernel_adapter import MoEKernelAdapter
from .soft_moe_router import SoftMoERouter

# SRP components (refactored from HDIMPipeline)
from .domain_encoder import DomainEncoder
from .invariant_processor import InvariantProcessor, InvariantMemoryState
from .transfer_engine import TransferEngine

__all__ = [
    # Hypercomplex
    'CliffordAlgebra', 'QuaternionLinear', 'QLayerNorm',
    # Domain operators
    'DomainRotationOperator', 'InvariantExtractor', 'sandwich_transfer',
    # Memory
    'TitansMemoryModule',
    'HBMAMemory', 'WorkingMemory', 'EpisodicMemory', 'SemanticMemory', 'ProceduralMemory',
    # Pipeline (backward compatible)
    'HDIMPipeline', 'HDIMEncoder', 'HDIMDecoder',
    'TransferState',  # moved to separate module
    # MoE
    'MoEKernel', 'MoEKernelConfig', 'MoEKernelState',
    'DomainExpert', 'MathExpert', 'LanguageExpert', 'CodeExpert', 'ScienceExpert',
    'create_expert', 'EXPERT_REGISTRY',
    'MoERouter', 'MoEKernelAdapter', 'SoftMoERouter',
    # SRP Components
    'DomainEncoder', 'InvariantProcessor', 'InvariantMemoryState', 'TransferEngine',
]
