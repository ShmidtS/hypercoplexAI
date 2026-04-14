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
    MLPExpert,
    DomainExpert,
    MathExpert,
    LanguageExpert,
    CodeExpert,
    ScienceExpert,
    create_expert,
    register_expert,
    EXPERT_REGISTRY,
    EXPERT_CONFIGS,
)
from .moe_interface import MoERouter
from .moe_kernel_adapter import MoEKernelAdapter
from .soft_moe_router import SoftMoERouter
from .hallucination_detector import HallucinationDetector, HallucinationDetectionResult
from .semantic_entropy_probe import SemanticEntropyProbe
from .online_lora import (
    OnlineLoRA,
    OnlineLoRALinear,
    OnlineLoRAConv,
    OnlineLoRAConfig,
    OnlineLoRAManager,
    wrap_with_online_lora,
)
from .online_learner import OnlineLearner, ReplayBuffer, OnlineLearnerConfig

# SRP components (refactored from HDIMPipeline)
from .domain_encoder import DomainEncoder
from .invariant_processor import InvariantProcessor, InvariantMemoryState
from .transfer_engine import TransferEngine

# Memory interface
from .memory_interface import MemoryInterface, TitansAdapter, HBMAMemoryAdapter
from .prototype_memory import MSAMemory, PrototypeMemory, PrototypeIndex

from .nars_truth import NarsTruth

# Clifford interaction layers
from .clifford_interaction import CliffordInteractionLayer, CliffordFFN

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
    'MLPExpert', 'DomainExpert', 'MathExpert', 'LanguageExpert', 'CodeExpert', 'ScienceExpert',
    'create_expert', 'register_expert', 'EXPERT_REGISTRY', 'EXPERT_CONFIGS',
    'MoERouter', 'MoEKernelAdapter', 'SoftMoERouter',
    # Hallucination detection
    'HallucinationDetector', 'HallucinationDetectionResult', 'SemanticEntropyProbe',
    # Online-LoRA
    'OnlineLoRA', 'OnlineLoRALinear', 'OnlineLoRAConv', 'OnlineLoRAConfig', 'OnlineLoRAManager',
    'wrap_with_online_lora',
    # Online Learner
    'OnlineLearner', 'ReplayBuffer', 'OnlineLearnerConfig',
    # SRP Components
    'DomainEncoder', 'InvariantProcessor', 'InvariantMemoryState', 'TransferEngine',
    # Memory interface
    'MemoryInterface', 'TitansAdapter', 'HBMAMemoryAdapter', 'MSAMemory', 'PrototypeMemory', 'PrototypeIndex',
    # Clifford interaction layers
    'CliffordInteractionLayer', 'CliffordFFN',
    # NARS truth
    'NarsTruth',
]
