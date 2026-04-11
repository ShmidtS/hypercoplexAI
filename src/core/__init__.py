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
from .maxscore_router import MaxScoreRouter, RouterCheckpoint, RouterResult
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
from .msa_attention import MSAMemory

# Continual normalization
from .continual_norm import ContinualNorm
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
    'DomainExpert', 'MathExpert', 'LanguageExpert', 'CodeExpert', 'ScienceExpert',
    'create_expert', 'EXPERT_REGISTRY',
    'MoERouter', 'MoEKernelAdapter', 'SoftMoERouter', 'MaxScoreRouter', 'RouterCheckpoint', 'RouterResult',
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
    'MemoryInterface', 'TitansAdapter', 'HBMAMemoryAdapter', 'MSAMemory',
    # Continual normalization
    'ContinualNorm',
    # Clifford interaction layers
    'CliffordInteractionLayer', 'CliffordFFN',
    # NARS truth
    'NarsTruth',
]
