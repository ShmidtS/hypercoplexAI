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
from .memory import TitansMemory, TitansMemoryModule, HBMAMemory, WorkingMemory, EpisodicMemory, SemanticMemory, ProceduralMemory, NarsTruth
from .hdim_pipeline import HDIMPipeline, HDIMEncoder, HDIMDecoder
from .transfer_state import TransferState
from .moe import (
    MoEKernel,
    MoEKernelConfig,
    MoEKernelState,
    MLPExpert,
    EXPERT_CONFIGS,
    MoERouter,
    RouterState,
    SoftMoERouter,
    _create_mlp_expert,
    load_balance_loss,
    entropy_load_balance_loss,
    z_loss,
    aux_loss_free_update,
    expert_orthogonalization_loss,
)
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
from .per_domain_lora import PerDomainLoRA

# SRP components (refactored from HDIMPipeline)
from .domain_encoder import DomainEncoder
from .invariant_processor import InvariantProcessor, InvariantMemoryState
from .transfer_engine import TransferEngine

# Memory interface
from .memory import MemoryInterface, MSAMemory

# Clifford interaction layers
from .clifford_interaction import CliffordInteractionLayer, CliffordFFN

__all__ = [
    # Hypercomplex
    'CliffordAlgebra', 'QuaternionLinear', 'QLayerNorm',
    # Domain operators
    'DomainRotationOperator', 'InvariantExtractor', 'sandwich_transfer',
    # Memory
    'TitansMemory', 'TitansMemoryModule',
    'HBMAMemory', 'WorkingMemory', 'EpisodicMemory', 'SemanticMemory', 'ProceduralMemory',
    # Pipeline (backward compatible)
    'HDIMPipeline', 'HDIMEncoder', 'HDIMDecoder',
    'TransferState',  # moved to separate module
    # MoE
    'MoEKernel', 'MoEKernelConfig', 'MoEKernelState',
    'MLPExpert', 'EXPERT_CONFIGS',
    'MoERouter', 'RouterState', 'SoftMoERouter', '_create_mlp_expert',
    'load_balance_loss', 'entropy_load_balance_loss', 'z_loss',
    'aux_loss_free_update', 'expert_orthogonalization_loss',
    # Hallucination detection
    'HallucinationDetector', 'HallucinationDetectionResult', 'SemanticEntropyProbe',
    # Online-LoRA
    'OnlineLoRA', 'OnlineLoRALinear', 'OnlineLoRAConv', 'OnlineLoRAConfig', 'OnlineLoRAManager',
    'wrap_with_online_lora',
    # Online Learner
    'OnlineLearner', 'ReplayBuffer', 'OnlineLearnerConfig',
    # Per-domain LoRA
    'PerDomainLoRA',
    # SRP Components
    'DomainEncoder', 'InvariantProcessor', 'InvariantMemoryState', 'TransferEngine',
    # Memory interface
    'MemoryInterface', 'MSAMemory',
    # Clifford interaction layers
    'CliffordInteractionLayer', 'CliffordFFN',
    # NARS truth
    'NarsTruth',
]
