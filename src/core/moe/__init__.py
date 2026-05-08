"""HDIM — MoE subpackage: unified Mixture-of-Experts implementations.

Exports:
    MoERouter — abstract router interface
    RouterState — structured routing state type
    MLPExpert — parameterized FFN expert
    MoEKernel — domain-specific MoE with dispatch/combine
    MoEKernelConfig — configuration for MoEKernel
    MoEKernelState — forward-pass state for MoEKernel
    SoftMoERouter — soft routing without token dropping
    EXPERT_CONFIGS — built-in domain expert configurations
    load_balance_loss — Switch Transformer load balance loss
    entropy_load_balance_loss — dynamic load balance loss
    z_loss — router z-loss for stability
    aux_loss_free_update — DeepSeek-V3 bias update
    expert_orthogonalization_loss — gram-matrix ortho loss
"""

from .interface import MoERouter, RouterState
from .kernel import (
    MoEKernel,
    MoEKernelConfig,
    MoEKernelState,
    MLPExpert,
    EXPERT_CONFIGS,
    _create_mlp_expert,
)
from .soft_router import SoftMoERouter
from .utils import (
    load_balance_loss,
    entropy_load_balance_loss,
    z_loss,
    aux_loss_free_update,
    expert_orthogonalization_loss,
)

__all__ = [
    # Interface
    "MoERouter",
    "RouterState",
    # Kernel
    "MoEKernel",
    "MoEKernelConfig",
    "MoEKernelState",
    "MLPExpert",
    "EXPERT_CONFIGS",
    # Soft Router
    "SoftMoERouter",
    # Utils
    "load_balance_loss",
    "entropy_load_balance_loss",
    "z_loss",
    "aux_loss_free_update",
    "expert_orthogonalization_loss",
]
