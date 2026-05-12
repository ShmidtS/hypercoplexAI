"""HDIM optional Mixture-of-Experts extension modules."""

from .config import MoEConfig
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
    "MoEConfig",
    "MoERouter",
    "RouterState",
    "MoEKernel",
    "MoEKernelConfig",
    "MoEKernelState",
    "MLPExpert",
    "EXPERT_CONFIGS",
    "SoftMoERouter",
    "load_balance_loss",
    "entropy_load_balance_loss",
    "z_loss",
    "aux_loss_free_update",
    "expert_orthogonalization_loss",
]
