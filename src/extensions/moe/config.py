"""Configuration for optional Mixture-of-Experts extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MoEConfig:
    """Mixture of Experts configuration."""

    num_experts: Optional[int] = None
    top_k: int = 2
    n_shared_experts: int = 0
    z_loss_weight: float = 0.0
    use_aux_loss_free: bool = False
    aux_lr: float = 0.001
    use_expert_ortho: bool = False


@dataclass
class MoEKernelConfig:
    """Конфигурация MoE-ядра.

    num_experts вычисляется автоматически из expert_names если не указан явно.
    """

    input_dim: int = 128
    expert_hidden_dim: int = 256
    num_experts: Optional[int] = None  # None -> вычисляется из expert_names
    slots_per_expert: int = 1
    temperature: float = 1.0
    z_loss_weight: float = 0.01
    ortho_loss_weight: float = 0.01
    use_shared_expert: bool = True
    use_aux_loss_free: bool = True
    use_expert_ortho: bool = True
    aux_lr: float = 0.001
    dropout: float = 0.1
    expert_names: Optional[List[str]] = None
    use_can_experts: bool = False
    # Deprecated: auto-dispatch is always enabled; kept for backward compatibility with tests
    use_batched_experts: bool = True
    batched_fallback: bool = True
    expert_homogeneity_check: bool = True
    use_bias_balancing: bool = True  # Alias for use_aux_loss_free (DeepSeek-V3 style)
    bias_update_frequency: int = 100  # Steps between bias updates (0 = update every forward)
    dispatch_budget_threshold: float = 0.0  # 0 = disabled

    def __post_init__(self):
        if self.temperature < 0.1:
            raise ValueError(
                f"MoEKernelConfig.temperature={self.temperature} < 0.1: "
                "z_loss regulation is suppressed by clamp at such small values. "
                "Use temperature >= 0.1."
            )
        if self.expert_names is not None:
            computed_num = len(self.expert_names)
            if self.num_experts is not None and self.num_experts != computed_num:
                raise ValueError(
                    f"num_experts={self.num_experts} conflicts with "
                    f"len(expert_names)={computed_num}"
                )
            self.num_experts = computed_num
        else:
            if self.num_experts is None:
                self.num_experts = 4
            self.expert_names = [f"expert_{i}" for i in range(self.num_experts)]
