"""Expert modules for the MoE kernel."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn


EXPERT_CONFIGS: Dict[str, Dict] = {
    "math": {"activation": "gelu", "architecture": "bottleneck"},
    "language": {"activation": "gelu", "pre_norm": True},
    "code": {"activation": "silu"},
    "science": {"activation": "tanh"},
}

ACTIVATION_MAP = {
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
}


class MLPExpert(nn.Module):
    """
    Parameterized FFN-expert with configurable activation, architecture,
    and optional pre-hook transform.

    Supports variants via config dict:
      - activation: "gelu"|"silu"|"tanh"|"relu" (default: "gelu")
      - architecture: "standard"|"bottleneck" (default: "standard")
      - pre_norm: bool (default: False) — LayerNorm before FFN

    pre_hook: optional nn.Module applied before FFN (e.g., CliffordInteractionLayer).
    When use_can=True, a CliffordInteractionLayer is created as pre_hook.
    """

    ACTIVATION_MAP = ACTIVATION_MAP

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        name: str = "expert",
        use_can: bool = False,
        config: Optional[Dict] = None,
        pre_hook: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._config = config or {}
        self.architecture = self._config.get("architecture", "standard")
        self._pre_norm = self._config.get("pre_norm", False)

        if pre_hook is not None:
            self.pre_hook = pre_hook
        elif use_can:
            from src.extensions.moe.clifford_interaction import CliffordInteractionLayer
            self.pre_hook = CliffordInteractionLayer(dim=input_dim, dropout=dropout)
        else:
            self.pre_hook = None

        _can_replaces_ffn = use_can and pre_hook is None

        if not _can_replaces_ffn:
            if self._pre_norm:
                self.pre_norm = nn.LayerNorm(input_dim)

            act_name = self._config.get("activation", "gelu")
            act_cls = self.ACTIVATION_MAP.get(act_name, nn.GELU)

            if self.architecture == "bottleneck":
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim * 2),
                    act_cls(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, input_dim),
                )
            else:
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    act_cls(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, input_dim),
                )

            self._init_weights()

    @property
    def use_can(self) -> bool:
        """True if a pre_hook (CliffordInteractionLayer) is configured."""
        return self.pre_hook is not None

    def _init_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_hook is not None and not hasattr(self, 'net'):
            return self.pre_hook(x)
        if self.pre_hook is not None:
            x = self.pre_hook(x)
        if self._pre_norm:
            x = self.pre_norm(x)
        return self.net(x)


def _create_mlp_expert(
    name: str,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    use_can: bool = False,
) -> MLPExpert:
    """Create an MLPExpert by name, using EXPERT_CONFIGS for built-in domain configs."""
    config = EXPERT_CONFIGS.get(name, {})
    return MLPExpert(input_dim, hidden_dim, dropout, name=name, use_can=use_can, config=config)
