"""
HDIM — MoE Kernel: полноценное ядро Mixture-of-Experts роутера.

Реализует архитектуру с доменно-специализированными лёгкими экспертами
на базе единого параметризованного MLPExpert:
 - math: bottleneck архитектура, GELU активация
 - language: pre-norm + GELU активация
 - code: SiLU активация
 - science: Tanh активация

Архитектура:
 Input → DomainRouter (soft dispatch) → [Expert_0..Expert_K] → combine → Output
 SharedExpert (DeepSeek-V3 стиль) → residual добавляется к output

Особенности:
 - Auxiliary-Loss-Free балансировка (DeepSeek-V3, arXiv:2412.19437)
 - Expert Orthogonalization (arXiv:2505.22323)
 - Z-loss регуляризация (ST-MoE)
 - Similarity-Preserving routing (SIMBAL, arXiv:2506.14038)
 - Soft MoE dispatch/combine (Puigcerver et al., ICLR 2024)

MoEKernel directly implements the MoERouter abstract interface,
eliminating the need for a separate adapter layer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interface import MoERouter
from .utils import (
    load_balance_loss,
    z_loss as compute_z_loss,
    aux_loss_free_update,
    expert_orthogonalization_loss,
)

logger = logging.getLogger(__name__)


# ============================================================
# Конфигурация
# ============================================================

EXPERT_CONFIGS: Dict[str, Dict] = {
    "math": {"activation": "gelu", "architecture": "bottleneck"},
    "language": {"activation": "gelu", "pre_norm": True},
    "code": {"activation": "silu"},
    "science": {"activation": "tanh"},
}


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


@dataclass
class MoEKernelState:
    """Состояние одного forward-прохода MoE-ядра."""

    output: torch.Tensor
    router_loss: torch.Tensor
    z_loss: torch.Tensor
    ortho_loss: torch.Tensor
    expert_weights: torch.Tensor
    expert_usage: torch.Tensor
    routing_entropy: torch.Tensor
    dispatch_weights: torch.Tensor
    combine_weights: torch.Tensor
    expert_names: List[str]
    top_expert_idx: torch.Tensor
    expert_reliability: torch.Tensor
    slot_outputs: Optional[torch.Tensor] = None

    def total_loss(self) -> torch.Tensor:
        return self.router_loss + self.z_loss + self.ortho_loss

    def dominant_expert_names(self) -> List[str]:
        idx = self.top_expert_idx[..., 0] if self.top_expert_idx.dim() >= 2 else self.top_expert_idx
        return [self.expert_names[int(i)] for i in idx.flatten().tolist()]

    def to_dict(self, orig_shape: torch.Size, num_experts: int, slots_per_expert: int, top_k: int) -> Dict[str, Any]:
        """Convert to MoERouter-compatible dict."""
        expert_weights = self.expert_weights.reshape(*orig_shape[:-1], num_experts)
        topk_weights, topk_indices = self.expert_weights.topk(top_k, dim=-1)
        topk_weights_norm = topk_weights / topk_weights.sum(-1, keepdim=True).clamp_min(1e-8)

        return {
            "expert_load": self.expert_usage,
            "aux_loss": self.router_loss,
            "router_loss": self.router_loss,
            "z_loss": self.z_loss,
            "ortho_loss": self.ortho_loss,
            "expert_usage": self.expert_usage,
            "routing_entropy": self.routing_entropy,
            "expert_weights": expert_weights,
            "dispatch_weights": self.dispatch_weights.reshape(*orig_shape[:-1], num_experts * slots_per_expert),
            "combine_weights": self.combine_weights.reshape(*orig_shape[:-1], num_experts * slots_per_expert),
            "expert_names": self.expert_names,
            "top_expert_idx": self.top_expert_idx.reshape(*orig_shape[:-1], top_k),
            "total_loss": self.total_loss(),
            "dominant_expert_names": self.dominant_expert_names(),
            "slot_outputs": self.slot_outputs,
            "gate_weights": expert_weights,
            "topk_idx": topk_indices.reshape(*orig_shape[:-1], top_k),
            "topk_gate_weights": topk_weights_norm.reshape(*orig_shape[:-1], top_k),
            "train_scores_snapshot": self.expert_reliability.detach().clone(),
        }


# ============================================================
# Доменные эксперты
# ============================================================

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

    ACTIVATION_MAP = {
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
    }

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


# ============================================================
# MoE Kernel — основное ядро
# ============================================================

class MoEKernel(MoERouter):
    """
    Полноценное ядро MoE-роутера с доменными экспертами.

    Directly implements the MoERouter abstract interface, so no adapter is needed.

    Реализует Soft MoE dispatch/combine (Puigcerver ICLR 2024) с расширениями:
    - Auxiliary-Loss-Free балансировка per-expert bias (DeepSeek-V3)
    - Expert Orthogonalization loss (arXiv:2505.22323)
    - Shared Expert residual (DeepSeek-V3)
    - Router Z-loss (ST-MoE)
    - Similarity-Preserving routing через ортогонализацию весов

    Args:
        config: MoEKernelConfig с параметрами
    """

    def __init__(self, config: MoEKernelConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts  # type: ignore[assignment]
        self.num_slots = config.num_experts * config.slots_per_expert  # type: ignore[operator]
        self.slots_per_expert = config.slots_per_expert
        self.input_dim = config.input_dim
        self.expert_names = config.expert_names

        # --- Router projection: input → slot logits ---
        self.router_proj = nn.Linear(config.input_dim, self.num_slots, bias=False)
        nn.init.normal_(self.router_proj.weight, std=0.02)

        # --- Доменные эксперты (direct MLPExpert, no registry) ---
        self.experts = nn.ModuleList([
            _create_mlp_expert(
                name=config.expert_names[i],
                input_dim=config.input_dim,
                hidden_dim=config.expert_hidden_dim,
                dropout=config.dropout,
                use_can=config.use_can_experts,
            )
            for i in range(config.num_experts)  # type: ignore[arg-type]
        ])

        # Cache batched availability
        self._batched_available = self._can_use_batched()

        # --- Shared Expert (DeepSeek-V3 стиль) ---
        if config.use_shared_expert:
            self.shared_expert = nn.Sequential(
                nn.Linear(config.input_dim, config.expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.expert_hidden_dim, config.input_dim),
            )
        else:
            self.shared_expert = None

        # --- Auxiliary-Loss-Free: per-expert bias ---
        if config.use_aux_loss_free:
            self.register_buffer(
                "_expert_bias", torch.zeros(config.num_experts)  # type: ignore[arg-type]
            )
            self._aux_lr = config.aux_lr
        else:
            self._expert_bias = None

        # --- EMA train scores ---
        self.register_buffer(
            "train_scores",
            torch.ones(config.num_experts) / config.num_experts,  # type: ignore[operator]
        )
        # --- Per-expert accuracy tracking (EMA) ---
        self.register_buffer(
            "expert_accuracy",
            torch.ones(config.num_experts) * 0.5,  # type: ignore[operator]
        )
        # --- Target uniform load ---
        self.register_buffer(
            "_target_load",
            torch.ones(config.num_experts) / config.num_experts,  # type: ignore[operator]
        )

        self.temperature = config.temperature
        self.dispatch_budget_threshold = config.dispatch_budget_threshold
        self.z_loss_weight = config.z_loss_weight
        self.ortho_loss_weight = config.ortho_loss_weight
        self.use_expert_ortho = config.use_expert_ortho

        # --- Heterogeneous expert warning ---
        if config.use_batched_experts and config.expert_homogeneity_check:
            if not self._batched_available:
                logger.warning(
                    "MoEKernel: use_batched_experts=True but experts are heterogeneous. "
                    "Batched execution requires homogeneous MLPExpert instances. "
                    "Falling back to sequential execution."
                )

        # --- Bias update step counter ---
        self._bias_step = 0

    # ----------------------------------------------------------
    # Dispatch / Combine
    # ----------------------------------------------------------

    def _compute_dispatch_combine(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Вычисляет dispatch и combine матрицы и z_loss."""
        logits = self.router_proj(x) / max(self.temperature, 1e-8)

        if self._expert_bias is not None:
            bias_exp = self._expert_bias.repeat_interleave(self.slots_per_expert)
            logits = logits + bias_exp.unsqueeze(0)

        # Z-loss
        _z_loss = compute_z_loss(logits, self.z_loss_weight)

        # Upcast for softmax only when needed
        if logits.dtype == torch.float16 and not torch.is_autocast_enabled():
            logits = logits.float()

        T = x.shape[0]
        if T == 1:
            dispatch = torch.ones(1, self.num_slots, device=x.device, dtype=x.dtype)
        else:
            dispatch = F.softmax(logits, dim=0)

        # Budget threshold
        if self.dispatch_budget_threshold > 0.0:
            mask = (dispatch >= self.dispatch_budget_threshold).float()
            dispatch = dispatch * mask
            dispatch = dispatch / (dispatch.sum(dim=0, keepdim=True) + 1e-8)

        combine = F.softmax(logits, dim=-1)

        return dispatch, combine, _z_loss

    # ----------------------------------------------------------
    # Expert execution
    # ----------------------------------------------------------

    def _run_experts(self, slot_inputs: torch.Tensor) -> torch.Tensor:
        """Auto-dispatch: use batched if enabled and possible, else sequential."""
        if self.config.use_batched_experts and self._batched_available:
            return self._run_experts_batched(slot_inputs)
        return self._run_experts_sequential(slot_inputs)

    def _run_experts_sequential(self, slot_inputs: torch.Tensor) -> torch.Tensor:
        """Запускает каждый эксперт на соответствующих слотах (sequential loop)."""
        num_slots = slot_inputs.shape[0]
        D = slot_inputs.shape[1]
        outputs = torch.empty(num_slots, D, device=slot_inputs.device, dtype=slot_inputs.dtype)
        for e_idx, expert in enumerate(self.experts):
            start = e_idx * self.slots_per_expert
            end = start + self.slots_per_expert
            expert_input = slot_inputs[start:end]
            expert_output = expert(expert_input)
            expert_output = torch.nan_to_num(expert_output, nan=0.0)
            outputs[start:end] = expert_output
        return outputs

    def _run_experts_batched(self, slot_inputs: torch.Tensor) -> torch.Tensor:
        """Batched expert execution via stacked weights and torch.bmm."""
        for expert in self.experts:
            if not isinstance(expert, MLPExpert) or expert.use_can:
                raise RuntimeError(
                    f"_run_experts_batched requires homogeneous MLPExpert instances, "
                    f"got {type(expert).__name__}. Use _run_experts_sequential()."
                )
        first = self.experts[0]
        for expert in self.experts[1:]:
            if expert.architecture != first.architecture:
                raise RuntimeError(
                    "_run_experts_batched requires homogeneous architecture."
                )

        E = self.num_experts
        S = self.slots_per_expert
        D = self.input_dim

        def _first_linear(seq: nn.Sequential) -> nn.Linear:
            for m in seq:
                if isinstance(m, nn.Linear):
                    return m
            raise RuntimeError("No Linear layer found in expert Sequential")

        def _last_linear(seq: nn.Sequential) -> nn.Linear:
            for m in reversed(seq):
                if isinstance(m, nn.Linear):
                    return m
            raise RuntimeError("No Linear layer found in expert Sequential")

        W1_stack = torch.stack([_first_linear(expert.net).weight for expert in self.experts], dim=0)
        W2_stack = torch.stack([_last_linear(expert.net).weight for expert in self.experts], dim=0)
        b1_stack = torch.stack([_first_linear(expert.net).bias for expert in self.experts], dim=0)
        b2_stack = torch.stack([_last_linear(expert.net).bias for expert in self.experts], dim=0)

        x = slot_inputs.view(E, S, D)

        h = torch.bmm(x, W1_stack.transpose(1, 2)) + b1_stack.unsqueeze(1)

        act_name = self.experts[0]._config.get("activation", "gelu")
        _fn_map = {"gelu": F.gelu, "silu": F.silu, "tanh": torch.tanh, "relu": F.relu}
        h = _fn_map.get(act_name, F.gelu)(h)

        if self.training:
            h = F.dropout(h, p=self.config.dropout, training=True)

        y = torch.bmm(h, W2_stack.transpose(1, 2)) + b2_stack.unsqueeze(1)

        outputs = y.view(E * S, D)
        outputs = torch.nan_to_num(outputs, nan=0.0)

        return outputs

    def _can_use_batched(self) -> bool:
        """Check if batched expert execution is applicable."""
        if self.config.use_can_experts:
            return False

        for expert in self.experts:
            if not isinstance(expert, MLPExpert) or expert.use_can:
                return False
            if getattr(expert, "_pre_norm", False):
                return False
            if expert.architecture == "bottleneck":
                return False

        first = self.experts[0]
        if first.architecture != "standard":
            return False
        for expert in self.experts[1:]:
            if expert.architecture != first.architecture:
                return False
            if expert._config.get("activation", "gelu") != first._config.get("activation", "gelu"):
                return False

        for expert in self.experts:
            linear_count = sum(1 for m in expert.net.modules() if isinstance(m, nn.Linear))
            if linear_count > 2:
                return False

        return True

    # ----------------------------------------------------------
    # Router projection orthogonalization (SIMBAL)
    # ----------------------------------------------------------

    def router_similarity_loss(self) -> torch.Tensor:
        """Similarity-Preserving: ортогонализация строк router_proj.weight."""
        W = F.normalize(self.router_proj.weight, dim=-1)
        gram = W @ W.T
        I = torch.eye(self.num_slots, device=W.device, dtype=W.dtype)
        return ((gram - I) ** 2).mean()

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """MoE-ядро forward pass.

        Implements MoERouter.forward() returning (output, info_dict).
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.input_dim)
        T = x_flat.shape[0]

        # 1. Dispatch / Combine
        dispatch, combine, _z_loss = self._compute_dispatch_combine(x_flat)

        # 2. Агрегация токенов в слоты
        slot_inputs = dispatch.T @ x_flat

        # 3. Запуск экспертов
        slot_outputs = self._run_experts(slot_inputs)

        # 4. Combine
        output = combine @ slot_outputs
        output = torch.nan_to_num(output, nan=0.0)

        # 5. Shared Expert residual
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x_flat)
            shared_out = torch.nan_to_num(shared_out, nan=0.0)
            output = output + shared_out

        # 6. Лоссы
        lb_loss = load_balance_loss(combine, self.num_experts, self.slots_per_expert)
        z_loss_scaled = self.z_loss_weight * _z_loss
        if self.use_expert_ortho and self.training:
            ortho_loss = self.ortho_loss_weight * expert_orthogonalization_loss(self.experts)
        else:
            ortho_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        router_loss = lb_loss

        # 7. Метрики нагрузки
        combine_reshaped = combine.reshape(T, self.num_experts, self.slots_per_expert)
        expert_weights_per_token = combine_reshaped.sum(-1)
        expert_usage = expert_weights_per_token.mean(0).detach()
        routing_entropy = -(expert_weights_per_token * (expert_weights_per_token + 1e-8).log()).sum(-1).mean()
        _top_k = min(2, self.num_experts)
        top_expert_idx = expert_weights_per_token.topk(_top_k, dim=-1).indices

        # 8. EMA + bias update
        if self.training:
            with torch.no_grad():
                self.train_scores.mul_(0.9).add_(expert_usage, alpha=0.1)
                self._bias_step = aux_loss_free_update(
                    self._expert_bias,
                    expert_usage,
                    self._target_load,
                    aux_lr=self._aux_lr,
                    bias_update_frequency=self.config.bias_update_frequency,
                    bias_step=self._bias_step,
                ) if self._expert_bias is not None else self._bias_step + 1
                # Per-expert accuracy EMA
                dispatch_per_expert = dispatch.reshape(T, self.num_experts, self.slots_per_expert).mean(-1)
                dispatch_mean = dispatch_per_expert.mean(0)
                self.expert_accuracy.mul_(0.9).add_(dispatch_mean, alpha=0.1)

        # Build MoERouter-compatible dict
        expert_weights = expert_weights_per_token.reshape(*orig_shape[:-1], self.num_experts)
        topk_weights, topk_indices = expert_weights_per_token.topk(_top_k, dim=-1)
        topk_weights_norm = topk_weights / topk_weights.sum(-1, keepdim=True).clamp_min(1e-8)

        info: Dict[str, Any] = {
            "expert_load": expert_usage,
            "aux_loss": router_loss,
            "router_loss": router_loss,
            "z_loss": z_loss_scaled,
            "ortho_loss": ortho_loss,
            "expert_usage": expert_usage,
            "routing_entropy": routing_entropy,
            "expert_weights": expert_weights,
            "dispatch_weights": dispatch.reshape(*orig_shape[:-1], self.num_slots),
            "combine_weights": combine.reshape(*orig_shape[:-1], self.num_slots),
            "expert_names": self.expert_names,
            "top_expert_idx": top_expert_idx.reshape(*orig_shape[:-1], _top_k),
            "total_loss": router_loss + z_loss_scaled + ortho_loss,
            "slot_outputs": slot_outputs,
            "gate_weights": expert_weights,
            "topk_idx": topk_indices.reshape(*orig_shape[:-1], _top_k),
            "topk_gate_weights": topk_weights_norm.reshape(*orig_shape[:-1], _top_k),
            "train_scores_snapshot": self.train_scores.detach().clone(),
            "expert_reliability": self.expert_accuracy.detach(),
        }

        return output.reshape(orig_shape), info

    def get_expert_load(self) -> torch.Tensor:
        """Return current expert load statistics (EMA train_scores)."""
        return self.train_scores.clone()

    def reset_training_state(self) -> None:
        """Reset EMA train_scores and bias."""
        n = self.num_experts
        self.train_scores.fill_(1.0 / n)
        self.reset_bias()

    def reset_bias(self) -> None:
        """Сбрасывает per-expert bias к нулю."""
        if self._expert_bias is not None:
            self._expert_bias.data.zero_()
            self._bias_step = 0

    def expert_orthogonalization_loss(self) -> torch.Tensor:
        """Return orthogonalization loss for expert diversity."""
        return expert_orthogonalization_loss(self.experts)

    def expert_load_stats(self) -> Dict[str, float]:
        """Возвращает текущие EMA нагрузки по именам экспертов."""
        return {
            name: float(self.train_scores[i].item())
            for i, name in enumerate(self.expert_names)
        }

    # --- Runtime feature flags ---

    def enable_shared_expert(self) -> None:
        """Enable shared expert at runtime."""
        if getattr(self, 'shared_expert', None) is not None:
            return
        self.config.use_shared_expert = True
        input_dim = self.config.input_dim
        hidden_dim = self.config.expert_hidden_dim
        self.shared_expert = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def enable_aux_loss_free(self, aux_lr: float = 0.001) -> None:
        """Enable aux-loss-free load balancing at runtime."""
        self.config.use_aux_loss_free = True
        self.config.use_bias_balancing = True
        self._aux_lr = aux_lr
        bias = getattr(self, '_expert_bias', None)
        if bias is None:
            device = next(self.parameters()).device
            self.register_buffer(
                "_expert_bias", torch.zeros(self.num_experts, device=device)
            )

    def enable_expert_ortho(self) -> None:
        """Enable expert orthogonalization loss."""
        self.use_expert_ortho = True
        self.config.use_expert_ortho = True
