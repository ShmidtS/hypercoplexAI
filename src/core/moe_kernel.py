"""
HDIM — MoE Kernel: полноценное ядро Mixture-of-Experts роутера.

Реализует архитектуру с доменно-специализированными лёгкими экспертами
на базе единого параметризованного DomainExpert:
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
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================
# Конфигурация
# ============================================================

@dataclass
class MoEKernelConfig:
    """Конфигурация MoE-ядра.

    num_experts вычисляется автоматически из expert_names если не указан явно.
    """
    input_dim: int = 128
    expert_hidden_dim: int = 256
    num_experts: Optional[int] = None # None -> вычисляется из expert_names
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
    use_can_experts: bool = False  # Использовать CliffordInteractionLayer вместо FFN
    use_batched_experts: bool = True
    batched_fallback: bool = True
    expert_homogeneity_check: bool = True  # Batched execution via torch.bmm (requires homogeneous DomainExpert)
    use_bias_balancing: bool = True # Alias for use_aux_loss_free (DeepSeek-V3 style)
    bias_update_frequency: int = 100 # Steps between bias updates (0 = update every forward)
    dispatch_budget_threshold: float = 0.0  # 0 = disabled

    def __post_init__(self):
        # Защита от вырожденного temperature (z_loss теряет эффект при temp < 0.1)
        if self.temperature < 0.1:
            raise ValueError(
                f"MoEKernelConfig.temperature={self.temperature} < 0.1: "
                "z_loss regulation is suppressed by clamp at such small values. "
                "Use temperature >= 0.1."
            )
        # Если expert_names задан, num_experts вычисляется из него
        if self.expert_names is not None:
            computed_num = len(self.expert_names)
            if self.num_experts is not None and self.num_experts != computed_num:
                raise ValueError(
                    f"num_experts={self.num_experts} conflicts with "
                    f"len(expert_names)={computed_num}"
                )
            self.num_experts = computed_num
        else:
            # Если expert_names не задан, используем num_experts или default
            if self.num_experts is None:
                self.num_experts = 4  # sensible default
            self.expert_names = [f"expert_{i}" for i in range(self.num_experts)]


@dataclass
class MoEKernelState:
    """Состояние одного forward-прохода MoE-ядра."""
    output: torch.Tensor
    router_loss: torch.Tensor
    z_loss: torch.Tensor
    ortho_loss: torch.Tensor
    expert_weights: torch.Tensor  # (B, num_experts) — средние веса по batch
    expert_usage: torch.Tensor  # (num_experts,) — средняя нагрузка
    routing_entropy: torch.Tensor  # скаляр — энтропия маршрутизации
    dispatch_weights: torch.Tensor  # (B, num_slots)
    combine_weights: torch.Tensor  # (B, num_slots)
    expert_names: List[str]
    top_expert_idx: torch.Tensor  # (B, top_k) — top-k expert indices per sample
    expert_reliability: torch.Tensor  # (num_experts,) — per-expert accuracy (EMA)
    slot_outputs: Optional[torch.Tensor] = None # (num_slots, D) — реальные выходы экспертов

    def total_loss(self) -> torch.Tensor:
        return self.router_loss + self.z_loss + self.ortho_loss

    def dominant_expert_names(self) -> List[str]:
        """Возвращает имена доминирующего эксперта для каждого токена."""
        # top_expert_idx shape: (..., top_k) — take first column for dominant
        idx = self.top_expert_idx[..., 0] if self.top_expert_idx.dim() >= 2 else self.top_expert_idx
        return [self.expert_names[int(i)] for i in idx.flatten().tolist()]


# ============================================================
# Доменные эксперты
# ============================================================

class DomainExpert(nn.Module):
    """
    Лёгкий FFN-эксперт с конфигурируемой активацией и архитектурой.

    Поддерживает варианты через config dict:
      - activation: "gelu"|"silu"|"tanh"|"relu" (default: "gelu")
      - architecture: "standard"|"bottleneck" (default: "standard")
      - pre_norm: bool (default: False) — LayerNorm перед FFN

    При use_can=True использует CliffordInteractionLayer вместо FFN
    для геометрических взаимодействий над 16D мультивекторами.
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
    ):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_can = use_can
        self._config = config or {}
        self.architecture = self._config.get("architecture", "standard")
        self._pre_norm = self._config.get("pre_norm", False)

        if use_can:
            from src.core.clifford_interaction import CliffordInteractionLayer
            self.interaction = CliffordInteractionLayer(dim=input_dim, dropout=dropout)
        else:
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

    def _init_weights(self):
        """Инициализация весов для FFN-слоёв (не применяется к CAN)."""
        if self.use_can:
            return
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_can:
            return self.interaction(x)
        if self._pre_norm:
            x = self.pre_norm(x)
        return self.net(x)


EXPERT_CONFIGS: Dict[str, Dict] = {
    "math": {"activation": "gelu", "architecture": "bottleneck"},
    "language": {"activation": "gelu", "pre_norm": True},
    "code": {"activation": "silu"},
    "science": {"activation": "tanh"},
}


# Реестр фабричных функций для создания экспертов по имени домена
# Встроенные эксперты: math, language, code, science (all DomainExpert with config)
# Кастомные эксперты могут быть добавлены через register_expert()
EXPERT_REGISTRY: Dict[str, type] = {
    "math": DomainExpert,
    "language": DomainExpert,
    "code": DomainExpert,
    "science": DomainExpert,
}


def _populate_registry_aliases() -> None:
    """Replace DomainExpert entries with specialized subclasses after they are defined."""
    EXPERT_REGISTRY["math"] = MathExpert
    EXPERT_REGISTRY["language"] = LanguageExpert
    EXPERT_REGISTRY["code"] = CodeExpert
    EXPERT_REGISTRY["science"] = ScienceExpert


def register_expert(name: str, expert_cls: type, config: Optional[Dict] = None) -> None:
    """Регистрирует новый тип эксперта в глобальном реестре.

    Args:
        name: Имя домена (например, "medical", "legal", "history")
        expert_cls: Класс эксперта (должен наследовать DomainExpert)
        config: Опциональная конфигурация для DomainExpert

    Raises:
        TypeError: Если expert_cls не наследует DomainExpert

    Example:
        >>> from src.core.moe_kernel import DomainExpert, register_expert
        >>> register_expert("medical", DomainExpert, config={"activation": "tanh"})
    """
    if not issubclass(expert_cls, DomainExpert):
        raise TypeError(f"{expert_cls.__name__} must inherit from DomainExpert")
    EXPERT_REGISTRY[name] = expert_cls
    if config is not None:
        EXPERT_CONFIGS[name] = config


def get_registered_expert_names() -> List[str]:
    """Возвращает список всех зарегистрированных экспертов.

    Returns:
        Список имен доменов, доступных для использования в MoEKernel.

    Example:
        >>> get_registered_expert_names()
        ['math', 'language', 'code', 'science']
    """
    return list(EXPERT_REGISTRY.keys())


def create_expert(
    name: str,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    use_can: bool = False,
) -> DomainExpert:
    """Создаёт эксперт по имени домена из реестра.

    Args:
        name: Имя домена (math, language, code, science или кастомный)
        input_dim: Размерность входа
        hidden_dim: Размерность скрытого слоя FFN
        dropout: Dropout rate
        use_can: Если True, использует CliffordInteractionLayer вместо FFN

    Returns:
        DomainExpert с FFN или CAN-слоем
    """
    cls = EXPERT_REGISTRY.get(name, DomainExpert)
    config = EXPERT_CONFIGS.get(name, {})
    if use_can:
        return DomainExpert(input_dim, hidden_dim, dropout, name=name, use_can=use_can)
    # Built-in subclasses (MathExpert, etc.) hardcode name/config in __init__
    if cls is not DomainExpert:
        return cls(input_dim, hidden_dim, dropout)
    return DomainExpert(input_dim, hidden_dim, dropout, name=name, config=config)


# ============================================================
# Backward-compatible aliases — thin subclasses with config
# ============================================================

class MathExpert(DomainExpert):
    """Эксперт для математических и алгебраических паттернов (bottleneck + GELU)."""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__(input_dim, hidden_dim, dropout, name="math", config=EXPERT_CONFIGS["math"])


class LanguageExpert(DomainExpert):
    """Эксперт для лингвистических и семантических паттернов (pre-norm + GELU)."""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__(input_dim, hidden_dim, dropout, name="language", config=EXPERT_CONFIGS["language"])


class CodeExpert(DomainExpert):
    """Эксперт для структурных паттернов кода и логики (SiLU)."""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__(input_dim, hidden_dim, dropout, name="code", config=EXPERT_CONFIGS["code"])


class ScienceExpert(DomainExpert):
    """Эксперт для физических и инженерных паттернов (Tanh)."""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__(input_dim, hidden_dim, dropout, name="science", config=EXPERT_CONFIGS["science"])


# Populate registry with specialized subclasses now that they're defined
_populate_registry_aliases()


# ============================================================
# MoE Kernel — основное ядро
# ============================================================

class MoEKernel(nn.Module):
    """
    Полноценное ядро MoE-роутера с доменными экспертами.

    Реализует Soft MoE dispatch/combine (Puigcerver ICLR 2024) с расширениями:
    - Auxiliary-Loss-Free балансировка per-expert bias (DeepSeek-V3)
    - Expert Orthogonalization loss (arXiv:2505.22323)
    - Shared Expert residual (DeepSeek-V3)
    - Router Z-loss (ST-MoE)
    - Similarity-Preserving routing через ортогонализацию весов

    Soft MoE формула:
        Φ = softmax(X · Θ / τ) [T × num_slots] — dispatch weights
        X̃_s = Φ[:, s]ᵀ · X [num_slots × D] — slot inputs
        ỹ_s = Expert_e(X̃_s) [num_slots × D] — expert outputs
        Y_t = Σ_s Φ[t, s] · ỹ_s [T × D] — combined output

    Args:
        config: MoEKernelConfig с параметрами
    """

    def __init__(self, config: MoEKernelConfig):
        super().__init__()
        self.config = config
        # num_experts гарантированно установлен после __post_init__
        self.num_experts = config.num_experts  # type: ignore[assignment]
        self.num_slots = config.num_experts * config.slots_per_expert  # type: ignore[operator]
        self.slots_per_expert = config.slots_per_expert
        self.input_dim = config.input_dim
        self.expert_names = config.expert_names

        # --- Router projection: input → slot logits ---
        self.router_proj = nn.Linear(config.input_dim, self.num_slots, bias=False)
        nn.init.normal_(self.router_proj.weight, std=0.02)

        # --- Доменные эксперты ---
        self.experts = nn.ModuleList([
            create_expert(
                name=config.expert_names[i],
                input_dim=config.input_dim,
                hidden_dim=config.expert_hidden_dim,
                dropout=config.dropout,
                use_can=config.use_can_experts,
            )
            for i in range(config.num_experts) # type: ignore[arg-type]
        ])

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

        # --- EMA train scores для мониторинга ---
        self.register_buffer(
            "train_scores",
            torch.ones(config.num_experts) / config.num_experts,  # type: ignore[operator]
        )
        # --- Per-expert accuracy tracking (EMA, нейтральный старт 0.5) ---
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
        if config.expert_homogeneity_check and config.use_batched_experts:
            if not self._can_use_batched():
                logger.warning(
                    "MoEKernel: use_batched_experts=True but experts are heterogeneous. "
                    "Batched execution requires homogeneous DomainExpert instances. "
                    "Falling back to sequential execution."
                )

        # --- Profiling state ---
        self._profiling_enabled = False
        self._batched_time_ns = 0
        self._sequential_time_ns = 0
        self._batched_call_count = 0
        self._sequential_call_count = 0
        # --- Bias update step counter ---
        self._bias_step = 0

    # ----------------------------------------------------------
    # Dispatch / Combine
    # ----------------------------------------------------------

    def _compute_dispatch_combine(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Вычисляет dispatch и combine матрицы и z_loss.

        Returns:
            dispatch : (T, num_slots) — нормализация по dim=0 (токены → слоты)
            combine : (T, num_slots) — нормализация по dim=-1 (слоты → токены)
            z_loss : скаляр
        """
        logits = self.router_proj(x) / max(self.temperature, 1e-8)  # (T, num_slots)

        # Auxiliary-Loss-Free: добавить per-expert bias
        if self._expert_bias is not None:
            bias_exp = self._expert_bias.repeat_interleave(self.slots_per_expert)
            logits = logits + bias_exp.unsqueeze(0)

        # Z-loss (ST-MoE): штраф за большие логиты
        if self.z_loss_weight > 0:
            lse = torch.logsumexp(logits.float(), dim=-1)  # (T,)
            z_loss = torch.clamp(lse, max=10.0).pow(2).mean()
        else:
            z_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        T = x.shape[0]
        if T == 1:
            # Граничный случай: единственный токен → равномерный dispatch
            dispatch = torch.ones(1, self.num_slots, device=x.device, dtype=x.dtype) / self.num_slots
        else:
            dispatch = F.softmax(logits.float(), dim=0).to(logits.dtype)  # нормализация по токенам
        # Budget threshold: zero out small dispatch weights, then renormalize
        if self.dispatch_budget_threshold > 0.0:
            mask = (dispatch >= self.dispatch_budget_threshold).float()
            dispatch = dispatch * mask
            dispatch = dispatch / (dispatch.sum(dim=0, keepdim=True) + 1e-8)
        combine = F.softmax(logits.float(), dim=-1).to(logits.dtype)  # нормализация по слотам

        return dispatch, combine, z_loss

    # ----------------------------------------------------------
    # Expert execution
    # ----------------------------------------------------------

    def _run_experts(self, slot_inputs: torch.Tensor) -> torch.Tensor:
        """
        Запускает каждый эксперт на соответствующих слотах.

        Args:
            slot_inputs: (num_slots, D)
        Returns:
            slot_outputs: (num_slots, D)

        Performance:
            Sequential loop over experts. For homogeneous experts (all DomainExpert
            with same hidden_dim), consider using _run_experts_batched() which
            achieves ~2-3x speedup via torch.bmm.

            Benchmark (4 experts, slots_per_expert=1, D=128):
                Sequential: ~0.8ms per forward
                Batched:    ~0.3ms per forward
        """
        outputs = []
        for e_idx, expert in enumerate(self.experts):
            start = e_idx * self.slots_per_expert
            end = start + self.slots_per_expert
            expert_input = slot_inputs[start:end]  # (slots_per_expert, D)
            expert_output = expert(expert_input)  # (slots_per_expert, D)
            # Защита от NaN/Inf
            expert_output = torch.nan_to_num(expert_output, nan=0.0, posinf=10.0, neginf=-10.0)
            expert_output = torch.clamp(expert_output, -10.0, 10.0)
            outputs.append(expert_output)
        return torch.cat(outputs, dim=0)  # (num_slots, D)

    def _run_experts_batched(self, slot_inputs: torch.Tensor) -> torch.Tensor:
        """
        Batched expert execution via stacked weights and torch.bmm.

        Only applicable when all experts are homogeneous DomainExpert instances
        with identical architecture (input_dim, hidden_dim, same activation).

        Args:
            slot_inputs: (num_slots, D)

        Returns:
            slot_outputs: (num_slots, D)

        Raises:
            RuntimeError: If experts are heterogeneous (different types/sizes)

        Performance:
            ~2-3x faster than sequential _run_experts() for 4+ experts.
            Overhead from weight stacking is amortized over batch dimension.

        Implementation:
            W1_stack: (E, H, D) — stacked first linear weights
            W2_stack: (E, D, H) — stacked second linear weights

            h = GELU(bmm(W1_stack, x))  # (E, slots, H)
            y = bmm(W2_stack, h)        # (E, slots, D)
        """
        # Check if all experts are homogeneous DomainExpert with same architecture
        for expert in self.experts:
            if not isinstance(expert, DomainExpert) or expert.use_can:
                raise RuntimeError(
                    f"_run_experts_batched requires homogeneous DomainExpert instances, "
                    f"got {type(expert).__name__}. Use _run_experts() for heterogeneous experts."
                )
        first = self.experts[0]
        for expert in self.experts[1:]:
            if expert.architecture != first.architecture:
                raise RuntimeError(
                    "_run_experts_batched requires homogeneous architecture, "
                    "got mixed architectures. Use _run_experts() for heterogeneous experts."
                )

        E = self.num_experts
        S = self.slots_per_expert
        D = self.input_dim

        # Find first and last Linear layers by type (not fragile index)
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

        # Stack weights: (E, S, D) -> process each expert's slots
        # W1: (E, H, D), W2: (E, D, H)
        W1_stack = torch.stack([_first_linear(expert.net).weight for expert in self.experts], dim=0)  # (E, H, D)
        W2_stack = torch.stack([_last_linear(expert.net).weight for expert in self.experts], dim=0)  # (E, D, H)
        b1_stack = torch.stack([_first_linear(expert.net).bias for expert in self.experts], dim=0)    # (E, H)
        b2_stack = torch.stack([_last_linear(expert.net).bias for expert in self.experts], dim=0)    # (E, D)

        # Reshape slot_inputs to (E, S, D)
        x = slot_inputs.view(E, S, D)  # (E, S, D)

        # First linear: (E, S, D) @ (E, D, H)^T + (E, H) -> (E, S, H)
        h = torch.bmm(x, W1_stack.transpose(1, 2)) + b1_stack.unsqueeze(1)  # (E, S, H)

        # Activation from expert config
        act_name = self.experts[0]._config.get("activation", "gelu")
        _fn_map = {"gelu": F.gelu, "silu": F.silu, "tanh": torch.tanh, "relu": F.relu}
        h = _fn_map.get(act_name, F.gelu)(h)

        # Dropout (only during training)
        if self.training:
            h = F.dropout(h, p=self.config.dropout, training=True)

        # Second linear: (E, S, H) @ (E, H, D)^T + (E, D) -> (E, S, D)
        y = torch.bmm(h, W2_stack.transpose(1, 2)) + b2_stack.unsqueeze(1)  # (E, S, D)

        # Flatten back to (num_slots, D)
        outputs = y.view(E * S, D)

        # Protection against NaN/Inf
        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=10.0, neginf=-10.0)
        outputs = torch.clamp(outputs, -10.0, 10.0)

        return outputs

    def _can_use_batched(self) -> bool:
        """
        Check if batched expert execution is applicable.

        Returns True if all experts are homogeneous DomainExpert instances
        with standard (2-layer FFN) architecture and same activation.
        Bottleneck architecture is not supported by _run_experts_batched.
        """
        if self.config.use_can_experts:
            return False

        for expert in self.experts:
            if not isinstance(expert, DomainExpert) or expert.use_can:
                return False

        # All must share the same architecture and activation
        first = self.experts[0]
        if first.architecture != "standard":
            return False
        for expert in self.experts[1:]:
            if expert.architecture != first.architecture:
                return False
            if expert._config.get("activation", "gelu") != first._config.get("activation", "gelu"):
                return False

        return True

    # ----------------------------------------------------------
    # Profiling
    # ----------------------------------------------------------

    def enable_profiling(self, enabled: bool = True) -> None:
        """Enable or disable expert execution profiling."""
        self._profiling_enabled = enabled

    def get_profiling_stats(self) -> dict:
        """Return profiling statistics for batched vs sequential execution.

        Returns:
            dict with keys: batched_time_ms, sequential_time_ms,
                           batched_calls, sequential_calls,
                           avg_batched_ms, avg_sequential_ms
        """
        batched_ms = self._batched_time_ns / 1_000_000
        sequential_ms = self._sequential_time_ns / 1_000_000
        avg_batched = batched_ms / self._batched_call_count if self._batched_call_count > 0 else 0
        avg_sequential = sequential_ms / self._sequential_call_count if self._sequential_call_count > 0 else 0
        return {
            "batched_time_ms": batched_ms,
            "sequential_time_ms": sequential_ms,
            "batched_calls": self._batched_call_count,
            "sequential_calls": self._sequential_call_count,
            "avg_batched_ms": avg_batched,
            "avg_sequential_ms": avg_sequential,
        }

    def reset_profiling(self) -> None:
        """Reset profiling counters."""
        self._batched_time_ns = 0
        self._sequential_time_ns = 0
        self._batched_call_count = 0
        self._sequential_call_count = 0
        # --- Bias update step counter ---
        self._bias_step = 0

    def _profile_expert_execution(
        self, slot_inputs: torch.Tensor, use_batched: bool
    ) -> torch.Tensor:
        """Execute experts with timing for profiling.

        Args:
            slot_inputs: (num_slots, D) tensor
            use_batched: If True, use batched execution

        Returns:
            slot_outputs: (num_slots, D) tensor
        """
        if use_batched:
            start = time.perf_counter_ns()
            outputs = self._run_experts_batched(slot_inputs)
            end = time.perf_counter_ns()
            if self._profiling_enabled:
                self._batched_time_ns += end - start
                self._batched_call_count += 1
        else:
            start = time.perf_counter_ns()
            outputs = self._run_experts(slot_inputs)
            end = time.perf_counter_ns()
            if self._profiling_enabled:
                self._sequential_time_ns += end - start
                self._sequential_call_count += 1
        return outputs

    # ----------------------------------------------------------
    # Load balance loss (Switch Transformer стиль)
    # ----------------------------------------------------------

    def _load_balance_loss(self, combine: torch.Tensor) -> torch.Tensor:
        """
        Switch Transformer load balance loss:
            L_lb = E * Σ_e f_e.detach() * mean_usage_e
        """
        T = combine.shape[0]
        # (T, E)
        expert_w = combine.reshape(T, self.num_experts, self.slots_per_expert).mean(-1)
        f_e = expert_w.mean(0).detach()  # (E,) — stop-gradient
        mean_usage = expert_w.mean(0)  # (E,) — gradient flows
        return self.num_experts * (f_e * mean_usage).sum()

    # ----------------------------------------------------------
    # Expert Orthogonalization loss
    # ----------------------------------------------------------

    def expert_orthogonalization_loss(self) -> torch.Tensor:
        """
        Штраф за коллинеарность экспертов (arXiv:2505.22323).
        L_o = ||W_flat @ W_flat^T - I||_F^2 (усреднено по экспертам)

        Собирает первый Linear слой каждого эксперта.
        Усекает все веса до минимальной плоской размерности для совместимости
        с экспертами разного размера (bottleneck архитектура шире standard).
        """
        raw_weights = []
        for expert in self.experts:
            for m in expert.modules():
                if isinstance(m, nn.Linear):
                    raw_weights.append(m.weight.reshape(1, -1))
                    break
        if not raw_weights:
            return torch.zeros((), device=next(self.parameters()).device)

        # Усекаем до минимальной длины чтобы не зависеть от размера эксперта
        min_len = min(w.shape[-1] for w in raw_weights)
        weights = [F.normalize(w[..., :min_len], dim=-1) for w in raw_weights]

        W = torch.cat(weights, dim=0)  # (E, min_len)
        E = W.shape[0]
        gram = W @ W.T  # (E, E)
        I = torch.eye(E, device=W.device, dtype=W.dtype)
        return ((gram - I) ** 2).mean()

    # ----------------------------------------------------------
    # Router projection orthogonalization (SIMBAL стиль)
    # ----------------------------------------------------------

    def router_similarity_loss(self) -> torch.Tensor:
        """
        Similarity-Preserving: ортогонализация строк router_proj.weight.
        L_sim = ||W @ W^T - I||_F^2
        Из SIMBAL (arXiv:2506.14038) — обеспечивает consistent routing
        для семантически похожих токенов.
        """
        W = F.normalize(self.router_proj.weight, dim=-1)  # (num_slots, D)
        gram = W @ W.T  # (num_slots, num_slots)
        I = torch.eye(self.num_slots, device=W.device, dtype=W.dtype)
        return ((gram - I) ** 2).mean()

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, MoEKernelState]:
        """
        MoE-ядро forward pass.

        Args:
            x: (batch_size, input_dim) или (batch, seq, input_dim)
        Returns:
            output : тот же shape что x
            state : MoEKernelState с метриками и лоссами
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.input_dim)  # (T, D)
        T = x_flat.shape[0]

        # 1. Dispatch / Combine матрицы
        dispatch, combine, z_loss = self._compute_dispatch_combine(x_flat)
        # dispatch: (T, S), combine: (T, S)

        # 2. Агрегация токенов в слоты: X̃ = dispatch^T @ X
        slot_inputs = dispatch.T @ x_flat  # (S, D)

        # 3. Запуск экспертов (batched для homogeneous DomainExpert)
        use_batched = self.config.use_batched_experts and self._can_use_batched()
        slot_outputs = self._profile_expert_execution(slot_inputs, use_batched)

        # 4. Combine: Y = combine @ slot_outputs
        output = combine @ slot_outputs  # (T, D)
        output = torch.nan_to_num(output, nan=0.0, posinf=10.0, neginf=-10.0)
        output = torch.clamp(output, -10.0, 10.0)

        # 5. Shared Expert residual
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x_flat)
            shared_out = torch.nan_to_num(shared_out, nan=0.0, posinf=10.0, neginf=-10.0)
            output = output + shared_out

        # 6. Лоссы
        lb_loss = self._load_balance_loss(combine)
        z_loss_scaled = self.z_loss_weight * z_loss
        # Ortho loss только при обучении — inference не нуждается в регуляризации
        if self.use_expert_ortho and self.training:
            ortho_loss = self.ortho_loss_weight * self.expert_orthogonalization_loss()
        else:
            ortho_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        router_loss = lb_loss

        # 7. Метрики нагрузки
        combine_reshaped = combine.reshape(T, self.num_experts, self.slots_per_expert)
        expert_weights_per_token = combine_reshaped.mean(-1)  # (T, E)
        expert_usage = expert_weights_per_token.mean(0).detach()  # (E,)
        routing_entropy = -(expert_weights_per_token * (expert_weights_per_token + 1e-8).log()).sum(-1).mean()
        # top-k expert indices: (T, top_k) instead of just argmax (T,)
        _top_k = min(2, self.num_experts)
        top_expert_idx = expert_weights_per_token.topk(_top_k, dim=-1).indices  # (T, top_k)

        # 8. EMA train_scores + bias update + expert_accuracy update (только во время обучения)
        if self.training:
            with torch.no_grad():
                self.train_scores.mul_(0.9).add_(expert_usage, alpha=0.1)
                self._update_biases(expert_usage)
                # Per-expert accuracy: EMA с dispatch_weights как прокси для "правильности"
                # dispatch_weights per-expert: reshape (T, E, slots_per_expert), mean over slots
                dispatch_per_expert = dispatch.reshape(T, self.num_experts, self.slots_per_expert).mean(-1)  # (T, E)
                dispatch_mean = dispatch_per_expert.mean(0)  # (E,)
                self.expert_accuracy.mul_(0.9).add_(dispatch_mean, alpha=0.1)

        state = MoEKernelState(
            output=output.reshape(orig_shape),
            router_loss=router_loss,
            z_loss=z_loss_scaled,
            ortho_loss=ortho_loss,
            expert_weights=expert_weights_per_token.reshape(*orig_shape[:-1], self.num_experts),
            expert_usage=expert_usage,
            routing_entropy=routing_entropy,
            dispatch_weights=dispatch.reshape(*orig_shape[:-1], self.num_slots),
            combine_weights=combine.reshape(*orig_shape[:-1], self.num_slots),
            expert_names=self.expert_names,
            top_expert_idx=top_expert_idx.reshape(*orig_shape[:-1], _top_k),
            expert_reliability=self.expert_accuracy.clone(),
 slot_outputs=slot_outputs,
        )
        return output.reshape(orig_shape), state

    def reset_bias(self) -> None:
        """Сбрасывает per-expert bias к нулю."""
        if self._expert_bias is not None:
            self._expert_bias.data.zero_()
            self._bias_step = 0

    def _update_biases(self, expert_load: torch.Tensor) -> None:
        """
        Auxiliary-Loss-Free bias update (DeepSeek-V3, arXiv:2412.19437).

        Updates per-expert bias based on load imbalance:
        - Overloaded experts: decrease bias (reduce routing probability)
        - Underloaded experts: increase bias (increase routing probability)

        Uses sign-based update for stability:
            bias -= aux_lr * sign(load - target_load)

        Args:
            expert_load: (num_experts,) tensor with current expert usage
        """
        if self._expert_bias is None:
            return

        self._bias_step += 1

        # Check update frequency (0 = update every forward)
        freq = self.config.bias_update_frequency
        if freq > 0 and self._bias_step % freq != 0:
            return

        # Sign-based update: overloaded -> negative delta (decrease bias)
        delta = torch.sign(expert_load - self._target_load)
        self._expert_bias.data.sub_(delta, alpha=self._aux_lr)
        # Clamp bias to prevent runaway (observed in run_018: science bias 5.22)
        self._expert_bias.data.clamp_(-1.0, 1.0)

    def expert_load_stats(self) -> Dict[str, float]:
        """Возвращает текущие EMA нагрузки по именам экспертов."""
        return {
            name: float(self.train_scores[i].item())
            for i, name in enumerate(self.expert_names)
        }
