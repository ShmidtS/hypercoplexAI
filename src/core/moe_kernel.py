"""
HDIM — MoE Kernel: полноценное ядро Mixture-of-Experts роутера.

Реализует архитектуру с доменно-специализированными лёгкими экспертами:
 - MathExpert: алгебраические и численные паттерны
 - LanguageExpert: лингвистические и семантические паттерны
 - CodeExpert: структурные и логические паттерны программирования
 - ScienceExpert: физические и инженерные паттерны

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

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __post_init__(self):
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
    top_expert_idx: torch.Tensor  # (B,) — индекс наиболее используемого эксперта

    def total_loss(self) -> torch.Tensor:
        return self.router_loss + self.z_loss + self.ortho_loss

    def dominant_expert_names(self) -> List[str]:
        """Возвращает имена доминирующего эксперта для каждого токена."""
        return [self.expert_names[int(i)] for i in self.top_expert_idx.tolist()]


# ============================================================
# Доменные эксперты
# ============================================================

class DomainExpert(nn.Module):
    """
    Лёгкий FFN-эксперт с активацией GELU и Dropout.
    Архитектура: Linear(D→H) → GELU → Dropout → Linear(H→D)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        name: str = "expert",
    ):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MathExpert(DomainExpert):
    """
    Эксперт для математических и алгебраических паттернов.
    Использует двухуровневый bottleneck: input→hidden*2→hidden→input.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__(input_dim, hidden_dim, dropout, name="math")
        # Переопределяем net с расширенной промежуточной размерностью
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self._init_weights()


class LanguageExpert(DomainExpert):
    """
    Эксперт для лингвистических и семантических паттернов.
    Использует нормализацию для устойчивости к разным шкалам текста.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__(input_dim, hidden_dim, dropout, name="language")
        self.pre_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.pre_norm(x))


class CodeExpert(DomainExpert):
    """
    Эксперт для структурных паттернов кода и логики.
    Использует SiLU (Swish) вместо GELU для более жёстких нелинейностей.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__(input_dim, hidden_dim, dropout, name="code")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self._init_weights()


class ScienceExpert(DomainExpert):
    """
    Эксперт для физических и инженерных паттернов.
    Использует Tanh для ограниченного диапазона активаций (физические величины).
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__(input_dim, hidden_dim, dropout, name="science")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self._init_weights()


# Реестр фабричных функций для создания экспертов по имени домена
# Встроенные эксперты: math, language, code, science
# Кастомные эксперты могут быть добавлены через register_expert()
EXPERT_REGISTRY: Dict[str, type] = {
    "math": MathExpert,
    "language": LanguageExpert,
    "code": CodeExpert,
    "science": ScienceExpert,
}


def register_expert(name: str, expert_cls: type) -> None:
    """Регистрирует новый тип эксперта в глобальном реестре.

    Args:
        name: Имя домена (например, "medical", "legal", "history")
        expert_cls: Класс эксперта (должен наследовать DomainExpert)

    Raises:
        TypeError: Если expert_cls не наследует DomainExpert

    Example:
        >>> from src.core.moe_kernel import DomainExpert, register_expert
        >>> class MedicalExpert(DomainExpert):
        ...     def __init__(self, input_dim, hidden_dim, dropout=0.1):
        ...         super().__init__(input_dim, hidden_dim, dropout, name="medical")
        ...         self.net = nn.Sequential(
        ...             nn.Linear(input_dim, hidden_dim),
        ...             nn.Tanh(),
        ...             nn.Dropout(dropout),
        ...             nn.Linear(hidden_dim, input_dim),
        ...         )
        >>> register_expert("medical", MedicalExpert)
    """
    if not issubclass(expert_cls, DomainExpert):
        raise TypeError(f"{expert_cls.__name__} must inherit from DomainExpert")
    EXPERT_REGISTRY[name] = expert_cls


def get_registered_expert_names() -> List[str]:
    """Возвращает список всех зарегистрированных экспертов.

    Returns:
        Список имен доменов, доступных для использования в MoEKernel.

    Example:
        >>> get_registered_expert_names()
        ['math', 'language', 'code', 'science']
    """
    return list(EXPERT_REGISTRY.keys())


def create_expert(name: str, input_dim: int, hidden_dim: int, dropout: float) -> DomainExpert:
    """Создаёт эксперт по имени домена из реестра. При неизвестном имени — базовый DomainExpert."""
    cls = EXPERT_REGISTRY.get(name, DomainExpert)
    if cls is DomainExpert:
        return DomainExpert(input_dim, hidden_dim, dropout, name=name)
    return cls(input_dim, hidden_dim, dropout)


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
            )
            for i in range(config.num_experts)  # type: ignore[arg-type]
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
            self._expert_bias = nn.Parameter(
                torch.zeros(config.num_experts), requires_grad=False  # type: ignore[arg-type]
            )
            self._aux_lr = config.aux_lr
        else:
            self._expert_bias = None

        # --- EMA train scores для мониторинга ---
        self.register_buffer(
            "train_scores",
            torch.ones(config.num_experts) / config.num_experts,  # type: ignore[operator]
        )
        # --- Target uniform load ---
        self.register_buffer(
            "_target_load",
            torch.ones(config.num_experts) / config.num_experts,  # type: ignore[operator]
        )

        self.temperature = config.temperature
        self.z_loss_weight = config.z_loss_weight
        self.ortho_loss_weight = config.ortho_loss_weight
        self.use_expert_ortho = config.use_expert_ortho

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
        logits = self.router_proj(x) / self.temperature  # (T, num_slots)

        # Auxiliary-Loss-Free: добавить per-expert bias
        if self._expert_bias is not None:
            bias_exp = self._expert_bias.repeat_interleave(self.slots_per_expert)
            logits = logits + bias_exp.unsqueeze(0)

        # Z-loss (ST-MoE): штраф за большие логиты
        if self.z_loss_weight > 0:
            lse = torch.logsumexp(logits, dim=-1)  # (T,)
            z_loss = torch.clamp(lse, max=10.0).pow(2).mean()
        else:
            z_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        T = x.shape[0]
        if T == 1:
            # Граничный случай: единственный токен → равномерный dispatch
            dispatch = torch.ones(1, self.num_slots, device=x.device, dtype=x.dtype) / self.num_slots
        else:
            dispatch = F.softmax(logits, dim=0)  # нормализация по токенам
        combine = F.softmax(logits, dim=-1)  # нормализация по слотам

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
        с экспертами разного размера (MathExpert шире остальных).
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

        # 3. Запуск экспертов
        slot_outputs = self._run_experts(slot_inputs)  # (S, D)

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
        ortho_loss = (
            self.ortho_loss_weight * self.expert_orthogonalization_loss()
            if self.use_expert_ortho
            else torch.zeros((), device=x.device, dtype=x.dtype)
        )
        router_loss = lb_loss

        # 7. Метрики нагрузки
        combine_reshaped = combine.reshape(T, self.num_experts, self.slots_per_expert)
        expert_weights_per_token = combine_reshaped.mean(-1)  # (T, E)
        expert_usage = expert_weights_per_token.mean(0).detach()  # (E,)
        routing_entropy = -(expert_weights_per_token * (expert_weights_per_token + 1e-8).log()).sum(-1).mean()
        top_expert_idx = expert_weights_per_token.argmax(-1)  # (T,)

        # 8. EMA train_scores + bias update (только во время обучения)
        if self.training:
            with torch.no_grad():
                self.train_scores.mul_(0.9).add_(0.1 * expert_usage)
                if self._expert_bias is not None:
                    delta = torch.sign(expert_usage - self._target_load)
                    self._expert_bias.data -= self._aux_lr * delta

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
            top_expert_idx=top_expert_idx.reshape(orig_shape[:-1]),
        )
        return output.reshape(orig_shape), state

    def reset_bias(self) -> None:
        """Сбрасывает per-expert bias к нулю."""
        if self._expert_bias is not None:
            self._expert_bias.data.zero_()

    def expert_load_stats(self) -> Dict[str, float]:
        """Возвращает текущие EMA нагрузки по именам экспертов."""
        return {
            name: float(self.train_scores[i].item())
            for i, name in enumerate(self.expert_names)
        }
