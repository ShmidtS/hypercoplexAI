"""HDIM — Hypercomplex Domain Isomorphism Machine
Главный пайплайн: полный цикл кроссдоменного переноса знаний.

Архитектура HCT-MoE-R3:
 1. Encoder: вход → мультивектор Cl_{p,q,r}
 2. InvariantExtractor: мультивектор → U_inv (структурный инвариант)
 3. TitansMemory: кэширование и поиск инвариантов
 4. SoftMoERouter: маршрутизация к доменным экспертам (soft dispatch)
 5. Decoder: U_inv → выход целевого домена

Рефакторинг (SRP):
 - DomainEncoder: инкапсулирует encoder + domain_rotors + invariant_extractor
 - InvariantProcessor: инкапсулирует memory operations
 - TransferEngine: инкапсулирует MoE routing + decoder + sandwich_transfer
"""

import logging
import warnings
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .engine import CoreEngineConfig, HDIMCoreEngine
from .rotors import DomainRotationOperator
from .hypercomplex import CliffordAlgebra
from .domain_operators import sandwich_transfer
from .invariant_index import InvariantIndex
from .domain_encoder import DomainEncoder
from src.extensions.memory.invariant_processor import InvariantProcessor, InvariantMemoryState
from .transfer_engine import TransferEngine
from .transfer_state import TransferState


class _CoreTruth(SimpleNamespace):
    """Minimal truth-value placeholder for core transfer metadata."""

    def __init__(self, freq: float = 1.0, conf: float = 1.0):
        super().__init__(freq=freq, conf=conf)


class HDIMEncoder(nn.Module):
    """Кодирует входной вектор в мультивектор алгебры Клиффорда.

    Вход: (B, input_dim) → выход: (B, clifford_dim)
    """

    def __init__(self, input_dim: int, clifford_dim: int, use_quaternion: bool = True):
        super().__init__()
        self.use_quaternion = use_quaternion

        if use_quaternion and input_dim % 4 == 0 and clifford_dim % 4 == 0:
            from .hypercomplex import QuaternionLinear, QLayerNorm
            self.proj = QuaternionLinear(input_dim, clifford_dim)
            self.norm = QLayerNorm(clifford_dim // 4)
        else:
            self.proj = nn.Linear(input_dim, clifford_dim)
            self.norm = nn.LayerNorm(clifford_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class HDIMDecoder(nn.Module):
    """Декодирует мультивектор из алгебры Клиффорда в выходной вектор.

    Вход: (B, clifford_dim) → выход: (B, output_dim)
    """

    def __init__(self, clifford_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(clifford_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class _CoreMoEShim(nn.Module):
    """Minimal legacy MoE surface for HDIMModel compatibility in core mode."""

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.expert_names = [f"expert_{i}" for i in range(num_experts)]
        self.slots_per_expert = 1
        self.register_buffer("train_scores", torch.full((num_experts,), 1.0 / num_experts))

    def forward(self, u_inv: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = u_inv.shape[0]
        scores = u_inv[:, :self.num_experts]
        if scores.shape[1] < self.num_experts:
            pad = u_inv.new_zeros(batch_size, self.num_experts - scores.shape[1])
            scores = torch.cat([scores, pad], dim=1)
        gate_weights = torch.softmax(scores, dim=-1)
        topk_gate_weights, topk_idx = torch.topk(gate_weights, k=self.top_k, dim=-1)
        topk_gate_weights = topk_gate_weights / topk_gate_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        expert_usage = gate_weights.mean(dim=0)
        if self.training:
            self.train_scores.mul_(0.7).add_(expert_usage.detach().to(self.train_scores) * 0.3)
        zero = u_inv.new_tensor(0.0)
        entropy = -(gate_weights * gate_weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return u_inv, {
            "gate_weights": gate_weights,
            "topk_idx": topk_idx,
            "topk_gate_weights": topk_gate_weights,
            "router_loss": zero,
            "z_loss": zero,
            "routing_entropy": entropy,
            "train_scores_snapshot": self.train_scores.to(device=u_inv.device, dtype=u_inv.dtype),
            "expert_usage": expert_usage,
        }

    def enable_aux_loss_free(self, aux_lr: float = 0.01) -> None:
        return None

    def enable_expert_ortho(self) -> None:
        return None

    def expert_orthogonalization_loss(self) -> torch.Tensor:
        return self.train_scores.new_tensor(0.0)


class _CoreMemoryShim(nn.Module):
    """Minimal legacy memory surface for core mode."""

    def __init__(self, clifford_dim: int):
        super().__init__()
        self.memory = nn.Embedding(1, clifford_dim)
        nn.init.zeros_(self.memory.weight)
        self.register_buffer("momentum_S", torch.zeros(1, clifford_dim))
        self.use_gradient_surprise = False
        self.use_adaptive_forgetting = False

    def reset(self, strategy: str = "geometric") -> None:
        self.memory.weight.data.zero_()
        self.momentum_S.zero_()


class HDIMPipeline(nn.Module):
    """Полный пайплайн HDIM для кроссдоменного переноса знаний.

    Архитектура (composition):
    - DomainEncoder: кодирование входа в инвариант
    - InvariantProcessor: обработка через память
    - TransferEngine: MoE routing + декодирование
    """

    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 64,
        clifford_p: int = 3,
        clifford_q: int = 1,
        clifford_r: int = 0,
        domain_names: Optional[List[str]] = None,
        num_experts: Optional[int] = None,
        expert_names: Optional[List[str]] = None,
        top_k: int = 2,
        memory_key_dim: int = 32,
        memory_type: str = 'titans',
        msa_config: Optional[dict] = None,
        z_loss_weight: float = 0.0,
        n_shared_experts: int = 0,
    ):
        super().__init__()

        if domain_names is None:
            domain_names = ["source", "target"]

        valid_memory_types = ("titans", "hbma", "msa")
        if memory_type not in valid_memory_types:
            raise ValueError(
                f"Unsupported memory_type {memory_type!r}. Valid memory types: {list(valid_memory_types)}"
            )

        if memory_type != "titans" or msa_config is not None or memory_key_dim != 32:
            warnings.warn(
                "memory_type is deprecated in core mode; using InvariantIndex",
                DeprecationWarning,
                stacklevel=2,
            )

        # Compute num_experts from expert_names if provided, for compatibility metadata only.
        if expert_names is not None:
            num_experts = len(expert_names)
        elif num_experts is None:
            num_experts = 4

        engine_config = CoreEngineConfig(
            input_dim=input_dim,
            clifford_p=clifford_p,
            clifford_q=clifford_q,
            clifford_r=clifford_r,
            domain_names=tuple(domain_names),
        )
        self.engine = HDIMCoreEngine(engine_config)

        self.domain_names = list(domain_names)
        self.algebra = self.engine.algebra
        self.clifford_dim = self.algebra.dim
        self.memory_type = memory_type

        # Backward compatibility aliases. Legacy memory/MoE objects are not used in core mode.
        self.encoder = self.engine.encoder
        self.decoder = HDIMDecoder(self.clifford_dim, output_dim)
        self.domain_rotors = self.engine.domain_rotors
        self.invariant_extractor = self.engine.extractor
        self.invariant_norm = nn.Identity()
        self.moe = _CoreMoEShim(num_experts=num_experts, top_k=top_k)
        self.memory = _CoreMemoryShim(self.clifford_dim)
        self.invariant_index = self.engine.index

        self._use_gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing compatibility flag."""
        self._use_gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self._use_gradient_checkpointing = False

    def encode_domain(
        self,
        x: torch.Tensor,
        domain_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Кодирует вход и извлекает структурный инвариант для домена."""
        G = self.engine.encode(x)
        U_inv = self.engine.extract(G, domain_name)
        return G, U_inv

    def _apply_memory(
        self,
        u_inv: torch.Tensor,
        update_memory: bool,
        memory_mode: str,
    ) -> Tuple[torch.Tensor, Any]:
        """Legacy memory hook; core mode leaves invariants unchanged."""
        zero = u_inv.new_tensor(0.0)
        if update_memory and memory_mode == "update" and self.training:
            with torch.no_grad():
                mean_inv = u_inv.detach().mean(dim=0, keepdim=True).to(self.memory.memory.weight)
                self.memory.memory.weight.copy_(mean_inv[:1])
                self.memory.momentum_S.copy_(mean_inv[:1])
        return u_inv, SimpleNamespace(
            loss=zero,
            retrieved=None,
            updated=False,
            alpha=zero,
            eta=zero,
            theta=zero,
        )

    def transfer(
        self,
        x: torch.Tensor,
        source_domain: str,
        target_domain: str,
        *,
        update_memory: bool = True,
        memory_mode: str = "update",
        input_is_invariant: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Полный кроссдоменный перенос: source → target."""
        valid_memory_modes = ("none", "retrieve", "update")
        if memory_mode not in valid_memory_modes:
            raise ValueError(
                f"Unsupported memory_mode {memory_mode!r}. Valid memory modes: {list(valid_memory_modes)}"
            )

        if input_is_invariant:
            g_source = None
            u_inv = x
        else:
            g_source, u_inv = self.encode_domain(x, source_domain)

        g_target = self.engine.transfer(u_inv, target_domain)
        output = self.decoder(g_target)
        zero = u_inv.new_tensor(0.0)
        transfer_truth = _CoreTruth(freq=1.0, conf=1.0)

        state = {
            "g_source": g_source,
            "u_inv": u_inv,
            "u_mem": u_inv,
            "u_route": u_inv,
            "g_target": g_target,
            "output": output,
            "memory_loss": zero,
            "memory_retrieved": None,
            "memory_updated": False,
            "memory_alpha": zero,
            "memory_eta": zero,
            "memory_theta": zero,
            "router_state": {"g_target": g_target, "u_route": u_inv},
            "memory_mode": "none",
            "update_memory": update_memory,
            "input_is_invariant": input_is_invariant,
            "transfer_truth": transfer_truth,
            "raw_invariant": u_inv,
            "memory_augmented_invariant": u_inv,
            "exported_invariant": u_inv,
            "invariant": u_inv,
        }
        return output, state

    def forward(
        self,
        x: torch.Tensor,
        source_domain: str = "source",
        target_domain: str = "target",
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Алиас для transfer()."""
        return self.transfer(x, source_domain, target_domain, **kwargs)

    def reverse_infer(self, u_target, source_domain, target_domain):
        """Counterfactual: what source would produce this target?"""
        u_source_est = self.engine.transfer(u_target, source_domain)
        return u_source_est, _CoreTruth(freq=1.0, conf=1.0)

    def add_domain(self, domain_name: str) -> None:
        """Добавляет новый домен в pipeline в runtime.

        Args:
            domain_name: уникальное имя нового домена.
        """
        if domain_name in self.engine.domain_rotors:
            raise ValueError(f"Domain {domain_name!r} already exists")
        self.engine.domain_rotors[domain_name] = DomainRotationOperator(
            self.algebra,
            domain_name=domain_name,
        )
        self.domain_names.append(domain_name)
        self.domain_rotors = self.engine.domain_rotors

    def reset_memory(self, strategy: str = 'geometric') -> None:
        """Сбрасывает compatibility memory placeholder."""
        self.memory.reset(strategy=strategy)

    def compute_isomorphism_loss(
        self,
        domain_pairs: List[Tuple[torch.Tensor, str, str]],
    ) -> torch.Tensor:
        """Потеря изоморфизма L_iso."""
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)
        for x, domain_a, domain_b in domain_pairs:
            _, u_a = self.encode_domain(x, domain_a)
            _, u_b = self.encode_domain(x, domain_b)
            total_loss = total_loss + ((u_a - u_b) ** 2).mean()
        return total_loss / max(len(domain_pairs), 1)
