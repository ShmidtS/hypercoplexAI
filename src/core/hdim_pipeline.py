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

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .hypercomplex import CliffordAlgebra
from .domain_operators import sandwich_transfer
from .memory_interface import MemoryInterface, TitansAdapter, HBMAMemoryAdapter
from .domain_encoder import DomainEncoder
from .invariant_processor import InvariantProcessor, InvariantMemoryState
from .transfer_engine import TransferEngine
from .transfer_state import TransferState


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
    ):
        super().__init__()

        if domain_names is None:
            domain_names = ["source", "target"]

        self.domain_names = domain_names
        self.algebra = CliffordAlgebra(p=clifford_p, q=clifford_q, r=clifford_r)
        clifford_dim = self.algebra.dim
        self.clifford_dim = clifford_dim

        # Compute num_experts from expert_names if provided
        if expert_names is not None:
            num_experts = len(expert_names)
        elif num_experts is None:
            num_experts = 4

        self.memory_type = memory_type

        # === Component 1: DomainEncoder ===
        self.domain_encoder = DomainEncoder(
            input_dim=input_dim,
            clifford_dim=clifford_dim,
            algebra=self.algebra,
            domain_names=domain_names,
        )

        # === Component 2: Memory + InvariantProcessor ===
        if memory_type == 'titans':
            from .titans_memory import TitansMemoryModule
            _raw_memory = TitansMemoryModule(
                key_dim=memory_key_dim,
                val_dim=clifford_dim,
                hidden_dim=memory_key_dim * 2,
            )
            self.memory: MemoryInterface = TitansAdapter(
                _raw_memory,
                clifford_dim=clifford_dim,
                memory_key_dim=memory_key_dim,
            )
        elif memory_type == 'hbma':
            from .hbma_memory import HBMAMemory
            self.memory = HBMAMemoryAdapter(HBMAMemory(hidden_dim=clifford_dim))
        else:
            from .hbma_memory import CLSMemory
            self.memory = HBMAMemoryAdapter(CLSMemory(hidden_dim=clifford_dim))

        self.invariant_processor = InvariantProcessor(self.memory)

        # === Component 3: TransferEngine ===
        self.transfer_engine = TransferEngine(
            clifford_dim=clifford_dim,
            output_dim=output_dim,
            algebra=self.algebra,
            num_experts=num_experts,
            top_k=top_k,
        )

        # Backward compatibility aliases
        self.encoder = self.domain_encoder.encoder
        self.decoder = self.transfer_engine.decoder
        self.domain_rotors = self.domain_encoder.domain_rotors
        self.invariant_extractor = self.domain_encoder.invariant_extractor
        self.invariant_norm = self.domain_encoder.invariant_norm
        self.moe = self.transfer_engine.moe

        self._use_gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on memory and MoE forward paths."""
        self._use_gradient_checkpointing = True
        self.transfer_engine.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self._use_gradient_checkpointing = False
        self.transfer_engine.disable_gradient_checkpointing()

    def encode_domain(
        self,
        x: torch.Tensor,
        domain_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Кодирует вход и извлекает структурный инвариант для домена."""
        return self.domain_encoder.encode_domain(x, domain_name)

    def _apply_memory(
        self,
        u_inv: torch.Tensor,
        update_memory: bool,
        memory_mode: str,
    ) -> Tuple[torch.Tensor, InvariantMemoryState]:
        """Применяет память к инварианту."""
        return self.invariant_processor.process(u_inv, update_memory, memory_mode)

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
        # Run entire pipeline in fp32: geometric_product returns float32
        with torch.autocast(device_type='cuda', enabled=False):
            if input_is_invariant:
                g_source = None
                u_inv = x
            else:
                g_source, u_inv = self.encode_domain(x, source_domain)

            u_mem, memory_state = self._apply_memory(
                u_inv,
                update_memory=update_memory,
                memory_mode=memory_mode,
            )

            source_rotor = self.domain_encoder.get_rotor(source_domain)
            target_rotor = self.domain_encoder.get_rotor(target_domain)

            output, router_state = self.transfer_engine.transfer(
                u_mem=u_mem,
                source_rotor=source_rotor,
                target_rotor=target_rotor,
                g_source=g_source,
                input_is_invariant=input_is_invariant,
            )

            transfer_state = TransferState(
                g_source=g_source,
                u_inv=u_inv,
                u_mem=u_mem,
                u_route=router_state.get("u_route", u_mem),
                g_target=router_state["g_target"],
                output=output,
                memory_loss=memory_state.loss,
                memory_retrieved=memory_state.retrieved,
                memory_updated=memory_state.updated,
                memory_alpha=memory_state.alpha,
                memory_eta=memory_state.eta,
                memory_theta=memory_state.theta,
                router_state=router_state,
                memory_mode=memory_mode,
                update_memory=update_memory,
                input_is_invariant=input_is_invariant,
            )
            return output, transfer_state.to_dict()

    def forward(
        self,
        x: torch.Tensor,
        source_domain: str = "source",
        target_domain: str = "target",
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Алиас для transfer()."""
        return self.transfer(x, source_domain, target_domain, **kwargs)

    def add_domain(self, domain_name: str) -> None:
        """Добавляет новый домен в pipeline в runtime.

        Args:
            domain_name: уникальное имя нового домена.
        """
        self.domain_encoder.add_domain(domain_name)
        self.domain_names = self.domain_encoder.domain_names
        # Update backward compat alias
        self.domain_rotors = self.domain_encoder.domain_rotors

    def reset_memory(self, strategy: str = 'geometric') -> None:
        """Сбрасывает stateful память."""
        self.invariant_processor.reset_memory(strategy=strategy)

    def compute_isomorphism_loss(
        self,
        domain_pairs: List[Tuple[torch.Tensor, str, str]],
    ) -> torch.Tensor:
        """Потеря изоморфизма L_iso."""
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for x, domain_a, domain_b in domain_pairs:
            _, u_a = self.encode_domain(x, domain_a)
            _, u_b = self.encode_domain(x, domain_b)
            total_loss = total_loss + ((u_a - u_b) ** 2).mean()
        return total_loss / max(len(domain_pairs), 1)
