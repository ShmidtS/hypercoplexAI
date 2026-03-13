"""
HDIM — Hypercomplex Domain Isomorphism Machine
Главный пайплайн: полный цикл кроссдоменного переноса знаний.

Архитектура HCT-MoE-R3:
  1. Encoder: вход → мультивектор Cl_{p,q,r}
  2. InvariantExtractor: мультивектор → U_inv (структурный инвариант)
  3. TitansMemory: кэширование и поиск инвариантов
  4. SoftMoERouter: маршрутизация к доменным экспертам (soft dispatch)
  5. Decoder: U_inv → выход целевого домена
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

from .hypercomplex import CliffordAlgebra, QuaternionLinear, QLayerNorm
from .domain_operators import DomainRotationOperator, InvariantExtractor, sandwich_transfer
from .titans_memory import MemoryState, TitansMemoryModule
from .soft_moe_router import SoftMoERouter


@dataclass
class TransferState:
    g_source: Optional[torch.Tensor]
    u_inv: torch.Tensor
    u_mem: torch.Tensor
    u_route: torch.Tensor
    g_target: torch.Tensor
    output: torch.Tensor
    memory_loss: torch.Tensor
    memory_retrieved: torch.Tensor
    memory_updated: bool
    memory_alpha: Optional[torch.Tensor]
    memory_eta: Optional[torch.Tensor]
    memory_theta: Optional[torch.Tensor]
    router_state: Dict[str, Any]
    memory_mode: str
    update_memory: bool
    input_is_invariant: bool

    @property
    def routing_weights(self) -> torch.Tensor:
        return self.router_state["gate_weights"]

    @property
    def raw_invariant(self) -> torch.Tensor:
        return self.u_inv

    @property
    def memory_augmented_invariant(self) -> torch.Tensor:
        return self.u_mem

    @property
    def exported_invariant(self) -> torch.Tensor:
        return self.u_route

    @property
    def invariant(self) -> torch.Tensor:
        return self.exported_invariant

    def to_dict(self) -> Dict[str, Any]:
        return {
            "g_source": self.g_source,
            "u_inv": self.u_inv,
            "u_mem": self.u_mem,
            "u_route": self.u_route,
            "g_target": self.g_target,
            "output": self.output,
            "memory_loss": self.memory_loss,
            "memory_retrieved": self.memory_retrieved,
            "memory_updated": self.memory_updated,
            "memory_alpha": self.memory_alpha,
            "memory_eta": self.memory_eta,
            "memory_theta": self.memory_theta,
            "router_state": self.router_state,
            "routing_weights": self.routing_weights,
            "memory_mode": self.memory_mode,
            "update_memory": self.update_memory,
            "input_is_invariant": self.input_is_invariant,
            "raw_invariant": self.raw_invariant,
            "memory_augmented_invariant": self.memory_augmented_invariant,
            "exported_invariant": self.exported_invariant,
            "invariant": self.invariant,
        }


class HDIMEncoder(nn.Module):
    """
    Кодирует входной вектор в мультивектор алгебры Клиффорда.

    Вход: (B, input_dim) → выход: (B, clifford_dim)
    """

    def __init__(self, input_dim: int, clifford_dim: int, use_quaternion: bool = True):
        super().__init__()
        self.use_quaternion = use_quaternion

        if use_quaternion and input_dim % 4 == 0 and clifford_dim % 4 == 0:
            self.proj = QuaternionLinear(input_dim, clifford_dim)
            self.norm = QLayerNorm(clifford_dim // 4)
        else:
            self.proj = nn.Linear(input_dim, clifford_dim)
            self.norm = nn.LayerNorm(clifford_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class HDIMDecoder(nn.Module):
    """
    Декодирует мультивектор из алгебры Клиффорда в выходной вектор.

    Вход: (B, clifford_dim) → выход: (B, output_dim)
    """

    def __init__(self, clifford_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(clifford_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class HDIMPipeline(nn.Module):
    """
    Полный пайплайн HDIM для кроссдоменного переноса знаний.
    """

    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 64,
        clifford_p: int = 3,
        clifford_q: int = 1,
        clifford_r: int = 0,
        domain_names: Optional[List[str]] = None,
        num_experts: int = 4,
        top_k: int = 2,
        memory_key_dim: int = 32,
    ):
        super().__init__()

        if domain_names is None:
            domain_names = ["source", "target"]
        self.domain_names = domain_names
        self.algebra = CliffordAlgebra(p=clifford_p, q=clifford_q, r=clifford_r)
        clifford_dim = self.algebra.dim
        self.clifford_dim = clifford_dim

        self.encoder = HDIMEncoder(input_dim, clifford_dim)
        self.decoder = HDIMDecoder(clifford_dim, output_dim)
        self.domain_rotors = nn.ModuleDict({
            name: DomainRotationOperator(self.algebra, domain_name=name)
            for name in domain_names
        })
        self.invariant_extractor = InvariantExtractor(self.algebra)
        # LayerNorm после инварианта — критично для стабильности geometric_product цепочки
        self.invariant_norm = nn.LayerNorm(clifford_dim)
        self.memory = TitansMemoryModule(
            key_dim=memory_key_dim,
            val_dim=clifford_dim,
            hidden_dim=memory_key_dim * 2,
        )
        self.memory_key_proj = nn.Linear(clifford_dim, memory_key_dim)
        self.moe = SoftMoERouter(
            input_dim=clifford_dim,
            num_experts=num_experts,
            expert_dim=clifford_dim * 2,
            top_k=top_k,
        )

    def encode_domain(
        self,
        x: torch.Tensor,
        domain_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Кодирует вход и извлекает структурный инвариант для домена.
        """
        g_source = self.encoder(x)
        rotor = self.domain_rotors[domain_name]
        u_inv = self.invariant_extractor(g_source, rotor)
        # Normalize invariant to prevent geometric product cascade explosion
        u_inv = self.invariant_norm(u_inv)
        return g_source, u_inv

    def _apply_memory(
        self,
        u_inv: torch.Tensor,
        update_memory: bool,
        memory_mode: str,
    ) -> tuple[torch.Tensor, MemoryState]:
        if memory_mode not in {"none", "retrieve", "update"}:
            raise ValueError(f"Unsupported memory_mode: {memory_mode}")
        if memory_mode == "none":
            empty_state = MemoryState(
                retrieved=torch.zeros_like(u_inv),
                loss=torch.zeros((), device=u_inv.device, dtype=u_inv.dtype),
                updated=False,
            )
            return u_inv, empty_state

        mem_key = self.memory_key_proj(u_inv)
        memory_state = self.memory.retrieve_and_update(
            mem_key,
            u_inv,
            update_memory=update_memory and memory_mode == "update",
        )
        return u_inv + memory_state.retrieved, memory_state

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
        """
        Полный кроссдоменный перенос: source → target.
        """
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
        u_route, router_state = self.moe(u_mem)
        r_source = self.domain_rotors[source_domain]
        r_target = self.domain_rotors[target_domain]
        if input_is_invariant:
            g_target = r_target(u_route)
        else:
            _, g_target = sandwich_transfer(
                self.algebra,
                g_source,
                r_source,
                r_target,
                invariant_override=u_route,
            )
        output = self.decoder(g_target)

        transfer_state = TransferState(
            g_source=g_source,
            u_inv=u_inv,
            u_mem=u_mem,
            u_route=u_route,
            g_target=g_target,
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
        if domain_name in self.domain_rotors:
            raise ValueError(f"Domain '{domain_name}' already exists.")
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        new_rotor = DomainRotationOperator(self.algebra, domain_name=domain_name)
        new_rotor = new_rotor.to(device=device, dtype=dtype)
        self.domain_rotors[domain_name] = new_rotor
        self.domain_names.append(domain_name)

    def remove_domain(self, domain_name: str) -> None:
        """Удаляет домен из pipeline.

        Args:
            domain_name: имя домена для удаления.
        """
        if domain_name not in self.domain_rotors:
            raise KeyError(f"Domain '{domain_name}' not found.")
        if len(self.domain_names) <= 2:
            raise RuntimeError("Cannot remove domain: at least 2 domains required.")
        del self.domain_rotors[domain_name]
        self.domain_names.remove(domain_name)

    def reset_memory(self, strategy: str = 'geometric') -> None:
        """Сбрасывает stateful память. strategy передаётся в TitansMemoryModule."""
        self.memory.reset_memory(strategy=strategy)

    def compute_isomorphism_loss(
        self,
        domain_pairs: List[Tuple[torch.Tensor, str, str]],
    ) -> torch.Tensor:
        """
        Потеря изоморфизма L_iso.
        """
        total_loss = torch.tensor(0.0, device=self.memory_key_proj.weight.device)
        for x, domain_a, domain_b in domain_pairs:
            _, u_a = self.encode_domain(x, domain_a)
            _, u_b = self.encode_domain(x, domain_b)
            total_loss = total_loss + ((u_a - u_b) ** 2).mean()
        return total_loss / max(len(domain_pairs), 1)
