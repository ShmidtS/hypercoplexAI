"""
HDIM — Hypercomplex Domain Isomorphism Machine
Главный пайплайн: полный цикл кроссдоменного переноса знаний.

Архитектура HCT-MoE-R3:
  1. Encoder: вход → мультивектор Cl_{p,q,r}
  2. InvariantExtractor: мультивектор → U_inv (структурный инвариант)
  3. TitansMemory: кэширование и поиск инвариантов
  4. R3MoERouter: маршрутизация к доменным экспертам
  5. Decoder: U_inv → выход целевого домена
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

from .hypercomplex import CliffordAlgebra, QuaternionLinear, QLayerNorm
from .domain_operators import DomainRotationOperator, InvariantExtractor, sandwich_transfer
from .titans_memory import TitansMemoryModule
from .moe_router import R3MoERouter


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
        self.memory = TitansMemoryModule(
            key_dim=memory_key_dim,
            val_dim=clifford_dim,
            hidden_dim=memory_key_dim * 2,
        )
        self.memory_key_proj = nn.Linear(clifford_dim, memory_key_dim)
        self.moe = R3MoERouter(
            input_dim=clifford_dim,
            num_experts=num_experts,
            top_k=top_k,
            expert_dim=clifford_dim * 2,
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
        return g_source, u_inv

    def _apply_memory(
        self,
        u_inv: torch.Tensor,
        update_memory: bool,
        memory_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if memory_mode not in {"none", "retrieve", "update"}:
            raise ValueError(f"Unsupported memory_mode: {memory_mode}")
        should_update = update_memory and memory_mode == "update"
        if memory_mode == "none":
            mem_retrieved = torch.zeros_like(u_inv)
            memory_loss = torch.zeros((), device=u_inv.device, dtype=u_inv.dtype)
            return u_inv, mem_retrieved, memory_loss
        mem_key = self.memory_key_proj(u_inv)
        mem_retrieved, memory_loss = self.memory(mem_key, u_inv, update_memory=should_update)
        return u_inv + mem_retrieved, mem_retrieved, memory_loss

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

        u_mem, mem_retrieved, memory_loss = self._apply_memory(
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
            _, g_target = sandwich_transfer(self.algebra, g_source, r_source, r_target, invariant_override=u_route)
        output = self.decoder(g_target)

        transfer_state: Dict[str, Any] = {
            "g_source": g_source,
            "u_inv": u_inv,
            "u_mem": u_mem,
            "u_route": u_route,
            "g_target": g_target,
            "memory_loss": memory_loss,
            "memory_retrieved": mem_retrieved,
            "router_state": router_state,
            "routing_weights": router_state["gate_weights"],
            "invariant": u_inv,
            "processed_invariant": u_route,
            "memory_mode": memory_mode,
            "update_memory": update_memory,
            "input_is_invariant": input_is_invariant,
        }
        return output, transfer_state

    def forward(
        self,
        x: torch.Tensor,
        source_domain: str = "source",
        target_domain: str = "target",
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Алиас для transfer()."""
        return self.transfer(x, source_domain, target_domain, **kwargs)

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
