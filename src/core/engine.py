from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn

from .algebra import CliffordAlgebra
from .invariant_index import InvariantIndex
from .invariants import InvariantExtractor, sandwich_transfer
from .rotors import DomainRotationOperator
from .types import AnalogyMatch


@dataclass
class CoreEngineConfig:
    input_dim: int = 64
    clifford_p: int = 3
    clifford_q: int = 1
    clifford_r: int = 0
    domain_names: tuple[str, ...] = ("source", "target")
    dropout: float = 0.1
    use_text_adapter: bool = False


class HDIMCoreEngine(nn.Module):
    def __init__(self, config: CoreEngineConfig):
        super().__init__()
        self.config = config
        self.algebra = CliffordAlgebra(p=config.clifford_p, q=config.clifford_q, r=config.clifford_r)
        self.encoder = nn.Linear(config.input_dim, self.algebra.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.domain_rotors = nn.ModuleDict({
            name: DomainRotationOperator(self.algebra, domain_name=name)
            for name in config.domain_names
        })
        self.extractor = InvariantExtractor(self.algebra)
        self.index = InvariantIndex()
        self._text_adapter = None

    def encode(self, problem: Union[torch.Tensor, str]) -> torch.Tensor:
        """problem -> G (multivector in Clifford algebra)"""
        if isinstance(problem, str):
            if self._text_adapter is None:
                raise TypeError(
                    "String input requires a text adapter. Set use_text_adapter=True or provide an adapter."
                )
            problem = self._text_adapter(problem)

        if not torch.is_tensor(problem):
            raise TypeError("problem must be a torch.Tensor or str")

        return self.dropout(self.encoder(problem))

    def extract(self, G: torch.Tensor, domain_rotor: Union[DomainRotationOperator, str]) -> torch.Tensor:
        """G, domain -> U_inv (structural invariant via sandwich product)"""
        rotor = self._resolve_rotor(domain_rotor)
        return self.extractor.forward(G, rotor)

    def match(self, U_inv: torch.Tensor, expert_base: Optional[InvariantIndex] = None) -> List[List[AnalogyMatch]]:
        """U_inv -> ranked analogies"""
        index = expert_base if expert_base is not None else self.index
        return index.search(U_inv)

    def transfer(self, U_inv: torch.Tensor, target_rotor: Union[DomainRotationOperator, str]) -> torch.Tensor:
        """U_inv, target -> G_target"""
        rotor = self._resolve_rotor(target_rotor)
        _, target = sandwich_transfer(
            self.algebra,
            torch.empty_like(U_inv),
            rotor,
            rotor,
            invariant_override=U_inv,
        )
        return target

    def _resolve_rotor(self, rotor: Union[DomainRotationOperator, str]) -> DomainRotationOperator:
        if isinstance(rotor, str):
            return self.domain_rotors[rotor]
        return rotor
