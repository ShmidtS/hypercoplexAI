"""DomainEncoder — инкапсулирует кодирование входа в доменный инвариант.

Отвечает за:
- Кодирование входного вектора в мультивектор Клиффорда (encoder)
- Применение доменного ротора (domain_rotors)
- Извлечение структурного инварианта (invariant_extractor)
- Нормализацию инварианта (invariant_norm)

Контракт:
    encode_domain(x, domain_name) -> (g_source, u_inv)

    где:
    - x: входной тензор [B, input_dim]
    - g_source: мультивектор [B, clifford_dim]
    - u_inv: структурный инвариант [B, clifford_dim]
"""

import torch
import torch.nn as nn

from .algebra import CliffordAlgebra
from .invariants import InvariantExtractor
from .rotors import DomainRotationOperator


class DomainEncoder(nn.Module):
    """Кодирует вход в доменный структурный инвариант."""

    def __init__(
        self,
        input_dim: int,
        clifford_dim: int,
        algebra: CliffordAlgebra,
        domain_names: list[str] | None = None,
        use_quaternion: bool = True,
    ):
        super().__init__()

        if domain_names is None:
            domain_names = ["source", "target"]

        self.algebra = algebra
        self.clifford_dim = clifford_dim
        self.domain_names = list(domain_names)

        _ = use_quaternion
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, clifford_dim),
            nn.LayerNorm(clifford_dim),
        )

        self.domain_rotors = nn.ModuleDict({
            name: DomainRotationOperator(algebra, domain_name=name)
            for name in self.domain_names
        })

        with torch.no_grad():
            for rotor in self.domain_rotors.values():
                rotor.R.data = torch.nn.functional.normalize(rotor.R.data, dim=-1)

        self.invariant_extractor = InvariantExtractor(algebra)
        self.invariant_norm = nn.LayerNorm(clifford_dim)

    def encode_domain(
        self,
        x: torch.Tensor,
        domain_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Кодирует вход и извлекает структурный инвариант для домена."""
        if domain_name not in self.domain_rotors:
            raise KeyError(
                f"Domain '{domain_name}' not found. "
                f"Available: {list(self.domain_rotors.keys())}"
            )

        g_source = self.encoder(x)
        rotor = self.domain_rotors[domain_name]
        u_inv = self.invariant_extractor(g_source, rotor)
        u_inv = self.invariant_norm(u_inv)

        return g_source, u_inv

    def forward(
        self,
        x: torch.Tensor,
        domain_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Алиас для encode_domain()."""
        return self.encode_domain(x, domain_name)

    def add_domain(self, domain_name: str) -> None:
        """Добавляет новый домен в runtime."""
        if domain_name in self.domain_rotors:
            raise ValueError(f"Domain '{domain_name}' already exists.")

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        new_rotor = DomainRotationOperator(self.algebra, domain_name=domain_name)
        new_rotor = new_rotor.to(device=device, dtype=dtype)

        with torch.no_grad():
            new_rotor.R.data = torch.nn.functional.normalize(new_rotor.R.data, dim=-1)

        self.domain_rotors[domain_name] = new_rotor
        self.domain_names.append(domain_name)

    def get_rotor(self, domain_name: str) -> DomainRotationOperator:
        """Возвращает ротор для домена."""
        return self.domain_rotors[domain_name]
