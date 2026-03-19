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

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .hypercomplex import CliffordAlgebra
from .domain_operators import DomainRotationOperator, InvariantExtractor


class DomainEncoder(nn.Module):
    """Кодирует вход в доменный структурный инвариант.

    Pipeline:
        x -> encoder -> g_source
        g_source -> domain_rotor -> g_rotated
        g_rotated -> invariant_extractor -> u_inv
        u_inv -> norm -> u_inv_normalized
    """

    def __init__(
        self,
        input_dim: int,
        clifford_dim: int,
        algebra: CliffordAlgebra,
        domain_names: Optional[List[str]] = None,
        use_quaternion: bool = True,
    ):
        """Инициализация DomainEncoder.

        Args:
            input_dim: размерность входного вектора
            clifford_dim: размерность мультивектора Клиффорда
            algebra: алгебра Клиффорда для операций
            domain_names: список доменов для инициализации роторов
            use_quaternion: использовать ли кватернионные слои
        """
        super().__init__()

        if domain_names is None:
            domain_names = ["source", "target"]

        self.algebra = algebra
        self.clifford_dim = clifford_dim
        self.domain_names = list(domain_names)

        # Encoder: вход -> мультивектор
        from .hdim_pipeline import HDIMEncoder
        self.encoder = HDIMEncoder(input_dim, clifford_dim, use_quaternion=use_quaternion)

        # Domain rotors: вращения для каждого домена
        self.domain_rotors = nn.ModuleDict({
            name: DomainRotationOperator(algebra, domain_name=name)
            for name in self.domain_names
        })

        # Invariant extractor: мультивектор -> инвариант
        self.invariant_extractor = InvariantExtractor(algebra)

        # Normalization: стабилизация инварианта
        self.invariant_norm = nn.LayerNorm(clifford_dim)

    def encode_domain(
        self,
        x: torch.Tensor,
        domain_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Кодирует вход и извлекает структурный инвариант для домена.

        Args:
            x: входной тензор [B, input_dim]
            domain_name: имя домена для выбора ротора

        Returns:
            g_source: мультивектор [B, clifford_dim]
            u_inv: структурный инвариант [B, clifford_dim]

        Raises:
            KeyError: если домен не найден
        """
        if domain_name not in self.domain_rotors:
            raise KeyError(
                f"Domain '{domain_name}' not found. "
                f"Available: {list(self.domain_rotors.keys())}"
            )

        # Step 1: encode input to multivector
        g_source = self.encoder(x)

        # Step 2: apply domain rotation and extract invariant
        rotor = self.domain_rotors[domain_name]
        u_inv = self.invariant_extractor(g_source, rotor)

        # Step 3: normalize invariant to prevent cascade explosion
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
        """Добавляет новый домен в runtime.

        Args:
            domain_name: уникальное имя нового домена

        Raises:
            ValueError: если домен уже существует
        """
        if domain_name in self.domain_rotors:
            raise ValueError(f"Domain '{domain_name}' already exists.")

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        new_rotor = DomainRotationOperator(self.algebra, domain_name=domain_name)
        new_rotor = new_rotor.to(device=device, dtype=dtype)

        self.domain_rotors[domain_name] = new_rotor
        self.domain_names.append(domain_name)

    def get_rotor(self, domain_name: str) -> DomainRotationOperator:
        """Возвращает ротор для домена.

        Args:
            domain_name: имя домена

        Returns:
            DomainRotationOperator для указанного домена
        """
        return self.domain_rotors[domain_name]
