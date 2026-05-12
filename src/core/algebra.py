"""
HDIM — focused Clifford algebra primitives.
"""

from typing import Tuple

import torch
import torch.nn as nn


class CliffordAlgebra(nn.Module):
    """
    Вырожденная алгебра Клиффорда Cl_{p,q,r}(R).

    Базисные векторы:
      e_i^2 = +1  для i = 1..p
      e_j^2 = -1  для j = p+1..p+q
      e_k^2 =  0  для k = p+q+1..p+q+r  (нильпотентные, иерархические трансляции)

    dim = 2^n  где n = p + q + r
    """

    def __init__(self, p: int = 3, q: int = 1, r: int = 0):
        super().__init__()
        self.p = p
        self.q = q
        self.r = r
        self.n = p + q + r
        self.dim = 2 ** self.n
        self._build_metric()
        self._build_cayley_table()
        self._build_sign_buffers()

    @staticmethod
    def compute_signature(p: int, q: int, r: int = 0) -> str:
        return f"Cl({p},{q},{r})"

    def _build_metric(self):
        """Строит метрический тензор подписи (p, q, r)."""
        metric = torch.zeros(self.n)
        metric[:self.p] = 1.0
        metric[self.p:self.p + self.q] = -1.0
        # r нильпотентные остаются 0
        self.metric = metric  # shape: (n,)

        # Learnable metric scaling (CliffordNet, 2026) — Phase 22
        # Instead of rebuilding Cayley table, learn a per-blade scaling
        self.learnable_metric = nn.Parameter(torch.ones(self.dim))  # one per blade
        self.use_learnable_metric = False  # off by default

        self.signature_str = f"Cl({self.p},{self.q},{self.r})"

    def _blade_sign(self, a_idx: int, b_idx: int) -> Tuple[float, int]:
        """
        Вычисляет знак и индекс результата перестановки базисных элементов.
        Возвращает (sign, result_index) для e_a * e_b.

        Merge-based approach: insert b_bits into a_bits one at a time.
        When a duplicate b_bit is found, count the number of elements
        after it in result_bits (anticommutation swaps needed to bring it
        to the duplicate), then square via metric and remove.
        New bits are appended. Final sort into canonical order with parity.
        """
        a_bits = [i for i in range(self.n) if a_idx & (1 << i)]
        b_bits = [i for i in range(self.n) if b_idx & (1 << i)]

        sign = 1.0
        result_bits = list(a_bits)

        # Merge b_bits: duplicates get squared via metric, new bits appended
        for b in b_bits:
            if b in result_bits:
                pos = result_bits.index(b)
                num_after = len(result_bits) - pos - 1
                sign *= (-1.0) ** num_after  # anticommutation swaps
                sign *= float(self.metric[b].item())
                result_bits.remove(b)
            else:
                result_bits.append(b)

        # Sort result_bits into canonical order, tracking permutation parity
        sorted_bits, parity = self._sort_with_parity(result_bits)
        sign *= (-1.0) ** parity

        result_idx = sum(1 << bit for bit in sorted_bits)
        return sign, result_idx

    @staticmethod
    def _sort_with_parity(arr):
        """Bubble sort returning (sorted_list, parity) where parity is 0 (even) or 1 (odd)."""
        arr = list(arr)
        parity = 0
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    parity ^= 1
        return arr, parity

    def _build_cayley_table(self):
        """
        Строит таблицу Кэли: cayley[a, b] = (sign, c) для e_a * e_b = sign * e_c.
        Хранится как два тензора: signs и indices.
        """
        D = self.dim
        signs = torch.zeros(D, D)
        indices = torch.zeros(D, D, dtype=torch.long)

        for a in range(D):
            for b in range(D):
                s, c = self._blade_sign(a, b)
                signs[a, b] = s
                indices[a, b] = c

        self.register_buffer('cayley_signs', signs)    # (D, D)
        self.register_buffer('cayley_indices', indices)  # (D, D)

    def _validate_inputs(self, a: torch.Tensor, b: torch.Tensor, dim: int, sig: str) -> None:
        """Validate multivector trailing dimensions."""
        _ = sig
        if a.shape[-1] != dim:
            raise ValueError(f"Expected a.shape[-1] == {dim}, got {a.shape[-1]}")
        if b.shape[-1] != dim:
            raise ValueError(f"Expected b.shape[-1] == {dim}, got {b.shape[-1]}")

    def _compute_geometric_components(self, a: torch.Tensor, b: torch.Tensor, dim: int, sig: str) -> dict[str, torch.Tensor]:
        """Compute signed pairwise blade products before assembly."""
        _ = sig
        signs = self.cayley_signs
        indices = self.cayley_indices
        if a.dtype != torch.float32:
            a = a.float()
        if b.dtype != torch.float32:
            b = b.float()
        outer = a.unsqueeze(-1) * b.unsqueeze(-2)
        return {
            "a": a,
            "b": b,
            "weighted": outer * signs,
            "flat_indices": indices.reshape(dim * dim),
        }

    def _assemble_result(self, components: dict[str, torch.Tensor], dim: int) -> torch.Tensor:
        """Scatter signed blade components into a multivector tensor."""
        weighted = components["weighted"]
        result_shape = weighted.shape[:-1]
        result = torch.zeros(result_shape, dtype=weighted.dtype, device=weighted.device)
        flat_weighted = weighted.reshape(*result_shape[:-1], dim * dim)
        flat_indices = components["flat_indices"]
        result.scatter_add_(-1, flat_indices.expand(*result_shape[:-1], dim * dim), flat_weighted)
        return result

    def geometric_product(self, a: torch.Tensor, b: torch.Tensor, *, safe: bool = False) -> torch.Tensor:
        """
        Геометрическое произведение двух мультивекторов.

        Args:
            a: (..., dim) — мультивектор
            b: (..., dim) — мультивектор
            safe: если True, применяет nan_to_num и clamp к результату
        Returns:
            result: (..., dim)

        Формула: (a ⊗ b)_c = Σ_{i,j: e_i*e_j=±e_c} sign * a_i * b_j

        Note: arithmetic is always done in float32 to prevent fp16 overflow
        from the dim×dim outer product (Cl(p,q,r) dim=16 for p+q+r=4).
        """
        self._validate_inputs(a, b, self.dim, self.signature_str)
        D = self.dim
        components = self._compute_geometric_components(a, b, D, self.signature_str)
        result = self._assemble_result(components, D)
        if safe:
            result = torch.nan_to_num(result, nan=0.0, posinf=1e8, neginf=-1e8)
            result = torch.clamp(result, min=-1e8, max=1e8)

        # Learnable metric scaling (CliffordNet, 2026) — Phase 22
        if self.use_learnable_metric:
            result = result * self.learnable_metric

        # Always return float32 for numerical stability.
        # fp16 max (65504) is too small for dim×dim outer products
        # in sandwich/geometric_product chains (Cl410: dim=32, max²×32 > fp16).
        return result

    def geometric_product_batch(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Batch geometric product where b has an extra leading dimension.

        This is optimized for computing multiple geometric products in parallel,
        e.g., when computing interactions across multiple shifts in CliffordInteractionLayer.

        Args:
            a: (..., D) — multivector (e.g., (B, T, D))
            b: (N, ..., D) — N multivectors (e.g., (num_shifts, B, T, D))

        Returns:
            result: (N, ..., D) — N geometric products

        This avoids a Python loop by broadcasting a to match b's leading dimension
        and computing all products in a single tensor operation.
        """
        self._validate_inputs(a, b, self.dim, self.signature_str)
        D = self.dim
        a_broadcast = a.unsqueeze(0)
        components = self._compute_geometric_components(a_broadcast, b, D, self.signature_str)
        result = self._assemble_result(components, D)

        # Numerical stability
        result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
        result = torch.clamp(result, min=-1e8, max=1e8)

        # Learnable metric scaling
        if self.use_learnable_metric:
            result = result * self.learnable_metric

        return result

    def _build_sign_buffers(self):
        """Precompute involute and reverse sign vectors as buffers."""
        inv_signs = torch.ones(self.dim)
        rev_signs = torch.ones(self.dim)
        for i in range(self.dim):
            grade = bin(i).count('1')
            if grade % 2 == 1:
                inv_signs[i] = -1.0
            if grade % 4 in [2, 3]:
                rev_signs[i] = -1.0
        self.register_buffer('_involute_signs', inv_signs)
        self.register_buffer('_reverse_signs', rev_signs)

    def involute(self, x: torch.Tensor) -> torch.Tensor:
        """
        Главная инволюция: меняет знак у нечётных k-векторов.
        x̂ = Σ (-1)^k <x>_k
        """
        return x * self._involute_signs.to(device=x.device)

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Реверс (реверсия): меняет знак у k-векторов с k ≡ 2,3 (mod 4).
        x̃ = Σ (-1)^{k(k-1)/2} <x>_k
        """
        return x * self._reverse_signs.to(device=x.device)

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Норма мультивектора: ||x||_Cl = sqrt(<x * x̃>_0)
        Скалярная часть произведения x и его реверса.
        Возвращает magnitude (всегда >= 0); abs() берётся от scalar_part,
        знак теряется. Для inverse (где знак важен) см. sandwich() и
        domain_operators.get_inverse() — они используют scalar_part напрямую.
        """
        x_rev = self.reverse(x)
        product = self.geometric_product(x, x_rev)
        scalar_part = product[..., 0]  # Grade-0 компонент
        magnitude = torch.sqrt(torch.clamp(scalar_part.abs(), min=1e-8))
        return magnitude

    def sandwich(self, R: torch.Tensor, x: torch.Tensor, *, unit: bool = False) -> torch.Tensor:
        """
        Сэндвич-произведение: R x R^{-1}.

        R^{-1} = ~R / <R * ~R>_0 всегда. Знак квадратичной формы
        сохраняется для сигнатур Cl(p,q,0) с q>0 (timelike роторы).

        unit=True: epsilon НЕ добавляется. Для единичных роторов
          <R*~R>_0 = ±1, и R^{-1} = ~R/<R*~R>_0 корректно.
          Удовлетворяет теоремам sandwich_norm_preservation,
          sandwich_identity, sandwich_composition.

        unit=False: epsilon 1e-8 добавляется для численной стабильности.
          Для обучаемых/ненормализованных роторов, <R*~R>_0 может быть ≈0.
        """
        odd_mask = self._involute_signs.to(device=R.device) < 0
        if torch.any(R[..., odd_mask].abs() > 1e-6):
            import warnings
            warnings.warn(
                "sandwich() expected an even-grade rotor (scalar+bivector+...), got odd-grade components",
                RuntimeWarning,
                stacklevel=2,
            )

        R_rev = self.reverse(R)
        R_R_rev = self.geometric_product(R, R_rev)
        quad_form = R_R_rev[..., 0:1]  # <R * ~R>_0, shape (..., 1)
        if unit:
            denom = quad_form
        else:
            eps = torch.as_tensor(1e-8, dtype=quad_form.dtype, device=quad_form.device)
            denom = torch.sign(quad_form) * torch.clamp(quad_form.abs(), min=eps)
            denom = torch.where(quad_form == 0, eps, denom)
        R_inv = R_rev / denom
        Rx = self.geometric_product(R, x)
        result = self.geometric_product(Rx, R_inv)
        return result
