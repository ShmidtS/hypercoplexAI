"""
HDIM — Hypercomplex Domain Isomorphism Machine
Корневые гиперкомплексные операции: алгебра Клиффорда и кватернионные слои.

Математическая основа: Cl_{p,q,r}(R) — вырожденная алгебра Клиффорда.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ============================================================
#  1. CliffordAlgebra — вырожденная алгебра Cl_{p,q,r}(R)
# ============================================================

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
        # Валидация размерностей входных тензоров
        if a.shape[-1] != self.dim:
            raise ValueError(f"Expected a.shape[-1] == {self.dim}, got {a.shape[-1]}")
        if b.shape[-1] != self.dim:
            raise ValueError(f"Expected b.shape[-1] == {self.dim}, got {b.shape[-1]}")

        D = self.dim
        device = a.device
        signs = self.cayley_signs      # (D, D)
        indices = self.cayley_indices  # (D, D)

        # Upcast to float32 for numerical stability under AMP autocast
        if a.dtype != torch.float32:
            a = a.float()
            b = b.float()

        # a: (..., D), b: (..., D)
        # outer: (..., D, D)
        outer = a.unsqueeze(-1) * b.unsqueeze(-2)  # (..., D, D)
        weighted = outer * signs  # (..., D, D)

        # scatter по indices
        result = torch.zeros(*a.shape, dtype=a.dtype, device=device)
        # Flatten D,D -> D*D, scatter_add
        flat_weighted = weighted.reshape(*a.shape[:-1], D * D)
        flat_indices = indices.reshape(D * D)
        result.scatter_add_(-1, flat_indices.expand(*a.shape[:-1], D * D), flat_weighted)
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
        # Validate dimensions
        if a.shape[-1] != self.dim:
            raise ValueError(f"Expected a.shape[-1] == {self.dim}, got {a.shape[-1]}")
        if b.shape[-1] != self.dim:
            raise ValueError(f"Expected b.shape[-1] == {self.dim}, got {b.shape[-1]}")

        D = self.dim
        device = a.device
        signs = self.cayley_signs  # (D, D)
        indices = self.cayley_indices  # (D, D)

        # Upcast to float32 for numerical stability
        if a.dtype != torch.float32:
            a = a.float()
        if b.dtype != torch.float32:
            b = b.float()

        # a: (..., D), b: (N, ..., D)
        # We need to compute geometric_product(a, b[i]) for all i
        # Broadcast a to (N, ..., D)
        a_broadcast = a.unsqueeze(0)  # (1, ..., D)

        # outer: (N, ..., D, D)
        outer = a_broadcast.unsqueeze(-1) * b.unsqueeze(-2)  # (N, ..., D, D)
        weighted = outer * signs  # (N, ..., D, D)

        # Scatter by indices
        result_shape = b.shape  # (N, ..., D)
        result = torch.zeros(result_shape, dtype=torch.float32, device=device)

        # Flatten D,D -> D*D for scatter
        flat_weighted = weighted.reshape(*b.shape[:-1], D * D)  # (N, ..., D*D)
        flat_indices = indices.reshape(D * D)  # (D*D,)
        result.scatter_add_(-1, flat_indices.expand(*b.shape[:-1], D * D), flat_weighted)

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


# ============================================================
#  2. QuaternionLinear — кватернионный линейный слой
# ============================================================

class QuaternionLinear(nn.Module):
    """
    Drop-in замена для nn.Linear, использующий кватернионное (Hamilton) умножение.

    Вход x делится на 4 компоненты: [r, i, j, k].
    Веса также хранятся как 4 компоненты.

    Hamilton product:
      (a + bi + cj + dk)(e + fi + gj + hk) =
        (ae - bf - cg - dh) +
        (af + be + ch - dg)i +
        (ag - bh + ce + df)j +
        (ah + bg - cf + de)k
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        assert in_features % 4 == 0 and out_features % 4 == 0, \
            "in_features и out_features должны быть кратны 4"

        self.in_features = in_features
        self.out_features = out_features
        self.in_q = in_features // 4
        self.out_q = out_features // 4

        # 4 весовые матрицы (компоненты кватерниона)
        self.Wr = nn.Parameter(torch.Tensor(self.out_q, self.in_q))
        self.Wi = nn.Parameter(torch.Tensor(self.out_q, self.in_q))
        self.Wj = nn.Parameter(torch.Tensor(self.out_q, self.in_q))
        self.Wk = nn.Parameter(torch.Tensor(self.out_q, self.in_q))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self._reset_parameters()

    def _reset_parameters(self):
        """Инициализация по схеме Glorot для кватернионных слоёв."""
        std = 1.0 / math.sqrt(2 * self.in_q)
        nn.init.uniform_(self.Wr, -std, std)
        nn.init.uniform_(self.Wi, -std, std)
        nn.init.uniform_(self.Wj, -std, std)
        nn.init.uniform_(self.Wk, -std, std)

    def hamilton_product_weights(self) -> torch.Tensor:
        """
        Строит полную матрицу Hamilton-произведения из 4 компонент.
        Результирующая матрица: (out_features, in_features).
        """
        # Hamilton product matrix:
        # [[ Wr, -Wi, -Wj, -Wk],
        #  [ Wi,  Wr, -Wk,  Wj],
        #  [ Wj,  Wk,  Wr, -Wi],
        #  [ Wk, -Wj,  Wi,  Wr]]
        row1 = torch.cat([self.Wr, -self.Wi, -self.Wj, -self.Wk], dim=1)
        row2 = torch.cat([self.Wi,  self.Wr, -self.Wk,  self.Wj], dim=1)
        row3 = torch.cat([self.Wj,  self.Wk,  self.Wr, -self.Wi], dim=1)
        row4 = torch.cat([self.Wk, -self.Wj,  self.Wi,  self.Wr], dim=1)
        return torch.cat([row1, row2, row3, row4], dim=0)  # (out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_features)
        Returns:
            (..., out_features)
        """
        W = self.hamilton_product_weights()  # (out_features, in_features)
        out = F.linear(x, W, self.bias)
        return out


# ============================================================
#  5. QLayerNorm — покомпонентная нормализация кватерниона
# ============================================================

class QLayerNorm(nn.Module):
    """
    LayerNorm, применяемый независимо к каждой из 4 компонент кватерниона.

    Последнее измерение входа должно делиться на 4; каждая четверть
    (w, x, y, z) нормализуется отдельным nn.LayerNorm с собственными
    обучаемыми аффинными параметрами.

    Parameters
    ----------
    normalized_shape : int or sequence — форма, передаваемая в nn.LayerNorm
    eps              : epsilon для численной устойчивости
    """

    def __init__(self, normalized_shape, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.norms = nn.ModuleList(
            [nn.LayerNorm(normalized_shape, eps=eps) for _ in range(4)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor — последнее измерение делится на 4

        Returns
        -------
        Tensor той же формы
        """
        d = x.shape[-1]
        assert d % 4 == 0, "Last dimension must be divisible by 4"
        chunk = d // 4
        parts = [x[..., i * chunk:(i + 1) * chunk] for i in range(4)]
        normed = [self.norms[i](parts[i]) for i in range(4)]
        return torch.cat(normed, dim=-1)
