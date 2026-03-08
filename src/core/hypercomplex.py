"""
HDIM — Hypercomplex Domain Isomorphism Machine
Корневые гиперкомплексные операции: алгебра Клиффорда, кватернионные слои, PHM, роторы доменов.

Математическая основа: Cl_{p,q,r}(R) — вырожденная алгебра Клиффорда.
U_inv = R^{-1} ⊗_Cl G ⊗_Cl R  — структурный инвариант.
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

    def _build_metric(self):
        """Строит метрический тензор подписи (p, q, r)."""
        metric = torch.zeros(self.n)
        metric[:self.p] = 1.0
        metric[self.p:self.p + self.q] = -1.0
        # r нильпотентные остаются 0
        self.metric = metric  # shape: (n,)

    def _blade_sign(self, a_idx: int, b_idx: int) -> Tuple[float, int]:
        """
        Вычисляет знак и индекс результата перестановки базисных элементов.
        Возвращает (sign, result_index) для e_a * e_b.
        """
        a_bits = []
        b_bits = []
        for i in range(self.n):
            if a_idx & (1 << i):
                a_bits.append(i)
            if b_idx & (1 << i):
                b_bits.append(i)

        sign = 1.0
        result_bits = list(a_bits)

        for b in b_bits:
            # Переставляем b через result_bits
            pos = len(result_bits)
            swaps = 0
            for i in range(len(result_bits) - 1, -1, -1):
                if result_bits[i] < b:
                    break
                if result_bits[i] == b:
                    # Квадрат базисного вектора
                    sign *= float(self.metric[b].item())
                    result_bits.pop(i)
                    pos = -1
                    break
                swaps += 1
            if pos != -1:
                sign *= (-1.0) ** swaps
                result_bits.insert(pos - swaps, b)

        result_idx = sum(1 << b for b in result_bits)
        return sign, result_idx

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

    def geometric_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Геометрическое произведение двух мультивекторов.

        Args:
            a: (..., dim) — мультивектор
            b: (..., dim) — мультивектор
        Returns:
            result: (..., dim)

        Формула: (a ⊗ b)_c = Σ_{i,j: e_i*e_j=±e_c} sign * a_i * b_j
        """
        D = self.dim
        device = a.device
        signs = self.cayley_signs      # (D, D)
        indices = self.cayley_indices  # (D, D)

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
        return result

    def involute(self, x: torch.Tensor) -> torch.Tensor:
        """
        Главная инволюция: меняет знак у нечётных k-векторов.
        x̂ = Σ (-1)^k <x>_k
        """
        signs = torch.ones(self.dim, device=x.device)
        for i in range(self.dim):
            grade = bin(i).count('1')
            if grade % 2 == 1:
                signs[i] = -1.0
        return x * signs

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Реверс (реверсия): меняет знак у k-векторов с k ≡ 2,3 (mod 4).
        x̃ = Σ (-1)^{k(k-1)/2} <x>_k
        """
        signs = torch.ones(self.dim, device=x.device)
        for i in range(self.dim):
            grade = bin(i).count('1')
            if grade % 4 in [2, 3]:
                signs[i] = -1.0
        return x * signs

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Норма мультивектора: ||x||_Cl = sqrt(<x * x̃>_0)
        Скалярная часть произведения x и его реверса.
        """
        x_rev = self.reverse(x)
        product = self.geometric_product(x, x_rev)
        scalar_part = product[..., 0]  # Grade-0 компонент
        return torch.sqrt(torch.clamp(scalar_part.abs(), min=1e-8))

    def sandwich(self, R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Сэндвич-произведение: R x R^{-1}, где R^{-1} = reverse(R) / ||R||².
        Корректно для произвольных (ненормализованных) верзоров.
        """
        R_rev = self.reverse(R)
        # norm возвращает (...) без последнего dim, нужно unsqueeze
        norm_sq = (self.norm(R) ** 2 + 1e-8).unsqueeze(-1)  # (..., 1)
        R_inv = R_rev / norm_sq  # broadcast по последнему dim
        Rx = self.geometric_product(R, x)
        return self.geometric_product(Rx, R_inv)


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
#  3. PHMLinear — параметризованное гиперкомплексное умножение
# ============================================================

class PHMLinear(nn.Module):
    """
    Parameterized Hypercomplex Multiplication (PHM).
    Zhang et al., "Parameterized Hypercomplex Graph Neural Networks for Graph Classification"

    Матрица весов: W = Σ_{s=1}^n A_s ⊗ S_s
    где A_s — обучаемые скалярные матрицы (n x n)
          S_s — обучаемые блочные матрицы (out/n x in/n)
    n — размерность гиперкомплексной алгебры (4 для кватернионов)
    """

    def __init__(self, n: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        assert in_features % n == 0 and out_features % n == 0, \
            f"in_features и out_features должны быть кратны n={n}"

        self.in_features  = in_features
        self.out_features = out_features
        self.n            = n

        self.A = nn.ParameterList(
            [nn.Parameter(torch.Tensor(n, n)) for _ in range(n)]
        )
        self.S = nn.ParameterList(
            [nn.Parameter(torch.Tensor(out_features // n, in_features // n)) for _ in range(n)]
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        for a in self.A:
            nn.init.eye_(a)
        for s in self.S:
            nn.init.kaiming_uniform_(s, a=math.sqrt(5))

    def kronecker_product(self, A: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        return torch.kron(A, S)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = sum(torch.kron(a, s) for a, s in zip(self.A, self.S))
        return F.linear(x, W, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"n={self.n}, "
            f"bias={self.bias is not None}"
        )


# ============================================================
#  4. hamilton_product — standalone функция
# ============================================================

def hamilton_product(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Произведение Гамильтона двух кватернионных тензоров.

    Parameters
    ----------
    q1, q2 : Tensor shape (..., 4) — порядок компонент (w, x, y, z)

    Returns
    -------
    Tensor shape (..., 4)
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


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


# ============================================================
#  6. DomainRotor — обучаемый роторный оператор
# ============================================================

class DomainRotor(nn.Module):
    """
    Обучаемый ротационный оператор в алгебре Клиффорда через сэндвич-произведение.

    Parameters
    ----------
    algebra       : экземпляр CliffordAlgebra
    init_identity : если True — инициализирует R как единичный мультивектор (R[0]=1)
    """

    def __init__(self, algebra: CliffordAlgebra, init_identity: bool = True):
        super().__init__()
        self.algebra = algebra
        self.R = nn.Parameter(torch.zeros(algebra.dim))
        if init_identity:
            with torch.no_grad():
                self.R[0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Применяет сэндвич-произведение R x R^{-1}.

        Parameters
        ----------
        x : Tensor — вход в виде мультивектора

        Returns
        -------
        Tensor той же формы
        """
        return self.algebra.sandwich(self.R, x)

    def get_inverse(self) -> torch.Tensor:
        """
        Вычисляет обратный ротор: R^{-1} = reverse(R) / ||R||^2.

        Returns
        -------
        Tensor shape (algebra.dim,)
        """
        return self.algebra.reverse(self.R) / (self.algebra.norm(self.R) ** 2 + 1e-8)