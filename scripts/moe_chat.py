#!/usr/bin/env python
"""
Интерактивный чат с MoEKernel domain experts.

Показывает как модель маршрутизирует тексты по доменам:
 math / language / code / science

Команды:
 :transfer <домен> -- перенести последний текст в другой домен
 :compare <текст2> -- сравнить сходство с последним текстом
 :experts -- показать текущую нагрузку экспертов
 :quit -- выход

Пример регистрации кастомных экспертов:
 ------------------------------
 from src.core.moe_kernel import DomainExpert, register_expert
 import torch.nn as nn

 class HistoryExpert(DomainExpert):
     '''Expert for historical and temporal patterns.'''
     def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
         super().__init__(input_dim, hidden_dim, dropout, name="history")
         self.net = nn.Sequential(
             nn.Linear(input_dim, hidden_dim),
             nn.GELU(),
             nn.Dropout(dropout),
             nn.Linear(hidden_dim, input_dim),
         )

 class MedicalExpert(DomainExpert):
     '''Expert for medical and biological patterns.'''
     def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
         super().__init__(input_dim, hidden_dim, dropout, name="medical")
         self.net = nn.Sequential(
             nn.Linear(input_dim, hidden_dim),
             nn.Tanh(),
             nn.Dropout(dropout),
             nn.Linear(hidden_dim, input_dim),
         )

 # Register custom experts before building model
 register_expert("history", HistoryExpert)
 register_expert("medical", MedicalExpert)

 # Then use in _patch_moe_kernel:
 # _patch_moe_kernel(model.core_model, expert_names=["math", "medical", "history", "science"])
 ------------------------------
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Fix Windows console encoding for Cyrillic input via pipe
stdin_reconfigure = getattr(sys.stdin, "reconfigure", None)
if callable(stdin_reconfigure):
    stdin_reconfigure(encoding="utf-8", errors="replace")
stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.hdim_model import HDIMConfig
from src.models.model_factory import build_sbert_hdim_model, _patch_moe_kernel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAIN_NAMES = ["math", "language", "code", "science"]
DOMAIN_IDX = {n: i for i, n in enumerate(DOMAIN_NAMES)}
SEMANTIC_BLEND = 0.6

_DOMAIN_KEYWORDS = {
    "math": [
        "equation", "formula", "calculate", "integral", "derivative", "matrix",
        "proof", "theorem", "algebra", "geometry", "probability", "statistics",
        "solve", "number", "function", "sum", "plus", "minus", "multiply",
        "divide", "square", "root", "logarithm", "fraction", "percentage",
        "hypothesis", "conjecture", "riemann", "euler", "fibonacci", "pi",
        "prime", "infinity", "limit", "series", "polynomial", "tensor",
        "quaternion", "rotation", "vector", "coordinate", "dimension",
        "арифметика", "сложение", "умножение",
        "математика", "формула", "уравнение", "интеграл", "производная",
    ],
    "language": [
        "translate", "grammar", "sentence", "word", "meaning", "linguistic",
        "syntax", "semantics", "poem", "story", "text",
        "язык", "перевод", "переведи", "грамматика", "предложение", "текст",
        "слово", "значение", "перевести",
    ],
    "code": [
        "python", "javascript", "function", "class", "debug", "algorithm",
        "code", "program", "compile", "api", "variable", "loop", "script",
        "implement", "binary", "search", "sort", "recursion", "array",
        "hash", "tree", "graph", "stack", "queue", "pointer", "memory",
        "bfs", "dfs", "traversal", "breadth", "depth",
        "код", "программа", "функция", "класс", "отладка",
        "поиск в ширину", "поиск в глубину", "ширину", "глубину",
        "алгоритм", "сортировка", "обход",
    ],
    "science": [
        "quantum", "physics", "chemistry", "biology", "experiment",
        "atom", "molecule", "cell", "DNA", "gravity",
        "energy", "theory", "entropy", "photosynthesis", "light",
        "speed", "velocity", "radiation", "photon", "evolution",
        "species", "organism", "temperature", "pressure", "force",
        "crispr", "gene", "protein", "enzyme", "heisenberg",
        "uncertainty", "relativity", "newton", "telescope",
        "квантовая", "физика", "химия", "биология", "атом",
        "фотосинтез", "эволюция", "скорость", "свет",
        "ньютон", "закон ньютона", "механика", "энергия", "сила",
    ],
}


def semantic_domain_hints(text: str) -> dict[str, float]:
    """Compute domain weight hints from keyword matching on lowercased text."""
    import re
    low = text.lower()
    hits: dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in low)
        hits[domain] = count
    # Pattern-based boosts (math expressions, code snippets, etc.)
    if re.search(r'\d+\s*[+\-*/^=]\s*\d+', text) or re.search(r'[+\-*/=]\s*\d', text):
        hits["math"] = hits.get("math", 0) + 3
    # Code patterns: braces, def/class/import/return keywords, assignment operators
    # Parentheses alone (e.g. "sin(x)") are NOT code — they appear in math too
    if re.search(r'[{}]', text) or re.search(r'\bdef\b|\bclass\b|\bimport\b|\breturn\b', text):
        hits["code"] = hits.get("code", 0) + 3
    total_hits = sum(hits.values())
    if total_hits == 0:
        return {n: 0.25 for n in DOMAIN_NAMES}
    raw: dict[str, float] = {}
    for domain in DOMAIN_NAMES:
        raw[domain] = hits.get(domain, 0) / total_hits
    total_raw = sum(raw.values())
    if total_raw < 1e-8:
        return {n: 0.25 for n in DOMAIN_NAMES}
    return {n: raw[n] / total_raw for n in DOMAIN_NAMES}


LLM_BACKEND = os.environ.get("HDIM_LLM_BACKEND", "none")
LLM_MODEL = os.environ.get("HDIM_LLM_MODEL", "")


def query_llm(user_text: str, expert_info: str, model: str = "") -> str | None:
    """Query an LLM backend (openai or anthropic) for a grounded response.

    Returns the assistant reply text, or None if no backend is configured.
    """
    if LLM_BACKEND == "none":
        return None
    model = model or LLM_MODEL
    system_prompt = (
        "You are an HDIM AI assistant. The MoE router identified the following "
        f"expert routing for the user's input:\n{expert_info}\n"
        "Use this routing context to ground your response."
    )
    try:
        if LLM_BACKEND == "openai":
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                max_tokens=512,
            )
            return resp.choices[0].message.content
        if LLM_BACKEND == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=model or "claude-sonnet-4-20250514",
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": user_text}],
            )
            block = resp.content[0]
            if isinstance(block, anthropic.types.TextBlock):
                return block.text
            return str(block)
    except Exception as exc:
        print(f"[llm error] {exc}")
    return None


CHECKPOINT_CANDIDATES = [
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "titans_opt5"
    / "checkpoints"
    / "best.pt",
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "titans_opt4"
    / "checkpoints"
    / "best.pt",
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "titans_opt3"
    / "checkpoints"
    / "best.pt",
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "titans_30ep"
    / "checkpoints"
    / "best.pt",
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "gpu_training"
    / "checkpoints"
    / "best.pt",
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "run_018"
    / "checkpoints"
    / "best.pt",
]


def build_model():
    print(f"[init] Building SBERT + MoEKernel on {DEVICE}...")
    checkpoint_path = next((path for path in CHECKPOINT_CANDIDATES if path.exists()), None)

    if checkpoint_path is not None:
        ckpt = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        proj_bias = sd.get("text_encoder.projection.4.bias")
        hidden_dim = proj_bias.shape[0] if proj_bias is not None else 256
        # Infer clifford_p from checkpoint: learnable_metric shape gives clifford_dim
        # dim = 2^(p+q+r), with q=1,r=0 → p = log2(dim) - 1
        _lm = sd.get("core_model.pipeline.algebra.learnable_metric")
        if _lm is not None:
            clifford_dim = _lm.shape[0]
            import math
            _pq = int(math.log2(clifford_dim))
            clifford_p = _pq - 1  # q=1, r=0
        else:
            clifford_p = 3
        cfg = HDIMConfig(
            hidden_dim=hidden_dim,
            num_experts=4,
            num_domains=4,
            memory_type="titans",
            top_k=2,
            clifford_p=clifford_p,
            clifford_q=1,
        )
        model = build_sbert_hdim_model(cfg, soft_router=True, freeze_sbert=True)
        _patch_moe_kernel(
            model.core_model,
            expert_names=["math", "language", "code", "science"],
            z_loss_weight=0.01,
            ortho_loss_weight=0.01,
        )
        result = model.load_state_dict(sd, strict=False)
        score = ckpt.get("score", None)
        epoch = ckpt.get("current_epoch", ckpt.get("epoch", "?"))
        score_str = f", score={score:.4f}" if score is not None else ""
        print(
            f"[init] Checkpoint loaded from {checkpoint_path}: epoch={epoch}{score_str} ({len(result.missing_keys)} missing keys)"
        )
    else:
        cfg = HDIMConfig(hidden_dim=256, num_experts=4, num_domains=4, top_k=2)
        model = build_sbert_hdim_model(cfg, soft_router=False, freeze_sbert=True)
        _patch_moe_kernel(
            model.core_model,
            expert_names=["math", "language", "code", "science"],
            z_loss_weight=0.01,
            ortho_loss_weight=0.01,
        )
        print(
            "[init] WARNING: no checkpoint found in known artifacts paths — using random weights"
        )

    model.to(DEVICE)
    # Preserve trained weights — do NOT reset expert bias or router_proj at inference time
    model.eval()
    print("[init] Model ready.")
    print(f"[init] Experts: {DOMAIN_NAMES}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] Total params: {n_params:,}")
    return model


def encode_text(model, text: str):
    """Encode text and run through HDIMModel. Returns (exported_invariant, expert_weights, dominant_expert, out, aux_state).

    expert_weights are blended with semantic keyword hints (SEMANTIC_BLEND controls
    the weight of keyword hints vs model routing).
    """
    text = str(text)
    with torch.no_grad():
        enc = model.encode_texts([text], device=DEVICE)
        semantic_hints = semantic_domain_hints(text)
        hint_domain = max(semantic_hints, key=semantic_hints.get) if semantic_hints else "math"
        dom = torch.tensor([DOMAIN_IDX[hint_domain]], dtype=torch.long, device=DEVICE)
        out, _, _, _, aux_state = model(enc, dom, return_state=True, memory_mode="retrieve")
        expert_weights = aux_state.routing_weights[0].cpu()
        # Normalize routing scores to probabilities for display/dominant detection
        weights_sum = expert_weights.sum().clamp(min=1e-8)
        expert_probs = expert_weights / weights_sum
        # Blend with semantic keyword hints (reuse semantic_hints from line 303)
        blended: dict[str, float] = {}
        for name in DOMAIN_NAMES:
            model_w = expert_probs[DOMAIN_IDX[name]].item()
            sem_w = semantic_hints.get(name, 0.25)
            blended[name] = (1 - SEMANTIC_BLEND) * model_w + SEMANTIC_BLEND * sem_w
        total = sum(blended.values())
        for name in blended:
            blended[name] /= total
        blended_weights = torch.tensor([blended[n] for n in DOMAIN_NAMES])
        top_idx: int = int(blended_weights.argmax().item())
        exported_inv = aux_state.exported_invariant
        return exported_inv, blended_weights, DOMAIN_NAMES[top_idx], out, aux_state


def format_bar(weights, width=20):
    bars = []
    w_max = max(weights.tolist()) if weights.numel() > 0 else 1.0
    w_max = max(w_max, 1e-8)
    for name, w in zip(DOMAIN_NAMES, weights.tolist()):
        normed = w / w_max
        filled = int(normed * width)
        filled = max(0, min(filled, width))
        bar = "#" * filled + "-" * (width - filled)
        bars.append(f" {name:8s} [{bar}] {w:.3f}")
    return "\n".join(bars)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity with dimension alignment for potentially mismatched tensors."""
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    min_dim = min(a.shape[-1], b.shape[-1])
    a_aligned = a[..., :min_dim]
    b_aligned = b[..., :min_dim]
    return F.cosine_similarity(a_aligned, b_aligned, dim=-1).item()


# ---------------------------------------------------------------------------
# Local expert response engine -- generates domain-specific answers without
# external LLM backends.  Only uses Python stdlib (re, math, fractions).
# ---------------------------------------------------------------------------

import re as _re
import math as _math

_CYRILLIC_RE = _re.compile(r"[а-яА-ЯёЁ]")


def _detect_lang(text: str) -> str:
    """Return 'ru' if Cyrillic characters dominate, else 'en'."""
    cyrillic = len(_CYRILLIC_RE.findall(text))
    return "ru" if cyrillic > len(text) * 0.15 else "en"


def _safe_eval_arith(expr: str) -> str | None:
    """Evaluate simple arithmetic expressions via AST parsing (no eval/exec).

    Handles patterns like 2+3, 5*7, 12/4, 3^2, 2**3, sqrt(16).
    Returns a descriptive string or None.
    """
    import ast as _ast

    expr = expr.strip().replace("^", "**")
    cleaned = expr.replace("sqrt", "").replace("log", "").replace("abs", "").replace("pi", "").replace("e", "")
    if not _re.match(r'^[\d\+\-\*/\.\(\)\s]+$', cleaned):
        return None

    def _eval_node(node):
        if isinstance(node, _ast.Constant):
            return node.value
        if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.USub):
            return -_eval_node(node.operand)
        if isinstance(node, _ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            ops = {
                _ast.Add: lambda a, b: a + b,
                _ast.Sub: lambda a, b: a - b,
                _ast.Mult: lambda a, b: a * b,
                _ast.Div: lambda a, b: a / b,
                _ast.Pow: lambda a, b: a ** b,
                _ast.FloorDiv: lambda a, b: a // b,
                _ast.Mod: lambda a, b: a % b,
            }
            op_fn = ops.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op_fn(left, right)
        if isinstance(node, _ast.Call):
            if isinstance(node.func, _ast.Name) and node.func.id in ("sqrt", "log", "abs"):
                arg = _eval_node(node.args[0])
                if node.func.id == "sqrt":
                    return _math.sqrt(arg)
                if node.func.id == "log":
                    return _math.log10(arg)
                if node.func.id == "abs":
                    return abs(arg)
            raise ValueError("Unsupported function call")
        if isinstance(node, _ast.Name):
            if node.id == "pi":
                return _math.pi
            if node.id == "e":
                return _math.e
            raise ValueError(f"Unsupported name: {node.id}")
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    try:
        tree = _ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return str(result)
    except Exception:
        return None


# -- Math helpers -----------------------------------------------------------

_MATH_FORMULAS: dict[str, tuple[str, str]] = {
    # key (lowercase): (formula, explanation)
    "pythagorean": (
        "a^2 + b^2 = c^2",
        "In a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides.",
    ),
    "quadratic": (
        "x = (-b +- sqrt(b^2 - 4ac)) / 2a",
        "Solves ax^2 + bx + c = 0. Discriminant D = b^2 - 4ac determines the number of real roots.",
    ),
    "euler": (
        "e^(i*pi) + 1 = 0",
        "Euler's identity connects five fundamental constants: e, i, pi, 1, and 0.",
    ),
    "derivative_power": (
        "d/dx [x^n] = n * x^(n-1)",
        "Power rule: derivative of x^n is n*x^(n-1). Applies for any real n != 0.",
    ),
    "chain_rule": (
        "d/dx [f(g(x))] = f'(g(x)) * g'(x)",
        "Chain rule for composite functions: differentiate the outer function, then multiply by the inner derivative.",
    ),
    "integral_power": (
        "int x^n dx = x^(n+1)/(n+1) + C  (n != -1)",
        "Basic power rule for integration. For n=-1, the result is ln|x| + C.",
    ),
    "bayes": (
        "P(A|B) = P(B|A) * P(A) / P(B)",
        "Bayes' theorem updates the probability of hypothesis A given evidence B.",
    ),
    "gaussian": (
        "int e^(-x^2) dx from -inf to inf = sqrt(pi)",
        "The Gaussian integral is fundamental in probability and statistics (normal distribution).",
    ),
}

_MATH_FORMULAS_RU: dict[str, tuple[str, str]] = {
    "пифагор": (
        "a^2 + b^2 = c^2",
        "В прямоугольном треугольнике квадрат гипотенузы равен сумме квадратов катетов.",
    ),
    "квадратное": (
        "x = (-b +- sqrt(b^2 - 4ac)) / 2a",
        "Решение ax^2 + bx + c = 0. Дискриминант D = b^2 - 4ac определяет число действительных корней.",
    ),
    "эйлер": (
        "e^(i*pi) + 1 = 0",
        "Тождество Эйлера связывает пять фундаментальных констант: e, i, pi, 1 и 0.",
    ),
    "производная": (
        "d/dx [x^n] = n * x^(n-1)",
        "Правило дифференцирования степенной функции. Применимо для любого вещественного n != 0.",
    ),
    "интеграл": (
        "int x^n dx = x^(n+1)/(n+1) + C  (n != -1)",
        "Основное правило интегрирования степенной функции. При n=-1 результат ln|x| + C.",
    ),
    "байес": (
        "P(A|B) = P(B|A) * P(A) / P(B)",
        "Теорема Байеса обновляет вероятность гипотезы A при наличии свидетельства B.",
    ),
}


def _math_response(text: str, lang: str) -> str:
    low = text.lower()
    parts: list[str] = []

    # --- High-priority conceptual topics (checked BEFORE arithmetic) ---
    # Derivative-related questions (must precede arith to avoid x^3+2x -> 3+2=5)
    if any(w in low for w in ("derivative", "derivada", "производн", "дифференци")):
        # Try to parse polynomial derivative: d/dx of a*x^n + b*x + c
        # Match patterns like "x^3 + 2x", "3x^2 - 5x + 1"
        poly_m = _re.search(r'([+-]?\s*\d*x\^?\d*[^,;.!?]*)(?:$|[,;.!?])', text)
        if lang == "ru":
            parts.append("Производная f'(x0) = lim[h->0] (f(x0+h)-f(x0))/h измеряет мгновенную скорость изменения.")
            parts.append("Правила: (x^n)'=n*x^(n-1), (cf)'=cf', (u+v)'=u'+v', (uv)'=u'v+uv', цепное правило: (f(g(x)))'=f'(g(x))*g'(x).")
            if poly_m:
                parts.append(f"Для выражения {poly_m.group(1).strip()}: примените правило степени (x^n)'=n*x^(n-1) к каждому слагаемому.")
        else:
            parts.append("The derivative f'(x0) = lim[h->0] (f(x0+h)-f(x0))/h measures instantaneous rate of change.")
            parts.append("Key rules: (x^n)'=n*x^(n-1), (cf)'=cf', (u+v)'=u'+v', (uv)'=u'v+uv', (u/v)'=(u'v-uv')/v^2, chain rule: (f(g(x)))'=f'(g(x))*g'(x).")
            if poly_m:
                expr = poly_m.group(1).strip()
                parts.append(f"For {expr}: apply the power rule (x^n)'=n*x^(n-1) term-by-term.")

    # Integral-related questions
    if any(w in low for w in ("integral", "интеграл", "antiderivative", "первообразн")):
        if lang == "ru":
            parts.append("Интеграл -- обратная операция к дифференцированию. Определённый интеграл int_a^b f(x)dx = F(b)-F(a), где F'(x)=f(x) (Формула Ньютона-Лейбница).")
        else:
            parts.append("Integration is the inverse of differentiation. The Fundamental Theorem of Calculus: int_a^b f(x)dx = F(b)-F(a) where F'(x)=f(x).")

    # Matrix/linear algebra
    if any(w in low for w in ("matrix", "матриц", "determinant", "определител", "eigenvalue", "собственн")):
        if lang == "ru":
            parts.append("Матрица -- прямоугольная таблица чисел. Определитель det(A) квадратной матрицы -- скаляр, определяющий обратимость (det != 0 => обратима). Собственные значения: Av = lv, где l -- eigenvalue, v -- eigenvector.")
        else:
            parts.append("A matrix is a rectangular array of numbers. det(A) determines invertibility (det!=0 => invertible). Eigenvalues satisfy Av = lv; found via det(A - lI) = 0.")

    # Probability/statistics
    if any(w in low for w in ("probability", "вероятност", "statistics", "статистик", "distribution", "распредел", "mean", "variance", "дисперси", "матожидан")):
        if lang == "ru":
            parts.append("Вероятность P(A) события A -- число от 0 до 1. Матожидание E[X] = sum(x_i * p_i). Дисперсия Var(X) = E[(X-E[X])^2]. Нормальное распределение N(mu, sigma^2): f(x) = (1/(sigma*sqrt(2pi))) * exp(-(x-mu)^2/(2*sigma^2)).")
        else:
            parts.append("Probability P(A) in [0,1]. Expectation E[X]=sum(x_i*p_i). Variance Var(X)=E[(X-E[X])^2]. Normal distribution N(mu,sigma^2): f(x) = (1/(sigma*sqrt(2pi))) * exp(-(x-mu)^2/(2*sigma^2)).")

    # Fibonacci
    if "fibonacci" in low or "фибоначч" in low:
        if lang == "ru":
            parts.append("Числа Фибоначчи: F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2). Отношение F(n+1)/F(n) стремится к золотому сечению phi = (1+sqrt(5))/2 ~ 1.618.")
        else:
            parts.append("Fibonacci numbers: F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2). Ratio F(n+1)/F(n) converges to the golden ratio phi = (1+sqrt(5))/2 ~ 1.618.")

    # Prime numbers
    if "prime" in low or "простое числ" in low or "простых чисел" in low:
        if lang == "ru":
            parts.append("Простое число -- натуральное число > 1, делящееся только на 1 и на себя. Основная теорема арифметики: каждое целое > 1 единственным образом представляется как произведение простых.")
        else:
            parts.append("A prime number is a natural number > 1 divisible only by 1 and itself. Fundamental Theorem of Arithmetic: every integer > 1 has a unique prime factorization.")

    # --- Pure arithmetic (only if no conceptual topic matched) ---
    if not parts:
        arith_m = _re.search(r'(\d+\s*[+\-*/^]\s*\d+(?:\s*[+\-*/^]\s*\d+)*)', text)
        if arith_m:
            result = _safe_eval_arith(arith_m.group(1))
            if result is not None:
                if lang == "ru":
                    parts.append(f"Вычисление: {arith_m.group(1)} = {result}")
                else:
                    parts.append(f"Computation: {arith_m.group(1)} = {result}")

    # Match known formula requests
    formula_found = False
    for key, (formula, explanation) in _MATH_FORMULAS.items():
        if key in low:
            parts.append(f"Formula: {formula}")
            parts.append(explanation)
            formula_found = True
            break
    if not formula_found:
        for key, (formula, explanation) in _MATH_FORMULAS_RU.items():
            if key in low:
                parts.append(f"Формула: {formula}")
                parts.append(explanation)
                formula_found = True
                break

    # Fallback: structural analysis of the question
    if not parts:
        nums = _re.findall(r'\b\d+(?:\.\d+)?\b', text)
        ops = _re.findall(r'[+\-*/^=]', text)
        if lang == "ru":
            parts.append(f"Вопрос относится к области математики. Обнаружены: {len(nums)} числовых значений, {len(ops)} операторов.")
            parts.append("Для точного ответа уточните тип задачи: арифметика, алгебра, геометрия, анализ, теория вероятностей?")
        else:
            parts.append(f"Question falls in the math domain. Detected: {len(nums)} numeric values, {len(ops)} operators.")
            parts.append("For a precise answer, clarify the problem type: arithmetic, algebra, geometry, analysis, probability?")

    return "\n".join(parts)


# -- Code helpers -----------------------------------------------------------

_ALGO_INFO: dict[str, dict[str, str]] = {
    "sort": {
        "en": "Sorting algorithms compared: QuickSort O(n log n) avg (in-place, unstable), MergeSort O(n log n) worst (stable, needs O(n) extra), HeapSort O(n log n) worst (in-place, unstable), TimSort O(n log n) (hybrid merge+insertion, stable, Python/Java default).",
        "ru": "Сравнение сортировок: QuickSort O(n log n) ср. (на месте, нестабильная), MergeSort O(n log n) худ. (стабильная, O(n) памяти), HeapSort O(n log n) худ. (на месте, нестабильная), TimSort O(n log n) (гибрид, стабильная, стандарт в Python/Java).",
    },
    "search": {
        "en": "Search algorithms: Binary search O(log n) on sorted arrays. Linear search O(n) on unsorted. Hash table lookup O(1) avg. BFS/DFS for graph traversal -- both O(V+E).",
        "ru": "Алгоритмы поиска: бинарный поиск O(log n) на отсортированных массивах, линейный O(n), хеш-таблица O(1) в среднем. BFS/DFS для обхода графов -- оба O(V+E).",
    },
    "recursion": {
        "en": "Recursion solves a problem by reducing it to smaller instances of itself. Base case prevents infinite descent. Tail recursion can be optimized to iteration (not in Python). Memoization caches subproblem results (dynamic programming).",
        "ru": "Рекурсия решает задачу сведением к меньшим экземплярам себя. Базовый случай предотвращает бесконечный спуск. Хвостовая рекурсия оптимизируется в итерацию (не в Python). Мемоизация кэширует результаты подзадач (динамическое программирование).",
    },
    "bfs": {
        "en": "BFS (Breadth-First Search): uses a queue, explores neighbors level by level. Guarantees shortest path in unweighted graphs. Pseudocode: queue = [start]; while queue: node = dequeue(); for neighbor in adj[node]: if not visited: visited.add(neighbor); enqueue(neighbor).",
        "ru": "BFS (обход в ширину): использует очередь, обходит соседей по уровням. Гарантирует кратчайший путь во взвешенном графе с равными весами. Псевдокод: queue=[start]; while queue: node=dequeue(); for neighbor in adj[node]: if not visited: visited.add(neighbor); enqueue(neighbor).",
    },
    "dfs": {
        "en": "DFS (Depth-First Search): uses a stack (or recursion), explores as deep as possible first. Useful for topological sort, cycle detection, connected components. Pseudocode: stack=[start]; while stack: node=pop(); if not visited: visited.add(node); push all adj[node].",
        "ru": "DFS (обход в глубину): использует стек или рекурсию, уходит максимально глубоко. Применяется для топологической сортировки, поиска циклов, компонент связности. Псевдокод: stack=[start]; while stack: node=pop(); if not visited: visited.add(node); push adj[node].",
    },
    "hash": {
        "en": "Hash tables map keys to values via a hash function h(k)->index. Average O(1) insert/lookup/delete. Collisions resolved by chaining (linked lists at each bucket) or open addressing (probing). Load factor alpha = n/m should stay below ~0.75 for good performance.",
        "ru": "Хеш-таблицы отображают ключи в значения через хеш-функцию h(k)->индекс. В среднем O(1) вставка/поиск/удаление. Коллизии решаются цепочками (списки в бакетах) или открытой адресацией (пробы). Коэффициент заполнения alpha=n/m лучше держать < 0.75.",
    },
    "tree": {
        "en": "Binary Search Tree: each node has key, left subtree (smaller keys), right subtree (larger keys). Avg O(log n) search/insert, worst O(n) if unbalanced. AVL/Red-Black trees guarantee O(log n) by self-balancing.",
        "ru": "Бинарное дерево поиска: каждый узел имеет ключ, левое поддерево (меньшие ключи), правое (большие). В среднем O(log n) поиск/вставка, худшее O(n) без балансировки. AVL/Red-Black деревья гарантируют O(log n) самобалансировкой.",
    },
    "graph": {
        "en": "Graph representations: adjacency matrix O(V^2) space (fast edge lookup), adjacency list O(V+E) space (efficient for sparse graphs). Key algorithms: Dijkstra (shortest path, O((V+E)log V)), Kruskal/Prim (MST), Bellman-Ford (negative weights).",
        "ru": "Представления графов: матрица смежности O(V^2) (быстрый поиск рёбер), список смежности O(V+E) (эффективно для разреженных). Ключевые алгоритмы: Дейкстра (кратчайший путь, O((V+E)log V)), Краскал/Прим (MST), Беллман-Форд (отрицательные веса).",
    },
    "dynamic programming": {
        "en": "Dynamic Programming (DP): solve overlapping subproblems by storing results. Two approaches: top-down (memoization) and bottom-up (tabulation). Key: identify state, transition, base case. Classic problems: knapsack, LCS, edit distance, coin change.",
        "ru": "Динамическое программирование: решение перекрывающихся подзадач с сохранением результатов. Два подхода: сверху вниз (мемоизация) и снизу вверх (табуляция). Ключевое: определить состояние, переход, базу. Классика: рюкзак, НОП, редакторское расстояние, размен монет.",
    },
    "stack": {
        "en": "Stack (LIFO): push/pop O(1). Uses: function call stack, expression evaluation, undo operations, DFS, balanced parentheses check. Implemented via array (amortized O(1) push) or linked list.",
        "ru": "Стек (LIFO): push/pop O(1). Применения: стек вызовов, вычисление выражений, отмена операций, DFS, проверка скобок. Реализация: массив (амортизированное O(1)) или связный список.",
    },
    "queue": {
        "en": "Queue (FIFO): enqueue/dequeue O(1). Uses: BFS, task scheduling, buffering, producer-consumer patterns. Circular buffer or linked list implementation avoids O(n) dequeue from front of array.",
        "ru": "Очередь (FIFO): enqueue/dequeue O(1). Применения: BFS, планирование задач, буферизация, паттерн producer-consumer. Кольцевой буфер или связный список исключает O(n) dequeue из начала массива.",
    },
}


_ALGO_RU_KEYS: dict[str, str] = {
    "сортиров": "sort",
    "поиск": "search",
    "рекурси": "recursion",
    "стек": "stack",
    "очеред": "queue",
    "граф": "graph",
    "дерев": "tree",
    "хеш": "hash",
    "хэш": "hash",
    "динамическ": "dynamic programming",
    "динамич": "dynamic programming",
}

_SCIENCE_RU_KEYS: dict[str, str] = {
    "квантов": "quantum",
    "гравитац": "gravity",
    "тяготен": "gravity",
    "энерг": "energy",
    "энтроп": "entropy",
    "свет": "light",
    "эволюц": "evolution",
    "днк": "dna",
    "атом": "atom",
    "фотосинтез": "photosynthesis",
    "относител": "relativity",
    "теори относ": "relativity",
    "сила": "force",
    "ньютон": "force",
    "температ": "temperature",
    "белк": "protein",
    "фермент": "protein",
}


def _code_response(text: str, lang: str) -> str:
    low = text.lower()
    parts: list[str] = []

    # Detect comparison questions (BFS vs DFS, sort comparison, etc.)
    is_comparison = any(w in low for w in (" vs ", " versus ", " vs.", "difference between", "сравн", "разниц", "отличи"))
    # If comparison AND both BFS and DFS mentioned, give combined response
    if is_comparison and ("bfs" in low or "breadth" in low or "ширину" in low) and ("dfs" in low or "depth" in low or "глубину" in low):
        if lang == "ru":
            parts.append("BFS (обход в ширину) vs DFS (обход в глубину):")
            parts.append("BFS: очередь, обход по уровням, кратчайший путь в невзвешенном графе. O(V+E).")
            parts.append("DFS: стек/рекурсия, уходит максимально глубоко, топологическая сортировка, поиск циклов. O(V+E).")
            parts.append("Выбор: BFS -- кратчайший путь; DFS -- перебор с возвратом, связность, циклы.")
        else:
            parts.append("BFS (Breadth-First) vs DFS (Depth-First):")
            parts.append("BFS: uses a queue, explores level-by-level. Finds shortest path in unweighted graphs. O(V+E).")
            parts.append("DFS: uses a stack/recursion, goes deep first. Used for topological sort, cycle detection, backtracking. O(V+E).")
            parts.append("Choose: BFS for shortest path; DFS for exhaustive search, connectivity, cycles.")
    elif is_comparison and ("sort" in low or "сортиров" in low):
        if lang == "ru":
            parts.append("Сравнение сортировок: QuickSort O(n log n) ср. (на месте, нестабильная), MergeSort O(n log n) худ. (стабильная, O(n) памяти), HeapSort O(n log n) худ. (на месте, нестабильная), TimSort O(n log n) (гибрид, стабильная, стандарт Python/Java).")
        else:
            parts.append("Sorting compared: QuickSort O(n log n) avg (in-place, unstable), MergeSort O(n log n) worst (stable, O(n) extra), HeapSort O(n log n) worst (in-place, unstable), TimSort O(n log n) (hybrid, stable, Python/Java default).")
    else:
        # Match known algorithm/DS topics (English keys first, then Russian aliases)
        matched_key = None
        for key, info in _ALGO_INFO.items():
            if key in low:
                matched_key = key
                break
        if matched_key is None:
            for ru_key, en_key in _ALGO_RU_KEYS.items():
                if ru_key in low and en_key in _ALGO_INFO:
                    matched_key = en_key
                    break
        if matched_key is not None:
            parts.append(_ALGO_INFO[matched_key].get(lang, _ALGO_INFO[matched_key]["en"]))

    # Detect code patterns
    has_def = _re.search(r'\bdef\b', text)
    has_class = _re.search(r'\bclass\b', text)
    has_import = _re.search(r'\bimport\b', text)
    has_return = _re.search(r'\breturn\b', text)
    has_braces = _re.search(r'[{}]', text)

    # Language detection in the query
    lang_name = None
    for ln in ("python", "javascript", "typescript", "java", "c++", "c#", "rust", "go", "golang", "ruby"):
        if ln in low:
            lang_name = ln
            break

    if lang_name and not parts:
        lang_facts = {
            "python": ("Python: dynamic typing, GIL for threads, list comprehensions, generators for lazy evaluation, decorators for metaprogramming.",
                       "Python: динамическая типизация, GIL для потоков, списковые включения, генераторы для ленивых вычислений, декораторы для метапрограммирования."),
            "javascript": ("JavaScript: single-threaded event loop, closures for state encapsulation, promises/async-await for async, prototypal inheritance.",
                           "JavaScript: однопоточный цикл событий, замыкания для инкапсуляции состояния, промисы/async-await для асинхронности, прототипное наследование."),
            "rust": ("Rust: ownership model (no GC), borrow checker, zero-cost abstractions, traits for polymorphism, pattern matching.",
                     "Rust: модель владения (без GC), borrow checker, абстракции нулевой стоимости, типажи для полиморфизма, сопоставление образцов."),
            "java": ("Java: JVM bytecode, strong static typing, garbage collection, interfaces + abstract classes, streams API.",
                     "Java: байт-код JVM, строгая статическая типизация, сборка мусора, интерфейсы + абстрактные классы, Stream API."),
            "go": ("Go: goroutines + channels for concurrency, compiled to native, interfaces are implicit, no inheritance, defer for cleanup.",
                   "Go: горутины + каналы для конкурентности, компиляция в нативный код, неявные интерфейсы, нет наследования, defer для очистки."),
            "golang": ("Go: goroutines + channels for concurrency, compiled to native, interfaces are implicit, no inheritance, defer for cleanup.",
                       "Go: горутины + каналы для конкурентности, компиляция в нативный код, неявные интерфейсы, нет наследования, defer для очистки."),
        }
        if lang_name in lang_facts:
            en_txt, ru_txt = lang_facts[lang_name]
            parts.append(ru_txt if lang == "ru" else en_txt)

    # Code structure analysis
    if has_def or has_class or has_import:
        detected = []
        if has_import:
            detected.append("import statement")
        if has_class:
            detected.append("class definition")
        if has_def:
            detected.append("function definition")
        if has_return:
            detected.append("return statement")
        if lang == "ru":
            parts.append(f"Обнаружены конструкции: {', '.join(detected)}. Структура корректна для Python-подобного синтаксиса.")
        else:
            parts.append(f"Detected constructs: {', '.join(detected)}. Structure is consistent with Python-like syntax.")

    if has_braces and not parts:
        if lang == "ru":
            parts.append("Обнаружены фигурные скобки -- синтаксис C-подобного языка (C/C++/Java/JS/Rust/Go). Отличается от Python отступами и блоками кода.")
        else:
            parts.append("Detected curly braces -- C-family syntax (C/C++/Java/JS/Rust/Go). Unlike Python, uses braces for code blocks instead of indentation.")

    # Complexity analysis if mentioned
    if "complexity" in low or "o(" in low or "сложност" in low or "big-o" in low or "big o" in low:
        if lang == "ru":
            parts.append("O-нотация описывает рост времени/памяти: O(1) < O(log n) < O(n) < O(n log n) < O(n^2) < O(2^n) < O(n!). Оптимальные алгоритмы стремятся к O(n log n) для сортировки, O(n) для поиска максимума.")
        else:
            parts.append("Big-O notation describes growth: O(1) < O(log n) < O(n) < O(n log n) < O(n^2) < O(2^n) < O(n!). Optimal algorithms target O(n log n) for sorting, O(n) for max-finding.")

    # Fallback
    if not parts:
        if lang == "ru":
            parts.append("Вопрос относится к программированию. Уточните тему: алгоритмы (сортировка, поиск, графы), структуры данных (список, дерево, хеш-таблица), языки, паттерны проектирования?")
        else:
            parts.append("Question relates to programming. Clarify the topic: algorithms (sorting, searching, graphs), data structures (list, tree, hash table), languages, design patterns?")

    return "\n".join(parts)


# -- Science helpers ---------------------------------------------------------

_SCIENCE_TOPICS: dict[str, dict[str, str]] = {
    "quantum": {
        "en": "Quantum mechanics: particles exhibit wave-particle duality. Heisenberg uncertainty: delta_x * delta_p >= hbar/2. Schrodinger equation: ih d(psi)/dt = H(psi). Superposition: a system exists in multiple states until measured.",
        "ru": "Квантовая механика: частицы проявляют корпускулярно-волновой дуализм. Принцип неопределённости Гейзенберга: delta_x * delta_p >= hbar/2. Уравнение Шрёдингера: ih d(psi)/dt = H(psi). Суперпозиция: система находится во многих состояниях до измерения.",
    },
    "gravity": {
        "en": "Newton's law of gravitation: F = G*m1*m2/r^2. Einstein's General Relativity: mass curves spacetime, objects follow geodesics. Gravitational constant G = 6.674e-11 N*m^2/kg^2. Escape velocity: v = sqrt(2GM/r).",
        "ru": "Закон всемирного тяготения Ньютона: F = G*m1*m2/r^2. ОТО Эйнштейна: масса искривляет пространство-время, тела движутся по геодезическим. G = 6.674e-11 Н*м^2/кг^2. Вторая космическая: v = sqrt(2GM/r).",
    },
    "energy": {
        "en": "Energy conservation: total energy in a closed system is constant. Kinetic E_k = mv^2/2. Potential E_p = mgh (gravitational), kx^2/2 (elastic). First law of thermodynamics: dU = Q - W (energy = heat minus work).",
        "ru": "Закон сохранения энергии: полная энергия замкнутой системы постоянна. Кинетическая E_k = mv^2/2. Потенциальная E_p = mgh (гравитационная), kx^2/2 (упругая). Первое начало термодинамики: dU = Q - W.",
    },
    "entropy": {
        "en": "Entropy (S) measures disorder. Second law of thermodynamics: total entropy of an isolated system never decreases. Boltzmann: S = k_B * ln(W), where W is the number of microstates. Information entropy (Shannon): H = -sum(p_i * log2(p_i)).",
        "ru": "Энтропия (S) измеряет хаос. Второе начало термодинамики: энтропия изолированной системы не убывает. Больцман: S = k_B * ln(W), W -- число микросостояний. Информационная энтропия Шеннона: H = -sum(p_i * log2(p_i)).",
    },
    "light": {
        "en": "Light: electromagnetic wave, speed c = 3e8 m/s in vacuum. Wave-particle duality: E = hf (photon energy), lambda = c/f. Refraction: n1*sin(theta1) = n2*sin(theta2) (Snell's law). Photoelectric effect: KE = hf - phi (Einstein).",
        "ru": "Свет: электромагнитная волна, скорость c = 3e8 м/с в вакууме. Корпускулярно-волновой дуализм: E = hf (энергия фотона), lambda = c/f. Преломление: n1*sin(theta1) = n2*sin(theta2) (закон Снеллиуса). Фотоэффект: KE = hf - phi (Эйнштейн).",
    },
    "evolution": {
        "en": "Evolution by natural selection (Darwin): organisms with traits better suited to their environment reproduce more. Mutation introduces variation; selection filters it. Speciation occurs when populations become reproductively isolated.",
        "ru": "Эволюция путём естественного отбора (Дарвин): организмы с лучшими адаптациями оставляют больше потомков. Мутации создают вариации, отбор фильтрует. Видообразование происходит при репродуктивной изоляции популяций.",
    },
    "dna": {
        "en": "DNA: double helix of complementary base pairs (A-T, G-C). Central dogma: DNA -> RNA -> Protein. Replication: semi-conservative (each strand templates a new one). Transcription: DNA -> mRNA. Translation: mRNA -> amino acid chain via ribosomes.",
        "ru": "ДНК: двойная спираль комплементарных пар оснований (А-Т, Г-Ц). Центральная догма: ДНК -> РНК -> Белок. Репликация: полуконсервативная. Транскрипция: ДНК -> мРНК. Трансляция: мРНК -> аминокислотная цепь через рибосомы.",
    },
    "atom": {
        "en": "Atom: nucleus (protons + neutrons) + electron cloud. Atomic number Z = proton count. Mass number A = protons + neutrons. Bohr model: E_n = -13.6/Z^2 * 1/n^2 eV for hydrogen-like atoms. Electron config follows Pauli exclusion + Hund's rules.",
        "ru": "Атом: ядро (протоны + нейтроны) + электронное облако. Атомный номер Z = число протонов. Массовое число A = протоны + нейтроны. Модель Бора: E_n = -13.6/Z^2 * 1/n^2 эВ. Конфигурация электронов подчиняется принципу Паули и правилу Хунда.",
    },
    "photosynthesis": {
        "en": "Photosynthesis: 6CO2 + 6H2O + light -> C6H12O6 + 6O2. Light-dependent reactions (thylakoid): H2O split, ATP + NADPH produced. Calvin cycle (stroma): CO2 fixed into glucose via Rubisco. Efficiency ~1-2% of solar energy.",
        "ru": "Фотосинтез: 6CO2 + 6H2O + свет -> C6H12O6 + 6O2. Световые реакции (тилакоиды): расщепление H2O, синтез АТФ + НАДФH. Цикл Кальвина (строма): CO2 фиксируется в глюкозу через Рубиско. КПД ~1-2% солнечной энергии.",
    },
    "relativity": {
        "en": "Special relativity (Einstein 1905): c is constant in all frames. Time dilation: t' = t * gamma, gamma = 1/sqrt(1-v^2/c^2). Length contraction: L' = L/gamma. Mass-energy: E = mc^2. General relativity: gravity = spacetime curvature.",
        "ru": "Специальная теория относительности (1905): c постоянна во всех системах отсчёта. Замедление времени: t' = t*gamma, gamma = 1/sqrt(1-v^2/c^2). Сокращение длины: L' = L/gamma. Эквивалентность массы и энергии: E = mc^2. ОТО: гравитация = кривизна пространства-времени.",
    },
    "force": {
        "en": "Newton's laws: (1) An object at rest stays at rest unless acted on by a force. (2) F = ma (net force = mass * acceleration). (3) Every action has an equal and opposite reaction. Weight: W = mg. Friction: f = mu*N.",
        "ru": "Законы Ньютона: (1) Тело покоится или движется равномерно без воздействия сил. (2) F = ma (равнодействующая = масса * ускорение). (3) Действие равно противодействию. Вес: W = mg. Трение: f = mu*N.",
    },
    "temperature": {
        "en": "Temperature scales: C = (F-32)*5/9, K = C + 273.15. Ideal gas law: PV = nRT (R = 8.314 J/(mol*K)). Kinetic theory: average KE = (3/2)k_B*T per molecule. Heat transfer: conduction, convection, radiation.",
        "ru": "Шкалы температур: C = (F-32)*5/9, K = C + 273.15. Уравнение Менделеева-Клапейрона: PV = nRT (R = 8.314 Дж/(моль*К)). Кинетическая теория: ср. KE = (3/2)k_B*T на молекулу. Теплопередача: теплопроводность, конвекция, излучение.",
    },
    "protein": {
        "en": "Proteins: polymers of 20 amino acids linked by peptide bonds. Four structure levels: primary (sequence), secondary (alpha-helix/beta-sheet), tertiary (3D fold), quaternary (multi-subunit). Folded by hydrophobic effect, H-bonds, disulfide bridges. Enzymes are catalytic proteins.",
        "ru": "Белки: полимеры из 20 аминокислот, связанных пептидными связями. Четыре уровня структуры: первичная (последовательность), вторичная (альфа-спираль/бета-лист), третичная (3D-складка), четвертичная (мультисубъединичный). Фолдинг обеспечивается гидрофобным эффектом, H-связями, дисульфидными мостиками. Ферменты -- каталитические белки.",
    },
}


def _science_response(text: str, lang: str) -> str:
    low = text.lower()
    parts: list[str] = []

    # Match science topics (English keys first, then Russian aliases)
    matched_key = None
    for key in _SCIENCE_TOPICS:
        if key in low:
            matched_key = key
            break
    if matched_key is None:
        for ru_key, en_key in _SCIENCE_RU_KEYS.items():
            if ru_key in low and en_key in _SCIENCE_TOPICS:
                matched_key = en_key
                break
    if matched_key is not None:
        parts.append(_SCIENCE_TOPICS[matched_key].get(lang, _SCIENCE_TOPICS[matched_key]["en"]))

    # Physics formula patterns
    if "newton" in low and not parts:
        if lang == "ru":
            parts.append("Законы Ньютона -- основа классической механики. F=ma -- второй закон. Для макроскопических тел при v << c. При релятивистских скоростях заменяется ОТО.")
        else:
            parts.append("Newton's laws are the foundation of classical mechanics. F=ma is the second law. Valid for macroscopic bodies at v << c. Replaced by GR at relativistic speeds.")

    # Chemistry-related
    if any(w in low for w in ("chemistry", "chemical", "reaction", "molecule", "хими", "реакц", "молекул")):
        if lang == "ru":
            parts.append("Химические реакции: реагенты -> продукты. Баланс: закон сохранения массы. Типы: соединение, разложение, замещение, обмен. Скорость реакции зависит от концентрации (закон действующих масс), температуры (правило Вант-Гоффа), катализаторов.")
        else:
            parts.append("Chemical reactions: reactants -> products. Balanced by mass conservation. Types: synthesis, decomposition, single/double replacement. Rate depends on concentration (law of mass action), temperature (Arrhenius), catalysts.")

    # Fallback
    if not parts:
        if lang == "ru":
            parts.append("Вопрос из области естественных наук. Уточните тему: физика (механика, термодинамика, электричество, оптика, квантовая), химия, биология, астрономия?")
        else:
            parts.append("Question falls in the natural sciences. Specify the topic: physics (mechanics, thermodynamics, electricity, optics, quantum), chemistry, biology, astronomy?")

    return "\n".join(parts)


# -- Language helpers -------------------------------------------------------

_COMMON_PHRASES_RU_EN: dict[str, str] = {
    "привет": "hello / hi",
    "спасибо": "thank you",
    "пожалуйста": "you're welcome / please",
    "извини": "sorry / excuse me",
    "да": "yes",
    "нет": "no",
    "как дела": "how are you",
    "до свидания": "goodbye",
    "доброе утро": "good morning",
    "добрый вечер": "good evening",
    "я не понимаю": "i don't understand",
    "сколько стоит": "how much does it cost",
    "где находится": "where is",
    "помогите": "help me",
}

_COMMON_PHRASES_EN_RU: dict[str, str] = {
    "hello": "привет",
    "hi": "привет",
    "thank you": "спасибо",
    "thanks": "спасибо",
    "please": "пожалуйста",
    "sorry": "извини",
    "yes": "да",
    "no": "нет",
    "how are you": "как дела",
    "goodbye": "до свидания",
    "good morning": "доброе утро",
    "good evening": "добрый вечер",
    "i don't understand": "я не понимаю",
    "how much": "сколько стоит",
    "where is": "где находится",
    "help me": "помогите",
}

# Multi-language translation dictionary (phrase -> {lang_code: translation})
_MULTILANG_DICT: dict[str, dict[str, str]] = {
    "hello": {"fr": "bonjour", "de": "hallo", "es": "hola", "it": "ciao", "ja": "konnichiwa", "ru": "привет"},
    "hello world": {"fr": "bonjour le monde", "de": "hallo welt", "es": "hola mundo", "it": "ciao mondo", "ja": "konnnichiwa sekai", "ru": "привет мир"},
    "goodbye": {"fr": "au revoir", "de": "auf wiedersehen", "es": "adios", "it": "arrivederci", "ja": "sayounara", "ru": "до свидания"},
    "thank you": {"fr": "merci", "de": "danke", "es": "gracias", "it": "grazie", "ja": "arigatou", "ru": "спасибо"},
    "please": {"fr": "s'il vous plait", "de": "bitte", "es": "por favor", "it": "per favore", "ja": "onegaishimasu", "ru": "пожалуйста"},
    "good morning": {"fr": "bonjour", "de": "guten morgen", "es": "buenos dias", "it": "buongiorno", "ja": "ohayou", "ru": "доброе утро"},
    "good evening": {"fr": "bonsoir", "de": "guten abend", "es": "buenas tardes", "it": "buonasera", "ja": "konbanwa", "ru": "добрый вечер"},
    "yes": {"fr": "oui", "de": "ja", "es": "si", "it": "si", "ja": "hai", "ru": "да"},
    "no": {"fr": "non", "de": "nein", "es": "no", "it": "no", "ja": "iie", "ru": "нет"},
    "i love you": {"fr": "je t'aime", "de": "ich liebe dich", "es": "te quiero", "it": "ti amo", "ja": "aishiteimasu", "ru": "я люблю тебя"},
    "how are you": {"fr": "comment allez-vous", "de": "wie geht es ihnen", "es": "como estas", "it": "come stai", "ja": "ogenki desu ka", "ru": "как дела"},
    "sorry": {"fr": "desole", "de": "entschuldigung", "es": "lo siento", "it": "mi dispiace", "ja": "gomen nasai", "ru": "извини"},
    "water": {"fr": "eau", "de": "wasser", "es": "agua", "it": "acqua", "ja": "mizu", "ru": "вода"},
    "friend": {"fr": "ami", "de": "freund", "es": "amigo", "it": "amico", "ja": "tomodachi", "ru": "друг"},
    "love": {"fr": "amour", "de": "liebe", "es": "amor", "it": "amore", "ja": "ai", "ru": "любовь"},
    "life": {"fr": "vie", "de": "leben", "es": "vida", "it": "vita", "ja": "inochi", "ru": "жизнь"},
}

# Reverse mapping: Russian phrase -> English key in _MULTILANG_DICT
_MULTILANG_RU_TO_EN: dict[str, str] = {}
for _en_key, _translations in _MULTILANG_DICT.items():
    _ru_val = _translations.get("ru")
    if _ru_val:
        _MULTILANG_RU_TO_EN[_ru_val] = _en_key

_LINGUISTIC_CONCEPTS: dict[str, dict[str, str]] = {
    "syntax": {
        "en": "Syntax: the set of rules governing how words combine into phrases and sentences. In generative grammar (Chomsky), syntax is described by rewrite rules (S -> NP VP, etc.). Universal Grammar hypothesizes innate syntactic constraints across all languages.",
        "ru": "Синтаксис: система правил, определяющих комбинирование слов в предложения. В порождающей грамматике (Хомский) синтаксис описывается правилами переписывания (S -> NP VP и т.д.). Универсальная грамматика -- гипотеза о врождённых синтаксических ограничениях.",
    },
    "semantics": {
        "en": "Semantics: the study of meaning in language. Lexical semantics (word meaning), compositional semantics (meaning of phrases from parts), truth-conditional semantics (meaning as truth conditions). Frege's principle: the meaning of a whole is a function of the meanings of its parts.",
        "ru": "Семантика: изучение значения в языке. Лексическая семантика (значение слов), композиционная (значение фраз из частей), истинностно-условная (значение как условия истинности). Принцип Фреге: значение целого есть функция значений частей.",
    },
    "morphology": {
        "en": "Morphology: the study of word structure and formation. Morphemes are the smallest meaning-bearing units (e.g., un-break-able = 3 morphemes). Inflectional (tense, number, case) vs derivational (creates new words: happy -> unhappy).",
        "ru": "Морфология: изучение структуры и образования слов. Морфема -- минимальная значимая единица (например, не-лом-аемый = 3 морфемы). Флективная (время, число, падеж) против словообразовательной (создаёт новые слова: счастливый -> несчастливый).",
    },
    "pragmatics": {
        "en": "Pragmatics: how context contributes to meaning beyond literal semantics. Grice's maxims (quantity, quality, relevance, manner) govern conversation. Speech acts (Austin/Searle): utterances perform actions (promising, requesting, asserting). Implicature: meaning implied but not said.",
        "ru": "Прагматика: как контекст влияет на значение сверх буквальной семантики. Максимы Грайса (количество, качество, релевантность, способ) управляют беседой. Речевые акты (Остин/Сёрль): высказывания выполняют действия. Импликатура: значение, подразумеваемое, но не высказанное.",
    },
    "phonetics": {
        "en": "Phonetics: the physical study of speech sounds. Three branches: articulatory (how sounds are produced), acoustic (physical properties of sound waves), auditory (how the ear perceives sounds). IPA: International Phonetic Alphabet for transcribing any language.",
        "ru": "Фонетика: физическое изучение звуков речи. Три раздела: артикуляционный (как производятся звуки), акустический (физические свойства звуковых волн), аудитивный (как ухо воспринимает звуки). МФА: международный фонетический алфавит для транскрипции любого языка.",
    },
    "grammar": {
        "en": "Grammar: the whole system of language rules, encompassing phonology, morphology, syntax. Descriptive grammar (how people actually speak) vs prescriptive grammar (how authorities say one should speak). All natural languages have complex, systematic grammars.",
        "ru": "Грамматика: вся система правил языка, включая фонологию, морфологию, синтаксис. Описательная грамматика (как люди реально говорят) против предписывающей (как авторитеты говорят, что надо). Все естественные языки имеют сложные систематические грамматики.",
    },
}


def _language_response(text: str, lang: str) -> str:
    low = text.lower()
    parts: list[str] = []

    # Detect target language for translation
    target_lang = None
    for tl_name, tl_keys in [("fr", ("french", "французск", "на французск")), ("de", ("german", "немецк", "на немецск")), ("es", ("spanish", "испанск", "на испанск")), ("it", ("italian", "итальянск", "на итальянск")), ("ja", ("japanese", "японск", "на японск"))]:
        if any(k in low for k in tl_keys):
            target_lang = tl_name
            break

    # Translation requests
    if any(w in low for w in ("translate", "перевод", "переведи", "how do you say", "как сказать", "как будет")):
        # Try to find a phrase to translate
        phrase = low
        for marker in ("translate", "перевод", "переведи", "how do you say", "как сказать", "как будет"):
            phrase = phrase.replace(marker, "").strip()
        # Also strip target language keywords from phrase
        if target_lang:
            for tl_keys_val in [("french", "французск", "на французск"), ("german", "немецк", "на немецск"), ("spanish", "испанск", "на испанск"), ("italian", "итальянск", "на итальянск"), ("japanese", "японск", "на японск")]:
                for k in tl_keys_val:
                    phrase = phrase.replace(k, "").strip()
        # Strip remaining noise words
        for noise in ("to ", "into ", "в ", "на "):
            phrase = phrase.replace(noise, " ").strip()
        phrase = phrase.strip("'\" ,.")

        translated = None
        # Try multi-language translation first
        if target_lang and phrase:
            # For Russian input, try reverse mapping RU->EN first
            if lang == "ru" and _MULTILANG_RU_TO_EN:
                sorted_ru_keys = sorted(_MULTILANG_RU_TO_EN.keys(), key=len, reverse=True)
                for ru_key in sorted_ru_keys:
                    if ru_key == phrase or ru_key in phrase:
                        en_key = _MULTILANG_RU_TO_EN[ru_key]
                        ml_match = _MULTILANG_DICT.get(en_key)
                        if ml_match:
                            translated = f"'{ru_key}' -> {ml_match.get(target_lang, "?")}"
                            break
            # Fallback to English-key lookup
            if not translated:
                # Sort dict keys by length (longest first) to match "hello world" before "hello"
                sorted_phrases = sorted(_MULTILANG_DICT.keys(), key=len, reverse=True)
                for dict_phrase in sorted_phrases:
                    if dict_phrase == phrase or dict_phrase in phrase:
                        ml_match = _MULTILANG_DICT[dict_phrase]
                        translated = f"'{dict_phrase}' -> {ml_match.get(target_lang, '?')}"
                        break
        # Fallback to RU/EN pairs
        if not translated:
            if lang == "ru":
                for ru_phrase, en_phrase in _COMMON_PHRASES_RU_EN.items():
                    if ru_phrase in phrase:
                        translated = f"'{ru_phrase}' -> EN: {en_phrase}"
                        break
            else:
                for en_phrase, ru_phrase in _COMMON_PHRASES_EN_RU.items():
                    if en_phrase in phrase:
                        translated = f"'{en_phrase}' -> RU: {ru_phrase}"
                        break
        if translated:
            parts.append(f"Translation: {translated}")
        else:
            if lang == "ru":
                parts.append("Запрос на перевод обнаружен, но фраза не в локальном словаре. Для точного перевода подключите HDIM_LLM_BACKEND=openai или anthropic.")
            else:
                parts.append("Translation request detected, but the phrase is not in the local dictionary. For accurate translation, set HDIM_LLM_BACKEND=openai or anthropic.")

    # Linguistic concept explanations
    for key, info in _LINGUISTIC_CONCEPTS.items():
        if key in low:
            parts.append(info.get(lang, info["en"]))
            break

    # Language identification
    has_cyrillic = bool(_CYRILLIC_RE.search(text))
    has_latin = bool(_re.search(r'[a-zA-Z]', text))
    if has_cyrillic and has_latin:
        parts.append("Mixed script detected: Cyrillic + Latin. This may indicate code-switching, a bilingual query, or a technical term embedded in a different-script language.")

    # Fallback
    if not parts:
        if lang == "ru":
            parts.append("Вопрос из области языкознания. Уточните тему: перевод, грамматика, синтаксис, семантика, морфология, прагматика, фонетика?")
        else:
            parts.append("Question falls in linguistics. Specify the topic: translation, grammar, syntax, semantics, morphology, pragmatics, phonetics?")

    return "\n".join(parts)


# -- Main response generator ------------------------------------------------

_DOMAIN_GENERATORS = {
    "math": _math_response,
    "code": _code_response,
    "science": _science_response,
    "language": _language_response,
}


def generate_expert_response(
    dominant: str, user_text: str, confidence: float, expert_weights: torch.Tensor
) -> str:
    """Generate a meaningful expert response with blended domain weights."""
    lang = _detect_lang(user_text)
    generator = _DOMAIN_GENERATORS.get(dominant)
    body = generator(user_text, lang) if generator else (
        "Unable to generate a domain-specific response for this expert." if lang == "en"
        else "Невозможно сгенерировать ответ для данного эксперта."
    )

    lang_label = "RU" if lang == "ru" else "EN"
    lines = [
        f"\nExpert [{dominant.upper()}] (confidence: {confidence:.1%}, lang: {lang_label}):",
        f"  {body}",
        f"  Routed via Clifford invariant + semantic blend (alpha={SEMANTIC_BLEND})",
    ]
    if confidence > 0.5:
        lines.append("  High routing confidence." if lang == "en" else "  Высокая уверенность маршрутизации.")
    elif confidence < 0.3:
        lines.append("  Low confidence -- query may span multiple domains." if lang == "en" else "  Низкая уверенность -- запрос может относиться к нескольким доменам.")
    lines.append("  Domain weights:")
    for name, w in zip(DOMAIN_NAMES, expert_weights.tolist()):
        lines.append(f"    {name:8s} {w:5.1%}")
    lines.append("  To get full AI responses, set HDIM_LLM_BACKEND=openai or anthropic")
    return "\n".join(lines)


def main():
    model = build_model()
    print()
    print("=" * 55)
    print(" MoEKernel Interactive Chat")
    print(" Experts: math | language | code | science")
    print(" Commands: :transfer <domain> | :compare <text> | :experts | :memory | :quit")
    print("=" * 55)
    print()

    last_invariant = None
    last_text = None
    last_expert_weights = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            break

        if not user_input:
            continue

        # Ensure str (defensive against pipe encoding issues)
        if isinstance(user_input, bytes):
            user_input = user_input.decode("utf-8", errors="replace")

        # --- Commands ---
        if user_input == ":quit":
            print("[bye]")
            break

        if user_input == ":experts":
            if last_expert_weights is not None:
                print(
                    f'Expert load for: "{last_text[:50]}..."'
                    if last_text and len(last_text) > 50
                    else f'Expert load for: "{last_text}"'
                )
                print(format_bar(last_expert_weights))
            else:
                print("[no text encoded yet]")
            continue

        if user_input == ":memory":
            memory = getattr(model.core_model, "memory", None)
            if memory is None:
                memory = getattr(model.core_model.pipeline, "memory", None)
            if memory is None:
                print("[memory] No memory module found")
            else:
                stats = memory.stats()
                for k, v in stats.items():
                    print(f"  {k}: {v}")
            continue

        if user_input.startswith(":transfer "):
            target = user_input[10:].strip().lower()
            if target not in DOMAIN_IDX:
                print(f"[error] Unknown domain. Choose from: {DOMAIN_NAMES}")
                continue
            if last_invariant is None:
                print("[error] Enter a text first.")
                continue
            # Transfer invariant to target domain (last_invariant is now in clifford space)
            with torch.no_grad():
                pipeline = model.core_model.pipeline
                tgt_name = f"domain_{DOMAIN_IDX[target]}"
                r_target = pipeline.domain_rotors[tgt_name]
                g_target = r_target(last_invariant)
                transferred = pipeline.decoder(g_target)
                print(f"[transfer -> {target}] embedding shape: {transferred.shape}")
                print(
                    f" Cosine sim to original: {cosine_sim(last_invariant, g_target):.4f}"
                )
                print(
                    f" Norm ratio: {g_target.norm().item():.4f} / {last_invariant.norm().item():.4f}"
                )
            continue

        if user_input.startswith(":compare "):
            text2 = user_input[9:].strip()
            if not text2:
                print("[error] Provide a text after :compare")
                continue
            if last_invariant is None:
                print("[error] Enter a text first.")
                continue
            inv2, ew2, dom2, _, _ = encode_text(model, text2)
            sim = cosine_sim(last_invariant, inv2)
            print("[compare]")
            print(f' Text 1: "{(last_text or "")[:60]}"')
            print(f' Text 2: "{text2[:60]}"')
            print(f" Cosine similarity (invariant space): {sim:.4f}")
            text1_expert = "unknown"
            if last_expert_weights is not None:
                text1_expert = DOMAIN_NAMES[int(last_expert_weights.argmax().item())]
            print(f" Text 1 dominant expert: {text1_expert}")
            print(f" Text 2 dominant expert: {dom2}")
            print(" Expert weights (text 2):")
            print(format_bar(ew2))
            continue

        # --- Regular text encoding ---
        invariant, expert_weights, dominant, _, _ = encode_text(model, user_input)
        last_invariant = invariant
        last_text = user_input
        last_expert_weights = expert_weights

        expert_info = (
            f"Dominant: {dominant.upper()}\n"
            f"Weights: {', '.join(f'{n}={w:.3f}' for n, w in zip(DOMAIN_NAMES, expert_weights.tolist()))}"
        )
        print("\nModel:")
        print(f" Dominant expert : {dominant.upper()}")
        print(f" Invariant norm : {invariant.norm().item():.4f}")
        print(" Expert routing :")
        print(format_bar(expert_weights))

        llm_reply = query_llm(user_input, expert_info)
        if llm_reply is not None:
            print(f"\nLLM ({LLM_BACKEND}): {llm_reply}")
        else:
            conf = expert_weights.max().item()
            print(generate_expert_response(dominant, user_input, conf, expert_weights))
        print()


if __name__ == "__main__":
    main()
