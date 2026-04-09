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
        "язык", "перевод", "грамматика", "предложение", "текст",
    ],
    "code": [
        "python", "javascript", "function", "class", "debug", "algorithm",
        "code", "program", "compile", "api", "variable", "loop", "script",
        "implement", "binary", "search", "sort", "recursion", "array",
        "hash", "tree", "graph", "stack", "queue", "pointer", "memory",
        "код", "программа", "функция", "класс", "отладка",
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
        ckpt = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=True)
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
    # Fix runaway expert bias from run_018 (science bias was 5.22)
    # and router_proj weight collapse (row norms range 0.2-1.0, rows collinear).
    # Root cause: training co-adapted router_proj + _expert_bias; clamping or
    # normalizing alone is insufficient because row directions remain collinear.
    # Fix: zero bias + orthogonal reinit of router_proj (expert weights kept).
    _moe = model.core_model.pipeline.moe
    _kernel = getattr(_moe, "kernel", _moe)
    # _expert_bias may be on MoEKernel directly or under .kernel (adapter pattern)
    for _attr in ("_expert_bias",):
        _target = None
        if hasattr(_moe, _attr) and getattr(_moe, _attr) is not None:
            _target = _moe
        elif hasattr(_moe, "kernel") and hasattr(_moe.kernel, _attr) and getattr(_moe.kernel, _attr) is not None:
            _target = _moe.kernel
        if _target is not None:
            bias_vals = getattr(_target, _attr).data
            max_bias = bias_vals.abs().max().item()
            if max_bias > 1.0:
                print(f"[init] Resetting expert bias (max={max_bias:.2f} -> 0.0)")
                bias_vals.zero_()
    # Re-initialize router_proj with orthogonal rows (fixes co-adaptation collapse)
    if hasattr(_kernel, "router_proj"):
        _rp = _kernel.router_proj.weight.data
        _row_norms = _rp.norm(dim=1)
        _norm_ratio = _row_norms.max() / _row_norms.min().clamp(min=1e-8)
        _cos = torch.nn.functional.cosine_similarity(_rp.unsqueeze(1), _rp.unsqueeze(0), dim=2)
        _offdiag = _cos[_cos != 1.0].abs().max().item()
        if _norm_ratio > 2.0 or _offdiag > 0.8:
            print(f"[init] Re-initializing router_proj (norm ratio={_norm_ratio:.2f}, max cosine={_offdiag:.2f})")
            torch.nn.init.orthogonal_(_kernel.router_proj.weight)
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
        dom = torch.zeros(1, dtype=torch.long, device=DEVICE)
        out, _, _, _, aux_state = model(enc, dom, return_state=True, memory_mode="retrieve")
        expert_weights = aux_state.routing_weights[0].cpu()
        # Normalize routing scores to probabilities for display/dominant detection
        weights_sum = expert_weights.sum().clamp(min=1e-8)
        expert_probs = expert_weights / weights_sum
        # Blend with semantic keyword hints
        semantic_hints = semantic_domain_hints(text)
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


_EXPERT_RESPONSE_TEMPLATES = {
    "math": (
        "Математический анализ: '{text}' — это задача из области математического "
        "рассуждения. Ключевые аспекты: формализация задачи, поиск инвариантов, "
        "построение доказательства."
    ),
    "language": (
        "Лингвистический анализ: '{text}' — текст с выраженной языковой структурой. "
        "Аспекты: семантическое поле, синтаксическая конструкция, прагматический контекст."
    ),
    "code": (
        "Программный анализ: '{text}' — задача из области программирования. "
        "Подход: декомпозиция, выбор алгоритма, реализация, тестирование."
    ),
    "science": (
        "Научный анализ: '{text}' — вопрос из области естественных наук. "
        "Метод: формулировка гипотезы, экспериментальная проверка, анализ данных."
    ),
}


def generate_expert_response(
    dominant: str, user_text: str, confidence: float, expert_weights: torch.Tensor
) -> str:
    """Generate a meaningful expert response with blended domain weights."""
    template = _EXPERT_RESPONSE_TEMPLATES.get(dominant, "Анализ: '{text}'")
    snippet = user_text[:80] + ("..." if len(user_text) > 80 else "")
    lines = [
        f"\nExpert [{dominant.upper()}] (confidence: {confidence:.1%}):",
        f"  {template.format(text=snippet)}",
        f"  Routed via Clifford invariant + semantic blend (alpha={SEMANTIC_BLEND})",
    ]
    if confidence > 0.5:
        lines.append("  Высокая уверенность маршрутизации.")
    elif confidence < 0.3:
        lines.append("  Низкая уверенность — запрос может относиться к нескольким доменам.")
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
