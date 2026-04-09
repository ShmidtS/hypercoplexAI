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
        cfg = HDIMConfig(
            hidden_dim=hidden_dim,
            num_experts=4,
            num_domains=4,
            memory_type="titans",
            top_k=2,
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
    _moe = model.core_model.pipeline.moe
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
                print(f"[init] Clamping expert bias (max={max_bias:.2f} -> clamp 1.0)")
                bias_vals.clamp_(-1.0, 1.0)
    model.eval()
    print("[init] Model ready.")
    print(f"[init] Experts: {DOMAIN_NAMES}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] Total params: {n_params:,}")
    return model


def encode_text(model, text: str):
    """Encode text and run through HDIMModel. Returns (exported_invariant, expert_weights, dominant_expert, out, aux_state)."""
    text = str(text)
    with torch.no_grad():
        enc = model.encode_texts([text], device=DEVICE)
        dom = torch.zeros(1, dtype=torch.long, device=DEVICE)
        out, _, _, _, aux_state = model(enc, dom, return_state=True, memory_mode="retrieve")
        expert_weights = aux_state.expert_usage.cpu()
        # Normalize routing scores to probabilities for display/dominant detection
        weights_sum = expert_weights.sum().clamp(min=1e-8)
        expert_probs = expert_weights / weights_sum
        top_idx = expert_probs.argmax().item()
        exported_inv = aux_state.exported_invariant
        return exported_inv, expert_probs, DOMAIN_NAMES[top_idx], out, aux_state


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
    return F.cosine_similarity(a, b, dim=-1).item()


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
                text1_expert = DOMAIN_NAMES[last_expert_weights.argmax().item()]
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
            # Built-in expert response when no LLM backend configured
            expert_descriptions = {
                "math": "Mathematical reasoning, equations, proofs, numerical analysis",
                "language": "Linguistic analysis, semantics, grammar, text comprehension",
                "code": "Programming, algorithms, software architecture, debugging",
                "science": "Scientific method, physics, chemistry, biology, experiments",
            }
            conf = expert_weights.max().item()
            print(f"\nExpert [{dominant.upper()}] (confidence: {conf:.1%}):")
            print(f"  Domain: {expert_descriptions.get(dominant, 'general')}")
            print(f"  Routed via Clifford invariant (norm={invariant.norm().item():.4f})")
            if conf > 0.5:
                print(f"  High confidence routing to {dominant} expert.")
            elif conf > 0.3:
                print(f"  Moderate confidence — input spans multiple domains.")
            else:
                print(f"  Low confidence — input is ambiguous across domains.")
            print(f"  To get full AI responses, set HDIM_LLM_BACKEND=openai or anthropic")
        print()


if __name__ == "__main__":
    main()
