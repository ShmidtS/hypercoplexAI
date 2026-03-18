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
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Fix Windows console encoding for Cyrillic input via pipe
import io

if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.hdim_model import HDIMConfig
from src.models.model_factory import build_sbert_hdim_model, _patch_moe_kernel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAIN_NAMES = ["math", "language", "code", "science"]
DOMAIN_IDX = {n: i for i, n in enumerate(DOMAIN_NAMES)}


CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "gpu_training"
    / "checkpoints"
    / "best.pt"
)


def build_model():
    print(f"[init] Building SBERT + MoEKernel on {DEVICE}...")
    cfg = HDIMConfig(hidden_dim=64, num_experts=4, num_domains=4, top_k=2)
    model = build_sbert_hdim_model(cfg, soft_router=False, freeze_sbert=True)
    _patch_moe_kernel(
        model.core_model,
        expert_names=["math", "language", "code", "science"],
        z_loss_weight=0.01,
        ortho_loss_weight=0.01,
    )
    # Load trained checkpoint if available
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE, weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt)
        result = model.load_state_dict(sd, strict=False)
        score = ckpt.get("score", None)
        epoch = ckpt.get("current_epoch", ckpt.get("epoch", "?"))
        score_str = f", score={score:.4f}" if score is not None else ""
        print(
            f"[init] Checkpoint loaded: epoch={epoch}{score_str} ({len(result.missing_keys)} missing keys)"
        )
    else:
        print(
            f"[init] WARNING: checkpoint not found at {CHECKPOINT_PATH} — using random weights"
        )
    model.to(DEVICE)
    model.eval()
    print("[init] Model ready.")
    print(f"[init] Experts: {DOMAIN_NAMES}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] Total params: {n_params:,}")
    return model


def encode_text(model, text: str):
    """Encode text and run through HDIMModel. Returns (exported_invariant, expert_weights, dominant_expert, out, aux_state)."""
    text = str(text)  # ensure str, not bytes
    with torch.no_grad():
        enc = model.encode_texts([text], device=DEVICE)  # (1, 64)
        dom = torch.zeros(1, dtype=torch.long, device=DEVICE)
        out, routing_weights, invariant, aux_state = model(
            enc, dom, return_state=True, memory_mode="none"
        )
        expert_weights = aux_state.expert_usage.cpu()  # (4,)
        top_idx = expert_weights.argmax().item()
        # Return exported_invariant (clifford space) for domain transfer operations
        exported_inv = aux_state.exported_invariant
        return exported_inv, expert_weights, DOMAIN_NAMES[top_idx], out, aux_state


def format_bar(weights, width=20):
    bars = []
    for i, (name, w) in enumerate(zip(DOMAIN_NAMES, weights.tolist())):
        filled = int(w * width)
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
    print(" Commands: :transfer <domain> | :compare <text> | :experts | :quit")
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
            print(f"[compare]")
            print(f' Text 1: "{last_text[:60]}"')
            print(f' Text 2: "{text2[:60]}"')
            print(f" Cosine similarity (invariant space): {sim:.4f}")
            print(
                f" Text 1 dominant expert: {last_expert_weights.argmax() and DOMAIN_NAMES[last_expert_weights.argmax().item()]}"
            )
            print(f" Text 2 dominant expert: {dom2}")
            print(f" Expert weights (text 2):")
            print(format_bar(ew2))
            continue

        # --- Regular text encoding ---
        invariant, expert_weights, dominant, out, aux_state = encode_text(
            model, user_input
        )
        last_invariant = invariant
        last_text = user_input
        last_expert_weights = expert_weights

        print(f"\nModel:")
        print(f" Dominant expert : {dominant.upper()}")
        print(f" Invariant norm : {invariant.norm().item():.4f}")
        print(f" Expert routing :")
        print(format_bar(expert_weights))
        print()


if __name__ == "__main__":
    main()
