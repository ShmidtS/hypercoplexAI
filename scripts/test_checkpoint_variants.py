#!/usr/bin/env python
"""
Checkpoint Variant Testing Script for HDIM.

Tests a trained checkpoint across multiple configurations:
 1. Standard HDIM model (baseline)
 2. With MoE kernel (--moe_kernel flag)
 3. With different memory types (titans, hbma, cls)

Evaluates on real_pairs_v10.json and outputs comparison table.

Usage:
 python scripts/test_checkpoint_variants.py --checkpoint artifacts/gpu_training/checkpoints/best.pt
 python scripts/test_checkpoint_variants.py --checkpoint path/to/ckpt.pt --encoder_type modernbert

Output:
 - Markdown table with metrics: pair_margin, STS, score
 - JSON results saved to output_dir
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.hdim_model import HDIMConfig
from src.models.model_factory import (
    build_sbert_hdim_model,
    build_modernbert_hdim_model,
    build_hdim_model,
    _patch_moe_kernel,
    _patch_soft_router,
)
from src.models.metrics import compute_all_metrics


# ============================================================================
# Configuration
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
PAIRS_LIMIT = 128

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "real_pairs_v10.json"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class VariantResult:
    """Results for a single configuration variant."""
    variant: str
    pair_margin: float
    sts_exported: float
    score: float
    expert_usage: List[float]
    notes: str


# ============================================================================
# Data Loading
# ============================================================================

def load_real_pairs(path: Path, n: int = 128):
    """Load real pairs from JSON dataset."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    pos = [d for d in data if d["relation"] == "positive"][: n // 2]
    neg = [d for d in data if d["relation"] == "negative"][: n // 2]
    pairs = pos + neg

    texts_a = [p["source_text"] for p in pairs]
    texts_b = [p["target_text"] for p in pairs]
    labels = torch.tensor([1.0 if p["relation"] == "positive" else 0.0 for p in pairs])
    domains = torch.zeros(len(pairs), dtype=torch.long)

    return texts_a, texts_b, labels, domains


def create_dataloader(texts_a: List[str], texts_b: List[str], labels: torch.Tensor, batch_size: int):
    """Create a simple dataloader for metrics computation."""
    from torch.utils.data import Dataset, DataLoader

    class PairsDataset(Dataset):
        def __init__(self, texts_a, texts_b, labels):
            self.texts_a = texts_a
            self.texts_b = texts_b
            self.labels = labels

        def __len__(self):
            return len(self.texts_a)

        def __getitem__(self, idx):
            return {
                "text": self.texts_a[idx],
                "pair_text": self.texts_b[idx],
                "pair_relation_label": self.labels[idx],
                "domain_id": torch.tensor(0, dtype=torch.long),
                "pair_domain_id": torch.tensor(0, dtype=torch.long),
                "pair_group_id": torch.tensor(idx, dtype=torch.long),
            }

    dataset = PairsDataset(texts_a, texts_b, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ============================================================================
# Model Builders
# ============================================================================

def build_standard_model(encoder_type: str, device: str, hidden_dim: int = 256):
    """Build standard HDIM model matching training config (soft_router, shared_expert, etc.)."""
    cfg = HDIMConfig(hidden_dim=hidden_dim, num_experts=4, num_domains=4, top_k=2)

    if encoder_type == "modernbert":
        model = build_modernbert_hdim_model(
            cfg,
            soft_router=True,
            freeze_modernbert=True,
        )
    else:
        model = build_sbert_hdim_model(
            cfg,
            soft_router=True,
            freeze_sbert=True,
        )

    # Enable Phase 26 features (must match training config)
    if hasattr(model, 'enable_shared_expert'):
        model.enable_shared_expert()
    if hasattr(model, 'enable_aux_loss_free'):
        model.enable_aux_loss_free(aux_lr=0.001)
    if hasattr(model, 'enable_expert_ortho'):
        model.enable_expert_ortho()

    model.to(device)
    return model, "Standard HDIM (soft_router + shared_expert)"


def build_moe_kernel_model(encoder_type: str, device: str, hidden_dim: int = 256):
    """Build model with MoE kernel."""
    cfg = HDIMConfig(hidden_dim=hidden_dim, num_experts=4, num_domains=4, top_k=2)

    if encoder_type == "modernbert":
        model = build_modernbert_hdim_model(
            cfg,
            soft_router=False,
            freeze_modernbert=True,
        )
    else:
        model = build_sbert_hdim_model(
            cfg,
            soft_router=False,
            freeze_sbert=True,
        )

    # Patch with MoEKernel
    _patch_moe_kernel(
        model.core_model,
        expert_names=["math", "language", "code", "science"],
        z_loss_weight=0.01,
        ortho_loss_weight=0.01,
    )

    model.to(device)
    return model, "HDIM + MoEKernel (Phase 28)"


def build_soft_router_model(encoder_type: str, device: str, hidden_dim: int = 256):
    """Build model with SoftMoERouter (matching training config)."""
    cfg = HDIMConfig(hidden_dim=hidden_dim, num_experts=4, num_domains=4, top_k=2)

    if encoder_type == "modernbert":
        model = build_modernbert_hdim_model(
            cfg,
            soft_router=True,
            freeze_modernbert=True,
        )
    else:
        model = build_sbert_hdim_model(
            cfg,
            soft_router=True,
            freeze_sbert=True,
        )

    # Enable Phase 26 features (must match training config)
    if hasattr(model, 'enable_shared_expert'):
        model.enable_shared_expert()
    if hasattr(model, 'enable_aux_loss_free'):
        model.enable_aux_loss_free(aux_lr=0.001)
    if hasattr(model, 'enable_expert_ortho'):
        model.enable_expert_ortho()

    model.to(device)
    return model, "HDIM + SoftMoERouter (Phase 26)"


def build_memory_variant(encoder_type: str, memory_type: str, device: str, hidden_dim: int = 256):
    """Build model with specific memory type (matching training config)."""
    cfg = HDIMConfig(
        hidden_dim=hidden_dim,
        num_experts=4,
        num_domains=4,
        top_k=2,
        memory_type=memory_type,
    )

    if encoder_type == "modernbert":
        model = build_modernbert_hdim_model(
            cfg,
            soft_router=True,
            freeze_modernbert=True,
        )
    else:
        model = build_sbert_hdim_model(
            cfg,
            soft_router=True,
            freeze_sbert=True,
        )

    # Enable Phase 26 features (must match training config)
    if hasattr(model, 'enable_shared_expert'):
        model.enable_shared_expert()
    if hasattr(model, 'enable_aux_loss_free'):
        model.enable_aux_loss_free(aux_lr=0.001)
    if hasattr(model, 'enable_expert_ortho'):
        model.enable_expert_ortho()

    model.to(device)
    return model, f"HDIM + {memory_type.upper()} memory"


# ============================================================================
# Evaluation
# ============================================================================

def load_checkpoint(model, checkpoint_path: Path, device: str):
    """Load checkpoint weights into model."""
    if not checkpoint_path.exists():
        return f"Checkpoint not found: {checkpoint_path}"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Load with strict=False to handle architecture differences
    result = model.load_state_dict(state_dict, strict=False)

    epoch = ckpt.get("current_epoch", ckpt.get("epoch", "?"))
    score = ckpt.get("score", None)

    notes = []
    if epoch != "?":
        notes.append(f"epoch={epoch}")
    if score is not None:
        notes.append(f"score={score:.4f}")
    if result.missing_keys:
        notes.append(f"{len(result.missing_keys)} missing keys")
    if result.unexpected_keys:
        notes.append(f"{len(result.unexpected_keys)} unexpected keys")

    return f"Loaded ({', '.join(notes)})"


def compute_metrics_simple(model, texts_a, texts_b, labels, device):
    """Compute pair_margin and STS metrics directly."""
    model.eval()
    dev = torch.device(device)

    all_inv_a = []
    all_inv_b = []

    with torch.no_grad():
        for i in range(0, len(texts_a), BATCH_SIZE):
            batch_a = texts_a[i:i + BATCH_SIZE]
            batch_b = texts_b[i:i + BATCH_SIZE]

            enc_a = model.encode_texts(batch_a, device=dev)
            enc_b = model.encode_texts(batch_b, device=dev)

            dom = torch.zeros(enc_a.shape[0], dtype=torch.long, device=dev)
            _, _, inv_a, _ = model(enc_a, dom, return_state=True, memory_mode="none")
            _, _, inv_b, _ = model(enc_b, dom, return_state=True, memory_mode="none")

            all_inv_a.append(inv_a.cpu())
            all_inv_b.append(inv_b.cpu())

    inv_a = torch.cat(all_inv_a, dim=0)
    inv_b = torch.cat(all_inv_b, dim=0)

    # Cosine similarity
    cos_sim = F.cosine_similarity(inv_a, inv_b, dim=-1)

    # Separate positive and negative
    labels = labels[:len(cos_sim)]
    pos_mask = labels > 0.5
    neg_mask = ~pos_mask

    pos_sim = cos_sim[pos_mask].mean().item() if pos_mask.any() else 0.0
    neg_sim = cos_sim[neg_mask].mean().item() if neg_mask.any() else 0.0
    pair_margin = pos_sim - neg_sim
    sts_exported = cos_sim.mean().item()

    # Compute score (same formula as gpu_train.py)
    score = pair_margin * (1.0 - abs(sts_exported - 0.5) * 2.0)

    return pair_margin, sts_exported, score


def get_expert_usage(model):
    """Get expert usage statistics for MoE models."""
    try:
        moe = model.core_model.pipeline.moe
        if hasattr(moe, "kernel"):
            # MoEKernel
            with torch.no_grad():
                dummy = torch.randn(1, moe.kernel.input_dim, device=next(model.parameters()).device)
                _, state = moe.kernel(dummy)
                return state.expert_usage.tolist()
        if hasattr(moe, "train_scores"):
            return moe.train_scores.tolist()
    except Exception:
        pass
    return []


# ============================================================================
# Output Formatting
# ============================================================================

def format_markdown_table(results: List[VariantResult]) -> str:
    """Format results as Markdown table."""
    lines = [
        "# Checkpoint Variant Comparison",
        "",
        f"**Device:** {DEVICE.upper()}",
        f"**Dataset:** `{DATA_PATH.name}` ({PAIRS_LIMIT} pairs)",
        "",
        "## Results",
        "",
        "| Variant | pair_margin | STS_exported | Score | Expert Usage | Notes |",
        "|---------|-------------|--------------|-------|--------------|-------|",
    ]

    for r in results:
        usage_str = ", ".join(f"{u:.3f}" for u in r.expert_usage) if r.expert_usage else "N/A"
        lines.append(
            f"| {r.variant} | {r.pair_margin:.4f} | {r.sts_exported:.4f} | "
            f"{r.score:.4f} | {usage_str} | {r.notes} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- **pair_margin**: Difference in cosine similarity between positive and negative pairs.",
        "- **STS_exported**: Mean cosine similarity of exported invariants.",
        "- **Score**: Combined metric (pair_margin * (1 - |STS - 0.5| * 2)).",
        "- **Expert Usage**: Distribution across domain experts (for MoE variants).",
    ])

    return "\n".join(lines)


def save_results(results: List[VariantResult], output_dir: Path):
    """Save results as Markdown and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Markdown
    md_path = output_dir / "checkpoint_comparison.md"
    md_content = format_markdown_table(results)
    md_path.write_text(md_content, encoding="utf-8")
    print(f"\nMarkdown saved: {md_path}")

    # JSON
    json_path = output_dir / "checkpoint_comparison.json"
    json_data = {"device": DEVICE, "pairs_limit": PAIRS_LIMIT, "results": [asdict(r) for r in results]}
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"JSON saved: {json_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test checkpoint across multiple configurations")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--encoder_type", type=str, default="sbert", choices=["sbert", "modernbert"],
                        help="Encoder type (default: sbert)")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension (must match checkpoint, default: 256)")
    parser.add_argument("--output_dir", type=str, default="artifacts/checkpoint_test",
                        help="Output directory for results (default: artifacts/checkpoint_test)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("HDIM Checkpoint Variant Testing")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Encoder: {args.encoder_type}")
    print(f"Dataset: {DATA_PATH}")

    # Load data
    print("\n[1] Loading real pairs...")
    texts_a, texts_b, labels, domains = load_real_pairs(DATA_PATH, n=PAIRS_LIMIT)
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    print(f"Loaded {len(texts_a)} pairs ({n_pos:.0f} positive, {n_neg:.0f} negative)")

    results: List[VariantResult] = []

    # Variant 1: Standard HDIM
    print("\n[2] Testing Standard HDIM...")
    model_std, notes_std = build_standard_model(args.encoder_type, DEVICE, args.hidden_dim)
    load_notes = load_checkpoint(model_std, checkpoint_path, DEVICE)
    pm_std, sts_std, score_std = compute_metrics_simple(model_std, texts_a, texts_b, labels, DEVICE)
    usage_std = get_expert_usage(model_std)
    results.append(VariantResult(
        variant="Standard HDIM",
        pair_margin=round(pm_std, 4),
        sts_exported=round(sts_std, 4),
        score=round(score_std, 4),
        expert_usage=[round(u, 4) for u in usage_std],
        notes=load_notes,
    ))
    print(f"  pair_margin={pm_std:.4f}, STS={sts_std:.4f}, score={score_std:.4f}")
    del model_std
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Variant 2: MoE Kernel
    print("\n[3] Testing HDIM + MoEKernel...")
    model_moe, notes_moe = build_moe_kernel_model(args.encoder_type, DEVICE, args.hidden_dim)
    load_notes_moe = load_checkpoint(model_moe, checkpoint_path, DEVICE)
    pm_moe, sts_moe, score_moe = compute_metrics_simple(model_moe, texts_a, texts_b, labels, DEVICE)
    usage_moe = get_expert_usage(model_moe)
    results.append(VariantResult(
        variant="HDIM + MoEKernel",
        pair_margin=round(pm_moe, 4),
        sts_exported=round(sts_moe, 4),
        score=round(score_moe, 4),
        expert_usage=[round(u, 4) for u in usage_moe],
        notes=load_notes_moe,
    ))
    print(f"  pair_margin={pm_moe:.4f}, STS={sts_moe:.4f}, score={score_moe:.4f}")
    print(f"  expert_usage: {usage_moe}")
    del model_moe
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Variant 3: SoftMoERouter
    print("\n[4] Testing HDIM + SoftMoERouter...")
    model_soft, notes_soft = build_soft_router_model(args.encoder_type, DEVICE, args.hidden_dim)
    load_notes_soft = load_checkpoint(model_soft, checkpoint_path, DEVICE)
    pm_soft, sts_soft, score_soft = compute_metrics_simple(model_soft, texts_a, texts_b, labels, DEVICE)
    usage_soft = get_expert_usage(model_soft)
    results.append(VariantResult(
        variant="HDIM + SoftMoERouter",
        pair_margin=round(pm_soft, 4),
        sts_exported=round(sts_soft, 4),
        score=round(score_soft, 4),
        expert_usage=[round(u, 4) for u in usage_soft],
        notes=load_notes_soft,
    ))
    print(f"  pair_margin={pm_soft:.4f}, STS={sts_soft:.4f}, score={score_soft:.4f}")
    del model_soft
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Variant 4-6: Memory types (titans, hbma, cls)
    for mem_type in ["titans", "hbma", "cls"]:
        print(f"\n[5] Testing HDIM + {mem_type.upper()} memory...")
        model_mem, notes_mem = build_memory_variant(args.encoder_type, mem_type, DEVICE, args.hidden_dim)
        load_notes_mem = load_checkpoint(model_mem, checkpoint_path, DEVICE)
        pm_mem, sts_mem, score_mem = compute_metrics_simple(model_mem, texts_a, texts_b, labels, DEVICE)
        usage_mem = get_expert_usage(model_mem)
        results.append(VariantResult(
            variant=f"HDIM + {mem_type.upper()}",
            pair_margin=round(pm_mem, 4),
            sts_exported=round(sts_mem, 4),
            score=round(score_mem, 4),
            expert_usage=[round(u, 4) for u in usage_mem],
            notes=load_notes_mem,
        ))
        print(f"  pair_margin={pm_mem:.4f}, STS={sts_mem:.4f}, score={score_mem:.4f}")
        del model_mem
        torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Save results
    print("\n[6] Saving results...")
    save_results(results, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(format_markdown_table(results))

    return 0


if __name__ == "__main__":
    sys.exit(main())
