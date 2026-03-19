#!/usr/bin/env python
"""SOTA Comparison Benchmark for HDIM.

Standard benchmarks:
- STS12-16 (Semantic Textual Similarity)
- SICK-R (Relatedness)
- MTEB tasks (optional)

Metrics reported:
- Spearman correlation (standard)
- Pearson correlation
- STS average

Comparison models:
- SBERT (all-MiniLM-L6-v2)
- E5 (intfloat/e5-base)
- BGE (BAAI/bge-base-en)

This script:
1. Loads test data from RealPairsDataset / real_pairs_v10.json
2. Computes embeddings through HDIM pipeline
3. Calculates cosine similarity between pairs
4. Computes Spearman correlation with ground truth
5. Compares with baseline models (SBERT, E5, BGE)
6. Outputs results table
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.hdim_model import HDIMConfig, HDIMModel
from src.models.model_factory import build_sbert_hdim_model, _patch_moe_kernel
from src.models.sbert_encoder import SBERTEncoder


# ============================================================================
# Configuration
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAIRS_LIMIT = 256  # Number of pairs to use for evaluation

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "real_pairs_v10.json"
CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "gpu_training" / "checkpoints" / "best.pt"

# Baseline model names (HuggingFace)
BASELINE_MODELS = {
    "SBERT": "sentence-transformers/all-MiniLM-L6-v2",
    "E5": "intfloat/e5-base",
    "BGE": "BAAI/bge-base-en",
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class BenchmarkResult:
    """Results for a single model."""
    model_name: str
    spearman_corr: float
    pearson_corr: float
    num_pairs: int
    positive_mean_sim: float
    negative_mean_sim: float
    pair_margin: float
    notes: str


# ============================================================================
# Data Loading
# ============================================================================

def load_real_pairs(path: Path, n: int = 256) -> Tuple[List[str], List[str], List[float]]:
    """Load real pairs from JSON dataset.

    Returns:
        texts_a: List of source texts
        texts_b: List of target texts
        scores: List of ground truth similarity scores (1.0 for positive, 0.0 for negative)
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    pos = [d for d in data if d["relation"] == "positive"][: n // 2]
    neg = [d for d in data if d["relation"] == "negative"][: n // 2]
    pairs = pos + neg

    texts_a = [p["source_text"] for p in pairs]
    texts_b = [p["target_text"] for p in pairs]
    scores = [1.0 if p["relation"] == "positive" else 0.0 for p in pairs]

    return texts_a, texts_b, scores


# ============================================================================
# Baseline Model Embeddings
# ============================================================================

def compute_baseline_embeddings(
    texts: List[str],
    model_name: str,
    device: str,
) -> torch.Tensor:
    """Compute embeddings using a baseline sentence-transformer model."""
    from sentence_transformers import SentenceTransformer

    # Load model on CPU, encode, then move to device
    model = SentenceTransformer(model_name, device="cpu")

    with torch.no_grad():
        embeddings = model.encode(
            texts,
            convert_to_tensor=True,
            device="cpu",
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    return embeddings


def compute_hdim_embeddings(
    model: HDIMModel,
    texts: List[str],
    device: str,
    batch_size: int = 16,
) -> torch.Tensor:
    """Compute embeddings using HDIM pipeline."""
    model.eval()
    dev = torch.device(device)
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = model.encode_texts(batch, device=dev)
            dom = torch.zeros(enc.shape[0], dtype=torch.long, device=dev)
            _, _, inv, _ = model(enc, dom, return_state=True, memory_mode="none")
            all_embeddings.append(inv.cpu())

    return torch.cat(all_embeddings, dim=0)


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_cosine_similarity(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
) -> torch.Tensor:
    """Compute cosine similarity between paired embeddings."""
    return F.cosine_similarity(emb_a, emb_b, dim=-1)


def compute_correlations(
    predictions: torch.Tensor,
    ground_truth: List[float],
) -> Tuple[float, float]:
    """Compute Spearman and Pearson correlations."""
    pred_np = predictions.numpy()
    gt_np = torch.tensor(ground_truth).numpy()

    spearman_corr, _ = stats.spearmanr(pred_np, gt_np)
    pearson_corr, _ = stats.pearsonr(pred_np, gt_np)

    return float(spearman_corr), float(pearson_corr)


def compute_pair_statistics(
    similarities: torch.Tensor,
    labels: List[float],
) -> Tuple[float, float, float]:
    """Compute mean similarity for positive/negative pairs and margin."""
    labels_t = torch.tensor(labels)
    pos_mask = labels_t > 0.5
    neg_mask = ~pos_mask

    pos_sim = similarities[pos_mask].mean().item() if pos_mask.any() else 0.0
    neg_sim = similarities[neg_mask].mean().item() if neg_mask.any() else 0.0
    margin = pos_sim - neg_sim

    return pos_sim, neg_sim, margin


# ============================================================================
# Model Builders
# ============================================================================

def build_hdim_model_with_checkpoint(device: str) -> Tuple[HDIMModel, str]:
    """Build HDIM model with MoEKernel and load trained checkpoint."""
    cfg = HDIMConfig(
        hidden_dim=64,
        num_experts=4,
        num_domains=4,
        top_k=2,
        expert_names=["math", "language", "code", "science"],
    )
    model = build_sbert_hdim_model(
        cfg,
        soft_router=False,
        freeze_sbert=True,
    )
    _patch_moe_kernel(
        model.core_model,
        expert_names=["math", "language", "code", "science"],
        z_loss_weight=0.01,
        ortho_loss_weight=0.01,
    )
    model.to(device)

    # Load checkpoint if available
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        notes = f"Loaded checkpoint: {CHECKPOINT_PATH.name}"
    else:
        notes = "No checkpoint, using random weights"

    return model, notes


# ============================================================================
# Benchmark Runners
# ============================================================================

def run_baseline_benchmark(
    model_name: str,
    model_path: str,
    texts_a: List[str],
    texts_b: List[str],
    scores: List[float],
    device: str,
) -> BenchmarkResult:
    """Run benchmark for a baseline model."""
    print(f"  Encoding with {model_name}...")

    # Compute embeddings
    emb_a = compute_baseline_embeddings(texts_a, model_path, device)
    emb_b = compute_baseline_embeddings(texts_b, model_path, device)

    # Compute similarities
    similarities = compute_cosine_similarity(emb_a, emb_b)

    # Compute metrics
    spearman, pearson = compute_correlations(similarities, scores)
    pos_sim, neg_sim, margin = compute_pair_statistics(similarities, scores)

    return BenchmarkResult(
        model_name=model_name,
        spearman_corr=round(spearman, 4),
        pearson_corr=round(pearson, 4),
        num_pairs=len(scores),
        positive_mean_sim=round(pos_sim, 4),
        negative_mean_sim=round(neg_sim, 4),
        pair_margin=round(margin, 4),
        notes=f"Pre-trained: {model_path}",
    )


def run_hdim_benchmark(
    model: HDIMModel,
    texts_a: List[str],
    texts_b: List[str],
    scores: List[float],
    device: str,
    notes: str,
) -> BenchmarkResult:
    """Run benchmark for HDIM model."""
    print("  Encoding with HDIM...")

    # Compute embeddings
    emb_a = compute_hdim_embeddings(model, texts_a, device)
    emb_b = compute_hdim_embeddings(model, texts_b, device)

    # Compute similarities
    similarities = compute_cosine_similarity(emb_a, emb_b)

    # Compute metrics
    spearman, pearson = compute_correlations(similarities, scores)
    pos_sim, neg_sim, margin = compute_pair_statistics(similarities, scores)

    return BenchmarkResult(
        model_name="HDIM (MoEKernel)",
        spearman_corr=round(spearman, 4),
        pearson_corr=round(pearson, 4),
        num_pairs=len(scores),
        positive_mean_sim=round(pos_sim, 4),
        negative_mean_sim=round(neg_sim, 4),
        pair_margin=round(margin, 4),
        notes=notes,
    )


# ============================================================================
# Output Formatting
# ============================================================================

def format_results_table(results: List[BenchmarkResult]) -> str:
    """Format results as Markdown table."""
    lines = [
        "# SOTA Comparison Benchmark Results",
        "",
        f"**Device:** {DEVICE.upper()}",
        f"**Dataset:** `{DATA_PATH.name}` ({PAIRS_LIMIT} pairs)",
        "",
        "## Results Table",
        "",
        "| Model | Spearman | Pearson | Pos Sim | Neg Sim | Margin | Notes |",
        "|-------|----------|---------|---------|---------|--------|-------|",
    ]

    # Sort by Spearman correlation (descending)
    sorted_results = sorted(results, key=lambda r: r.spearman_corr, reverse=True)

    for r in sorted_results:
        lines.append(
            f"| {r.model_name} | {r.spearman_corr:.4f} | {r.pearson_corr:.4f} | "
            f"{r.positive_mean_sim:.4f} | {r.negative_mean_sim:.4f} | "
            f"{r.pair_margin:.4f} | {r.notes} |"
        )

    lines.extend([
        "",
        "## Metrics Explanation",
        "",
        "- **Spearman**: Rank correlation coefficient (standard STS metric)",
        "- **Pearson**: Linear correlation coefficient",
        "- **Pos Sim**: Mean cosine similarity for positive pairs (semantically similar)",
        "- **Neg Sim**: Mean cosine similarity for negative pairs (semantically dissimilar)",
        "- **Margin**: Pos Sim - Neg Sim (higher = better discrimination)",
        "",
        "## Interpretation",
        "",
        "- Spearman > 0.7: Strong positive correlation",
        "- Spearman 0.4-0.7: Moderate correlation",
        "- Spearman < 0.4: Weak correlation",
        "- Higher margin indicates better separation between similar/dissimilar pairs",
    ])

    return "\n".join(lines)


def save_results(results: List[BenchmarkResult], output_dir: Path):
    """Save results as Markdown and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Markdown
    md_path = output_dir / "sota_benchmark_results.md"
    md_content = format_results_table(results)
    md_path.write_text(md_content, encoding="utf-8")
    print(f"\nMarkdown saved: {md_path}")

    # JSON
    json_path = output_dir / "sota_benchmark_results.json"
    results_data = {
        "device": DEVICE,
        "pairs_limit": PAIRS_LIMIT,
        "data_path": str(DATA_PATH),
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")
    print(f"JSON saved: {json_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("HDIM vs SOTA Models Benchmark")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATA_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    # Load data
    print("\n[1] Loading real pairs...")
    texts_a, texts_b, scores = load_real_pairs(DATA_PATH, n=PAIRS_LIMIT)
    n_pos = sum(1 for s in scores if s > 0.5)
    n_neg = len(scores) - n_pos
    print(f"  Loaded {len(texts_a)} pairs ({n_pos} positive, {n_neg} negative)")

    results: List[BenchmarkResult] = []

    # Run baseline benchmarks
    print("\n[2] Running baseline model benchmarks...")
    for model_name, model_path in BASELINE_MODELS.items():
        try:
            result = run_baseline_benchmark(
                model_name, model_path,
                texts_a, texts_b, scores,
                DEVICE,
            )
            results.append(result)
            print(f"    {model_name}: Spearman={result.spearman_corr:.4f}, Margin={result.pair_margin:.4f}")
        except Exception as e:
            print(f"    {model_name}: FAILED - {e}")

    # Run HDIM benchmark
    print("\n[3] Running HDIM benchmark...")
    try:
        hdim_model, hdim_notes = build_hdim_model_with_checkpoint(DEVICE)
        result = run_hdim_benchmark(
            hdim_model,
            texts_a, texts_b, scores,
            DEVICE, hdim_notes,
        )
        results.append(result)
        print(f"    HDIM: Spearman={result.spearman_corr:.4f}, Margin={result.pair_margin:.4f}")
        del hdim_model
        torch.cuda.empty_cache() if DEVICE == "cuda" else None
    except Exception as e:
        print(f"    HDIM: FAILED - {e}")

    # Save results
    print("\n[4] Saving results...")
    output_dir = Path(__file__).resolve().parents[1] / "artifacts" / "benchmark"
    save_results(results, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(format_results_table(results))

    return 0


if __name__ == "__main__":
    sys.exit(main())
