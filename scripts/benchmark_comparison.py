#!/usr/bin/env python
"""
Architecture Comparison Benchmark for HDIM.

Compares 4 architectures on real_pairs_v10.json:
  1. MoEKernel (Phase 28) — 4 domain experts (math/language/code/science)
  2. SoftMoERouter baseline — current soft routing
  3. Simple feedforward — HDIMModel without MoE (single expert)
  4. Pure SBERT — only sentence-transformers embeddings, no HDIM

Metrics:
  - pair_margin (difference in similarity positive vs negative pairs)
  - STS_exported (cosine similarity of invariants)
  - Training time (ms per batch)
  - Inference latency (ms per sample)
  - Params count
  - Expert load variance (for MoE models)

Output: Markdown table + JSON results
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.hdim_model import HDIMConfig, HDIMModel
from src.models.model_factory import (
    build_sbert_hdim_model,
    build_hdim_model,
    _patch_moe_kernel,
    _patch_soft_router,
)
from src.models.sbert_encoder import SBERTEncoder


# ============================================================================
# Configuration
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WARMUP_BATCHES = 3
TIMING_BATCHES = 10
PAIRS_LIMIT = 128  # Number of pairs to use for evaluation

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "real_pairs_v10.json"
CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "gpu_training" / "checkpoints" / "best.pt"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class BenchmarkResult:
    """Results for a single architecture."""
    architecture: str
    pair_margin: float
    sts_exported: float
    training_time_ms: float
    inference_latency_ms: float
    params_count: int
    expert_load_variance: float
    expert_usage: List[float]
    notes: str


# ============================================================================
# Data Loading
# ============================================================================

def load_real_pairs(path: Path, n: int = 128) -> tuple:
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


def create_batches(texts: List[str], batch_size: int) -> List[List[str]]:
    """Split texts into batches."""
    return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]


# ============================================================================
# Architecture Builders
# ============================================================================

def build_moe_kernel_model(device: str) -> tuple:
    """Build MoEKernel model (Phase 28) with trained checkpoint."""
    cfg = HDIMConfig(hidden_dim=64, num_experts=4, num_domains=4, top_k=2)
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

    # Load trained checkpoint if available
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        notes = "Loaded trained checkpoint (best.pt)"
    else:
        notes = "No checkpoint found, using random weights"

    return model, notes


def build_soft_moe_baseline(device: str) -> tuple:
    """Build SoftMoERouter baseline model."""
    cfg = HDIMConfig(hidden_dim=64, num_experts=4, num_domains=4, top_k=2)
    model = build_sbert_hdim_model(
        cfg,
        soft_router=True,
        freeze_sbert=True,
        z_loss_weight=0.01,
    )
    model.to(device)
    return model, "SoftMoERouter baseline (random weights)"


def build_simple_feedforward(device: str) -> tuple:
    """Build HDIMModel without MoE (single expert fallback)."""
    cfg = HDIMConfig(hidden_dim=64, num_experts=1, num_domains=4, top_k=1)
    model = build_sbert_hdim_model(
        cfg,
        soft_router=False,
        freeze_sbert=True,
    )
    model.to(device)
    return model, "Single expert (no MoE routing)"


def build_pure_sbert(device: str) -> tuple:
    """Build pure SBERT encoder without HDIM."""
    encoder = SBERTEncoder(
        output_dim=64,
        model_name="paraphrase-multilingual-mpnet-base-v2",
        freeze=True,
    )
    encoder.to(device)
    return encoder, "Pure SBERT embeddings (no HDIM pipeline)"


# ============================================================================
# Evaluation Functions
# ============================================================================

def count_parameters(model) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_pair_margin(
    model,
    texts_a: List[str],
    texts_b: List[str],
    labels: torch.Tensor,
    device: str,
) -> tuple:
    """Compute pair margin and STS metrics."""
    model.eval()
    dev = torch.device(device)

    all_inv_a = []
    all_inv_b = []

    with torch.no_grad():
        # Process in batches
        for i in range(0, len(texts_a), BATCH_SIZE):
            batch_a = texts_a[i:i + BATCH_SIZE]
            batch_b = texts_b[i:i + BATCH_SIZE]

            enc_a = model.encode_texts(batch_a, device=dev)
            enc_b = model.encode_texts(batch_b, device=dev)

            # Get invariants
            dom = torch.zeros(enc_a.shape[0], dtype=torch.long, device=dev)
            _, _, inv_a, _ = model(enc_a, dom, return_state=True, memory_mode="none")
            _, _, inv_b, _ = model(enc_b, dom, return_state=True, memory_mode="none")

            all_inv_a.append(inv_a.cpu())
            all_inv_b.append(inv_b.cpu())

    inv_a = torch.cat(all_inv_a, dim=0)
    inv_b = torch.cat(all_inv_b, dim=0)

    # Cosine similarity for each pair
    cos_sim = F.cosine_similarity(inv_a, inv_b, dim=-1)

    # Separate positive and negative pairs
    labels = labels[:len(cos_sim)]
    pos_mask = labels > 0.5
    neg_mask = ~pos_mask

    pos_sim = cos_sim[pos_mask].mean().item() if pos_mask.any() else 0.0
    neg_sim = cos_sim[neg_mask].mean().item() if neg_mask.any() else 0.0
    pair_margin = pos_sim - neg_sim

    # STS_exported: mean similarity
    sts_exported = cos_sim.mean().item()

    return pair_margin, sts_exported


def compute_pair_margin_pure_sbert(
    encoder,
    texts_a: List[str],
    texts_b: List[str],
    labels: torch.Tensor,
    device: str,
) -> tuple:
    """Compute metrics for pure SBERT encoder."""
    encoder.eval()
    dev = torch.device(device)

    with torch.no_grad():
        # SBERTEncoder uses forward(texts, device=device) API
        enc_a = encoder(texts_a, device=dev)
        enc_b = encoder(texts_b, device=dev)

    cos_sim = F.cosine_similarity(enc_a, enc_b, dim=-1)

    labels = labels[:len(cos_sim)]
    pos_mask = labels > 0.5
    neg_mask = ~pos_mask

    pos_sim = cos_sim[pos_mask].mean().item() if pos_mask.any() else 0.0
    neg_sim = cos_sim[neg_mask].mean().item() if neg_mask.any() else 0.0
    pair_margin = pos_sim - neg_sim
    sts_exported = cos_sim.mean().item()

    return pair_margin, sts_exported


def measure_training_time(
    model,
    texts: List[str],
    device: str,
) -> float:
    """Measure average training time per batch in milliseconds."""
    model.train()
    dev = torch.device(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    # Warmup
    for i in range(WARMUP_BATCHES):
        batch = texts[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        if not batch:
            break
        enc = model.encode_texts(batch, device=dev)
        dom = torch.zeros(enc.shape[0], dtype=torch.long, device=dev)
        optimizer.zero_grad()
        out, _, inv, aux = model(enc, dom, return_state=True, memory_mode="none")
        loss = F.mse_loss(inv, torch.zeros_like(inv))
        if hasattr(aux, "router_loss"):
            loss = loss + 0.01 * aux.router_loss
        loss.backward()
        optimizer.step()

    # Timing
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()

    for i in range(TIMING_BATCHES):
        batch_idx = (WARMUP_BATCHES + i) * BATCH_SIZE
        batch = texts[batch_idx:batch_idx + BATCH_SIZE]
        if not batch:
            break
        enc = model.encode_texts(batch, device=dev)
        dom = torch.zeros(enc.shape[0], dtype=torch.long, device=dev)
        optimizer.zero_grad()
        out, _, inv, aux = model(enc, dom, return_state=True, memory_mode="none")
        loss = F.mse_loss(inv, torch.zeros_like(inv))
        if hasattr(aux, "router_loss"):
            loss = loss + 0.01 * aux.router_loss
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize() if device == "cuda" else None
    elapsed_ms = (time.perf_counter() - start) * 1000

    return elapsed_ms / min(TIMING_BATCHES, len(texts) // BATCH_SIZE)


def measure_inference_latency(
    model,
    texts: List[str],
    device: str,
) -> float:
    """Measure average inference latency per sample in milliseconds."""
    model.eval()
    dev = torch.device(device)

    # Warmup
    with torch.no_grad():
        for i in range(WARMUP_BATCHES):
            batch = texts[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            if not batch:
                break
            enc = model.encode_texts(batch, device=dev)
            dom = torch.zeros(enc.shape[0], dtype=torch.long, device=dev)
            _ = model(enc, dom, return_state=False, memory_mode="none")

    # Timing
    total_samples = 0
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()

    with torch.no_grad():
        for i in range(TIMING_BATCHES):
            batch_idx = i * BATCH_SIZE
            batch = texts[batch_idx:batch_idx + BATCH_SIZE]
            if not batch:
                break
            enc = model.encode_texts(batch, device=dev)
            dom = torch.zeros(enc.shape[0], dtype=torch.long, device=dev)
            _ = model(enc, dom, return_state=False, memory_mode="none")
            total_samples += len(batch)

    torch.cuda.synchronize() if device == "cuda" else None
    elapsed_ms = (time.perf_counter() - start) * 1000

    return elapsed_ms / max(total_samples, 1)


def measure_inference_latency_pure_sbert(
    encoder,
    texts: List[str],
    device: str,
) -> float:
    """Measure inference latency for pure SBERT."""
    encoder.eval()
    dev = torch.device(device)

    # Warmup
    with torch.no_grad():
        for i in range(WARMUP_BATCHES):
            batch = texts[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            if not batch:
                break
            _ = encoder(batch, device=dev)

    total_samples = 0
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()

    with torch.no_grad():
        for i in range(TIMING_BATCHES):
            batch_idx = i * BATCH_SIZE
            batch = texts[batch_idx:batch_idx + BATCH_SIZE]
            if not batch:
                break
            _ = encoder(batch, device=dev)
            total_samples += len(batch)

    torch.cuda.synchronize() if device == "cuda" else None
    elapsed_ms = (time.perf_counter() - start) * 1000

    return elapsed_ms / max(total_samples, 1)


def get_expert_usage(model) -> tuple:
    """Get expert usage statistics for MoE models."""
    try:
        moe = model.core_model.pipeline.moe
        if hasattr(moe, "kernel"):
            # MoEKernel - run a forward pass to get state
            with torch.no_grad():
                dummy = torch.randn(1, moe.kernel.input_dim, device=next(model.parameters()).device)
                _, state = moe.kernel(dummy)
                usage = state.expert_usage.tolist()
                variance = torch.tensor(usage).var().item() if len(usage) > 1 else 0.0
                return variance, usage
        if hasattr(moe, "train_scores"):
            # SoftMoERouter or R3MoERouter
            usage = moe.train_scores.tolist()
            variance = torch.tensor(usage).var().item() if len(usage) > 1 else 0.0
            return variance, usage
    except Exception:
        pass
    return 0.0, []


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_architecture_benchmark(
    name: str,
    model,
    notes: str,
    texts_a: List[str],
    texts_b: List[str],
    labels: torch.Tensor,
    device: str,
    is_pure_sbert: bool = False,
) -> BenchmarkResult:
    """Run benchmark for a single architecture."""
    print(f"\n  Benchmarking: {name}")

    if is_pure_sbert:
        pair_margin, sts = compute_pair_margin_pure_sbert(model, texts_a, texts_b, labels, device)
        train_time = 0.0  # No training for pure encoder
        inf_latency = measure_inference_latency_pure_sbert(model, texts_a + texts_b, device)
        params = sum(p.numel() for p in model.parameters())
        expert_var = 0.0
        expert_usage = []
    else:
        pair_margin, sts = compute_pair_margin(model, texts_a, texts_b, labels, device)
        train_time = measure_training_time(model, texts_a + texts_b, device)
        inf_latency = measure_inference_latency(model, texts_a + texts_b, device)
        params = count_parameters(model)
        expert_var, expert_usage = get_expert_usage(model)

    result = BenchmarkResult(
        architecture=name,
        pair_margin=round(pair_margin, 4),
        sts_exported=round(sts, 4),
        training_time_ms=round(train_time, 2),
        inference_latency_ms=round(inf_latency, 3),
        params_count=params,
        expert_load_variance=round(expert_var, 6),
        expert_usage=[round(u, 4) for u in expert_usage],
        notes=notes,
    )

    print(f"    pair_margin={result.pair_margin:.4f}")
    print(f"    sts_exported={result.sts_exported:.4f}")
    print(f"    inference_latency={result.inference_latency_ms:.3f}ms/sample")
    print(f"    params={result.params_count:,}")

    return result


# ============================================================================
# Output Formatting
# ============================================================================

def format_markdown_table(results: List[BenchmarkResult]) -> str:
    """Format results as Markdown table."""
    lines = [
        "# Architecture Comparison Benchmark",
        "",
        f"Device: **{DEVICE.upper()}**",
        f"Dataset: `data/real_pairs_v10.json` ({PAIRS_LIMIT} pairs)",
        "",
        "## Results",
        "",
        "| Architecture | pair_margin | STS_exported | Train (ms/batch) | Inference (ms/sample) | Params | Expert Var |",
        "|--------------|-------------|--------------|------------------|----------------------|--------|------------|",
    ]

    for r in results:
        lines.append(
            f"| {r.architecture} | {r.pair_margin:.4f} | {r.sts_exported:.4f} | "
            f"{r.training_time_ms:.1f} | {r.inference_latency_ms:.3f} | "
            f"{r.params_count:,} | {r.expert_load_variance:.6f} |"
        )

    lines.extend([
        "",
        "## Expert Usage",
        "",
    ])

    for r in results:
        if r.expert_usage:
            lines.append(f"- **{r.architecture}**: `{r.expert_usage}`")
        lines.append(f"  - {r.notes}")

    lines.extend([
        "",
        "## Notes",
        "",
        "- **pair_margin**: Difference in cosine similarity between positive and negative pairs.",
        "- **STS_exported**: Mean cosine similarity of exported invariants.",
        "- **Expert Var**: Variance of expert load distribution (lower = more balanced).",
    ])

    return "\n".join(lines)


def save_results(results: List[BenchmarkResult], output_dir: Path):
    """Save results as Markdown and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Markdown
    md_path = output_dir / "benchmark_results.md"
    md_content = format_markdown_table(results)
    md_path.write_text(md_content, encoding="utf-8")
    print(f"\n  Markdown saved: {md_path}")

    # JSON - handle NaN values
    json_path = output_dir / "benchmark_results.json"
    import math
    results_data = []
    for r in results:
        d = asdict(r)
        # Replace NaN with None for JSON serialization
        for k, v in d.items():
            if isinstance(v, float) and math.isnan(v):
                d[k] = None
        results_data.append(d)
    json_data = {"device": DEVICE, "pairs_limit": PAIRS_LIMIT, "results": results_data}
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"  JSON saved: {json_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("HDIM Architecture Comparison Benchmark")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATA_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    # Load data
    print("\n[1] Loading real pairs...")
    texts_a, texts_b, labels, domains = load_real_pairs(DATA_PATH, n=PAIRS_LIMIT)
    print(f"  Loaded {len(texts_a)} pairs ({labels.sum().item():.0f} positive, {(labels < 0.5).sum().item():.0f} negative)")

    results: List[BenchmarkResult] = []

    # Architecture 1: MoEKernel (Phase 28)
    print("\n[2] Building MoEKernel (Phase 28)...")
    model_moe, notes_moe = build_moe_kernel_model(DEVICE)
    print(f"  {notes_moe}")
    results.append(run_architecture_benchmark(
        "MoEKernel (Phase 28)",
        model_moe,
        notes_moe,
        texts_a, texts_b, labels,
        DEVICE,
    ))
    del model_moe
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Architecture 2: SoftMoERouter baseline
    print("\n[3] Building SoftMoERouter baseline...")
    model_soft, notes_soft = build_soft_moe_baseline(DEVICE)
    results.append(run_architecture_benchmark(
        "SoftMoERouter",
        model_soft,
        notes_soft,
        texts_a, texts_b, labels,
        DEVICE,
    ))
    del model_soft
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Architecture 3: Simple feedforward
    print("\n[4] Building Simple feedforward (no MoE)...")
    model_simple, notes_simple = build_simple_feedforward(DEVICE)
    results.append(run_architecture_benchmark(
        "Simple Feedforward",
        model_simple,
        notes_simple,
        texts_a, texts_b, labels,
        DEVICE,
    ))
    del model_simple
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Architecture 4: Pure SBERT
    print("\n[5] Building Pure SBERT...")
    encoder_sbert, notes_sbert = build_pure_sbert(DEVICE)
    results.append(run_architecture_benchmark(
        "Pure SBERT",
        encoder_sbert,
        notes_sbert,
        texts_a, texts_b, labels,
        DEVICE,
        is_pure_sbert=True,
    ))
    del encoder_sbert
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Save results
    print("\n[6] Saving results...")
    output_dir = Path(__file__).resolve().parents[1] / "artifacts" / "benchmark"
    save_results(results, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(format_markdown_table(results))

    return 0


if __name__ == "__main__":
    sys.exit(main())
