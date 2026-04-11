#!/usr/bin/env python
"""GPU Memory Profiling Script for RTX 3070 Laptop (8.6GB)

Profile memory usage during training to identify bottlenecks and
validate batch_size=24 stability with MoE + TitansMemory.

Usage:
    python scripts/gpu_memory_profile.py
"""

import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def profile_gpu_memory():
    """Profile GPU memory allocation at different stages."""

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"=== GPU Memory Profile ===")
    print(f"Device: {gpu_name}")
    print(f"Total VRAM: {total_memory:.2f} GB")
    print()

    # Reset stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Stage 1: Baseline
    baseline = torch.cuda.memory_allocated() / 1024**3
    print(f"[1] Baseline (empty): {baseline:.3f} GB")

    # Stage 2: Import SBERT
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    sbert.to(device)
    after_sbert = torch.cuda.memory_allocated() / 1024**3
    print(f"[2] After SBERT load: {after_sbert:.3f} GB (+{after_sbert - baseline:.3f})")

    # Stage 3: Import HDIM model
    from src.models.model_factory import build_hdim_model
    from src.models.hdim_model import HDIMConfig

    config = HDIMConfig(
        hidden_dim=256,
        num_experts=4,
        num_domains=4,
    )

    # Build with MoE + TitansMemory flags (passed to model_factory)
    model = build_hdim_model(
        config,
        use_soft_router=True,
        use_shared_expert=True,
        use_aux_loss_free=True,
        use_expert_ortho=True,
        use_moe_kernel=True,
    )
    model.to(device)
    after_model = torch.cuda.memory_allocated() / 1024**3
    print(f"[3] After HDIM model: {after_model:.3f} GB (+{after_model - after_sbert:.3f})")

    # Stage 4: Simulate batch forward pass
    batch_size = 24
    seq_len = 128
    hidden_dim = 256

    # Simulate input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Forward pass (no grad)
    with torch.no_grad():
        # Simulate encoder output
        encoder_out = torch.randn(batch_size, seq_len, hidden_dim, device=device)

        # Track memory after forward
        torch.cuda.reset_peak_memory_stats()
        _ = model(encoder_out) if hasattr(model, 'forward') else None

    after_forward = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[4] Peak during forward (batch={batch_size}): {after_forward:.3f} GB")

    # Stage 5: With gradients
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Simulate training step
    encoder_out = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    output = model(encoder_out) if hasattr(model, 'forward') else encoder_out

    # Fake loss
    loss = output.mean()
    loss.backward()

    after_backward = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[5] Peak during backward: {after_backward:.3f} GB")

    # Stage 6: Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    after_step = torch.cuda.memory_allocated() / 1024**3
    print(f"[6] After optimizer step: {after_step:.3f} GB")

    # Summary
    print()
    print("=== Memory Summary ===")
    print(f"Baseline:       {baseline:.3f} GB")
    print(f"Peak (forward): {after_forward:.3f} GB")
    print(f"Peak (backward):{after_backward:.3f} GB")
    print(f"Current:        {after_step:.3f} GB")
    print()

    # Estimate headroom
    headroom = total_memory - after_backward
    print(f"Headroom: {headroom:.3f} GB ({headroom/total_memory*100:.1f}% of total)")

    # Batch size recommendation
    if headroom > 1.5:
        print("RECOMMENDATION: batch_size=32 may be feasible")
    elif headroom > 0.5:
        print("RECOMMENDATION: batch_size=24 is optimal")
    else:
        print("WARNING: batch_size=24 may cause OOM on edge cases")

    # Memory fragmentation check
    torch.cuda.empty_cache()
    final = torch.cuda.memory_allocated() / 1024**3
    fragmentation = after_step - final
    print(f"Fragmentation: {fragmentation:.3f} GB (recoverable after empty_cache)")

    return {
        "baseline_gb": baseline,
        "peak_forward_gb": after_forward,
        "peak_backward_gb": after_backward,
        "final_gb": final,
        "headroom_gb": headroom,
    }


if __name__ == "__main__":
    profile_gpu_memory()
