#!/usr/bin/env python
"""Benchmark script for CAN integration in HDIM.

Compares performance of FFN vs CAN experts:
- Parameter count
- Forward pass time
- Backward pass time
- Memory usage

Usage:
    python scripts/benchmark_can_integration.py
"""

import sys
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.clifford_interaction import CliffordInteractionLayer, CliffordFFN
from src.core.moe_kernel import MoEKernel, MoEKernelConfig, DomainExpert


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    params_count: int
    forward_time_ms: float
    backward_time_ms: float
    memory_mb: float
    output_shape: tuple


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def benchmark_forward_backward(
    model: nn.Module,
    x: torch.Tensor,
    num_warmup: int = 5,
    num_iterations: int = 20,
) -> tuple:
    """Benchmark forward and backward pass times.
    
    Returns:
        (forward_time_ms, backward_time_ms)
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            if isinstance(model, MoEKernel):
                output, _ = model(x)
            else:
                output = model(x)
    
    # Sync GPU if available
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Forward benchmark
    forward_times = []
    for _ in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        if isinstance(model, MoEKernel):
            output, _ = model(x)
        else:
            output = model(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        forward_times.append((end - start) * 1000)
    
    forward_time_ms = sum(forward_times) / len(forward_times)
    
    # Backward benchmark (need gradients)
    model.train()
    backward_times = []
    for _ in range(num_iterations):
        x_grad = x.detach().clone().requires_grad_(True)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        if isinstance(model, MoEKernel):
            output, state = model(x_grad)
            loss = output.sum() + state.total_loss()
        else:
            output = model(x_grad)
            loss = output.sum()
        
        loss.backward()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        backward_times.append((end - start) * 1000)
    
    backward_time_ms = sum(backward_times) / len(backward_times)
    
    return forward_time_ms, backward_time_ms


def benchmark_domain_expert(
    input_dim: int,
    hidden_dim: int,
    batch_size: int,
    seq_len: int,
    use_can: bool,
    device: str = "cpu",
) -> BenchmarkResult:
    """Benchmark a single DomainExpert."""
    expert = DomainExpert(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        use_can=use_can,
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    
    forward_ms, backward_ms = benchmark_forward_backward(expert, x)
    
    with torch.no_grad():
        if use_can:
            output = expert(x)
        else:
            output = expert(x)
    
    memory_mb = get_memory_usage()
    
    return BenchmarkResult(
        name=f"DomainExpert({'CAN' if use_can else 'FFN'})",
        params_count=count_parameters(expert),
        forward_time_ms=forward_ms,
        backward_time_ms=backward_ms,
        memory_mb=memory_mb,
        output_shape=tuple(output.shape),
    )


def benchmark_moe_kernel(
    input_dim: int,
    hidden_dim: int,
    num_experts: int,
    batch_size: int,
    seq_len: int,
    use_can: bool,
    device: str = "cpu",
) -> BenchmarkResult:
    """Benchmark MoEKernel with FFN or CAN experts."""
    config = MoEKernelConfig(
        input_dim=input_dim,
        expert_hidden_dim=hidden_dim,
        num_experts=num_experts,
        use_can_experts=use_can,
        expert_names=[f"expert_{i}" for i in range(num_experts)],
    )
    
    kernel = MoEKernel(config).to(device)
    
    if seq_len == 1:
        x = torch.randn(batch_size, input_dim, device=device)
    else:
        x = torch.randn(batch_size, seq_len, input_dim, device=device)
    
    forward_ms, backward_ms = benchmark_forward_backward(kernel, x)
    
    with torch.no_grad():
        output, _ = kernel(x)
    
    memory_mb = get_memory_usage()
    
    return BenchmarkResult(
        name=f"MoEKernel({'CAN' if use_can else 'FFN'})",
        params_count=count_parameters(kernel),
        forward_time_ms=forward_ms,
        backward_time_ms=backward_ms,
        memory_mb=memory_mb,
        output_shape=tuple(output.shape),
    )


def benchmark_clifford_layer(
    dim: int,
    batch_size: int,
    seq_len: int,
    device: str = "cpu",
) -> BenchmarkResult:
    """Benchmark CliffordInteractionLayer."""
    layer = CliffordInteractionLayer(dim=dim).to(device)
    
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    forward_ms, backward_ms = benchmark_forward_backward(layer, x)
    
    with torch.no_grad():
        output = layer(x)
    
    memory_mb = get_memory_usage()
    
    return BenchmarkResult(
        name="CliffordInteractionLayer",
        params_count=count_parameters(layer),
        forward_time_ms=forward_ms,
        backward_time_ms=backward_ms,
        memory_mb=memory_mb,
        output_shape=tuple(output.shape),
    )


def print_results_table(results: List[BenchmarkResult]):
    """Print results in a formatted table."""
    print("\n" + "=" * 90)
    print(f"{'Name':<30} {'Params':>12} {'Forward (ms)':>14} {'Backward (ms)':>14} {'Memory (MB)':>12}")
    print("=" * 90)
    
    for r in results:
        print(f"{r.name:<30} {r.params_count:>12,} {r.forward_time_ms:>14.3f} {r.backward_time_ms:>14.3f} {r.memory_mb:>12.1f}")
    
    print("=" * 90)


def print_comparison_table(ffn_results: List[BenchmarkResult], can_results: List[BenchmarkResult]):
    """Print comparison between FFN and CAN results."""
    print("\n" + "=" * 100)
    print("COMPARISON: FFN vs CAN")
    print("=" * 100)
    print(f"{'Metric':<25} {'FFN':>20} {'CAN':>20} {'Ratio (CAN/FFN)':>20}")
    print("-" * 100)
    
    for ffn, can in zip(ffn_results, can_results):
        if ffn.name.replace("FFN", "") == can.name.replace("CAN", ""):
            params_ratio = can.params_count / ffn.params_count if ffn.params_count > 0 else 0
            forward_ratio = can.forward_time_ms / ffn.forward_time_ms if ffn.forward_time_ms > 0 else 0
            backward_ratio = can.backward_time_ms / ffn.backward_time_ms if ffn.backward_time_ms > 0 else 0
            
            print(f"{'Parameters':<25} {ffn.params_count:>20,} {can.params_count:>20,} {params_ratio:>20.2f}x")
            print(f"{'Forward (ms)':<25} {ffn.forward_time_ms:>20.3f} {can.forward_time_ms:>20.3f} {forward_ratio:>20.2f}x")
            print(f"{'Backward (ms)':<25} {ffn.backward_time_ms:>20.3f} {can.backward_time_ms:>20.3f} {backward_ratio:>20.2f}x")
            print("-" * 100)


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("CAN Integration Benchmark for HDIM")
    print("=" * 60)
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Benchmark parameters
    batch_size = 8
    seq_len = 32
    
    # For CAN, input_dim must be 16 (CliffordInteractionLayer constraint)
    can_input_dim = 16
    can_hidden_dim = 32
    
    # For FFN, we can use larger dimensions
    ffn_input_dim = 64
    ffn_hidden_dim = 128
    
    num_experts = 4
    
    print(f"\nBatch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Number of experts: {num_experts}")
    
    # Reset memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # ========================================
    # Benchmark 1: DomainExpert comparison
    # ========================================
    print("\n" + "-" * 60)
    print("Benchmark 1: DomainExpert (FFN vs CAN)")
    print("-" * 60)
    
    # For fair comparison, use same input_dim
    expert_input_dim = 16  # CAN requires 16
    expert_hidden_dim = 32
    
    ffn_expert_result = benchmark_domain_expert(
        input_dim=expert_input_dim,
        hidden_dim=expert_hidden_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        use_can=False,
        device=device,
    )
    
    can_expert_result = benchmark_domain_expert(
        input_dim=expert_input_dim,
        hidden_dim=expert_hidden_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        use_can=True,
        device=device,
    )
    
    print_results_table([ffn_expert_result, can_expert_result])
    print_comparison_table([ffn_expert_result], [can_expert_result])
    
    # ========================================
    # Benchmark 2: MoEKernel comparison
    # ========================================
    print("\n" + "-" * 60)
    print("Benchmark 2: MoEKernel (FFN vs CAN)")
    print("-" * 60)
    
    # Use same input_dim for fair comparison
    moe_input_dim = 16
    moe_hidden_dim = 32
    
    ffn_moe_result = benchmark_moe_kernel(
        input_dim=moe_input_dim,
        hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        batch_size=batch_size,
        seq_len=seq_len,
        use_can=False,
        device=device,
    )
    
    can_moe_result = benchmark_moe_kernel(
        input_dim=moe_input_dim,
        hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        batch_size=batch_size,
        seq_len=seq_len,
        use_can=True,
        device=device,
    )
    
    print_results_table([ffn_moe_result, can_moe_result])
    print_comparison_table([ffn_moe_result], [can_moe_result])
    
    # ========================================
    # Benchmark 3: CliffordInteractionLayer standalone
    # ========================================
    print("\n" + "-" * 60)
    print("Benchmark 3: CliffordInteractionLayer (standalone)")
    print("-" * 60)
    
    clifford_result = benchmark_clifford_layer(
        dim=16,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )
    
    print_results_table([clifford_result])
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nDomainExpert comparison (input_dim={expert_input_dim}):")
    params_reduction = (1 - can_expert_result.params_count / ffn_expert_result.params_count) * 100
    print(f"  - CAN has {params_reduction:.1f}% fewer parameters than FFN")
    
    if can_expert_result.forward_time_ms < ffn_expert_result.forward_time_ms:
        speedup = ffn_expert_result.forward_time_ms / can_expert_result.forward_time_ms
        print(f"  - CAN is {speedup:.2f}x faster in forward pass")
    else:
        slowdown = can_expert_result.forward_time_ms / ffn_expert_result.forward_time_ms
        print(f"  - CAN is {slowdown:.2f}x slower in forward pass")
    
    print(f"\nMoEKernel comparison (input_dim={moe_input_dim}, {num_experts} experts):")
    moe_params_reduction = (1 - can_moe_result.params_count / ffn_moe_result.params_count) * 100
    print(f"  - CAN MoE has {moe_params_reduction:.1f}% fewer parameters than FFN MoE")
    
    print("\n" + "=" * 60)
    print("Benchmark completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
