"""Phase 27c: Performance profiling — torch.profiler for geometric_product, sandwich, memory."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import time
from pathlib import Path

import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.hypercomplex import CliffordAlgebra
from src.core.hbma_memory import HBMAMemory
from src.core.titans_memory import TitansMemoryModule
from src.models.hdim_model import HDIMModel, HDIMConfig


def profile_geometric_product(device='cpu', D=32):
    """Profile Clifford algebra geometric product."""
    print(f'\n=== Geometric Product (D={D}, {device}) ===')
    alg = CliffordAlgebra(p=4, q=1, r=0).to(device)
    a = torch.randn(8, D, device=device)
    b = torch.randn(8, D, device=device)

    # Warmup
    for _ in range(3):
        _ = alg.geometric_product(a, b)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
        with record_function("geometric_product"):
            for _ in range(10):
                c = alg.geometric_product(a, b)

    key_averages = prof.key_averages()
    gp_stats = None
    for entry in key_averages:
        if 'geometric_product' in entry.key:
            gp_stats = {
                'name': entry.key,
                'cpu_time_avg_ms': entry.cpu_time_total / max(entry.count, 1),
                'cpu_time_total_ms': entry.cpu_time_total,
                'count': entry.count,
                'flops': entry.flops,
            }
            break

    print(f'  avg time: {gp_stats["cpu_time_avg_ms"]:.3f} ms/call' if gp_stats else '  (no stats)')
    return gp_stats


def profile_sandwich(device='cpu', D=32):
    """Profile sandwich product."""
    print(f'\n=== Sandwich Product (D={D}, {device}) ===')
    alg = CliffordAlgebra(p=4, q=1, r=0).to(device)
    R = torch.randn(8, D, device=device)
    x = torch.randn(8, D, device=device)

    # Warmup
    for _ in range(3):
        _ = alg.sandwich(R, x)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("sandwich"):
            for _ in range(10):
                s = alg.sandwich(R, x)

    key_averages = prof.key_averages()
    sw_stats = None
    for entry in key_averages:
        if 'sandwich' in entry.key:
            sw_stats = {
                'name': entry.key,
                'cpu_time_avg_ms': entry.cpu_time_total / max(entry.count, 1),
                'cpu_time_total_ms': entry.cpu_time_total,
                'count': entry.count,
            }
            break

    print(f'  avg time: {sw_stats["cpu_time_avg_ms"]:.3f} ms/call' if sw_stats else '  (no stats)')
    return sw_stats


def profile_memory_systems(device='cpu', hidden=64):
    """Profile Titans vs HBMA memory forward passes."""
    results = {}

    for name, mtype in [('Titans', 'titans'), ('HBMA', 'hbma')]:
        print(f'\n=== {name} Memory (hidden={hidden}) ===')
        cfg = HDIMConfig(hidden_dim=hidden, num_domains=4, memory_type=mtype)
        model = HDIMModel(cfg).to(device)
        model.eval()
        x = torch.randn(16, hidden, device=device)
        d = torch.zeros(16, dtype=torch.long, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(x, d)

        t0 = time.time()
        n_iter = 20
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function(f"{name}_forward"):
                with torch.no_grad():
                    for _ in range(n_iter):
                        out, rw, inv = model(x, d)

        elapsed_ms = (time.time() - t0) / n_iter * 1000

        # Count top time consumers
        top_ops = []
        for entry in prof.key_averages().table(sort_by="cpu_time_total", row_limit=5).split('\n'):
            if entry.strip():
                top_ops.append(entry.strip())

        results[mtype] = {
            'forward_time_ms': round(elapsed_ms, 2),
            'n_params': sum(p.numel() for p in model.parameters()),
        }
        print(f'  forward time: {elapsed_ms:.2f} ms')
        print(f'  params: {results[mtype]["n_params"]:,}')

    return results


def profile_vram_comparison():
    """Profile VRAM if CUDA available."""
    if not torch.cuda.is_available():
        print('\n=== VRAM Comparison === (CUDA not available)')
        return {'gpu': False}

    print('\n=== VRAM Comparison (CUDA) ===')
    results = {}

    for name, mtype in [('Titans', 'titans'), ('HBMA', 'hbma')]:
        torch.cuda.reset_peak_memory_stats()
        cfg = HDIMConfig(hidden_dim=64, num_domains=4, memory_type=mtype)
        model = HDIMModel(cfg).cuda()
        model.train()
        x = torch.randn(64, 64, requires_grad=True).cuda()
        d = torch.zeros(64, dtype=torch.long).cuda()

        torch.cuda.synchronize()
        for _ in range(5):
            out, rw, inv = model(x, d)
            loss = out.sum()
            loss.backward()
        torch.cuda.synchronize()

        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        results[mtype] = {'peak_vram_mb': round(peak_mb, 2)}
        print(f'  {name}: peak VRAM = {peak_mb:.2f} MB')
        del model
        torch.cuda.empty_cache()

    results['gpu'] = True
    return results


def main():
    print('='*60)
    print('HDIM Phase 27c: Performance Profiling')
    print('='*60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    results = {}

    # 1. Geometric product
    results['geometric_product'] = profile_geometric_product(device)

    # 2. Sandwich product
    results['sandwich'] = profile_sandwich(device)

    # 3. Memory systems
    results['memory'] = profile_memory_systems(device)

    # 4. VRAM comparison
    results['vram'] = profile_vram_comparison()

    # Save results
    out_path = Path('artifacts/perf_profile.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
