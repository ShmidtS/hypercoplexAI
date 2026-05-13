from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extensions.moe.config import MoEKernelConfig
from src.extensions.moe.kernel import MoEKernel


def _imbalance_trace(kernel: MoEKernel, inputs: list[torch.Tensor]) -> list[float]:
    trace: list[float] = []
    kernel.train()
    for x in inputs:
        _, info = kernel(x)
        usage = info["expert_usage"]
        trace.append(float((usage.max() - usage.min()).item()))
    return trace


def main() -> None:
    config = MoEKernelConfig(
        routing_seed=0,
        input_dim=64,
        expert_hidden_dim=128,
        num_experts=4,
        expert_names=["a", "b", "c", "d"],
        use_aux_loss_free=True,
        bias_update_frequency=1,
        aux_lr=0.01,
    )
    kernel_a = MoEKernel(config)
    kernel_b = MoEKernel(config)

    generator = torch.Generator().manual_seed(1234)
    inputs = [torch.randn(32, 64, generator=generator) for _ in range(50)]

    trace_a = _imbalance_trace(kernel_a, inputs)
    trace_b = _imbalance_trace(kernel_b, inputs)
    if trace_a != trace_b:
        raise AssertionError("same routing_seed produced different imbalance traces")

    early_avg = sum(trace_a[:10]) / 10
    late_avg = sum(trace_a[-10:]) / 10
    if late_avg > early_avg * 1.5:
        raise AssertionError(
            f"late imbalance grew too much: late_avg={late_avg}, early_avg={early_avg}"
        )

    print("probe_passed")


if __name__ == "__main__":
    main()
