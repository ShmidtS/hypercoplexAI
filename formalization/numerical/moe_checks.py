"""Numerical checks for HDIM formalization."""
# Lean4 theorem mapping: formalization/Extensions.lean MoE section
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from src.core.algebra import CliffordAlgebra
from src.core.rotors import DomainRotationOperator
from src.core.invariants import InvariantExtractor, sandwich_transfer
from src.extensions.memory import TitansMemory, HBMAMemory, WorkingMemory, SemanticMemory, MemoryResult
from src.extensions.moe import SoftMoERouter


def _memory_output(result):
    return result.output if isinstance(result, MemoryResult) else result


def make_plane_rotor(ca, i, j, angle):
    blade_idx = (1 << i) | (1 << j)
    b_square = -float(ca.metric[i].item() * ca.metric[j].item())
    rot = torch.zeros(1, ca.dim)
    if b_square > 0:
        angle = min(angle, 1.0)
        rot[0, 0] = math.cosh(angle)
        rot[0, blade_idx] = math.sinh(angle)
    elif b_square < 0:
        rot[0, 0] = math.cos(angle)
        rot[0, blade_idx] = math.sin(angle)
    else:
        rot[0, 0] = 1.0
        rot[0, blade_idx] = angle
    return rot / ca.norm(rot)


def make_bivector_rotor(ca, n_trials=1):
    """Build unit bivector rotor R from metric-aware bivector exponentials."""
    Rs = []
    for _ in range(n_trials):
        R = torch.zeros(1, ca.dim); R[0,0] = 1.0
        for k in range(ca.n // 2):
            i = 2*k
            if i+1 >= ca.n: break
            angle = torch.rand(1).item() * math.pi
            rot = make_plane_rotor(ca, i, i + 1, angle)
            R = ca.geometric_product(R, rot)
        R = R / ca.norm(R)
        Rs.append(R)
    return Rs[0] if n_trials == 1 else Rs


def run_checks() -> list[tuple[str, str]]:
    results = []
    from src.extensions.moe import MoEKernel, MoEKernelConfig
    cfg116 = MoEKernelConfig(
        input_dim=64, expert_hidden_dim=128, num_experts=4,
        expert_names=['math','language','code','science']
    )

    # ===== 124. MoEKernel: shared expert output is finite =====
    print('\n--- 124. moe_kernel_shared_expert_finite ---')
    all_ok = True
    torch.manual_seed(1240)
    cfg124_shared = MoEKernelConfig(
        input_dim=64, expert_hidden_dim=128, num_experts=4,
        use_shared_expert=True, expert_names=['math','language','code','science']
    )
    k124 = MoEKernel(cfg124_shared)
    k124.eval()
    x124 = torch.randn(16, 64)
    with torch.no_grad():
        out124, _ = k124(x124)
    if torch.isnan(out124).any() or torch.isinf(out124).any():
        all_ok = False
    status = 'PASS' if all_ok else 'FAIL'
    print(f' Shared expert output is finite: [{status}]')
    results.append(('moe_kernel_shared_expert_finite', status))

    # ===== 125. MoEKernel: 3D seq input (B, T, D) =====
    print('\n--- 125. moe_kernel_seq_input ---')
    all_ok = True
    torch.manual_seed(1250)
    k125 = MoEKernel(cfg116)
    k125.eval()
    with torch.no_grad():
        x125 = torch.randn(4, 16, 64)
        out125, st125 = k125(x125)
        if out125.shape != (4, 16, 64):
            all_ok = False
        if torch.isnan(out125).any():
            all_ok = False
        if st125["expert_weights"].shape != (4, 16, 4):
            all_ok = False
    status = 'PASS' if all_ok else 'FAIL'
    print(f' Seq input (4,16,64) -> (4,16,64): [{status}]')
    results.append(('moe_kernel_seq_input', status))

    # ===== 126. MoEKernel: domain expert type consistency =====
    print('\n--- 126. moe_kernel_expert_types ---')
    from src.extensions.moe import MLPExpert as MLPExpert126
    all_ok = True
    torch.manual_seed(1260)
    k126 = MoEKernel(cfg116)
    type_ok = [
        isinstance(k126.experts[0], MLPExpert126),
        isinstance(k126.experts[1], MLPExpert126),
        isinstance(k126.experts[2], MLPExpert126),
        isinstance(k126.experts[3], MLPExpert126),
    ]
    if not all(type_ok):
        all_ok = False
    status = 'PASS' if all_ok else 'FAIL'
    print(f' Expert types correct {type_ok}: [{status}]')
    results.append(('moe_kernel_expert_types', status))
    return results
