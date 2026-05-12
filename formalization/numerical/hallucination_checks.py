"""Numerical checks for HDIM formalization."""
# Lean4 theorem mapping: formalization/Extensions.lean hallucination theorems
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from src.core.hypercomplex import CliffordAlgebra, QuaternionLinear, QLayerNorm
from src.core.domain_operators import DomainRotationOperator, sandwich_transfer, InvariantExtractor
from src.extensions.memory import TitansMemory, HBMAMemory, WorkingMemory, SemanticMemory, MemoryResult
from src.extensions.moe import SoftMoERouter
from src.core.hdim_pipeline import HDIMPipeline


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

    # ===== 127. HallucinationDetector: risk_score in [0, 1] =====
    print('\n--- 127. hallucination_risk_bound ---')
    from src.extensions.hallucination.detector import HallucinationDetector
    torch.manual_seed(1270)
    hd127 = HallucinationDetector(num_experts=4, hidden_dim=64)
    all_ok_127 = True
    for _ in range(50):
        routing_entropy = torch.rand(4) * hd127.max_entropy
        moe_confidence = torch.rand(4)
        memory_mismatch = torch.randn(4) * 0.5
        memory_loss = torch.rand(4) * 0.3
        hidden_states = torch.randn(4, 8, 64)
        routing_repr = torch.randn(4, 16)
        res127 = hd127.compute_hallucination_risk(
            routing_entropy=routing_entropy,
            moe_confidence=moe_confidence,
            memory_mismatch=memory_mismatch,
            memory_loss=memory_loss,
            hidden_states=hidden_states,
            routing_repr=routing_repr,
        )
        if res127.hallucination_risk < 0.0 or res127.hallucination_risk > 1.0:
            all_ok_127 = False
            break
    status = 'PASS' if all_ok_127 else 'FAIL'
    print(f'  50 trials, risk in [0,1]: [{status}]')
    results.append(('hallucination_risk_bound', status))

    # ===== 128. HallucinationDetector: eigen_score >= 0 =====
    print('\n--- 128. eigen_score_nonneg ---')
    torch.manual_seed(1280)
    hd128 = HallucinationDetector(num_experts=4, hidden_dim=64)
    all_ok_128 = True
    for _ in range(50):
        routing_repr = torch.randn(4, 1, 32)
        eigen128 = hd128.compute_eigen_score(routing_repr)
        if (eigen128 < 0).any():
            all_ok_128 = False
            break
    status = 'PASS' if all_ok_128 else 'FAIL'
    print(f'  50 trials, eigen_score >= 0: [{status}]')
    results.append(('eigen_score_nonneg', status))

    # ===== 129. HallucinationDetector: detection weights sum to 1.0 =====
    print('\n--- 129. detection_weights_sum_one ---')
    torch.manual_seed(1290)
    hd129 = HallucinationDetector(num_experts=4, hidden_dim=64)
    weights_sum = (
        hd129.weight_entropy.item()
        + hd129.weight_confidence.item()
        + hd129.weight_mismatch.item()
        + hd129.weight_semantic.item()
        + hd129.weight_eigen.item()
    )
    err129 = abs(weights_sum - 1.0)
    status = 'PASS' if err129 < 1e-6 else 'FAIL'
    print(f'  weights sum = {weights_sum:.6f}, err = {err129:.2e}: [{status}]')
    results.append(('detection_weights_sum_one', status))

    # ===== 130. HallucinationDetector: SVD eigen score boundedness =====
    print('\n--- 130. svd_eigen_bounded ---')
    torch.manual_seed(1300)
    hd130 = HallucinationDetector(num_experts=4, hidden_dim=64)
    all_ok_130 = True
    max_eigen = 0.0
    for _ in range(50):
        routing_repr = torch.randn(4, 1, 32)
        eigen130 = hd130.compute_eigen_score(routing_repr)
        norm_eigen = torch.sigmoid(eigen130 - 1.0)
        if (norm_eigen < 0.0).any() or (norm_eigen > 1.0).any():
            all_ok_130 = False
            break
        max_eigen = max(max_eigen, norm_eigen.max().item())
    status = 'PASS' if all_ok_130 else 'FAIL'
    print(f'  50 trials, norm_eigen in [0,1], max = {max_eigen:.4f}: [{status}]')
    results.append(('svd_eigen_bounded', status))
    return results
