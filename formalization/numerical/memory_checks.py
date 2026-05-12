"""Numerical checks for current HDIM memory/core formalization.

Lean4 theorem mapping: formalization/Extensions.lean memory section plus
current canonical core modules in src/core/{algebra,rotors,invariants,engine}.py.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from src.core.algebra import CliffordAlgebra
from src.core.engine import CoreEngineConfig, HDIMCoreEngine
from src.core.rotors import DomainRotationOperator
from src.extensions.memory import HBMAMemory, MemoryResult, SemanticMemory, TitansMemory, WorkingMemory
from src.extensions.moe import SoftMoERouter
from src.models.modern_text_encoder import MatryoshkaProjection
from src.training.invariant_losses import compute_infonce_loss


def _memory_output(result):
    return result.output if isinstance(result, MemoryResult) else result


def make_plane_rotor(ca: CliffordAlgebra, i: int, j: int, angle: float) -> torch.Tensor:
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


def make_bivector_rotor(ca: CliffordAlgebra) -> torch.Tensor:
    R = torch.zeros(1, ca.dim)
    R[0, 0] = 1.0
    for k in range(ca.n // 2):
        i = 2 * k
        if i + 1 >= ca.n:
            break
        angle = torch.rand(1).item() * math.pi
        R = ca.geometric_product(R, make_plane_rotor(ca, i, i + 1, angle))
    return R / ca.norm(R)


def run_checks() -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []

    print('\n--- 7. HBMAMemory properties ---')
    hm = HBMAMemory(hidden_dim=64)
    x = torch.randn(4, 64)
    out = _memory_output(hm(x))
    shape_ok = out.shape == (4, 64)
    print(f'  forward shape: {out.shape} [{"PASS" if shape_ok else "FAIL"}]')
    results.append(('hbma_shape', 'PASS' if shape_ok else 'FAIL'))

    loss = out.sum()
    loss.backward()
    grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in hm.parameters())
    print(f'  gradient flow: [{"PASS" if grad_ok else "FAIL"}]')
    results.append(('hbma_gradients', 'PASS' if grad_ok else 'FAIL'))

    hm2 = HBMAMemory(hidden_dim=32)
    _ = hm2(torch.randn(2, 32))
    hm2.reset()
    reset_ok = hm2.working.buf.abs().sum() == 0 and hm2.episodic.mem_vals.abs().sum() == 0
    print(f'  reset clears buffers: [{"PASS" if reset_ok else "FAIL"}]')
    results.append(('hbma_reset', 'PASS' if reset_ok else 'FAIL'))

    print('\n--- 8. MemoryInterface direct ---')
    titans_mem = TitansMemory(clifford_dim=64, memory_key_dim=32)
    result_t = titans_mem(torch.randn(4, 64), update_memory=True)
    t_ok = result_t.output.shape == (4, 64) and isinstance(result_t.loss, torch.Tensor)
    print(f'  TitansMemory forward: [{"PASS" if t_ok else "FAIL"}]')
    results.append(('titans_memory', 'PASS' if t_ok else 'FAIL'))

    hbma_mem = HBMAMemory(hidden_dim=64)
    result_h = hbma_mem(torch.randn(4, 64), update_memory=True)
    h_ok = result_h.output.shape == (4, 64) and isinstance(result_h.loss, torch.Tensor)
    print(f'  HBMAMemory forward: [{"PASS" if h_ok else "FAIL"}]')
    results.append(('hbma_memory', 'PASS' if h_ok else 'FAIL'))

    titans_mem.reset()
    hbma_mem.reset()
    print('  memory reset: [PASS]')
    results.append(('memory_reset', 'PASS'))

    print('\n--- 9. DomainRotor built-in norm preservation ---')
    for p, q, r in [(2, 0, 0), (3, 1, 0)]:
        ca = CliffordAlgebra(p, q, r)
        dro = DomainRotationOperator(ca, 'test')
        r_norm = ca.norm(dro._normalized_R()).item()
        status = 'PASS' if abs(r_norm - 1.0) < 0.01 else 'FAIL'
        print(f'  Cl({p},{q},{r}): ||R||={r_norm:.6f} [{status}]')
        results.append((f'domain_rotor_norm_Cl{p}{q}{r}', status))

    print('\n--- 10. Geometric product and reverse properties ---')
    ca = CliffordAlgebra(3, 1, 0)
    max_assoc = 0.0
    max_reverse = 0.0
    for _ in range(20):
        a = torch.randn(1, ca.dim)
        b = torch.randn(1, ca.dim)
        c = torch.randn(1, ca.dim)
        max_assoc = max(max_assoc, (ca.geometric_product(ca.geometric_product(a, b), c) - ca.geometric_product(a, ca.geometric_product(b, c))).abs().max().item())
        max_reverse = max(max_reverse, (ca.reverse(ca.geometric_product(a, b)) - ca.geometric_product(ca.reverse(b), ca.reverse(a))).abs().max().item())
    assoc_status = 'PASS' if max_assoc < 1e-4 else 'FAIL'
    rev_status = 'PASS' if max_reverse < 1e-4 else 'FAIL'
    print(f'  associativity max_diff={max_assoc:.2e} [{assoc_status}]')
    print(f'  reverse product max_diff={max_reverse:.2e} [{rev_status}]')
    results.append(('geom_assoc', assoc_status))
    results.append(('reverse_product', rev_status))

    print('\n--- 11. Sandwich inverse and composition ---')
    all_roundtrip = True
    all_composition = True
    for p, q, r in [(2, 0, 0), (3, 0, 0), (3, 1, 0)]:
        ca = CliffordAlgebra(p, q, r)
        for _ in range(10):
            R1 = make_bivector_rotor(ca)
            R2 = make_bivector_rotor(ca)
            x = torch.randn(1, ca.dim)
            y = ca.sandwich(R1, x, unit=True)
            z = ca.sandwich(ca.reverse(R1), y, unit=True)
            if (z - x).abs().max().item() > 0.15:
                all_roundtrip = False
            sequential = ca.sandwich(R2, ca.sandwich(R1, x, unit=True), unit=True)
            composed = ca.sandwich(ca.geometric_product(R2, R1), x, unit=False)
            if (sequential - composed).abs().max().item() > 0.15:
                all_composition = False
    rt_status = 'PASS' if all_roundtrip else 'FAIL'
    comp_status = 'PASS' if all_composition else 'FAIL'
    print(f'  sandwich inverse roundtrip: [{rt_status}]')
    print(f'  sandwich composition: [{comp_status}]')
    results.append(('sandwich_inverse_roundtrip', rt_status))
    results.append(('sandwich_composition_sequence', comp_status))

    print('\n--- 12. Core engine transfer path ---')
    engine = HDIMCoreEngine(CoreEngineConfig(input_dim=64, domain_names=('A', 'B')))
    engine.eval()
    with torch.no_grad():
        G = engine.encode(torch.randn(2, 64))
        U = engine.extract(G, 'A')
        G_target = engine.transfer(U, 'B')
        U_back = engine.extract(G_target, 'B')
    engine_ok = G.shape[-1] == engine.algebra.dim and G_target.shape == U.shape and torch.isfinite(U_back).all()
    inv_diff = (U - U_back).abs().max().item()
    invariant_ok = inv_diff < 1e-3
    status = 'PASS' if engine_ok and invariant_ok else 'FAIL'
    print(f'  encode/extract/transfer finite, invariant diff={inv_diff:.2e}: [{status}]')
    results.append(('core_engine_transfer_path', status))

    print('\n--- 13. Matryoshka and losses ---')
    proj = MatryoshkaProjection(64, [16, 32, 48, 64])
    scales = proj(torch.randn(5, 64))
    mat_ok = all((scales[s] - scales[l][:, :s]).abs().max().item() < 1e-6 for s, l in [(16, 32), (32, 48), (48, 64)])
    src = torch.randn(16, 64)
    tgt = torch.randn(16, 64)
    labels = torch.ones(16)
    labels[:8] = 0.0
    infonce = compute_infonce_loss(src, tgt, labels, torch.ones(16), temperature=0.1)
    loss_ok = torch.isfinite(infonce) and infonce.item() >= 0
    print(f'  Matryoshka prefix nesting: [{"PASS" if mat_ok else "FAIL"}]')
    print(f'  InfoNCE finite non-negative: [{"PASS" if loss_ok else "FAIL"}]')
    results.append(('matryoshka_nesting', 'PASS' if mat_ok else 'FAIL'))
    results.append(('infonce_non_negative', 'PASS' if loss_ok else 'FAIL'))

    print('\n--- 14. Memory and MoE extension invariants ---')
    wm = WorkingMemory(hidden_dim=32, capacity=4)
    for i in range(6):
        wm._write(torch.zeros(1, 32) + i * 100.0)
    fifo_ok = wm.buf[0].mean().item() >= 300 and wm.buf[1].mean().item() >= 300
    sem = SemanticMemory(hidden_dim=64, num_prototypes=8, ema_momentum=0.95)
    fixed_vec = F.normalize(torch.randn(1, 64), dim=-1)
    with torch.no_grad():
        sem.prototypes.copy_(F.normalize(fixed_vec.repeat(8, 1) + 0.1 * torch.randn(8, 64), dim=-1))
        for _ in range(100):
            sem._update_prototypes(fixed_vec)
    ema_ok = (fixed_vec @ F.normalize(sem.prototypes, dim=-1).T).max().item() > 0.95
    router = SoftMoERouter(input_dim=64, num_experts=4, expert_dim=128)
    x_moe = torch.randn(8, 64, requires_grad=True)
    out_moe, state = router(x_moe)
    z_loss_ok = state['z_loss'].item() >= 0 and torch.isfinite(state['z_loss'])
    out_moe.sum().backward()
    grad_ok = x_moe.grad is not None and torch.isfinite(x_moe.grad).all()
    print(f'  WorkingMemory FIFO reuse: [{"PASS" if fifo_ok else "FAIL"}]')
    print(f'  SemanticMemory EMA convergence: [{"PASS" if ema_ok else "FAIL"}]')
    print(f'  SoftMoE z_loss and gradient: [{"PASS" if z_loss_ok and grad_ok else "FAIL"}]')
    results.append(('hbma_working_slot_reuse', 'PASS' if fifo_ok else 'FAIL'))
    results.append(('hbma_ema_convergence', 'PASS' if ema_ok else 'FAIL'))
    results.append(('soft_moe_z_loss_gradient', 'PASS' if z_loss_ok and grad_ok else 'FAIL'))

    return results
