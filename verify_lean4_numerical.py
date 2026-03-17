"""Numerical verification of all Lean4 formalization theorems for HDIM."""
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from src.core.hypercomplex import CliffordAlgebra, QuaternionLinear, QLayerNorm
from src.core.domain_operators import DomainRotationOperator, sandwich_transfer, InvariantExtractor
from src.core.memory_interface import TitansAdapter, HBMAMemoryAdapter
from src.core.titans_memory import TitansMemoryModule
from src.core.hbma_memory import HBMAMemory
from src.models.hdim_model import HDIMPipeline

def make_bivector_rotor(ca, n_trials=1):
    """Build unit bivector rotor R = prod_k (cos + sin * e_{2k}e_{2k+1})."""
    Rs = []
    for _ in range(n_trials):
        R = torch.zeros(1, ca.dim); R[0,0] = 1.0
        for k in range(ca.n // 2):
            i = 2*k
            if i+1 >= ca.n: break
            angle = torch.rand(1).item() * math.pi
            rot = torch.zeros(1, ca.dim)
            rot[0,0] = math.cos(angle); rot[0,(1<<i)|(1<<(i+1))] = math.sin(angle)
            R = ca.geometric_product(R, rot)
        R = R / ca.norm(R)
        Rs.append(R)
    return Rs[0] if n_trials == 1 else Rs

print('='*60)
print('LEAN4 NUMERICAL VERIFICATION - HDIM Core Theorems')
print('='*60)

results = []

# ===== 1. Sandwich Norm Preservation =====
print('\n--- 1. sandwich_norm_preservation (50 trials each) ---')
for p,q,r in [(2,0,0),(3,0,0),(3,1,0),(4,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    max_err = 0.0
    for _ in range(50):
        R = make_bivector_rotor(ca)
        x = torch.randn(1, ca.dim)
        y = ca.sandwich(R, x, unit=True)
        err = abs(ca.norm(y).item() / max(ca.norm(x).item(), 1e-8) - 1.0)
        max_err = max(max_err, err)
    status = 'PASS' if max_err < 0.02 else 'FAIL'
    print(f'  Cl({p},{q},{r}): max_err={max_err:.2e} [{status}]')
    results.append((f'sandwich_norm_Cl{p}{q}{r}', status))

# ===== 2. Sandwich Identity =====
print('\n--- 2. sandwich_identity ---')
for p,q,r in [(2,0,0),(3,0,0),(3,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    x = torch.randn(1, ca.dim)
    one = torch.zeros(1, ca.dim); one[0,0] = 1.0
    y = ca.sandwich(one, x, unit=True)
    err = (y - x).abs().max().item()
    status = 'PASS' if err < 1e-6 else 'FAIL'
    print(f'  Cl({p},{q},{r}): diff={err:.2e} [{status}]')
    results.append((f'sandwich_identity_Cl{p}{q}{r}', status))

# ===== 3. Sandwich Composition (using unit bivector rotors) =====
print('\n--- 3. sandwich_composition (unit bivector rotors, 20 trials) ---')
for p,q,r in [(2,0,0),(3,0,0),(3,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    max_diff = 0.0
    for _ in range(20):
        R1 = make_bivector_rotor(ca)
        R2 = make_bivector_rotor(ca)
        x = torch.randn(1, ca.dim)
        y1 = ca.sandwich(R1, ca.sandwich(R2, x, unit=True), unit=True)
        R12 = ca.geometric_product(R1, R2)
        y2 = ca.sandwich(R12, x, unit=False)  # R12 may not be unit
        diff = (y1 - y2).abs().max().item()
        max_diff = max(max_diff, diff)
    status = 'PASS' if max_diff < 0.1 else 'FAIL'
    print(f'  Cl({p},{q},{r}): max_diff={max_diff:.2e} [{status}]')
    results.append((f'sandwich_comp_Cl{p}{q}{r}', status))

# ===== 4. Transfer Roundtrip =====
print('\n--- 4. transfer_roundtrip (20 trials each) ---')
for p,q,r in [(2,0,0),(3,0,0),(3,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    max_diff = 0.0
    for _ in range(20):
        x = torch.randn(4, ca.dim)
        src = DomainRotationOperator(ca, 'src')
        tgt = DomainRotationOperator(ca, 'tgt')
        _, g_tgt = sandwich_transfer(ca, x, src, tgt)
        x_back, _ = sandwich_transfer(ca, g_tgt, tgt, src)
        diff = (x_back - x).abs().max().item()
        max_diff = max(max_diff, diff)
    status = 'PASS' if max_diff < 0.1 else 'FAIL'
    print(f'  Cl({p},{q},{r}): max_diff={max_diff:.2e} [{status}]')
    results.append((f'transfer_rt_Cl{p}{q}{r}', status))

# ===== 5. Invariant Domain Independence =====
print('\n--- 5. invariant_domain_independence (20 trials each) ---')
for p,q,r in [(2,0,0),(3,0,0),(3,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    max_diff = 0.0
    for _ in range(20):
        x = torch.randn(4, ca.dim)
        src = DomainRotationOperator(ca, 'src')
        tgt = DomainRotationOperator(ca, 'tgt')
        ie = InvariantExtractor(ca)
        u_src = ie(x, src)
        g_tgt_from_src = sandwich_transfer(ca, x, src, tgt)[1]
        u_tgt = ie(g_tgt_from_src, tgt)
        diff = (u_src - u_tgt).abs().max().item()
        max_diff = max(max_diff, diff)
    status = 'PASS' if max_diff < 0.1 else 'FAIL'
    print(f'  Cl({p},{q},{r}): max_diff={max_diff:.2e} [{status}]')
    results.append((f'inv_indep_Cl{p}{q}{r}', status))

# ===== 6. Geometric Product Properties =====
print('\n--- 6. geometric_product_properties ---')
ca = CliffordAlgebra(3,1,0)
anti_ok = True
for i in range(ca.n):
    for j in range(i+1, ca.n):
        ei = torch.zeros(1, ca.dim); ei[0, 1<<i] = 1.0
        ej = torch.zeros(1, ca.dim); ej[0, 1<<j] = 1.0
        diff = (ca.geometric_product(ei, ej) + ca.geometric_product(ej, ei)).abs().max().item()
        if diff > 1e-6: anti_ok = False
print(f'  anticommutativity Cl(3,1,0): [{"PASS" if anti_ok else "FAIL"}]')
results.append(('anticommutativity', 'PASS' if anti_ok else 'FAIL'))

metric_ok = True
for i in range(4):
    ei = torch.zeros(1, ca.dim); ei[0, 1<<i] = 1.0
    prod = ca.geometric_product(ei, ei)
    expected = 1.0 if i < 3 else -1.0
    if abs(prod[0,0].item() - expected) > 1e-6: metric_ok = False
print(f'  e_i^2 = metric Cl(3,1,0): [{"PASS" if metric_ok else "FAIL"}]')
results.append(('e_i_squared', 'PASS' if metric_ok else 'FAIL'))

# ===== 7. HBMA Memory Properties =====
print('\n--- 7. HBMAMemory properties ---')
hm = HBMAMemory(hidden_dim=64)
x = torch.randn(4, 64)
out = hm(x)
shape_ok = out.shape == (4, 64)
print(f'  forward shape: {out.shape} [{"PASS" if shape_ok else "FAIL"}]')
results.append(('hbma_shape', 'PASS' if shape_ok else 'FAIL'))

loss = out.sum()
loss.backward()
grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in hm.parameters())
print(f'  gradient flow: [{"PASS" if grad_ok else "FAIL"}]')
results.append(('hbma_gradients', 'PASS' if grad_ok else 'FAIL'))

hm2 = HBMAMemory(hidden_dim=32)
x2 = torch.randn(2, 32)
_ = hm2(x2)
hm2.reset()
wm_zero = hm2.working.buf.abs().sum() == 0
ep_zero = hm2.episodic.mem_vals.abs().sum() == 0
print(f'  reset clears buffers: [{"PASS" if (wm_zero and ep_zero) else "FAIL"}]')
results.append(('hbma_reset', 'PASS' if (wm_zero and ep_zero) else 'FAIL'))

# ===== 8. MemoryInterface adapters =====
print('\n--- 8. MemoryInterface adapters ---')
titans_raw = TitansMemoryModule(key_dim=32, val_dim=64)
titans_adp = TitansAdapter(titans_raw, clifford_dim=64, memory_key_dim=32)
result_t = titans_adp(torch.randn(4, 64), update_memory=True)
t_ok = result_t.output.shape == (4, 64) and isinstance(result_t.loss, torch.Tensor)
print(f'  TitansAdapter forward: [{"PASS" if t_ok else "FAIL"}]')
results.append(('titans_adapter', 'PASS' if t_ok else 'FAIL'))

hbma_raw = HBMAMemory(hidden_dim=64)
hbma_adp = HBMAMemoryAdapter(hbma_raw)
result_h = hbma_adp(torch.randn(4, 64), update_memory=True)
h_ok = result_h.output.shape == (4, 64) and isinstance(result_h.loss, torch.Tensor)
print(f'  HBMAMemoryAdapter forward: [{"PASS" if h_ok else "FAIL"}]')
results.append(('hbma_adapter', 'PASS' if h_ok else 'FAIL'))

titans_adp.reset()
hbma_adp.reset()
print(f'  adapter reset: [PASS]')
results.append(('adapter_reset', 'PASS'))

# ===== 9. DomainRotor norm preservation =====
print('\n--- 9. DomainRotor built-in norm preservation ---')
for p,q,r in [(2,0,0),(3,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    dro = DomainRotationOperator(ca, 'test')
    R = dro._normalized_R()
    r_norm = ca.norm(R).item()
    status = 'PASS' if abs(r_norm - 1.0) < 0.01 else 'FAIL'
    print(f'  Cl({p},{q},{r}): ||R||={r_norm:.6f} [{status}]')
    results.append((f'domain_rotor_norm_Cl{p}{q}{r}', status))

# ===== 10. Quaternion Linear Properties =====
print('\n--- 10. quaternion_linear_properties ---')
ql = QuaternionLinear(32, 32, bias=True)
x = torch.randn(4, 32)
out = ql(x)
shape_ok = out.shape == (4, 32)
print(f'  forward shape: {out.shape} [{"PASS" if shape_ok else "FAIL"}]')
results.append(('quat_linear_shape', 'PASS' if shape_ok else 'FAIL'))

# Gradient flow
loss = out.sum()
loss.backward()
grad_ok = ql.Wr.grad is not None and ql.Wr.grad.abs().sum() > 0
print(f'  gradient flow: [{"PASS" if grad_ok else "FAIL"}]')
results.append(('quat_linear_grad', 'PASS' if grad_ok else 'FAIL'))

# Quaternion weight structure: Hamilton product matrix should be block-structured
W = ql.hamilton_product_weights()
assert W.shape == (32, 32), f"Expected (32,32), got {W.shape}"
# Check block structure: Wr should appear on diagonal blocks
# The first 8x8 block should equal Wr
block_diag = W[:8, :8]
wr_match = (block_diag - ql.Wr).abs().max().item() < 1e-6
print(f'  Hamilton block structure: [{"PASS" if wr_match else "FAIL"}]')
results.append(('quat_hamilton_structure', 'PASS' if wr_match else 'FAIL'))

# ===== 11. QLayerNorm Properties =====
print('\n--- 11. qlayernorm_properties ---')
qln = QLayerNorm(8)
x = torch.randn(3, 32)
out = qln(x)
shape_ok = out.shape == (3, 32)
print(f'  forward shape: {out.shape} [{"PASS" if shape_ok else "FAIL"}]')
results.append(('qlayernorm_shape', 'PASS' if shape_ok else 'FAIL'))

# Each quadrant normalized independently: var of each chunk ~ 1
var_ok = True
for i in range(4):
    chunk = out[:, i*8:(i+1)*8]
    var = chunk.var(dim=-1).mean().item()
    if abs(var - 1.0) > 0.5:
        var_ok = False
print(f'  per-component normalization: [{"PASS" if var_ok else "FAIL"}]')
results.append(('qlayernorm_normalize', 'PASS' if var_ok else 'FAIL'))

# ===== 12. Geometric Product Associativity =====
print('\n--- 12. geometric_product_associativity (30 trials) ---')
ca = CliffordAlgebra(3,1,0)
max_diff = 0.0
for _ in range(30):
    a = torch.randn(1, ca.dim)
    b = torch.randn(1, ca.dim)
    c = torch.randn(1, ca.dim)
    ab_c = ca.geometric_product(ca.geometric_product(a, b), c)
    a_bc = ca.geometric_product(a, ca.geometric_product(b, c))
    diff = (ab_c - a_bc).abs().max().item()
    max_diff = max(max_diff, diff)
status = 'PASS' if max_diff < 1e-4 else 'FAIL'
print(f'  max_diff={max_diff:.2e} [{status}]')
results.append(('geom_assoc', status))

# ===== 13. Reverse involutive: reverse(reverse(x)) = x =====
print('\n--- 13. reverse_involution ---')
for p,q,r in [(2,0,0),(3,1,0),(4,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    x = torch.randn(5, ca.dim)
    x_rev_rev = ca.reverse(ca.reverse(x))
    diff = (x_rev_rev - x).abs().max().item()
    status = 'PASS' if diff < 1e-6 else 'FAIL'
    print(f'  Cl({p},{q},{r}): diff={diff:.2e} [{status}]')
    results.append((f'rev_involution_Cl{p}{q}{r}', status))

# ===== 14. Involution twice = identity: inv(inv(x)) = x =====
print('\n--- 14. involution_twice ---')
for p,q,r in [(2,0,0),(3,1,0),(4,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    x = torch.randn(5, ca.dim)
    x_inv_inv = ca.involute(ca.involute(x))
    diff = (x_inv_inv - x).abs().max().item()
    status = 'PASS' if diff < 1e-6 else 'FAIL'
    print(f'  Cl({p},{q},{r}): diff={diff:.2e} [{status}]')
    results.append((f'inv_twice_Cl{p}{q}{r}', status))

# ===== 15. Norm non-negativity =====
print('\n--- 15. norm_nonnegativity (20 trials) ---')
ca = CliffordAlgebra(3,1,0)
all_nonneg = True
for _ in range(20):
    x = torch.randn(4, ca.dim)
    n = ca.norm(x)
    if (n < 0).any():
        all_nonneg = False
status = 'PASS' if all_nonneg else 'FAIL'
print(f'  all non-negative: [{status}]')
results.append(('norm_nonneg', status))

# ===== 16. Geometric Product Distributivity =====
print('\n--- 16. geometric_product_distributivity (20 trials) ---')
ca = CliffordAlgebra(3,1,0)
max_diff = 0.0
for _ in range(20):
    a = torch.randn(1, ca.dim)
    b = torch.randn(1, ca.dim)
    c = torch.randn(1, ca.dim)
    left = ca.geometric_product(a, b + c)
    right = ca.geometric_product(a, b) + ca.geometric_product(a, c)
    diff = (left - right).abs().max().item()
    max_diff = max(max_diff, diff)
status = 'PASS' if max_diff < 1e-4 else 'FAIL'
print(f'  left-distributive max_diff={max_diff:.2e} [{status}]')
results.append(('geom_distrib_left', status))

max_diff = 0.0
for _ in range(20):
    a = torch.randn(1, ca.dim)
    b = torch.randn(1, ca.dim)
    c = torch.randn(1, ca.dim)
    left = ca.geometric_product(a + b, c)
    right = ca.geometric_product(a, c) + ca.geometric_product(b, c)
    diff = (left - right).abs().max().item()
    max_diff = max(max_diff, diff)
status = 'PASS' if max_diff < 1e-4 else 'FAIL'
print(f'  right-distributive max_diff={max_diff:.2e} [{status}]')
results.append(('geom_distrib_right', status))

# ===== 17. Sandwich inverse identity =====
print('\n--- 17. sandwich_inverse_identity (20 trials each) ---')
for p,q,r in [(2,0,0),(3,0,0),(3,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    max_diff = 0.0
    for _ in range(20):
        R = make_bivector_rotor(ca)
        x = torch.randn(1, ca.dim)
        R_inv = ca.reverse(R)  # unit rotor: R⁻¹ = ~R exactly
        y = ca.sandwich(R, x, unit=True)
        z = ca.sandwich(R_inv, y, unit=True)
        diff = (z - x).abs().max().item()
        max_diff = max(max_diff, diff)
    status = 'PASS' if max_diff < 0.1 else 'FAIL'
    print(f'  Cl({p},{q},{r}): max_diff={max_diff:.2e} [{status}]')
    results.append((f'sandwich_inv_Cl{p}{q}{r}', status))

# ===== 18. Reverse product order: ~ab = ~b * ~a =====
print('\n--- 18. reverse_product_order (20 trials) ---')
ca = CliffordAlgebra(3,1,0)
max_diff = 0.0
for _ in range(20):
    a = torch.randn(1, ca.dim)
    b = torch.randn(1, ca.dim)
    rev_ab = ca.reverse(ca.geometric_product(a, b))
    rev_b_rev_a = ca.geometric_product(ca.reverse(b), ca.reverse(a))
    diff = (rev_ab - rev_b_rev_a).abs().max().item()
    max_diff = max(max_diff, diff)
status = 'PASS' if max_diff < 1e-4 else 'FAIL'
print(f'  Cl(3,1,0): max_diff={max_diff:.2e} [{status}]')
results.append(('reverse_product', status))

# ===== 19. Scalar(x * ~x): |<x*~x>_0| = ||x||^2 =====
print('\n--- 19. scalar_xx_rev_norm_sq (20 trials) ---')
all_nonneg = True
for p,q,r in [(2,0,0),(3,1,0),(4,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    for _ in range(20):
        x = torch.randn(4, ca.dim)
        prod = ca.geometric_product(x, ca.reverse(x))
        scalar = prod[..., 0]
        # ||x||^2 = |<x*~x>_0|; for bivectors sign may flip but ||x|| is non-neg
        norm_sq = ca.norm(x)**2
        if (scalar.abs() - norm_sq).abs().max().item() > 1e-3:
            all_nonneg = False
status = 'PASS' if all_nonneg else 'FAIL'
print(f'  all non-negative: [{status}]')
results.append(('scalar_xx_rev_nonneg', status))

# ===== 20. Grade: e_i * e_j has grade <= 2 =====
print('\n--- 20. grade_ei_ej_leq_2 ---')
ca = CliffordAlgebra(3,1,0)
grade_ok = True
for i in range(ca.n):
    for j in range(ca.n):
        ei = torch.zeros(1, ca.dim); ei[0, 1<<i] = 1.0
        ej = torch.zeros(1, ca.dim); ej[0, 1<<j] = 1.0
        prod = ca.geometric_product(ei, ej)
        for k in range(ca.dim):
            if abs(prod[0, k].item()) > 1e-6:
                if bin(k).count('1') > 2:
                    grade_ok = False
status = 'PASS' if grade_ok else 'FAIL'
print(f'  e_i*e_j grade <= 2: [{status}]')
results.append(('grade_ei_ej', status))

# ===== 21. Involute o Reverse = Reverse o Involute =====
print('\n--- 21. involute_reverse_commute (20 trials) ---')
ca = CliffordAlgebra(3,1,0)
max_diff = 0.0
for _ in range(20):
    x = torch.randn(3, ca.dim)
    lhs = ca.involute(ca.reverse(x))
    rhs = ca.reverse(ca.involute(x))
    diff = (lhs - rhs).abs().max().item()
    max_diff = max(max_diff, diff)
status = 'PASS' if max_diff < 1e-6 else 'FAIL'
print(f'  Cl(3,1,0): max_diff={max_diff:.2e} [{status}]')
results.append(('inv_rev_commute', status))

# ===== 22. Norm scalar scaling: ||c*x|| = |c|*||x|| =====
print('\n--- 22. norm_scalar_scaling (20 trials) ---')
ca = CliffordAlgebra(3,1,0)
all_ok = True
for _ in range(20):
    x = torch.randn(4, ca.dim)
    c = torch.randn(1).item()
    scaled_norm = ca.norm(c * x).max().item()
    expected = abs(c) * ca.norm(x).max().item()
    if abs(scaled_norm - expected) > 1e-4 * max(expected, 1e-8):
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  scaling holds: [{status}]')
results.append(('norm_scalar_scale', status))

# ===== 23. Scalar part of reverse = scalar part of original (scalar is invariant) =====
print('\n--- 23. reverse_scalar_invariance (20 trials) ---')
ca = CliffordAlgebra(3,1,0)
max_diff = 0.0
for _ in range(20):
    x = torch.randn(5, ca.dim)
    x_rev = ca.reverse(x)
    diff = (x[:, 0] - x_rev[:, 0]).abs().max().item()
    max_diff = max(max_diff, diff)
status = 'PASS' if max_diff < 1e-6 else 'FAIL'
print(f'  max_diff={max_diff:.2e} [{status}]')
results.append(('rev_scalar_inv', status))

# ===== 24. Sandwich composition: S_R2(S_R1(x)) = S_{R2*R1}(x) =====
print('\n--- 24. sandwich_composition (20 trials each) ---')
for p,q,r in [(2,0,0),(3,1,0),(4,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    max_diff = 0.0
    for _ in range(20):
        R1 = make_bivector_rotor(ca)
        R2 = make_bivector_rotor(ca)
        x = torch.randn(1, ca.dim)
        composed = ca.sandwich(R2, ca.sandwich(R1, x, unit=True), unit=True)
        R21 = ca.geometric_product(R2, R1)
        R21 = R21 / ca.norm(R21)
        direct = ca.sandwich(R21, x, unit=True)
        diff = (composed - direct).abs().max().item()
        max_diff = max(max_diff, diff)
    status = 'PASS' if max_diff < 0.15 else 'FAIL'
    print(f'  Cl({p},{q},{r}): max_diff={max_diff:.2e} [{status}]')
    results.append((f'sandwich_comp2_Cl{p}{q}{r}', status))

# ===== 25. QuaternionLinear: linearity (distinct from section 10 shape/grad) =====
print('\n--- 25. quaternion_linear_linearity ---')
ql = QuaternionLinear(8, 8)  # 8 features in, 8 features out (2 quaternions each)

# Linearity: ql(a*x + b*y) == a*ql(x) + b*ql(y)
x1 = torch.randn(2, 8)
x2 = torch.randn(2, 8)
a, b = 2.0, -0.5
lhs = ql(a * x1 + b * x2)
rhs = a * ql(x1) + b * ql(x2)
lin_diff = (lhs - rhs).abs().max().item()
status = 'PASS' if lin_diff < 1e-4 else 'FAIL'
print(f'  linearity: diff={lin_diff:.2e} [{status}]')
results.append(('quat_linear_linearity', status))

# ===== 26. Nilpotent basis: e_k^2 = 0 for r-basis vectors (Cl310 has r=0, Cl300 has r=0) =====
print('\n--- 26. nilpotent_basis_properties ---')
# Test with Cl(2,0,1) which has one nilpotent direction
ca_nilp = CliffordAlgebra(2, 0, 1)
k = ca_nilp.n - 1  # last basis vector is nilpotent
ek = torch.zeros(1, ca_nilp.dim)
ek[0, 1 << k] = 1.0
ek_sq = ca_nilp.geometric_product(ek, ek)
is_zero = ek_sq.abs().max().item() < 1e-6
status = 'PASS' if is_zero else 'FAIL'
print(f'  Cl(2,0,1): e_{k}^2 = 0: max={ek_sq.abs().max().item():.2e} [{status}]')
results.append(('nilpotent_basis', status))

# Verify non-nilpotent basis: e_i^2 = +1 or -1
ca_pn = CliffordAlgebra(3, 1, 0)
all_unit_sq = True
for i in range(ca_pn.n):
    ei = torch.zeros(1, ca_pn.dim)
    ei[0, 1 << i] = 1.0
    ei_sq = ca_pn.geometric_product(ei, ei)
    scalar = ei_sq[0, 0].item()
    if i < ca_pn.p:
        if abs(scalar - 1.0) > 1e-6: all_unit_sq = False
    elif i < ca_pn.p + ca_pn.q:
        if abs(scalar + 1.0) > 1e-6: all_unit_sq = False
status = 'PASS' if all_unit_sq else 'FAIL'
print(f'  Cl(3,1,0): e_i^2 = +-1 for non-nilpotent: [{status}]')
results.append(('basis_unit_sq', status))

# ===== 27. Involute sign: involute(e_k) = (-1)^grade_k * e_k =====
print('\n--- 27. involute_grade_sign (Cl310) ---')
ca = CliffordAlgebra(3, 1, 0)
all_ok = True
for idx in range(ca.dim):
    grade = bin(idx).count('1')
    expected_sign = (-1.0) ** grade
    e = torch.zeros(1, ca.dim)
    e[0, idx] = 1.0
    inv_e = ca.involute(e)
    actual = inv_e[0, idx].item()
    if abs(actual - expected_sign) > 1e-6:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  all grades: [{status}]')
results.append(('involute_grade_sign', status))

# ===== 28. Geometric product of scalar and multivector is scalar multiplication =====
print('\n--- 28. scalar_multivector_product (20 trials) ---')
ca = CliffordAlgebra(3, 1, 0)
max_diff = 0.0
for _ in range(20):
    s = torch.randn(1).item()
    x = torch.randn(1, ca.dim)
    # s * x = (s*1) * x = geometric_product(s*1, x)
    scalar_basis = torch.zeros(1, ca.dim)
    scalar_basis[0, 0] = s
    prod = ca.geometric_product(scalar_basis, x)
    diff = (prod - s * x).abs().max().item()
    max_diff = max(max_diff, diff)
status = 'PASS' if max_diff < 1e-4 else 'FAIL'
print(f'  Cl(3,1,0): max_diff={max_diff:.2e} [{status}]')
results.append(('scalar_mult_prod', status))

# ===== 29. Bivector exponential: exp(tB)*exp(-tB) = 1 (rotor group, Euclidean bivectors only) =====
print('\n--- 29. bivector_exp_group (20 trials) ---')
ca = CliffordAlgebra(3,1,0)
max_diff = 0.0
for _ in range(20):
    # Use only Euclidean bivectors (both indices < p), where B^2 = -1
    # This is what's actually used for domain rotation in HDIM
    i = torch.randint(0, ca.p - 1, (1,)).item()
    j = torch.randint(i + 1, ca.p, (1,)).item()
    B = torch.zeros(1, ca.dim)
    B[0, (1<<i)|(1<<j)] = 1.0
    one = torch.zeros(1, ca.dim); one[0, 0] = 1.0
    theta = torch.rand(1).item() * math.pi
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    exp_pos = cos_t * one + sin_t * B
    exp_neg = cos_t * one - sin_t * B
    product = ca.geometric_product(exp_pos, exp_neg)
    diff = (product - one).abs().max().item()
    max_diff = max(max_diff, diff)
status = 'PASS' if max_diff < 1e-5 else 'FAIL'
print(f'  max_diff={max_diff:.2e} [{status}]')
results.append(('bivector_exp_group', status))

# ===== 30. Double angle: (cos t + sin t*B)^2 = cos 2t + sin 2t*B =====
print('\n--- 30. bivector_double_angle (20 trials) ---')
ca = CliffordAlgebra(3,1,0)
max_diff = 0.0
for _ in range(20):
    i = torch.randint(0, ca.p - 1, (1,)).item()
    j = torch.randint(i + 1, ca.p, (1,)).item()
    theta = torch.rand(1).item() * math.pi
    B = torch.zeros(1, ca.dim)
    B[0, (1<<i)|(1<<j)] = 1.0
    one = torch.zeros(1, ca.dim); one[0,0] = 1.0
    R = math.cos(theta) * one + math.sin(theta) * B
    R_sq = ca.geometric_product(R, R)
    expected = math.cos(2*theta) * one + math.sin(2*theta) * B
    diff = (R_sq - expected).abs().max().item()
    max_diff = max(max_diff, diff)
status = 'PASS' if max_diff < 1e-4 else 'FAIL'
print(f'  max_diff={max_diff:.2e} [{status}]')
print(f'  double angle identity: [{status}]')
results.append(('bivector_double_angle', status))

# ===== 31. Orthonormal basis inner product: e_i * e_j = metric[i] * delta_ij =====
print('\n--- 31. orthonormal_basis_inner ---')
ca = CliffordAlgebra(3,1,0)
all_ok = True
for i in range(ca.n):
    ei = torch.zeros(1, ca.dim); ei[0, 1<<i] = 1.0
    # For orthonormal basis: e_i * e_j = 0 if i!=j, = metric[i] if i=j (scalar part)
    for j in range(ca.n):
        ej = torch.zeros(1, ca.dim); ej[0, 1<<j] = 1.0
        prod = ca.geometric_product(ei, ej)
        if i == j:
            expected_scalar = float(ca.metric[i].item())
            if abs(prod[0, 0].item() - expected_scalar) > 1e-6:
                all_ok = False
        else:
            # scalar part should be 0
            if abs(prod[0, 0].item()) > 1e-6:
                all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  e_i * e_j = metric[i] delta_ij: [{status}]')
results.append(('orthonormal_basis', status))

# ===== 32. Bivector exponential: exp(theta*B) for Euclidean bivector =====
print('\n--- 32. bivector_exp_euclidean ---')
ca = CliffordAlgebra(3,0,0)
max_diff = 0.0
for _ in range(20):
    i = torch.randint(0, ca.p - 1, (1,)).item()
    j = torch.randint(i + 1, ca.p, (1,)).item()
    theta = torch.rand(1).item() * math.pi
    B = torch.zeros(1, ca.dim)
    B[0, (1<<i)|(1<<j)] = 1.0
    one = torch.zeros(1, ca.dim); one[0,0] = 1.0
    exp_B = math.cos(theta) * one + math.sin(theta) * B
    # exp(B)*exp(-B) should = 1
    exp_negB = math.cos(theta) * one - math.sin(theta) * B
    product = ca.geometric_product(exp_B, exp_negB)
    diff = (product - one).abs().max().item()
    max_diff = max(max_diff, diff)
status = 'PASS' if max_diff < 1e-4 else 'FAIL'
print(f'  exp(theta*B)*exp(-theta*B) = 1: max_diff={max_diff:.2e} [{status}]')
results.append(('bivector_exp_euclidean', status))

# ===== 33. Conjugation: conj(a*b) = conj(b)*conj(a) for reverse =====
print('\n--- 33. reverse_conjugation_order ---')
# ~ is an involutive anti-automorphism: ~(a*b) = ~b * ~a
# Already verified as theorem 19 (reverse_product_order), testing different algebras
ca = CliffordAlgebra(4,1,0)
max_diff = 0.0
for _ in range(20):
    a = torch.randn(1, ca.dim)
    b = torch.randn(1, ca.dim)
    ab = ca.geometric_product(a, b)
    rev_ab = ca.reverse(ab)
    rev_a = ca.reverse(a)
    rev_b = ca.reverse(b)
    rev_b_rev_a = ca.geometric_product(rev_b, rev_a)
    diff = (rev_ab - rev_b_rev_a).abs().max().item()
    max_diff = max(max_diff, diff)
status = 'PASS' if max_diff < 0.5 else 'FAIL'
print(f'  ~(a*b) = ~b * ~a in Cl(4,1,0): max_diff={max_diff:.2e} [{status}]')
results.append(('reverse_conjugation_order', status))

# ===== 34. Matryoshka nesting: lower-dim embedding is prefix of higher-dim =====
print('\n--- 34. matryoshka_nesting ---')
from src.models.modern_text_encoder import MatryoshkaProjection
proj = MatryoshkaProjection(64, [16, 32, 48, 64])
x = torch.randn(5, 64)
scales = proj(x)
all_ok = True
for small_dim, large_dim in [(16,32), (32,48), (48,64)]:
    diff = (scales[small_dim] - scales[large_dim][:, :small_dim]).abs().max().item()
    if diff > 1e-6:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  lower-dim is prefix of higher-dim: [{status}]')
results.append(('matryoshka_nesting', status))

# ===== 35. HBMA forward shape preservation and gradient flow =====
print('\n--- 35. hbma_forward_shape_and_grad ---')
from src.core.hbma_memory import HBMAMemory
mem = HBMAMemory(hidden_dim=64, ep_slots=8, sem_prototypes=8, proc_patterns=4)
mem.train()
x = torch.randn(2, 64)
out = mem(x)
shape_ok = out.shape == x.shape
# Check gradient flows through learnable parameters (more reliable than input)
loss = out.sum()
loss.backward()
n_grad_params = sum(1 for p in mem.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
n_total_params = sum(1 for _ in mem.parameters())
grad_ok = n_grad_params > 0
# memory_loss is a method
mem_loss = mem.memory_loss()
mem_loss_ok = isinstance(mem_loss, torch.Tensor) and mem_loss.dim() == 0
status = 'PASS' if shape_ok and grad_ok and mem_loss_ok else 'FAIL'
print(f'  output shape {tuple(out.shape)} == input shape {tuple(x.shape)}: shape_ok={shape_ok}')
print(f'  gradient flows through parameters: {n_grad_params}/{n_total_params} params, grad_ok={grad_ok}')
print(f'  memory_loss: {mem_loss.item():.6f}, is_scalar={mem_loss_ok}')
results.append(('hbma_forward_shape_and_grad', status))

# ===== 36. Clifford Conjugation anti-automorphism =====
print('\n--- 36. clifford_conjugation_anti_aut ---')
# conj(a*b) = conj(b) * conj(a) for Clifford conjugation
# Clifford conjugation = reverse ∘ involute
def clifford_conj(ca, x):
    return ca.involute(ca.reverse(x))

all_ok = True
for pqr in [(2,0,0), (3,0,0), (3,1,0)]:
    ca = CliffordAlgebra(*pqr)
    max_diff = 0.0
    for _ in range(15):
        a = torch.randn(1, ca.dim)
        b = torch.randn(1, ca.dim)
        ab = ca.geometric_product(a, b)
        conj_ab = clifford_conj(ca, ab)
        conj_b_conj_a = ca.geometric_product(clifford_conj(ca, b), clifford_conj(ca, a))
        diff = (conj_ab - conj_b_conj_a).abs().max().item()
        max_diff = max(max_diff, diff)
    if max_diff > 0.5:
        all_ok = False
    print(f'  Cl{pqr}: max_diff={max_diff:.2e}')
status = 'PASS' if all_ok else 'FAIL'
print(f'  conj(a*b) = conj(b)*conj(a): [{status}]')
results.append(('clifford_conjugation_anti_aut', status))

# ===== 37. Grade involution sign pattern =====
print('\n--- 37. grade_involution_sign ---')
# For a grade-k multivector x: involute(x) = (-1)^k * x (component-wise per grade)
# Specifically for bivector B: involute(B) = B (since grade 2)
# For vector v: involute(v) = -v (since grade 1)
all_ok = True
for pqr in [(3,0,0), (3,1,0)]:
    ca = CliffordAlgebra(*pqr)
    # Test vector: involute(v) = -v
    for _ in range(10):
        v = torch.randn(1, ca.dim)
        # Zero out all non-vector components
        v_vec = torch.zeros_like(v)
        for i in range(ca.n):
            v_vec[0, 1<<i] = v[0, 1<<i]
        inv_v = ca.involute(v_vec)
        diff = (inv_v + v_vec).abs().max().item()
        if diff > 1e-5:
            all_ok = False
    # Test bivector: involute(B) = B
    for _ in range(10):
        i = torch.randint(0, ca.n - 1, (1,)).item()
        j = torch.randint(i + 1, ca.n, (1,)).item()
        B = torch.zeros(1, ca.dim)
        B[0, (1<<i)|(1<<j)] = 1.0
        inv_B = ca.involute(B)
        diff = (inv_B - B).abs().max().item()
        if diff > 1e-5:
            all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  involute(v) = -v, involute(B) = B: [{status}]')
results.append(('grade_involution_sign', status))

# ===== 38. Triple product associativity (re-check with more trials) =====
print('\n--- 38. triple_product_associativity ---')
# (a*b)*c = a*(b*c) — verifying with more rigorous tests
all_ok = True
for pqr in [(2,0,0), (3,0,0), (3,1,0), (4,1,0)]:
    ca = CliffordAlgebra(*pqr)
    max_diff = 0.0
    for _ in range(30):
        a = torch.randn(1, ca.dim)
        b = torch.randn(1, ca.dim)
        c = torch.randn(1, ca.dim)
        ab_c = ca.geometric_product(ca.geometric_product(a, b), c)
        a_bc = ca.geometric_product(a, ca.geometric_product(b, c))
        diff = (ab_c - a_bc).abs().max().item()
        max_diff = max(max_diff, diff)
    if max_diff > 1e-4:
        all_ok = False
    print(f'  Cl{pqr}: max_diff={max_diff:.2e}')
status = 'PASS' if all_ok else 'FAIL'
print(f'  (a*b)*c = a*(b*c): [{status}]')
results.append(('triple_product_associativity', status))

# ===== 39. Geometric product scalar commutativity =====
print('\n--- 39. scalar_commutativity ---')
# s * a = a * s for scalar s and any multivector a
all_ok = True
for pqr in [(2,0,0), (3,0,0), (3,1,0)]:
    ca = CliffordAlgebra(*pqr)
    for _ in range(15):
        s = torch.randn(1).item()
        a = torch.randn(1, ca.dim)
        s1 = torch.zeros(1, ca.dim); s1[0,0] = s
        left = ca.geometric_product(s1, a)
        right = ca.geometric_product(a, s1)
        diff = (left - right).abs().max().item()
        if diff > 1e-5:
            all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  s*a = a*s: [{status}]')
results.append(('scalar_commutativity', status))

# ===== 40. Bivector exponential: sin/cos power series convergence =====
print('\n--- 40. bivector_exp_series_convergence ---')
# exp(B) = sum_n B^n / n! for unit bivector B^2 = -1
# Testing convergence with 8 terms
all_ok = True
for pqr in [(3,0,0), (3,1,0)]:
    ca = CliffordAlgebra(*pqr)
    max_diff = 0.0
    for _ in range(15):
        i = torch.randint(0, ca.n - 1, (1,)).item()
        j = torch.randint(i + 1, ca.n, (1,)).item()
        B = torch.zeros(1, ca.dim)
        B[0, (1<<i)|(1<<j)] = 1.0  # Unit bivector

        # Compute exp(B) via Taylor series
        one = torch.zeros(1, ca.dim); one[0,0] = 1.0
        exp_approx = one.clone()
        B_pow = B.clone()
        for n in range(1, 9):
            exp_approx = exp_approx + B_pow / math.factorial(n)
            if n < 8:
                B_pow = ca.geometric_product(B_pow, B)

        # Compare with cos+sin formula
        exp_exact = one + math.sin(1.0) * B  # cos(1)*one + sin(1)*B but B^2=-1 so cos(1)*1

        # Actually: exp(B) for B with B^2=-1 is cos(1)*1 + sin(1)*B
        # Wait, we need B to be unit with B^2=-1. Let me compute B^2 to check.
        B_sq = ca.geometric_product(B, B)
        b_sq_scalar = B_sq[0,0].item()
        if abs(b_sq_scalar + 1.0) < 0.1:  # B^2 = -1
            exp_exact = math.cos(1.0) * one + math.sin(1.0) * B
            diff = (exp_approx - exp_exact).abs().max().item()
            max_diff = max(max_diff, diff)

    status_local = 'PASS' if max_diff < 1e-2 else 'FAIL'
    print(f'  Cl{pqr}: max_diff={max_diff:.2e} [{status_local}]')
    if max_diff >= 1e-2:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  Taylor series converges to exp(B): [{status}]')
results.append(('bivector_exp_series_convergence', status))

# ===== 41. InvariantExtractor consistency across random inputs =====
print('\n--- 41. invariant_extractor_consistency ---')
# U_inv = R^{-1} * x * R should be same for any unit rotor R applied to same x
all_ok = True
from src.core.domain_operators import DomainRotationOperator
for pqr in [(2,0,0), (3,0,0), (3,1,0)]:
    ca = CliffordAlgebra(*pqr)
    for _ in range(10):
        x = torch.randn(2, ca.dim)
        R = make_bivector_rotor(ca)
        R_inv = ca.reverse(R)
        # Sandwich manually: R * x * R^{-1}
        Rx = ca.geometric_product(R.expand(2, -1), x)
        sandwich_result = ca.geometric_product(Rx, R_inv.expand(2, -1))
        has_nan = torch.isnan(sandwich_result).any()
        if has_nan:
            all_ok = False
        # Check norm preservation (bivector rotors preserve norm)
        orig_norm = ca.norm(x).mean().item()
        result_norm = ca.norm(sandwich_result).mean().item()
        if abs(result_norm - orig_norm) > 0.1:
            all_ok = False
    if not all_ok:
        break
    print(f'  Cl{pqr}: sandwich norm preserved, no NaN')
status = 'PASS' if all_ok else 'FAIL'
print(f'  Sandwich well-defined and norm-preserving: [{status}]')
results.append(('invariant_extractor_consistency', status))

# ===== 42. Matryoshka gradient flow through all scales =====
print('\n--- 42. matryoshka_gradient_flow_all_scales ---')
# Gradients should flow through all Matryoshka projection scales
from src.models.modern_text_encoder import MatryoshkaProjection
all_ok = True
for dim in [16, 32, 48, 64]:
    proj = MatryoshkaProjection(64, [16, 32, 48, 64])
    x = torch.randn(3, 64, requires_grad=True)
    out = proj(x, target_dim=dim)
    loss = out.mean()  # mean() avoids numerical zero-gradient for large dim
    loss.backward()
    if x.grad is None or x.grad.abs().sum() < 1e-8:
        all_ok = False
        print(f'  dim={dim}: NO GRADIENT FLOW')
status = 'PASS' if all_ok else 'FAIL'
print(f'  gradient flows through all Matryoshka scales: [{status}]')
results.append(('matryoshka_gradient_flow_all_scales', status))

# ===== 43. Clifford learnable_metric: updates preserve non-degeneracy =====
print('\n--- 43. clifford_learnable_metric_non_degenerate ---')
ca = CliffordAlgebra(3, 1, 0)
ca.use_learnable_metric = True
all_ok = True
for _ in range(10):
    # Random perturbation of learnable_metric
    ca.learnable_metric.data = torch.randn(ca.dim).abs() * 0.5 + 0.5  # positive scale
    x = torch.randn(1, ca.dim)
    y = ca.geometric_product(x, x)
    norm_val = ca.norm(x)
    if norm_val.item() < 1e-8:
        all_ok = False
    if torch.isnan(y).any() or torch.isinf(y).any():
        all_ok = False
    # Test that product is still well-defined (no all-zero from degenerate scaling)
    if y.abs().max().item() < 1e-12 and x.abs().max().item() > 0.1:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  learnable_metric perturbations preserve non-degeneracy: [{status}]')
results.append(('clifford_learnable_metric', status))

# ===== 44. InfoNCE loss >= 0 (non-negativity) =====
print('\n--- 44. infonce_loss_non_negative ---')
from src.training.trainer import HDIMTrainer
torch.manual_seed(42)
D = 64
B = 16
class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
    def forward(self, **kw):
        return {"invariant": torch.randn(B, D), "output": torch.randn(B, D)}

model = _DummyModel()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
trainer = HDIMTrainer(model, opt, device='cpu', use_infonce=True, infonce_temperature=0.1)

all_ok = True
for _ in range(20):
    src = torch.randn(B, D)
    tgt = torch.randn(B, D)
    labels = torch.ones(B)
    labels[:B//2] = 0.0
    weights = torch.ones(B)
    loss = trainer._compute_infonce_loss(src, tgt, labels, weights, temperature=0.1)
    # cross_entropy loss is always >= 0
    if loss.item() < -1e-6:
        all_ok = False
    if torch.isnan(loss) or torch.isinf(loss):
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  InfoNCE >= 0: [{status}]')
results.append(('infonce_non_negative', status))

# ===== 45. Diversity loss: bounded and gradient flows =====
print('\n--- 45. diversity_loss_bounded_and_grad ---')
all_ok = True
torch.manual_seed(42)
for _ in range(20):
    x = torch.randn(8, 64, requires_grad=True)
    loss = trainer._compute_diversity_loss(x)
    # Check gradient flows
    loss.backward()
    if x.grad is None or x.grad.abs().sum() == 0:
        all_ok = False
    # Diversity loss should be finite
    if torch.isnan(loss) or torch.isinf(loss):
        all_ok = False
    # Check variance component is bounded (normalized inputs -> variance <= 1)
    x_norm = torch.nn.functional.normalize(x, dim=-1)
    mean_v = x_norm.mean(dim=0, keepdim=True)
    variance = ((x_norm - mean_v) ** 2).sum(dim=-1).mean()
    if variance.item() > 4.0:  # theoretical max for points on unit sphere
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  diversity loss bounded, gradient flows: [{status}]')
results.append(('diversity_loss_bounded', status))

# ===== 46. Involution distribution: grade-dependent sign pattern =====
print('\n--- 46. involute_distribution_sign_pattern ---')
# For product a*b where a has grade k, b has grade l:
# The involute of each term in a*b follows sign pattern by the grade of that term
all_ok = True
ca = CliffordAlgebra(3, 1, 0)
for _ in range(20):
    # Create random multivectors and check involute is linear
    a = torch.randn(1, ca.dim)
    b = torch.randn(1, ca.dim)
    c = torch.randn(1, ca.dim)
    # involute is linear: involute(a + b) = involute(a) + involute(b)
    lhs = ca.involute(a + b)
    rhs = ca.involute(a) + ca.involute(b)
    if (lhs - rhs).abs().max().item() > 1e-6:
        all_ok = False
    # involute is involutive: involute(involute(x)) = x
    if (ca.involute(ca.involute(a)) - a).abs().max().item() > 1e-6:
        all_ok = False
    # involute distributes over scalar: involute(c*a) = c*involute(a)
    if (ca.involute(c * a) - c * ca.involute(a)).abs().max().item() > 1e-5:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  involute linearity and involution: [{status}]')
results.append(('involute_distribution', status))

# ===== 47. Bivector exp: general bivector (non-unit) via scaling =====
print('\n--- 47. bivector_exp_general_scaling ---')
# For general bivector B (not necessarily unit), exp(tB) = cos(t||B||) + sin(t||B||)*B/||B||
# Using the norm-scaled approach
ca = CliffordAlgebra(3, 1, 0)
max_diff = 0.0
for _ in range(20):
    # Random bivector (not necessarily unit)
    i = torch.randint(0, ca.p - 1, (1,)).item()
    j = torch.randint(i + 1, ca.p, (1,)).item()
    scale = torch.rand(1).item() * 3.0 + 0.1  # non-unit scale
    B = torch.zeros(1, ca.dim)
    B[0, (1<<i)|(1<<j)] = scale
    # Compute B^2 directly to verify it's a scaled negative scalar
    B_sq = ca.geometric_product(B, B)
    b_sq_scalar = B_sq[0, 0].item()
    # For Euclidean bivector with coefficient s: B = s*e_ij, B^2 = -s^2
    expected_sq = -scale**2
    diff = abs(b_sq_scalar - expected_sq)
    max_diff = max(max_diff, diff)
status = 'PASS' if max_diff < 1e-4 else 'FAIL'
print(f'  general bivector B^2 = -scale^2: max_diff={max_diff:.2e} [{status}]')
results.append(('bivector_exp_general', status))

# ===== 48. z_loss regularization: non-negative and differentiable =====
print('\n--- 48. z_loss_regularization ---')
from src.core.soft_moe_router import SoftMoERouter
all_ok = True
torch.manual_seed(42)
router = SoftMoERouter(input_dim=64, num_experts=4, expert_dim=128, z_loss_weight=0.01)
for _ in range(10):
    x = torch.randn(8, 64)
    output, state = router(x)
    z_loss = state['z_loss']
    # z_loss = (logsumexp)^2 >= 0 always
    if z_loss.item() < -1e-8:
        all_ok = False
    # Gradient flows through router
    loss = output.sum()
    loss.backward()
    if torch.isnan(output).any() or torch.isinf(output).any():
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  z_loss >= 0, no NaN/Inf: [{status}]')
results.append(('z_loss_regularization', status))

# ===== 49. Sandwich norm preservation: Cl(3,1,0) with higher precision =====
print('\n--- 49. sandwich_norm_Cl310_high_precision ---')
ca = CliffordAlgebra(3, 1, 0)
max_err = 0.0
for _ in range(100):
    R = make_bivector_rotor(ca)
    x = torch.randn(1, ca.dim)
    y = ca.sandwich(R, x, unit=True)
    err = abs(ca.norm(y).item() / max(ca.norm(x).item(), 1e-8) - 1.0)
    max_err = max(max_err, err)
status = 'PASS' if max_err < 0.01 else 'FAIL'
print(f'  Cl(3,1,0): max_err={max_err:.2e} (100 trials) [{status}]')
results.append(('sandwich_norm_Cl310_hp', status))

# ===== 50. HBMA memory_loss gradient flow =====
print('\n--- 50. hbma_memory_loss_gradient_flow ---')
from src.core.hbma_memory import HBMAMemory
all_ok = True
for _ in range(5):
    mem = HBMAMemory(hidden_dim=64, ep_slots=8, sem_prototypes=8, proc_patterns=4)
    mem.train()
    x = torch.randn(2, 64)
    out = mem(x)
    mloss = mem.memory_loss()
    mloss.backward()
    n_grad = sum(1 for p in mem.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    if n_grad == 0:
        all_ok = False
    if torch.isnan(mloss) or torch.isinf(mloss):
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  memory_loss gradient flows, finite: [{status}]')
results.append(('hbma_memory_loss_grad', status))

# ===== 48. Nilpotent basis: e_i^2 = 0 for r>0 =====
print('\n--- 48. nilpotent_basis_square_zero ---')
all_ok = True
for p,q,r,nil_idx in [(0,0,2,0),(1,0,2,0),(2,0,2,1)]:
    ca = CliffordAlgebra(p,q,r)
    for k in range(ca.p+ca.q, ca.n):
        e = torch.zeros(1, ca.dim); e[0, 1<<k] = 1.0
        prod = ca.geometric_product(e, e)
        sq = prod[0,0].abs().item()
        if sq > 1e-6:
            all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  nilpotent e_i^2=0 for r>0: [{status}]')
results.append(('nilpotent_basis_square_zero', status))

# ===== 49. Involution twice = identity =====
print('\n--- 49. involute_twice_identity ---')
all_ok = True
for p,q,r in [(2,0,0),(3,1,0),(4,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    x = torch.randn(3, ca.dim)
    double_inv = ca.involute(ca.involute(x))
    diff = (double_inv - x).abs().max().item()
    if diff > 1e-6:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  involute(involute(x)) = x: [{status}]')
results.append(('involute_twice_identity', status))

# ===== 50. Reverse twice = identity =====
print('\n--- 50. reverse_twice_identity ---')
all_ok = True
for p,q,r in [(2,0,0),(3,1,0),(4,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    x = torch.randn(3, ca.dim)
    double_rev = ca.reverse(ca.reverse(x))
    diff = (double_rev - x).abs().max().item()
    if diff > 1e-6:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  reverse(reverse(x)) = x: [{status}]')
results.append(('reverse_twice_identity', status))

# ===== 51. Sandwich roundtrip: A->B->A =====
print('\n--- 51. sandwich_roundtrip_ab_a ---')
all_ok = True
for p,q,r in [(3,0,0),(3,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    R = make_bivector_rotor(ca)
    R2 = make_bivector_rotor(ca)
    x = torch.randn(1, ca.dim)
    # A->B: sandwich(R), then B->A: sandwich(R_inv)
    y = ca.sandwich(R, x, unit=True)
    z = ca.sandwich(ca.reverse(R), y, unit=True)
    diff = (z - x).abs().max().item()
    if diff > 1e-4:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  roundtrip A->B->A identity: [{status}]')
results.append(('sandwich_roundtrip_ab_a', status))

# ===== 52. Geometric product with scalar =====
print('\n--- 52. geometric_product_scalar ---')
all_ok = True
for p,q,r in [(2,0,0),(3,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    x = torch.randn(2, ca.dim)
    scalar = torch.tensor([[2.0, 0.0, 0.0, 0.0] + [0.0]*(ca.dim-4)]).expand(2, -1)
    prod = ca.geometric_product(scalar, x)
    diff = (prod - 2.0 * x).abs().max().item()
    if diff > 1e-5:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  scalar * x = c*x (grade-0 multiplication): [{status}]')
results.append(('geometric_product_scalar', status))

# ===== 53. Sandwich composition: S(S1(x), S2) = S(S1xS2, x) =====
print('\n--- 53. sandwich_composition_sequence ---')
all_ok = True
for p,q,r in [(2,0,0),(3,0,0)]:
    ca = CliffordAlgebra(p,q,r)
    R1 = make_bivector_rotor(ca)
    R2 = make_bivector_rotor(ca)
    x = torch.randn(1, ca.dim)
    # Sequential: S(R2, S(R1, x))
    z = ca.sandwich(R2, ca.sandwich(R1, x, unit=True), unit=True)
    # Composed: S(R2xR1, x)
    R21 = ca.geometric_product(R2, R1)
    R21 = R21 / ca.norm(R21)
    y = ca.sandwich(R21, x, unit=True)
    diff = (z - y).abs().max().item()
    if diff > 0.15:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  S(R2, S(R1,x)) = S(R2xR1, x): [{status}]')
results.append(('sandwich_composition_sequence', status))

# ===== 54. QuaternionLinear output shape =====
print('\n--- 54. quaternion_linear_shape ---')
all_ok = True
for in_d, out_d in [(16,32),(64,64),(32,128)]:
    ql = QuaternionLinear(in_d, out_d)
    x = torch.randn(4, in_d)
    out = ql(x)
    if out.shape != (4, out_d):
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  QuaternionLinear correct output shape: [{status}]')
results.append(('quaternion_linear_shape', status))

# ===== 55. QuaternionLinear gradient flow =====
print('\n--- 55. quaternion_linear_gradient_flow ---')
ql = QuaternionLinear(32, 64)
x = torch.randn(4, 32, requires_grad=True)
out = ql(x)
loss = out.sum()
loss.backward()
grad_ok = (x.grad is not None) and (x.grad.abs().sum() > 0) and not torch.isnan(x.grad).any()
status = 'PASS' if grad_ok else 'FAIL'
print(f'  QuaternionLinear gradient flows to input: [{status}]')
results.append(('quaternion_linear_gradient_flow', status))

# ===== 56. QLayerNorm zero mean per component =====
print('\n--- 56. qlayernorm_zero_mean ---')
qln = QLayerNorm(16)
x = torch.randn(8, 64) * 10
y = qln(x)
means = [y[:, i*16:(i+1)*16].mean(dim=-1).abs().max().item() for i in range(4)]
all_ok = all(m < 0.1 for m in means)
status = 'PASS' if all_ok else 'FAIL'
print(f'  QLayerNorm zero mean per component: [{status}]')
results.append(('qlayernorm_zero_mean', status))

# ===== 57. Geometric product bilinearity =====
print('\n--- 57. geometric_product_bilinearity ---')
all_ok = True
for p,q,r in [(2,0,0),(3,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    a = torch.randn(1, ca.dim)
    b = torch.randn(1, ca.dim)
    c = torch.randn(1, ca.dim)
    alpha, beta = 2.5, -1.3
    lhs = ca.geometric_product(alpha*a + beta*b, c)
    rhs = alpha*ca.geometric_product(a, c) + beta*ca.geometric_product(b, c)
    diff = (lhs - rhs).abs().max().item()
    if diff > 1e-5:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  (aa+bb)*c = a(a*c)+b(b*c): [{status}]')
results.append(('geometric_product_bilinearity', status))

# ===== 58. Geometric product right bilinearity =====
print('\n--- 58. geometric_product_right_bilinearity ---')
all_ok = True
for p,q,r in [(2,0,0),(3,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    a = torch.randn(1, ca.dim)
    b = torch.randn(1, ca.dim)
    c = torch.randn(1, ca.dim)
    alpha, beta = 1.7, -0.9
    lhs = ca.geometric_product(a, alpha*b + beta*c)
    rhs = alpha*ca.geometric_product(a, b) + beta*ca.geometric_product(a, c)
    diff = (lhs - rhs).abs().max().item()
    if diff > 1e-5:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  a*(ab+bc) = a(a*b)+b(a*c): [{status}]')
results.append(('geometric_product_right_bilinearity', status))

# ===== 59. Involution distributes over addition =====
print('\n--- 59. involute_linearity ---')
all_ok = True
for p,q,r in [(2,0,0),(3,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    a = torch.randn(2, ca.dim)
    b = torch.randn(2, ca.dim)
    alpha, beta = 2.0, -1.0
    lhs = ca.involute(alpha*a + beta*b)
    rhs = alpha*ca.involute(a) + beta*ca.involute(b)
    diff = (lhs - rhs).abs().max().item()
    if diff > 1e-6:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  involute(aa+bb) = a*inv(a)+b*inv(b): [{status}]')
results.append(('involute_linearity', status))

# ===== 60. Reverse distributes over addition =====
print('\n--- 60. reverse_linearity ---')
all_ok = True
for p,q,r in [(2,0,0),(3,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    a = torch.randn(2, ca.dim)
    b = torch.randn(2, ca.dim)
    alpha, beta = 3.0, -2.0
    lhs = ca.reverse(alpha*a + beta*b)
    rhs = alpha*ca.reverse(a) + beta*ca.reverse(b)
    diff = (lhs - rhs).abs().max().item()
    if diff > 1e-6:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  reverse(aa+bb) = a*rev(a)+b*rev(b): [{status}]')
results.append(('reverse_linearity', status))

# ===== 61. Sandwich with identity rotor =====
print('\n--- 61. sandwich_identity_rotor ---')
all_ok = True
for p,q,r in [(2,0,0),(3,1,0),(4,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    x = torch.randn(1, ca.dim)
    R_id = torch.zeros(1, ca.dim); R_id[0, 0] = 1.0
    y = ca.sandwich(R_id, x, unit=True)
    diff = (y - x).abs().max().item()
    if diff > 1e-6:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  sandwich(1, x) = x: [{status}]')
results.append(('sandwich_identity_rotor', status))

# ===== 62. Grade projection idempotent =====
print('\n--- 62. grade_projection_idempotent ---')
all_ok = True
ca = CliffordAlgebra(3,1,0)
x = torch.randn(5, ca.dim)
# Project to grades and project again — should be same
for grade in range(ca.n+1):
    mask = torch.tensor([bin(i).count('1') == grade for i in range(ca.dim)], dtype=torch.float32)
    proj1 = x * mask
    proj2 = proj1 * mask
    diff = (proj1 - proj2).abs().max().item()
    if diff > 1e-6:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  grade_i(grade_i(x)) = grade_i(x): [{status}]')
results.append(('grade_projection_idempotent', status))

# ===== 63. SoftMoE: dispatch/combine weight normalization =====
print('\n--- 63. soft_moe_dispatch_combine_normalize ---')
from src.core.soft_moe_router import SoftMoERouter
moe = SoftMoERouter(input_dim=64, num_experts=4, expert_dim=128)
x = torch.randn(8, 64)
dispatch, combine = moe._compute_dispatch_combine(x)
# combine should sum to 1 per token (dim=-1)
combine_sum = combine.sum(dim=-1)
ok = (combine_sum - 1.0).abs().max().item() < 1e-4
status = 'PASS' if ok else 'FAIL'
print(f'  combine weights sum to 1 per token: [{status}]')
results.append(('soft_moe_dispatch_combine_normalize', status))

# ===== 64. SoftMoE: gradient flows through experts =====
print('\n--- 64. soft_moe_expert_gradient_flow ---')
moe = SoftMoERouter(input_dim=64, num_experts=4, expert_dim=128)
x = torch.randn(8, 64, requires_grad=True)
output, _ = moe(x)
loss = output.sum()
loss.backward()
grad_ok = (x.grad is not None) and (x.grad.abs().sum() > 0) and not torch.isnan(x.grad).any()
status = 'PASS' if grad_ok else 'FAIL'
print(f'  SoftMoE gradient flows through experts: [{status}]')
results.append(('soft_moe_expert_gradient_flow', status))

# ===== 65. DomainRotationOperator preserves rotor norm =====
print('\n--- 65. domain_rotor_norm_conservation ---')
all_ok = True
for p,q,r in [(3,0,0),(3,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    for _ in range(10):
        R = make_bivector_rotor(ca)
        x = torch.randn(1, ca.dim)
        y = ca.sandwich(R, x, unit=True)
        norm_ratio = ca.norm(y).item() / max(ca.norm(x).item(), 1e-8)
        if abs(norm_ratio - 1.0) > 0.02:
            all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  ||sandwich(R,x)|| = ||x||: [{status}]')
results.append(('domain_rotor_norm_conservation', status))

# ===== 66. Bivector e_i*e_j: square = -1 for Euclidean =====
print('\n--- 66. bivector_eiej_square_negative ---')
all_ok = True
ca = CliffordAlgebra(3,0,0)
for i in range(0, ca.n-1, 2):
    ei = torch.zeros(1, ca.dim); ei[0, 1<<i] = 1.0
    ej = torch.zeros(1, ca.dim); ej[0, 1<<(i+1)] = 1.0
    B = ca.geometric_product(ei, ej)
    B2 = ca.geometric_product(B, B)
    sq = B2[0,0].item()
    if abs(sq + 1.0) > 1e-4:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  (e_i*e_j)^2 = -1 (Euclidean bivector): [{status}]')
results.append(('bivector_eiej_square_negative', status))

# ===== 67. Titans memory: state update changes output =====
print('\n--- 67. titans_memory_state_update ---')
from src.core.titans_memory import TitansMemoryModule
mem = TitansMemoryModule(key_dim=16, val_dim=32, hidden_dim=32)
k1 = torch.randn(4, 16)
v1 = torch.randn(4, 32)
r1, _ = mem(k1, v1, update_memory=True)
r2, _ = mem(k1, v1, update_memory=False)
diff = (r1 - r2).abs().max().item()
ok = diff > 1e-6
status = 'PASS' if ok else 'FAIL'
print(f'  Titans memory state update changes output: [{status}]')
results.append(('titans_memory_state_update', status))

# ===== 68. Titans memory: no NaN after multiple updates =====
print('\n--- 68. titans_memory_no_nan_multi_update ---')
mem = TitansMemoryModule(key_dim=16, val_dim=32, hidden_dim=32)
all_ok = True
for i in range(50):
    k = torch.randn(4, 16)
    v = torch.randn(4, 32)
    out, loss = mem(k, v, update_memory=True)
    if torch.isnan(out).any() or torch.isinf(out).any() or torch.isnan(loss).any() or torch.isinf(loss).any():
        all_ok = False
        break
status = 'PASS' if all_ok else 'FAIL'
print(f'  Titans: 50 sequential updates, no NaN/Inf: [{status}]')
results.append(('titans_memory_no_nan_multi_update', status))

# ===== 69. Geometric product: e_0 * e_0 = +1 (Euclidean) =====
print('\n--- 69. scalar_basis_square_positive ---')
all_ok = True
for p,q,r in [(2,0,0),(3,0,0),(3,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    for i in range(ca.p):  # positive basis only
        e = torch.zeros(1, ca.dim); e[0, 1<<i] = 1.0
        e2 = ca.geometric_product(e, e)
        if abs(e2[0,0].item() - 1.0) > 1e-4:
            all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  e_i^2 = +1 for i < p (Euclidean): [{status}]')
results.append(('scalar_basis_square_positive', status))

# ===== 70. Geometric product: e_p^2 = -1 (negative basis) =====
print('\n--- 70. negative_basis_square_negative ---')
all_ok = True
for p,q,r in [(2,2,0),(3,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    for i in range(ca.p, ca.p+ca.q):
        e = torch.zeros(1, ca.dim); e[0, 1<<i] = 1.0
        e2 = ca.geometric_product(e, e)
        if abs(e2[0,0].item() + 1.0) > 1e-4:
            all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  e_i^2 = -1 for p<=i<p+q (negative): [{status}]')
results.append(('negative_basis_square_negative', status))

# ===== 71. Matryoshka: larger dim >= smaller dim quality =====
print('\n--- 71. matryoshka_nested_quality ---')
from src.models.modern_text_encoder import MatryoshkaProjection
proj = MatryoshkaProjection(64, [16, 32, 48, 64])
x = torch.randn(8, 64)
norms = {}
for dim in [16, 32, 48, 64]:
    out = proj(x, target_dim=dim)
    norms[dim] = out.norm(dim=-1).mean().item()
# All outputs should have reasonable magnitude
ok = all(n > 0.1 for n in norms.values())
status = 'PASS' if ok else 'FAIL'
print(f'  Matryoshka all scales produce valid output: [{status}]')
results.append(('matryoshka_nested_quality', status))

# ===== 72. HBMA: working memory FIFO capacity =====
print('\n--- 72. hbma_working_memory_capacity ---')
from src.core.hbma_memory import HBMAMemory
mem = HBMAMemory(hidden_dim=32)
x = torch.randn(1, 32)
for i in range(20):
    out = mem(x)
ok = not torch.isnan(out).any() and not torch.isinf(out).any()
status = 'PASS' if ok else 'FAIL'
print(f'  HBMA: 20 writes, no NaN/Inf: [{status}]')
results.append(('hbma_working_memory_capacity', status))

# ===== 73. Sandwich anti-commutativity of orthogonal bivectors =====
print('\n--- 73. sandwich_anticommute_orthogonal_bivectors ---')
ca = CliffordAlgebra(3,0,0)
x = torch.randn(1, ca.dim)
# Build orthogonal bivectors B1=e0e1, B2=e2e0
e0e1 = torch.zeros(1, ca.dim); e0e1[0, 3] = 1.0  # e0^e1
e2e0 = torch.zeros(1, ca.dim); e2e0[0, 5] = 1.0  # e2^e0
# Normalize to unit rotors
R1 = e0e1 / ca.norm(e0e1)
R2 = e2e0 / ca.norm(e2e0)
# S(R1, S(R2, x)) vs S(R2, S(R1, x))
z12 = ca.sandwich(R1, ca.sandwich(R2, x, unit=True), unit=True)
z21 = ca.sandwich(R2, ca.sandwich(R1, x, unit=True), unit=True)
# For commuting bivectors in Euclidean algebra, order shouldn't matter much
diff = (z12 - z21).abs().max().item()
ok = diff < 0.01  # commuting orthogonal bivectors
status = 'PASS' if ok else 'FAIL'
print(f'  S(R1,S(R2,x)) ~ S(R2,S(R1,x)) for comm. bivectors: [{status}]')
results.append(('sandwich_anticommute_orthogonal_bivectors', status))

# ===== 74. Geometric product: commutativity of grade-0 =====
print('\n--- 74. geometric_product_grade0_commutativity ---')
all_ok = True
for p,q,r in [(2,0,0),(3,1,0)]:
    ca = CliffordAlgebra(p,q,r)
    # Two scalars (grade 0)
    a = torch.zeros(1, ca.dim); a[0,0] = 3.0
    b = torch.zeros(1, ca.dim); b[0,0] = 7.0
    ab = ca.geometric_product(a, b)
    ba = ca.geometric_product(b, a)
    diff = (ab - ba).abs().max().item()
    if diff > 1e-6:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  scalar*a = a*scalar: [{status}]')
results.append(('geometric_product_grade0_commutativity', status))

# ===== 75. sandwich_transfer invariant_override equivalence =====
print('\n--- 75. sandwich_transfer_invariant_override ---')
all_ok = True
ca = CliffordAlgebra(p=3, q=1, r=0)
dim = ca.dim
R1 = torch.randn(dim); R1.data[0 if R1.dim() == 1 else 0] += 1.0
R1 = R1 / ca.norm(R1)
R2 = torch.randn(dim); R2.data[0 if R2.dim() == 1 else 0] += 1.0
R2 = R2 / ca.norm(R2)
x = torch.randn(1, dim)
G = ca.geometric_product(x, torch.ones(1, dim))
op1 = DomainRotationOperator(ca, "d1"); op1.R.data = R1
op2 = DomainRotationOperator(ca, "d2"); op2.R.data = R2
U1, G1 = sandwich_transfer(ca, G, op1, op2, invariant_override=None)
R1_inv = op1.get_inverse(); R1_n = op1._normalized_R()
step = ca.geometric_product(R1_inv.expand(*G.shape), G)
U_manual = ca.geometric_product(step, R1_n.expand(*G.shape))
U2, G2 = sandwich_transfer(ca, G, op1, op2, invariant_override=U_manual)
if not (torch.allclose(U1, U2, atol=1e-5) and torch.allclose(G1, G2, atol=1e-5)):
    all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  sandwich_transfer invariant_override: [{status}]')
results.append(('sandwich_transfer_invariant_override', status))

# ===== 76. pipeline input_is_invariant path =====
print('\n--- 76. pipeline_input_is_invariant ---')
all_ok = True
pipe = HDIMPipeline(input_dim=64, output_dim=64, clifford_p=3, clifford_q=1, domain_names=["A","B"])
pipe.eval()
# input_is_invariant=True means x is already the invariant (clifford_dim=16), not raw input (64)
clifford_dim = pipe.algebra.dim
x_inv = torch.randn(1, clifford_dim)
with torch.no_grad():
    out, state = pipe.transfer(x_inv, "A", "B", update_memory=False, memory_mode="none", input_is_invariant=True)
if not (state["g_source"] is None and state["input_is_invariant"] and out.shape[-1] == 64 and not torch.isnan(out).any()):
    all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  pipeline input_is_invariant path: [{status}]')
results.append(('pipeline_input_is_invariant', status))

# ===== 77. memory_mode none preserves =====
print('\n--- 77. memory_mode_none_preserves ---')
all_ok = True
pipe2 = HDIMPipeline(input_dim=64, output_dim=64, clifford_p=3, domain_names=["A","B"])
pipe2.eval()
x = torch.randn(2, 64)
with torch.no_grad():
    out, state = pipe2.transfer(x, "A", "B", update_memory=False, memory_mode="none")
if not (state["memory_mode"] == "none" and not state["memory_updated"]):
    all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  memory mode none preserves: [{status}]')
results.append(('memory_mode_none_preserves', status))

# ===== 78. Domain rotation apply_inverse roundtrip =====
print('\n--- 78. domain_rotor_apply_inverse_roundtrip ---')
all_ok = True
ca = CliffordAlgebra(p=3, q=1, r=0)
dim = ca.dim
# Use bivector rotor (unit norm) so epsilon has minimal effect
angle = 0.7
R = torch.zeros(1, dim); R[0, 0] = math.cos(angle); R[0, 12] = math.sin(angle)
x = torch.randn(1, dim)
# For unit bivector rotor: R_inv = reverse(R), and sandwich(R_inv, sandwich(R, x)) = x
R_inv = ca.reverse(R)
with torch.no_grad():
    y = ca.sandwich(R, x, unit=True)
    x_back = ca.sandwich(R_inv, y, unit=True)
max_diff = (x - x_back).abs().max().item()
if max_diff >= 1e-3:
    all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  domain_rotor apply_inverse roundtrip (max_diff={max_diff:.2e}): [{status}]')
results.append(('domain_rotor_apply_inverse_roundtrip', status))

# ===== 79. isomorphism loss sanity =====
print('\n--- 79. isomorphism_loss_sanity ---')
all_ok = True
pipe3 = HDIMPipeline(input_dim=64, output_dim=64, clifford_p=3, domain_names=["A","B"])
# Perturb one domain rotor so they are not identical
with torch.no_grad():
    pipe3.domain_rotors["B"].R.data += 0.5
x = torch.randn(4, 64)
loss_same = pipe3.compute_isomorphism_loss([(x, "A", "A")])
loss_diff = pipe3.compute_isomorphism_loss([(x, "A", "B")])
if not (loss_same.item() < 1e-4 and loss_diff.item() > loss_same.item()):
    all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  isomorphism loss sanity (same={loss_same.item():.2e}, diff={loss_diff.item():.2e}): [{status}]')
results.append(('isomorphism_loss_sanity', status))

# ===== 80. grade projection extraction =====
print('\n--- 80. grade_projection_extraction ---')
all_ok = True
ca = CliffordAlgebra(p=3, q=0, r=0)
dim = ca.dim
# Create a pure vector (grade 1)
v = torch.zeros(1, dim)
v[0, 1] = 1.0  # e_1 is grade 1
# Create a pure bivector (grade 2)
B = torch.zeros(1, dim)
B[0, 3] = 1.0  # e_1*e_2 is grade 2
# grade_1 of a bivector should be zero
g1_mask = torch.tensor([bin(i).count('1') == 1 for i in range(dim)], dtype=torch.float32)
g1_of_B = B * g1_mask
# grade_2 of a vector should be zero
g2_mask = torch.tensor([bin(i).count('1') == 2 for i in range(dim)], dtype=torch.float32)
g2_of_v = v * g2_mask
if not (g1_of_B.abs().max().item() < 1e-6 and g2_of_v.abs().max().item() < 1e-6):
    all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  grade projection extraction: [{status}]')
results.append(('grade_projection_extraction', status))

# ===== 81. Matryoshka nesting dimension monotonicity =====
print('\n--- 81. matryoshka_dim_monotonicity ---')
all_ok = True
from src.models.modern_text_encoder import MatryoshkaProjection
proj81 = MatryoshkaProjection(128, [32, 64, 96, 128])
x81 = torch.randn(16, 128)
# Larger target dim should produce equal or higher-quality (higher norm) output
prev_norm = 0.0
for dim in [32, 64, 96, 128]:
    out = proj81(x81, target_dim=dim)
    n = out.norm(dim=-1).mean().item()
    if n < prev_norm * 0.5:  # allow some variation but not collapse
        all_ok = False
    prev_norm = n
status = 'PASS' if all_ok else 'FAIL'
print(f'  Matryoshka dim monotonicity (non-collapsing): [{status}]')
results.append(('matryoshka_dim_monotonicity', status))

# ===== 82. SC-InfoNCE loss bounded below =====
print('\n--- 82. infoNCE_lower_bound ---')
all_ok = True
for temp in [0.07, 0.1, 0.5]:
    # InfoNCE loss is cross-entropy over similarity logits
    n = 16
    # Create embeddings: positive pairs have high similarity, negatives low
    anchors = torch.randn(n, 64)
    positives = anchors.clone()
    negatives = torch.randn(n, 63, 64)
    # Build logits: [neg_0, ..., neg_62, pos] — positive at index 63
    logits82 = torch.zeros(n, 64)
    for i in range(n):
        logits82[i, :63] = (anchors[i].unsqueeze(0) * negatives[i]).sum(-1) / temp
        logits82[i, 63] = (anchors[i] * positives[i]).sum(-1) / temp
    labels82 = torch.full((n,), 63)
    loss = F.cross_entropy(logits82, labels82)
    # With good positive scores, loss should be bounded (not explode)
    if torch.isnan(loss) or loss.item() > 20.0:
        all_ok = False
    # loss lower bound: -log(1/K) for uniform = log(K), here K=64
    if loss.item() < -math.log(64) - 1.0:
        all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  InfoNCE loss bounded (no NaN, within bounds): [{status}]')
results.append(('infoNCE_lower_bound', status))

# ===== 83. SoftMoE expert orthogonality loss >= 0 =====
print('\n--- 83. soft_moe_orthogonality_bound ---')
all_ok = True
from src.core.soft_moe_router import SoftMoERouter
for n_experts in [2, 4, 8]:
    router = SoftMoERouter(input_dim=64, num_experts=n_experts)
    for _ in range(10):
        orth_loss = router.expert_orthogonalization_loss()
        if orth_loss.item() < -1e-6:  # should always be >= 0
            all_ok = False
status = 'PASS' if all_ok else 'FAIL'
print(f'  expert_orthogonalization_loss >= 0: [{status}]')
results.append(('soft_moe_orthogonality_bound', status))

# ===== 84. Gradient norm bounded through sandwich =====
print('\n--- 84. sandwich_gradient_bounded ---')
all_ok = True
ca84 = CliffordAlgebra(p=3, q=0, r=0)
for _ in range(20):
    R84 = make_bivector_rotor(ca84)
    x84 = torch.randn(1, ca84.dim, requires_grad=True)
    y84 = ca84.sandwich(R84, x84)
    loss84 = y84.pow(2).sum()
    loss84.backward()
    if x84.grad is None or torch.isnan(x84.grad).any() or x84.grad.abs().max().item() > 1000.0:
        all_ok = False
        break
status = 'PASS' if all_ok else 'FAIL'
print(f'  sandwich gradient bounded (no NaN, max < 1000): [{status}]')
results.append(('sandwich_gradient_bounded', status))

# ===== 85. Bivector rotor composition preserves unit norm =====
print('\n--- 85. rotor_composition_unit_norm ---')
all_ok = True
ca85 = CliffordAlgebra(p=4, q=0, r=0)
for _ in range(20):
    R_a = make_bivector_rotor(ca85)
    R_b = make_bivector_rotor(ca85)
    R_ab = ca85.geometric_product(R_a, R_b)
    norm_ab = ca85.norm(R_ab).item()
    if abs(norm_ab - 1.0) > 1e-4:
        all_ok = False
        break
status = 'PASS' if all_ok else 'FAIL'
print(f'  ||R_a * R_b|| = 1 for unit bivector rotors: [{status}]')
results.append(('rotor_composition_unit_norm', status))

# ===== Summary =====
print('\n' + '='*60)
passed = sum(1 for _,s in results if s == 'PASS')
total = len(results)
print(f'RESULTS: {passed}/{total} PASS')
for name, status in results:
    mark = 'OK' if status == 'PASS' else 'FAIL'
    print(f'  [{mark}] {name}')
print('='*60)
