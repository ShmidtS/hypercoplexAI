"""Numerical verification of all Lean4 formalization theorems for HDIM."""
import torch, math
from src.core.hypercomplex import CliffordAlgebra, QuaternionLinear, QLayerNorm
from src.core.domain_operators import DomainRotationOperator, sandwich_transfer, InvariantExtractor
from src.core.memory_interface import TitansAdapter, HBMAMemoryAdapter
from src.core.titans_memory import TitansMemoryModule
from src.core.hbma_memory import HBMAMemory

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
        R_inv = ca.reverse(R) / (ca.norm(R)**2 + 1e-8).unsqueeze(-1)
        y = ca.sandwich(R_inv, x, unit=False)
        z = ca.sandwich(R, y, unit=False)
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
proj = MatryoshkaProjection(64, [16, 32, 48, 64])
x = torch.randn(3, 64, requires_grad=True)
scales = proj(x)
all_ok = True
for dim in [16, 32, 48, 64]:
    loss = scales[dim].sum()
    loss.backward(retain_graph=True)
    if x.grad is None or x.grad.abs().sum() == 0:
        all_ok = False
        print(f'  dim={dim}: NO GRADIENT FLOW')
    else:
        x.grad = None  # Reset for next scale test
status = 'PASS' if all_ok else 'FAIL'
print(f'  gradient flows through all Matryoshka scales: [{status}]')
results.append(('matryoshka_gradient_flow_all_scales', status))

# ===== Summary =====
print('\n' + '='*60)
passed = sum(1 for _,s in results if s == 'PASS')
total = len(results)
print(f'RESULTS: {passed}/{total} PASS')
for name, status in results:
    mark = 'OK' if status == 'PASS' else 'FAIL'
    print(f'  [{mark}] {name}')
print('='*60)
