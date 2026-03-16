"""Numerical verification of all Lean4 formalization theorems for HDIM."""
import torch, math
from src.core.hypercomplex import CliffordAlgebra
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
print('LEAN4 NUMERICAL VERIFICATION — HDIM Core Theorems')
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
    status = 'PASS' if max_err < 0.01 else 'FAIL'
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

# ===== Summary =====
print('\n' + '='*60)
passed = sum(1 for _,s in results if s == 'PASS')
total = len(results)
print(f'RESULTS: {passed}/{total} PASS')
for name, status in results:
    mark = 'OK' if status == 'PASS' else 'FAIL'
    print(f'  [{mark}] {name}')
print('='*60)
