"""Comprehensive module test suite for HDIM after Clifford algebra fix."""
import sys, os
sys.path.insert(0, '.')
os.environ['PYTHONIOENCODING'] = 'utf-8'
import torch

results = []

def test(name, ok):
    results.append((name, ok))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}')

# 1. CliffordAlgebra norm preservation
print('=== 1. CliffordAlgebra + Norm Preservation ===')
from src.core.hypercomplex import CliffordAlgebra, QuaternionLinear, QLayerNorm

ca = CliffordAlgebra(3, 1, 0)
test('CliffordAlgebra(3,1,0) init', ca.dim == 16)

# Proper bivector rotor norm test
def make_rotor(ca, angles):
    R = torch.zeros(1, ca.dim); R[0, 0] = 1.0
    for k, angle in enumerate(angles):
        i = 2 * k
        if i + 1 >= ca.n: break
        blade_idx = (1 << i) | (1 << (i + 1))
        rot = torch.zeros(1, ca.dim)
        rot[0, 0] = float(torch.cos(torch.tensor(angle)))
        rot[0, blade_idx] = float(torch.sin(torch.tensor(angle)))
        R = ca.geometric_product(R, rot)
    return R

max_err = 0
for _ in range(20):
    angles = [float(torch.rand(1).item() * 3.14159) for _ in range(ca.n // 2)]
    R = make_rotor(ca, angles) / ca.norm(make_rotor(ca, angles))
    x = torch.randn(1, ca.dim)
    nx = ca.norm(x).item()
    nr = ca.norm(ca.sandwich(R, x, unit=True)).item()
    max_err = max(max_err, abs(nr / max(nx, 1e-8) - 1.0))
test(f'Norm preservation Cl(3,1,0) err={max_err:.2e}', max_err < 0.01)

# Roundtrip
max_diff = 0
for _ in range(10):
    angles = [float(torch.rand(1).item() * 3.14159) for _ in range(ca.n // 2)]
    R = make_rotor(ca, angles) / ca.norm(make_rotor(ca, angles))
    x = torch.randn(1, ca.dim)
    y = ca.sandwich(R, x, unit=True)
    z = ca.sandwich(ca.reverse(R), y, unit=True)
    max_diff = max(max_diff, (z - x).abs().max().item())
test(f'Roundtrip Cl(3,1,0) diff={max_diff:.2e}', max_diff < 0.01)

# 2. QuaternionLinear + QLayerNorm
print('\n=== 2. QuaternionLinear + QLayerNorm ===')
ql = QuaternionLinear(16, 16)
y = ql(torch.randn(4, 16))
test('QuaternionLinear(16->16)', y.shape == (4, 16))
qn = QLayerNorm(4)
y = qn(torch.randn(4, 16))
test('QLayerNorm(4)', y.shape == (4, 16))

# 3. Domain operators
print('\n=== 3. Domain Operators ===')
from src.core.domain_operators import DomainRotationOperator, InvariantExtractor, sandwich_transfer

dro = DomainRotationOperator(ca, 'test')
x = torch.randn(4, 16)
y = dro(x)
R_inv = dro.get_inverse()
y_inv = ca.sandwich(R_inv.expand_as(y), y, unit=True)
diff = (y_inv - x).abs().max().item()
test(f'DomainRotationOperator roundtrip diff={diff:.2e}', diff < 1e-3)

src_op = DomainRotationOperator(ca, 'src')
tgt_op = DomainRotationOperator(ca, 'tgt')
_, g_tgt = sandwich_transfer(ca, x, src_op, tgt_op)
x_back, _ = sandwich_transfer(ca, g_tgt, tgt_op, src_op)
diff2 = (x_back - x).abs().max().item()
test(f'sandwich_transfer roundtrip diff={diff2:.2e}', diff2 < 0.1)

# 4. Memory modules
print('\n=== 4. Memory Modules ===')
from src.core.titans_memory import TitansMemoryModule
from src.core.hbma_memory import HBMAMemory

u = torch.randn(4, 256)
tm = TitansMemoryModule(key_dim=256, val_dim=256, hidden_dim=64)
out, loss = tm(u, u)
test('TitansMemoryModule', out.shape == u.shape)

hm = HBMAMemory(hidden_dim=256)
out = hm(u)
test('HBMAMemory', out.shape == u.shape)

# 5. SoftMoERouter
print('\n=== 5. SoftMoERouter ===')
from src.core.soft_moe_router import SoftMoERouter
router = SoftMoERouter(input_dim=256, num_experts=4, top_k=2)
out, aux = router(u)
test('SoftMoERouter', out.shape == u.shape)

# 6. HDIMModel (all memory types)
print('\n=== 6. HDIMModel ===')
from src.models.hdim_model import HDIMConfig, HDIMModel
for mtype in ['titans', 'hbma', 'cls', 'hippocampus']:
    cfg = HDIMConfig(hidden_dim=256, num_domains=4, num_experts=4, memory_type=mtype)
    model = HDIMModel(cfg)
    model.eval()
    x = torch.randn(4, 256)
    domain = torch.randint(0, 4, (4,))
    with torch.no_grad():
        out, rw, inv = model(x, domain)
    n_params = sum(p.numel() for p in model.parameters())
    test(f'HDIMModel({mtype}) params={n_params:,}', out.shape == x.shape)

# 7. Training step
print('\n=== 7. Training Step ===')
cfg = HDIMConfig(hidden_dim=128, num_domains=2, num_experts=2, memory_type='hbma')
model = HDIMModel(cfg)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(8, 128)
domain = torch.randint(0, 2, (8,))
out, rw, inv = model(x, domain)
loss = torch.nn.functional.mse_loss(out, x)
loss.backward()
opt.step()
test('Training step (forward+backward+step)', True)

# Summary
print('\n' + '=' * 60)
n_pass = sum(1 for _, ok in results if ok)
n_fail = len(results) - n_pass
print(f'RESULTS: {n_pass}/{len(results)} passed, {n_fail} failed')
for name, ok in results:
    if not ok:
        print(f'  FAIL: {name}')
print('=' * 60)
