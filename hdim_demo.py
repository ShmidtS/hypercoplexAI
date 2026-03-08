"""
HDIM Demo — Hypercomplex Domain Isomorphism Machine
Демонстрация кроссдоменного переноса знаний.

Запуск: python hdim_demo.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.optim as optim

from src.core import (
    CliffordAlgebra,
    QuaternionLinear,
    HDIMPipeline,
    sandwich_transfer,
    DomainRotationOperator,
    hamilton_product,
)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Возвращает сопряжённый кватернион для тензора (..., 4)."""
    signs = q.new_tensor([1.0, -1.0, -1.0, -1.0])
    return q * signs

def quaternion_norm(q: torch.Tensor) -> torch.Tensor:
    """Возвращает норму кватерниона для тензора (..., 4)."""
    return torch.linalg.vector_norm(q, dim=-1)


def demo_clifford_algebra():
    """Демонстрация алгебры Клиффорда Cl_{2,0,0}."""
    print("\n=== CliffordAlgebra Cl_{2,0,0} ===")
    alg = CliffordAlgebra(p=2, q=0, r=0)  # dim=4
    print(f"  dim = {alg.dim}")

    a = torch.tensor([1.0, 0.5, 0.0, 0.0])  # мультивектор a
    b = torch.tensor([0.0, 1.0, 0.0, 0.0])  # мультивектор b
    ab = alg.geometric_product(a, b)
    print(f"  a = {a.tolist()}")
    print(f"  b = {b.tolist()}")
    print(f"  a*b = {ab.tolist()}")
    
    norm_a = alg.norm(a)
    print(f"  ||a|| = {norm_a.item():.4f}")
    print("  OK")


def demo_quaternion_linear():
    """Демонстрация кватернионного линейного слоя."""
    print("\n=== QuaternionLinear ===")
    layer = QuaternionLinear(in_features=8, out_features=8)
    x = torch.randn(4, 8)  # batch=4, features=8
    y = layer(x)
    print(f"  input shape:  {x.shape}")
    print(f"  output shape: {y.shape}")
    print(f"  params: {sum(p.numel() for p in layer.parameters())}")
    print("  OK")


def demo_quaternion_ops():
    """Демонстрация базовых кватернионных операций."""
    print("\n=== Quaternion Ops ===")
    q1 = torch.tensor([1.0, 2.0, -1.0, 0.5])
    q2 = torch.tensor([0.5, -1.0, 0.0, 2.0])
    q_prod = hamilton_product(q1, q2)
    q_conj = quaternion_conjugate(q1)
    q_norm = quaternion_norm(q1)
    print(f" q1 = {q1.tolist()}")
    print(f" q2 = {q2.tolist()}")
    print(f" q1 * q2 = {q_prod.tolist()}")
    print(f" conj(q1) = {q_conj.tolist()}")
    print(f" ||q1|| = {q_norm.item():.4f}")
    print(" OK")


def demo_sandwich_transfer():
    """Демонстрация кроссдоменного переноса инварианта."""
    print("\n=== Sandwich Transfer (engineering -> biology) ===")
    alg = CliffordAlgebra(p=3, q=1, r=0)  # dim=16
    print(f"  Clifford algebra dim = {alg.dim}")

    R_eng = DomainRotationOperator(alg, domain_name='engineering')
    R_bio = DomainRotationOperator(alg, domain_name='biology')

    # Рандомный мультивектор исходного домена
    G_source = torch.randn(2, alg.dim)  # batch=2
    print(f"  G_source shape: {G_source.shape}")

    U_inv, G_target = sandwich_transfer(alg, G_source, R_eng, R_bio)
    print(f"  U_inv shape:   {U_inv.shape}")
    print(f"  G_target shape:{G_target.shape}")
    
    # Проверяем что инвариант != источнику (операция нетривиальна)
    diff = (U_inv - G_source).abs().mean().item()
    print(f"  |U_inv - G_source| = {diff:.4f}")
    print("  OK")


def demo_hdim_pipeline():
    """Демонстрация полного пайплайна HDIM."""
    print("\n=== HDIMPipeline ===")
    
    model = HDIMPipeline(
        input_dim=32,
        output_dim=32,
        clifford_p=2,
        clifford_q=0,
        clifford_r=0,
        domain_names=['math', 'biology', 'engineering'],
        num_experts=4,
        top_k=2,
        memory_key_dim=16,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Clifford dim: {model.clifford_dim}")
    
    # Forward pass
    x = torch.randn(8, 32)  # batch=8
    model.eval()
    with torch.no_grad():
        output, state = model(x, source_domain='math', target_domain='biology')

    print(f"  input shape:  {x.shape}")
    print(f"  output shape: {output.shape}")
    print(
        "  state: "
        f"memory_loss={state['memory_loss'].item():.4f}, "
        f"router_loss={state['router_state']['router_loss'].item():.4f}"
    )
    print("  OK")


def demo_training_step():
    """Демонстрация шага обучения с L_iso."""
    print("\n=== Training Step (L_iso) ===")
    
    model = HDIMPipeline(
        input_dim=16,
        output_dim=16,
        clifford_p=2,
        clifford_q=0,
        clifford_r=0,
        domain_names=['source', 'target'],
        num_experts=2,
        top_k=1,
        memory_key_dim=8,
    )
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    x = torch.randn(4, 16)
    
    for step in range(3):
        optimizer.zero_grad()

        _, state = model(x, 'source', 'target')

        iso_loss = model.compute_isomorphism_loss([
            (x, 'source', 'target')
        ])
        total_loss = state['memory_loss'] + 0.01 * state['router_state']['router_loss'] + iso_loss
        total_loss.backward()
        optimizer.step()

        print(f"  step {step+1}: loss={total_loss.item():.4f}, iso={iso_loss.item():.4f}")
    
    print("  OK")


if __name__ == '__main__':
    print("HDIM — Hypercomplex Domain Isomorphism Machine")
    print("=" * 50)
    
    try:
        demo_clifford_algebra()
        demo_quaternion_linear()
        demo_quaternion_ops()
        demo_sandwich_transfer()
        demo_hdim_pipeline()
        demo_training_step()
        print("\n" + "=" * 50)
        print("ALL DEMOS PASSED")
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
