"""
Test for NaN/Inf detection in HDIM forward pass with AMP enabled.

Tests the following operations for numerical stability:
1. geometric_product outer product (always fp32, but checking for overflow)
2. SoftMoERouter dispatch/combine matrices
3. HBMA memory scatter operations (SemanticMemory prototype update)
4. training_invariant final output

Run: python tests/test_nan_inf_forward.py
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.hypercomplex import CliffordAlgebra
from src.core.soft_moe_router import SoftMoERouter
from src.core.hbma_memory import SemanticMemory, HBMAMemory
from src.models.hdim_model import HDIMModel, HDIMConfig


def check_tensor(tensor: torch.Tensor, name: str) -> dict:
    """Check tensor for NaN/Inf values and return diagnostics."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()

    if tensor.numel() > 0:
        min_val = tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].min().item() if not has_nan and not has_inf else float('nan')
        max_val = tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].max().item() if not has_nan and not has_inf else float('nan')
    else:
        min_val = max_val = float('nan')

    return {
        "name": name,
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "min": min_val,
        "max": max_val,
    }


def print_diagnostics(diag: dict):
    """Print diagnostic info for a tensor."""
    status = "OK"
    if diag["has_nan"] or diag["has_inf"]:
        status = "FAIL"
    print(f"  [{status}] {diag['name']}: shape={diag['shape']}, dtype={diag['dtype']}")
    if diag["has_nan"]:
        print(f"    NaN count: {diag['nan_count']}")
    if diag["has_inf"]:
        print(f"    Inf count: {diag['inf_count']}")
    if status == "OK":
        print(f"    range: [{diag['min']:.4e}, {diag['max']:.4e}]")


def test_geometric_product_overflow():
    """Test geometric_product for fp16 overflow in outer product."""
    print("\n" + "="*60)
    print("TEST 1: geometric_product outer product overflow")
    print("="*60)

    # Cl(4,1,0) -> dim = 32
    ca = CliffordAlgebra(p=4, q=1, r=0).cuda()

    # Create large multivectors that could overflow fp16
    # fp16 max = 65504, so values ~256 could overflow when squared * 32
    batch_size = 16
    x = torch.randn(batch_size, ca.dim, device='cuda')
    y = torch.randn(batch_size, ca.dim, device='cuda')

    # Scale to create potential overflow
    x = x * 100  # Values ~100 -> outer product values ~10000 -> could overflow fp16
    y = y * 100

    results = []

    # Test in fp32 (should be safe)
    with torch.no_grad():
        result_fp32 = ca.geometric_product(x, y)
        diag = check_tensor(result_fp32, "geometric_product_fp32")
        print_diagnostics(diag)
        results.append(diag)

    # Test with AMP autocast (should still be fp32 per code)
    with torch.autocast('cuda', dtype=torch.float16):
        with torch.no_grad():
            result_amp = ca.geometric_product(x, y)
            diag = check_tensor(result_amp, "geometric_product_amp")
            print_diagnostics(diag)
            results.append(diag)

    # Verify both are fp32 (per code design)
    assert result_fp32.dtype == torch.float32, f"Expected fp32, got {result_fp32.dtype}"
    assert result_amp.dtype == torch.float32, f"AMP should return fp32, got {result_amp.dtype}"

    print("\n  VERIFICATION: geometric_product always returns fp32 (correct)")

    assert all(not r["has_nan"] and not r["has_inf"] for r in results), \
        "geometric_product produced NaN/Inf"


def test_soft_moe_router_nan():
    """Test SoftMoERouter for NaN in dispatch/combine matrices."""
    print("\n" + "="*60)
    print("TEST 2: SoftMoERouter dispatch/combine NaN detection")
    print("="*60)

    input_dim = 256
    num_experts = 4
    expert_dim = 256

    router = SoftMoERouter(
        input_dim=input_dim,
        num_experts=num_experts,
        expert_dim=expert_dim,
        z_loss_weight=0.001,
    ).cuda()

    batch_size = 32
    x = torch.randn(batch_size, input_dim, device='cuda')

    results = []

    # Test without AMP
    with torch.no_grad():
        output, state = router(x)

        # Check dispatch and combine from state
        if "dispatch" in state:
            diag = check_tensor(state["dispatch"], "dispatch_no_amp")
            print_diagnostics(diag)
            results.append(diag)

        if "combine" in state:
            diag = check_tensor(state["combine"], "combine_no_amp")
            print_diagnostics(diag)
            results.append(diag)

        # Check output
        diag = check_tensor(output, "router_output_no_amp")
        print_diagnostics(diag)
        results.append(diag)

    # Test with AMP
    with torch.autocast('cuda', dtype=torch.float16):
        with torch.no_grad():
            output_amp, state_amp = router(x)

            if "dispatch" in state_amp:
                diag = check_tensor(state_amp["dispatch"], "dispatch_amp")
                print_diagnostics(diag)
                results.append(diag)

            if "combine" in state_amp:
                diag = check_tensor(state_amp["combine"], "combine_amp")
                print_diagnostics(diag)
                results.append(diag)

            diag = check_tensor(output_amp, "router_output_amp")
            print_diagnostics(diag)
            results.append(diag)

    assert all(not r["has_nan"] and not r["has_inf"] for r in results), \
        "SoftMoERouter produced NaN/Inf"


def test_hbma_semantic_scatter():
    """Test HBMA SemanticMemory scatter operations for dtype mismatch."""
    print("\n" + "="*60)
    print("TEST 3: HBMA SemanticMemory prototype scatter operations")
    print("="*60)

    hidden_dim = 256
    num_prototypes = 64

    semantic = SemanticMemory(
        hidden_dim=hidden_dim,
        num_prototypes=num_prototypes,
    ).cuda()
    semantic.train()

    batch_size = 32
    x = torch.randn(batch_size, hidden_dim, device='cuda')

    results = []

    # Test without AMP
    with torch.no_grad():
        out = semantic(x)
        diag = check_tensor(out, "semantic_output_no_amp")
        print_diagnostics(diag)
        results.append(diag)

        # Check prototypes for NaN
        diag = check_tensor(semantic.prototypes, "prototypes_no_amp")
        print_diagnostics(diag)
        results.append(diag)

    # Test with AMP (this is where dtype mismatch can occur)
    semantic.reset()
    with torch.autocast('cuda', dtype=torch.float16):
        # Forward pass will trigger _update_prototypes
        out_amp = semantic(x)

    with torch.no_grad():
        diag = check_tensor(out_amp, "semantic_output_amp")
        print_diagnostics(diag)
        results.append(diag)

        diag = check_tensor(semantic.prototypes, "prototypes_amp")
        print_diagnostics(diag)
        results.append(diag)

    assert all(not r["has_nan"] and not r["has_inf"] for r in results), \
        "HBMA SemanticMemory produced NaN/Inf"


def test_full_hdim_forward():
    """Test full HDIM model forward pass with Phase 26 config."""
    print("\n" + "="*60)
    print("TEST 4: Full HDIM forward pass (Phase 26 config)")
    print("="*60)

    config = HDIMConfig(
        hidden_dim=256,
        num_domains=4,
        num_experts=4,
        clifford_p=4,
        clifford_q=1,
        clifford_r=0,
        top_k=2,
        memory_type="hbma",
    )

    model = HDIMModel(config).cuda()
    model.train()

    batch_size = 16
    x = torch.randn(batch_size, config.hidden_dim, device='cuda')
    domain_id = torch.randint(0, config.num_domains, (batch_size,), device='cuda')

    results = []

    # Test without AMP
    with torch.no_grad():
        output, routing_weights, invariant, _, aux = model(
            x, domain_id=domain_id, return_state=True, update_memory=True, memory_mode="update"
        )

        # Check all key tensors
        diag = check_tensor(output, "model_output_no_amp")
        print_diagnostics(diag)
        results.append(diag)

        diag = check_tensor(aux.training_invariant, "training_invariant_no_amp")
        print_diagnostics(diag)
        results.append(diag)

        diag = check_tensor(aux.raw_invariant, "raw_invariant_no_amp")
        print_diagnostics(diag)
        results.append(diag)

        diag = check_tensor(routing_weights, "routing_weights_no_amp")
        print_diagnostics(diag)
        results.append(diag)

    # Test with AMP
    model.zero_grad()
    with torch.autocast('cuda', dtype=torch.float16):
        output_amp, routing_weights_amp, invariant_amp, _, aux_amp = model(
            x, domain_id=domain_id, return_state=True, update_memory=True, memory_mode="update"
        )

    with torch.no_grad():
        diag = check_tensor(output_amp, "model_output_amp")
        print_diagnostics(diag)
        results.append(diag)

        diag = check_tensor(aux_amp.training_invariant, "training_invariant_amp")
        print_diagnostics(diag)
        results.append(diag)

        diag = check_tensor(aux_amp.raw_invariant, "raw_invariant_amp")
        print_diagnostics(diag)
        results.append(diag)

        diag = check_tensor(routing_weights_amp, "routing_weights_amp")
        print_diagnostics(diag)
        results.append(diag)

    assert all(not r["has_nan"] and not r["has_inf"] for r in results), \
        "Full HDIM forward produced NaN/Inf"


def test_extreme_values():
    """Test model behavior with extreme input values."""
    print("\n" + "="*60)
    print("TEST 5: Extreme input values (potential overflow)")
    print("="*60)

    config = HDIMConfig(
        hidden_dim=256,
        num_domains=4,
        num_experts=4,
        clifford_p=4,
        clifford_q=1,
        clifford_r=0,
        top_k=2,
        memory_type="hbma",
    )

    model = HDIMModel(config).cuda()
    model.eval()

    batch_size = 8

    results = []

    # Test 1: Large values
    x_large = torch.randn(batch_size, config.hidden_dim, device='cuda') * 100
    domain_id = torch.zeros(batch_size, dtype=torch.long, device='cuda')

    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.float16):
            output, routing_weights, invariant, _, aux = model(x_large, domain_id=domain_id, return_state=True)

        diag = check_tensor(output, "output_large_inputs")
        print_diagnostics(diag)
        results.append(diag)

        diag = check_tensor(aux.training_invariant, "training_invariant_large")
        print_diagnostics(diag)
        results.append(diag)

    # Test 2: Very small values (underflow)
    x_small = torch.randn(batch_size, config.hidden_dim, device='cuda') * 1e-6

    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.float16):
            output, routing_weights, invariant, _, aux = model(x_small, domain_id=domain_id, return_state=True)

        diag = check_tensor(output, "output_small_inputs")
        print_diagnostics(diag)
        results.append(diag)

        diag = check_tensor(aux.training_invariant, "training_invariant_small")
        print_diagnostics(diag)
        results.append(diag)

    # Test 3: Mixed scale (can cause issues)
    x_mixed = torch.randn(batch_size, config.hidden_dim, device='cuda')
    x_mixed[0] *= 1e3
    x_mixed[1] *= 1e-3

    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.float16):
            output, routing_weights, invariant, _, aux = model(x_mixed, domain_id=domain_id, return_state=True)

        diag = check_tensor(output, "output_mixed_scale")
        print_diagnostics(diag)
        results.append(diag)

        diag = check_tensor(aux.training_invariant, "training_invariant_mixed")
        print_diagnostics(diag)
        results.append(diag)

    assert all(not r["has_nan"] and not r["has_inf"] for r in results), \
        "Extreme values produced NaN/Inf"


def analyze_overflow_sources():
    """Analyze potential sources of fp16 overflow."""
    print("\n" + "="*60)
    print("ANALYSIS: Potential fp16 overflow sources in HDIM")
    print("="*60)

    fp16_max = 65504.0
    fp16_min = -65504.0

    print(f"\nfp16 range: [{fp16_min:.0f}, {fp16_max:.0f}]")
    print(f"fp32 range: [{torch.finfo(torch.float32).min:.2e}, {torch.finfo(torch.float32).max:.2e}]")

    print("\n--- Potential overflow sources ---")

    # 1. geometric_product outer product
    print("\n1. geometric_product outer product (hypercomplex.py:139):")
    print("   outer = a.unsqueeze(-1) * b.unsqueeze(-2)  # (..., D, D)")
    dim = 32  # Cl(4,1,0)
    print(f"   Cl(4,1,0) dim = {dim}")
    print(f"   Outer product scales: if |a|,|b| ~ 100, outer values ~ 10000")
    print(f"   Mitigation: Code upcasts to fp32 (line 133-135)")
    print(f"   Status: SAFE (always fp32)")

    # 2. SoftMoERouter logits
    print("\n2. SoftMoERouter dispatch_proj (soft_moe_router.py:123):")
    print("   logits = self.dispatch_proj(x) / self.temperature")
    print("   Large logits can cause softmax overflow/underflow")
    print(f"   fp16 softmax safe range: logits should be < ~10")
    print(f"   Mitigation: Check weight initialization, add z_loss")
    print(f"   Status: POTENTIAL RISK if weights explode")

    # 3. HBMA scatter_add
    print("\n3. HBMA SemanticMemory scatter_add (hbma_memory.py:437-438):")
    print("   proto_sum.scatter_add_(0, assigns.unsqueeze(-1), h_norm_f32)")
    print("   Issue: Under AMP, h_norm could be fp16, scatter_add needs fp32")
    print(f"   Mitigation: Code explicitly uses h_norm.float() (line 434)")
    print(f"   Status: SAFE (explicit fp32 conversion)")

    # 4. Sandwich product chain
    print("\n4. sandwich product chain (hypercomplex.py:214-215):")
    print("   Rx = geometric_product(R, x)")
    print("   result = geometric_product(Rx, R_inv)")
    print("   Chain: 2 sequential gp calls, each returns fp32")
    print(f"   Status: SAFE (geometric_product always fp32)")

    # 5. Training invariant projection
    print("\n5. training_inv_head projection (hdim_model.py:153):")
    print("   self.training_inv_head = nn.Linear(clifford_dim, hidden_dim)")
    print("   Under AMP: input fp32, weight fp16 -> output fp16")
    print(f"   Status: POTENTIAL RISK if input has large values")

    print("\n--- Summary ---")
    print("SAFE operations (explicit fp32):")
    print("  - geometric_product outer product")
    print("  - HBMA scatter_add (explicit .float())")
    print("  - sandwich chains (composed of fp32 gp calls)")

    print("\nPOTENTIAL RISKS:")
    print("  - SoftMoERouter dispatch_proj weights exploding")
    print("  - training_inv_head input values (clifford_dim=32)")


def main():
    print("="*60)
    print("HDIM NaN/Inf Forward Pass Test Suite")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    all_passed = True

    try:
        passed = test_geometric_product_overflow()
        all_passed &= passed
        print(f"\n  Test 1: {'PASS' if passed else 'FAIL'}")
    except Exception as e:
        print(f"\n  Test 1: ERROR - {e}")
        all_passed = False

    try:
        passed = test_soft_moe_router_nan()
        all_passed &= passed
        print(f"\n  Test 2: {'PASS' if passed else 'FAIL'}")
    except Exception as e:
        print(f"\n  Test 2: ERROR - {e}")
        all_passed = False

    try:
        passed = test_hbma_semantic_scatter()
        all_passed &= passed
        print(f"\n  Test 3: {'PASS' if passed else 'FAIL'}")
    except Exception as e:
        print(f"\n  Test 3: ERROR - {e}")
        all_passed = False

    try:
        passed = test_full_hdim_forward()
        all_passed &= passed
        print(f"\n  Test 4: {'PASS' if passed else 'FAIL'}")
    except Exception as e:
        print(f"\n  Test 4: ERROR - {e}")
        all_passed = False

    try:
        passed = test_extreme_values()
        all_passed &= passed
        print(f"\n  Test 5: {'PASS' if passed else 'FAIL'}")
    except Exception as e:
        print(f"\n  Test 5: ERROR - {e}")
        all_passed = False

    analyze_overflow_sources()

    print("\n" + "="*60)
    print(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
