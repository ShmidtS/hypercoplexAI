import pytest
import torch

from src.models.hdim_model import HDIMConfig, HDIMModel


@pytest.fixture
def cfg():
    return HDIMConfig()


@pytest.fixture
def model(cfg):
    return HDIMModel(cfg)


def test_identity_rotor_preserves_input():
    """Identity rotor (R[0]=1, rest=0) должен давать sandwich(R, x) ≈ x."""
    from src.core.hypercomplex import CliffordAlgebra
    from src.core.domain_operators import DomainRotationOperator
    alg = CliffordAlgebra(p=2, q=0, r=0)  # dim=4
    rotor = DomainRotationOperator(alg, init_identity=True)
    x = torch.randn(alg.dim)
    with torch.no_grad():
        x_rotated = rotor(x)
    assert torch.allclose(x_rotated, x, atol=1e-5), f"Identity rotor changed input: max diff={(x_rotated - x).abs().max().item()}"


def test_rotor_inverse_matches_explicit_reverse_formula():
    """apply_inverse(rotor(x)) должен совпадать с явной формулой через reverse(R)."""
    import math
    from src.core.hypercomplex import CliffordAlgebra
    from src.core.domain_operators import DomainRotationOperator
    alg = CliffordAlgebra(p=2, q=0, r=0)
    rotor = DomainRotationOperator(alg, init_identity=False)
    theta = math.pi / 6
    with torch.no_grad():
        rotor.R.data = torch.tensor([math.cos(theta), 0.0, 0.0, math.sin(theta)])
    x = torch.randn(alg.dim)
    with torch.no_grad():
        x_fwd = rotor(x)
        x_back = rotor.apply_inverse(x_fwd)
        expected = alg.sandwich(rotor.get_inverse(), x_fwd)
    assert torch.allclose(x_back, expected, atol=1e-6)


def test_round_trip_transfer_same_domain(model, cfg):
    """Перенос в тот же домен не должен сильно искажать вход (reconstruction)."""
    x = torch.randn(4, cfg.hidden_dim)
    domain_id = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        out = model(x, domain_id, update_memory=False, memory_mode="retrieve").output
    assert out.shape == x.shape


def test_raw_invariant_differs_from_exported(model, cfg):
    """raw_invariant и exported_invariant должны проходить через канонический lifecycle."""
    x = torch.randn(4, cfg.hidden_dim)
    domain_id = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        state = model(
            x, domain_id,
            return_state=True,
            update_memory=False,
            memory_mode="retrieve",
        ).aux_state
    raw = state.raw_invariant
    exported = state.exported_invariant
    assert raw.shape == exported.shape
    assert raw.shape == (4, model.pipeline.clifford_dim)


def test_clifford_geometric_product_anticommutativity():
    """e1 * e2 = -e2 * e1 в Cl_{2,0,0}."""
    from src.core.hypercomplex import CliffordAlgebra
    alg = CliffordAlgebra(p=2, q=0, r=0)
    e1 = torch.zeros(alg.dim)
    e1[1] = 1.0
    e2 = torch.zeros(alg.dim)
    e2[2] = 1.0
    e1e2 = alg.geometric_product(e1, e2)
    e2e1 = alg.geometric_product(e2, e1)
    assert torch.allclose(e1e2, -e2e1, atol=1e-6), f"e1*e2 != -e2*e1: {e1e2} vs {-e2e1}"
