import torch

from src.core.algebra import CliffordAlgebra
from src.core.invariants import InvariantExtractor, sandwich_transfer
from src.core.rotors import DomainRotationOperator


def _rotor(algebra: CliffordAlgebra) -> DomainRotationOperator:
    rotor = DomainRotationOperator(algebra)
    with torch.no_grad():
        values = torch.linspace(0.05, 0.15, steps=rotor.R.numel(), dtype=rotor.R.dtype)
        rotor.R.copy_(values)
    return rotor


def test_extract_identity_rotor_matches_input_without_layernorm():
    algebra = CliffordAlgebra(p=3, q=0, r=0)
    extractor = InvariantExtractor(algebra)
    rotor = DomainRotationOperator(algebra, init_identity=True)
    x = torch.randn(2, algebra.dim)

    extracted = extractor(x, rotor)

    torch.testing.assert_close(extracted, x, rtol=1e-5, atol=1e-6)


def test_transfer_then_extract_roundtrip():
    algebra = CliffordAlgebra(p=3, q=0, r=0)
    rotor = _rotor(algebra)
    extractor = InvariantExtractor(algebra)
    u = torch.randn(4, algebra.dim)
    r_n = rotor._normalized_R()
    r_inv = rotor.get_inverse()

    g = algebra.geometric_product(r_n.expand(*u.shape), u)
    g = algebra.geometric_product(g, r_inv.expand(*u.shape))
    extracted = extractor(g, rotor)

    torch.testing.assert_close(extracted, u, rtol=1e-4, atol=1e-5)


def test_unit_rotor_preserves_norm():
    algebra = CliffordAlgebra(p=3, q=0, r=0)
    rotor = _rotor(algebra)
    x = torch.randn(4, algebra.dim)

    r_n = rotor._normalized_R()
    rotated = algebra.sandwich(r_n.expand(*x.shape), x, unit=True)

    torch.testing.assert_close(algebra.norm(rotated), algebra.norm(x), rtol=1e-4, atol=1e-5)


def test_sandwich_transfer_matches_manual_products():
    algebra = CliffordAlgebra(p=3, q=0, r=0)
    source = _rotor(algebra)
    target = _rotor(algebra)
    u = torch.randn(3, algebra.dim)

    invariant, transferred = sandwich_transfer(
        algebra,
        torch.empty_like(u),
        source,
        target,
        invariant_override=u,
    )
    r_tgt_n = target._normalized_R()
    manual = algebra.geometric_product(r_tgt_n.expand(*u.shape), u)
    manual = algebra.geometric_product(manual, target.get_inverse().expand(*u.shape))

    torch.testing.assert_close(invariant, u, rtol=0, atol=0)
    torch.testing.assert_close(transferred, manual, rtol=1e-5, atol=1e-6)
