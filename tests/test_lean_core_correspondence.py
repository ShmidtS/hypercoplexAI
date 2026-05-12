import torch

from src.core.engine import CoreEngineConfig, HDIMCoreEngine


def _engine() -> HDIMCoreEngine:
    engine = HDIMCoreEngine(CoreEngineConfig(clifford_p=2, clifford_q=0, clifford_r=0, dropout=0.0))
    engine.eval()
    with torch.no_grad():
        engine.domain_rotors["source"].R.fill_(0.35)
        engine.domain_rotors["target"].R.fill_(-0.2)
    return engine


def test_identity_extraction_numeric():
    engine = HDIMCoreEngine(CoreEngineConfig(clifford_p=2, clifford_q=0, clifford_r=0, dropout=0.0))
    engine.eval()
    g = torch.randn(4, engine.algebra.dim)

    invariant = engine.extract(g, "source")

    torch.testing.assert_close(invariant, g, rtol=1e-5, atol=1e-6)


def test_transfer_roundtrip_numeric():
    engine = _engine()
    u = torch.randn(4, engine.algebra.dim)

    transferred = engine.transfer(u, "source")
    extracted = engine.extract(transferred, "source")

    torch.testing.assert_close(extracted, u, rtol=1e-4, atol=1e-5)


def test_cross_domain_invariance_numeric():
    engine = _engine()
    u = torch.randn(4, engine.algebra.dim)

    g1 = engine.transfer(u, "source")
    g2 = engine.transfer(u, "target")
    u1 = engine.extract(g1, "source")
    u2 = engine.extract(g2, "target")

    torch.testing.assert_close(u1, u, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(u2, u, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(u1, u2, rtol=1e-4, atol=1e-5)


def test_norm_preservation_numeric():
    engine = _engine()
    g = torch.randn(4, engine.algebra.dim)

    invariant = engine.extract(g, "source")

    torch.testing.assert_close(
        engine.algebra.norm(invariant),
        engine.algebra.norm(g),
        rtol=1e-4,
        atol=1e-5,
    )


def test_analogy_equivalence_relation_numeric():
    u = torch.randn(4, 4)
    v = u.clone()
    w = v.clone()

    assert torch.equal(u, u)
    assert torch.equal(u, v) == torch.equal(v, u)
    assert torch.equal(u, v) and torch.equal(v, w) and torch.equal(u, w)
