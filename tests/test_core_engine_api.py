import pytest
import torch

from src.core.engine import CoreEngineConfig, HDIMCoreEngine
from src.core.invariant_index import InvariantIndex


def _engine(dropout: float = 0.0) -> HDIMCoreEngine:
    engine = HDIMCoreEngine(CoreEngineConfig(dropout=dropout))
    engine.eval()
    return engine


def test_encode_tensor():
    engine = _engine()
    x = torch.randn(2, 64)

    encoded = engine.encode(x)

    assert encoded.shape == (2, engine.algebra.dim)


def test_encode_string_raises_without_adapter():
    engine = _engine()

    with pytest.raises(TypeError, match="String input requires a text adapter"):
        engine.encode("problem statement")


def test_extract_identity_rotor():
    engine = _engine()
    g = torch.randn(2, engine.algebra.dim)

    invariant = engine.extract(g, "source")

    torch.testing.assert_close(invariant, g, rtol=1e-5, atol=1e-6)


def test_match_empty_index_returns_empty():
    engine = _engine()
    u_inv = torch.randn(2, engine.algebra.dim)

    matches = engine.match(u_inv)

    assert matches == [[], []]


def test_transfer_roundtrip():
    engine = HDIMCoreEngine(CoreEngineConfig(clifford_q=0, dropout=0.0))
    engine.eval()
    rotor = engine.domain_rotors["source"]
    with torch.no_grad():
        rotor.R.copy_(torch.linspace(0.05, 0.15, steps=rotor.R.numel()))
    g = torch.randn(3, engine.algebra.dim)

    invariant = engine.extract(g, "source")
    transferred = engine.transfer(invariant, "source")

    torch.testing.assert_close(transferred, g, rtol=1e-4, atol=1e-5)


def test_full_pipeline_encode_extract_match_transfer():
    engine = _engine()
    expert_base = InvariantIndex()
    problem = torch.randn(2, 64)

    encoded = engine.encode(problem)
    invariant = engine.extract(encoded, "source")
    expert_base.add("case_0", invariant[0], {"domain": "source"})
    matches = engine.match(invariant, expert_base=expert_base)
    transferred = engine.transfer(invariant, "target")

    assert transferred.shape == encoded.shape
    assert len(matches) == 2
    assert matches[0][0].key == "case_0"
    assert matches[0][0].score == pytest.approx(1.0)
