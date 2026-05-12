import pytest
import torch

from src.adapters.text import SimpleTextEncoder, TextAdapter
from src.core.engine import CoreEngineConfig, HDIMCoreEngine


def _adapter() -> TextAdapter:
    engine = HDIMCoreEngine(CoreEngineConfig(input_dim=16, dropout=0.0))
    encoder = SimpleTextEncoder(output_dim=16, max_length=16, dropout=0.0)
    adapter = TextAdapter(encoder, engine)
    adapter.encoder.eval()
    adapter.engine.eval()
    return adapter


def test_adapter_encode_texts_shape():
    adapter = _adapter()

    embeddings = adapter.encode_texts(["alpha", "beta", "gamma"])

    assert embeddings.shape == (3, adapter.engine.config.input_dim)


def test_adapter_extract_returns_invariant():
    adapter = _adapter()

    invariant = adapter.extract_texts(["alpha", "beta"], "source")

    assert invariant.shape == (2, adapter.engine.algebra.dim)


def test_adapter_match_returns_list():
    adapter = _adapter()

    matches = adapter.match_texts(["alpha", "beta"], "source")

    assert matches == [[], []]


def test_adapter_engine_integration():
    adapter = _adapter()
    texts = ["alpha", "beta"]

    encoded = adapter.encode(texts)
    invariant = adapter.extract_texts(texts, "source")
    adapter.engine.index.add("case_alpha", invariant[0], {"domain": "source"})
    matches = adapter.match_texts(texts, "source")

    assert encoded.shape == (2, adapter.engine.algebra.dim)
    assert invariant.shape == encoded.shape
    assert len(matches) == 2
    assert matches[0][0].key == "case_alpha"
    assert matches[0][0].score == pytest.approx(1.0)


def test_string_encode_raises_without_adapter():
    engine = HDIMCoreEngine(CoreEngineConfig(dropout=0.0))

    with pytest.raises(TypeError, match="String input requires a text adapter"):
        engine.encode("problem statement")
