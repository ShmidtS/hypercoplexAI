import pytest
import torch

from src.core.invariant_index import InvariantIndex


def test_empty_index_returns_empty():
    index = InvariantIndex()
    query = torch.randn(2, 4)

    matches = index.search(query)

    assert matches == [[], []]


def test_exact_match_ranks_first():
    index = InvariantIndex()
    vector = torch.tensor([1.0, 0.0, 0.0])
    index.add("match", vector, {"domain": "test"})

    matches = index.search(vector, top_k=1)

    assert len(matches) == 1
    assert len(matches[0]) == 1
    assert matches[0][0].key == "match"
    assert matches[0][0].score == pytest.approx(1.0)
    assert torch.equal(matches[0][0].invariant, vector)
    assert matches[0][0].metadata == {"domain": "test"}


def test_batched_query():
    index = InvariantIndex()
    index.add("x", torch.tensor([1.0, 0.0]))
    index.add("y", torch.tensor([0.0, 1.0]))

    query = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    matches = index.search(query, top_k=1)

    assert len(matches) == 3
    assert all(len(batch_matches) == 1 for batch_matches in matches)
    assert matches[0][0].key == "x"
    assert matches[1][0].key == "y"


def test_clear_removes_all():
    index = InvariantIndex()
    index.add("match", torch.tensor([1.0, 0.0]))

    index.clear()
    matches = index.search(torch.tensor([[1.0, 0.0]]))

    assert len(index) == 0
    assert matches == [[]]


def test_top_k_limits_results():
    index = InvariantIndex()
    for i in range(10):
        index.add(f"record_{i}", torch.tensor([float(i + 1), 1.0]))

    matches = index.search(torch.tensor([[1.0, 0.0]]), top_k=3)

    assert len(matches) == 1
    assert len(matches[0]) == 3
