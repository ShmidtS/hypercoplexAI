from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.core.algebra import CliffordAlgebra
from src.core.domain_encoder import DomainEncoder
from src.models.sbert_encoder import SBERTEncoder
from src.models.text_encoder_protocol import TextEncoder
from src.training.invariant_losses import (
    compute_infonce_loss,
    compute_iso_loss,
    compute_pair_iso_loss,
)
from src.training.real_dataset import RealPairsDataset, split_real_pairs
from src.training.results_logger import (
    append_ledger_row,
    latest_ledger_row,
    read_jsonl,
    write_json,
)


def test_domain_encoder_forward_and_encode_domain_shapes_match() -> None:
    torch.manual_seed(0)
    algebra = CliffordAlgebra(p=2, q=0, r=0)
    encoder = DomainEncoder(
        input_dim=3,
        clifford_dim=algebra.dim,
        algebra=algebra,
        domain_names=["source", "target"],
    )
    x = torch.randn(4, 3)

    forward_g, forward_inv = encoder(x, "source")
    encoded_g, encoded_inv = encoder.encode_domain(x, "source")

    assert forward_g.shape == (4, algebra.dim)
    assert forward_inv.shape == (4, algebra.dim)
    assert torch.allclose(forward_g, encoded_g)
    assert torch.allclose(forward_inv, encoded_inv)


def test_domain_encoder_add_domain_registers_normalized_rotor_dtype_device() -> None:
    algebra = CliffordAlgebra(p=2, q=0, r=0)
    encoder = DomainEncoder(
        input_dim=2,
        clifford_dim=algebra.dim,
        algebra=algebra,
        domain_names=["source"],
    ).to(dtype=torch.float64)

    encoder.add_domain("new_domain")

    rotor = encoder.get_rotor("new_domain")
    assert "new_domain" in encoder.domain_rotors
    assert rotor.R.dtype == torch.float64
    assert rotor.R.device == next(encoder.parameters()).device
    normalized_rotor = rotor._normalized_R()
    assert normalized_rotor.dtype == torch.float64
    assert torch.allclose(normalized_rotor.norm(dim=-1), torch.ones((), dtype=torch.float64))


def test_domain_encoder_unknown_domain_raises_key_error() -> None:
    algebra = CliffordAlgebra(p=2, q=0, r=0)
    encoder = DomainEncoder(input_dim=2, clifford_dim=algebra.dim, algebra=algebra)

    with pytest.raises(KeyError):
        encoder.encode_domain(torch.zeros(1, 2), "missing")


def _real_pairs() -> list[dict[str, object]]:
    return [
        {
            "source_text": "alpha source",
            "target_text": "alpha target",
            "source_domain": 0,
            "target_domain": 1,
            "relation": "positive",
            "group_id": 10,
        },
        {
            "source_text": "beta source",
            "target_text": "beta target",
            "source_domain": 0,
            "target_domain": 1,
            "relation": "positive",
            "group_id": 20,
        },
        {
            "source_text": "gamma source",
            "target_text": "gamma target",
            "source_domain": 1,
            "target_domain": 0,
            "relation": "positive",
            "group_id": 30,
        },
    ]


def test_real_pairs_dataset_emits_forward_backward_and_negative_items() -> None:
    dataset = RealPairsDataset(
        _real_pairs(),
        augment_factor=1,
        seed=0,
        add_negatives=True,
        negative_ratio=1.0,
    )
    items = [dataset[i] for i in range(len(dataset))]
    labels = [float(item["pair_relation_label"]) for item in items]

    assert labels.count(1.0) == 6
    assert labels.count(0.0) >= 1
    assert any(item["text"] == "alpha source" and item["pair_text"] == "alpha target" for item in items)
    assert any(item["text"] == "alpha target" and item["pair_text"] == "alpha source" for item in items)
    assert all(item["domain_id"].dtype == torch.long for item in items)
    assert all(item["pair_domain_id"].dtype == torch.long for item in items)
    assert all(item["pair_relation_label"].dtype == torch.float32 for item in items)
    assert all(item["pair_weight"].dtype == torch.float32 for item in items)


def test_real_pairs_split_keeps_group_ids_disjoint() -> None:
    dataset = RealPairsDataset(
        _real_pairs(),
        augment_factor=1,
        seed=1,
        add_negatives=True,
        negative_ratio=1.0,
    )
    train_subset, val_subset = split_real_pairs(dataset, train_fraction=0.5, seed=3)

    train_groups = {int(dataset[idx]["pair_group_id"]) for idx in train_subset.indices}
    val_groups = {int(dataset[idx]["pair_group_id"]) for idx in val_subset.indices}

    assert train_groups
    assert val_groups
    assert train_groups.isdisjoint(val_groups)


def test_results_logger_appends_jsonl_and_finds_latest_run(tmp_path: Path) -> None:
    ledger = tmp_path / "nested" / "ledger.jsonl"

    append_ledger_row(ledger, {"run_id": "a", "value": 1})
    append_ledger_row(ledger, {"run_id": "b", "value": 2})
    append_ledger_row(ledger, {"run_id": "a", "value": 3})
    ledger.write_text(ledger.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    rows = read_jsonl(ledger)
    assert len(rows) == 3
    assert latest_ledger_row(rows, run_id="a") == {"run_id": "a", "value": 3}
    assert latest_ledger_row(rows, run_id="missing") is None


def test_results_logger_write_json_round_trips_utf8_payload(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "payload.json"
    payload = {"message": "привет", "values": [1, 2, 3]}

    returned = write_json(target, payload)

    assert returned == target
    assert json.loads(target.read_text(encoding="utf-8")) == payload


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class _FakeEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class _FakeAutoModel(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.encoder = types.SimpleNamespace(layer=nn.ModuleList([_FakeLayer() for _ in range(num_layers)]))
        self.embeddings = _FakeEmbeddings()


class _FakeTransformerModule:
    def __init__(self, auto_model: _FakeAutoModel) -> None:
        self.auto_model = auto_model


class _FakeSentenceTransformer(nn.Module):
    instances: list["_FakeSentenceTransformer"] = []

    def __init__(self, model_name: str, device: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.encode_calls = 0
        self.auto_model = _FakeAutoModel(num_layers=4)
        _FakeSentenceTransformer.instances.append(self)

    def get_sentence_embedding_dimension(self) -> int:
        return 6

    def _first_module(self) -> _FakeTransformerModule:
        return _FakeTransformerModule(self.auto_model)

    def encode(
        self,
        texts: list[str],
        *,
        convert_to_tensor: bool,
        device: str,
        show_progress_bar: bool,
        normalize_embeddings: bool,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        self.encode_calls += 1
        assert convert_to_tensor is True
        assert show_progress_bar is False
        rows = []
        for text in texts:
            base = float(len(text))
            rows.append(torch.arange(6, dtype=torch.float32) + base)
        embeddings = torch.stack(rows) if rows else torch.empty(0, 6)
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.to(device=device)


@pytest.fixture()
def fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSentenceTransformer]:
    _FakeSentenceTransformer.instances = []
    fake_module = types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return _FakeSentenceTransformer


def test_sbert_frozen_mode_keeps_model_out_of_child_modules_and_caches(
    fake_sentence_transformers: type[_FakeSentenceTransformer],
) -> None:
    encoder = SBERTEncoder(output_dim=3, freeze=True, dropout=0.0)
    fake_model = fake_sentence_transformers.instances[-1]

    first = encoder(["same", "other"])
    second = encoder(["same", "other"])

    assert first.shape == (2, 3)
    assert second.shape == (2, 3)
    assert fake_model.encode_calls == 1
    assert "_sbert" not in dict(encoder.named_children())
    assert set(encoder._embedding_cache) == {"same", "other"}


def test_sbert_empty_input_returns_empty_projected_shape(
    fake_sentence_transformers: type[_FakeSentenceTransformer],
) -> None:
    encoder = SBERTEncoder(output_dim=5, freeze=True, dropout=0.0)

    result = encoder([])

    assert result.shape == (0, 5)
    assert result.device.type == "cpu"


def test_sbert_freeze_bottom_frac_toggles_bottom_and_top_layers(
    fake_sentence_transformers: type[_FakeSentenceTransformer],
) -> None:
    encoder = SBERTEncoder(output_dim=3, freeze=True, freeze_bottom_frac=0.5, dropout=0.0)
    fake_model = fake_sentence_transformers.instances[-1]
    layers = fake_model.auto_model.encoder.layer

    assert "_sbert" in dict(encoder.named_children())
    assert [layer.weight.requires_grad for layer in layers] == [False, False, True, True]
    assert fake_model.auto_model.embeddings.weight.requires_grad is False


class _MinimalRuntimeEncoder:
    def forward(self, texts: list, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(len(texts), 2, device=device, dtype=dtype)

    def encode(self, texts: list) -> torch.Tensor:
        return torch.ones(len(texts), 2)


class _MissingEncode:
    def forward(self, texts: list, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(len(texts), 2, device=device, dtype=dtype)


def test_text_encoder_protocol_accepts_minimal_runtime_encoder() -> None:
    assert isinstance(_MinimalRuntimeEncoder(), TextEncoder)


def test_text_encoder_protocol_rejects_missing_encode() -> None:
    assert not isinstance(_MissingEncode(), TextEncoder)


def test_infonce_returns_zero_for_batch_size_one() -> None:
    source = torch.ones(1, 3)
    target = torch.ones(1, 3)
    loss = compute_infonce_loss(source, target, torch.ones(1), torch.ones(1))

    assert loss.shape == ()
    assert loss.item() == 0.0


def test_infonce_returns_zero_for_nan_inputs() -> None:
    source = torch.tensor([[float("nan"), 0.0], [1.0, 0.0]])
    target = torch.eye(2)
    loss = compute_infonce_loss(source, target, torch.ones(2), torch.ones(2))

    assert loss.item() == 0.0


def test_infonce_returns_zero_when_no_positive_labels() -> None:
    source = torch.eye(2)
    target = torch.eye(2)
    labels = torch.zeros(2)
    loss = compute_infonce_loss(source, target, labels, torch.ones(2))

    assert loss.item() == 0.0


def test_infonce_group_masking_keeps_finite_positive_loss() -> None:
    source = torch.eye(3)
    target = torch.eye(3)
    labels = torch.tensor([1.0, 1.0, 0.0])
    weights = torch.ones(3)
    group_ids = torch.tensor([7, 7, 9])

    loss = compute_infonce_loss(source, target, labels, weights, pair_group_id=group_ids)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_pair_iso_loss_falls_back_when_pair_metadata_missing() -> None:
    training = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.zeros_like(training)

    loss = compute_pair_iso_loss(training, target, batch={})

    assert torch.allclose(loss, compute_iso_loss(training, target))


def test_pair_iso_loss_negative_pairs_penalize_inside_margin() -> None:
    training = torch.zeros(2, 2)
    target = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
    batch = {
        "pair_relation_label": torch.tensor([0.0, 0.0]),
        "pair_weight": torch.tensor([1.0, 1.0]),
    }

    loss = compute_pair_iso_loss(training, target, batch=batch, negative_margin=1.0)

    assert torch.allclose(loss, torch.tensor(0.875))
