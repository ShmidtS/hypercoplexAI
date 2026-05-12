import pytest
import torch

from src.training.dataset import (
    DomainProblemDataset,
    create_demo_dataset,
    create_group_aware_split,
    create_paired_demo_dataset,
)


def test_dataset():
    ds = create_demo_dataset()
    assert len(ds) == 100
    sample = ds[0]
    assert "encoding" in sample
    assert "domain_id" in sample
    assert sample["encoding"].shape == (64,)
    assert sample["domain_id"].dtype == torch.long


def test_paired_dataset():
    ds = create_paired_demo_dataset(n_samples=40)
    sample = ds[0]
    assert "pair_encoding" in sample
    assert "pair_domain_id" in sample
    assert "pair_group_id" in sample
    assert sample["pair_encoding"].shape == (64,)
    assert sample["pair_domain_id"].dtype == torch.long
    assert sample["pair_group_id"].dtype == torch.long
    assert sample["pair_family_id"].dtype == torch.long
    assert sample["pair_relation_type"] == "positive"
    assert sample["pair_relation_label"].dtype == torch.float32
    assert sample["pair_relation_label"].item() == 1.0
    assert sample["pair_weight"].dtype == torch.float32
    assert sample["pair_weight"].item() > 0.0
    assert sample["pair_domain_id"].item() != sample["domain_id"].item()


def test_dataset_rejects_same_domain_pairs():
    samples = [("a", 0), ("b", 0)]
    with pytest.raises(ValueError):
        DomainProblemDataset(
            samples,
            pair_indices=[1, 0],
            pair_group_ids=[0, 0],
            pair_relation_types=["positive", "positive"],
            pair_weights=[1.0, 1.0],
        )


def test_dataset_rejects_misaligned_pair_groups():
    samples = [("a", 0), ("b", 1), ("c", 2)]
    with pytest.raises(ValueError):
        DomainProblemDataset(
            samples,
            pair_indices=[1, 0, 1],
            pair_group_ids=[0, 1, 2],
            pair_relation_types=["positive", "positive", "positive"],
            pair_weights=[1.0, 1.0, 1.0],
        )


def test_dataset_rejects_non_positive_pair_weights():
    samples = [("a", 0), ("b", 1)]
    with pytest.raises(ValueError):
        DomainProblemDataset(
            samples,
            pair_indices=[1, 0],
            pair_group_ids=[0, 0],
            pair_relation_types=["positive", "positive"],
            pair_weights=[0.0, 1.0],
        )


def test_dataset_rejects_mismatched_pair_relation_types():
    samples = [("a", 0), ("b", 1)]
    with pytest.raises(ValueError):
        DomainProblemDataset(
            samples,
            pair_indices=[1, 0],
            pair_group_ids=[0, 0],
            pair_relation_types=["positive", "negative"],
            pair_weights=[1.0, 1.0],
        )


def test_dataset_exposes_negative_pair_metadata():
    ds = create_paired_demo_dataset(n_samples=40, negative_ratio=1.0)
    sample = next(item for item in (ds[idx] for idx in range(len(ds))) if item["pair_relation_type"] == "negative")
    assert sample["pair_relation_label"].item() == 0.0
    assert sample["pair_family_id"].item() == sample["pair_group_id"].item()
    assert sample["pair_domain_id"].item() != sample["domain_id"].item()


def test_group_aware_split_keeps_pair_groups_separate():
    ds = create_paired_demo_dataset(n_samples=40, negative_ratio=0.0)
    train_ds, val_ds = create_group_aware_split(ds, train_fraction=0.8, seed=42)
    pair_group_ids = ds.pair_group_ids
    assert pair_group_ids is not None
    train_group_ids = {pair_group_ids[idx] for idx in train_ds.indices}
    val_group_ids = {pair_group_ids[idx] for idx in val_ds.indices}
    assert train_group_ids
    assert val_group_ids
    assert train_group_ids.isdisjoint(val_group_ids)
