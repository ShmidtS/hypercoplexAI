"""DomainProblemDataset and demo dataset factories for HDIM training."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Simple character-level tokeniser / embedder
# ---------------------------------------------------------------------------

def _text_to_embedding(text: str, dim: int, seed: Optional[int] = None) -> torch.Tensor:
    """Convert a text string to a fixed-size float embedding.

    Uses a deterministic hash-based projection so the same text always
    produces the same vector. This is intentionally minimal — suitable
    for integration tests and demos without external dependencies.
    """
    assert dim % 4 == 0, f"dim must be divisible by 4, got {dim}"

    char_vec = torch.zeros(256)
    for ch in text:
        char_vec[ord(ch) % 256] += 1.0
    if char_vec.sum() > 0:
        char_vec = char_vec / char_vec.sum()

    rng = torch.Generator()
    stable_seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rng.manual_seed(stable_seed if seed is None else seed)
    proj = torch.randn(dim, 256, generator=rng)
    embedding = proj @ char_vec

    norm = embedding.norm()
    if norm > 0:
        embedding = embedding / norm

    return embedding


def texts_to_tensor(
    texts: Sequence[str],
    dim: int,
    *,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Encode raw texts into a batch tensor for the minimal text-mode scaffold."""
    if len(texts) == 0:
        return torch.empty(0, dim)
    return torch.stack([_text_to_embedding(text, dim, seed=seed) for text in texts], dim=0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DomainProblemDataset(Dataset):
    """Dataset of domain-labelled problem samples for HDIM training.

    Optionally exposes cross-domain pair metadata for pair-supervised
    isomorphism training, including explicit relation types, group ids,
    and per-pair weights.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        embed_dim: int = 64,
        pair_indices: Optional[List[int]] = None,
        pair_group_ids: Optional[List[int]] = None,
        pair_relation_types: Optional[List[str]] = None,
        pair_weights: Optional[List[float]] = None,
    ) -> None:
        assert embed_dim % 4 == 0, f"embed_dim must be divisible by 4, got {embed_dim}"
        self.samples = samples
        self.embed_dim = embed_dim
        self.pair_indices = pair_indices
        self.pair_group_ids = pair_group_ids
        self.pair_relation_types = pair_relation_types
        self.pair_weights = pair_weights
        self._encodings: List[torch.Tensor] = [
            _text_to_embedding(text, embed_dim) for text, _ in samples
        ]
        self._labels: List[int] = [label for _, label in samples]

        if self.pair_indices is not None:
            if len(self.pair_indices) != len(self.samples):
                raise ValueError("pair_indices must have the same length as samples")
            if self.pair_group_ids is None:
                raise ValueError("pair_group_ids must be provided with pair_indices")
            if len(self.pair_group_ids) != len(self.samples):
                raise ValueError("pair_group_ids must have the same length as samples")
            if self.pair_relation_types is None:
                raise ValueError("pair_relation_types must be provided with pair_indices")
            if len(self.pair_relation_types) != len(self.samples):
                raise ValueError("pair_relation_types must have the same length as samples")
            if self.pair_weights is None:
                raise ValueError("pair_weights must be provided with pair_indices")
            if len(self.pair_weights) != len(self.samples):
                raise ValueError("pair_weights must have the same length as samples")
            for idx, pair_idx in enumerate(self.pair_indices):
                if pair_idx < 0 or pair_idx >= len(self.samples):
                    raise IndexError("pair_indices contains an out-of-range sample index")
                if pair_idx == idx:
                    raise ValueError("pair_indices must reference a different sample")
                if self._labels[pair_idx] == self._labels[idx]:
                    raise ValueError("pair_indices must point to a sample from a different domain")
                relation_type = self.pair_relation_types[idx]
                if relation_type not in {"positive", "negative"}:
                    raise ValueError("pair_relation_types must be either 'positive' or 'negative'")
                if self.pair_weights[idx] <= 0:
                    raise ValueError("pair_weights must be positive for all paired samples")
                paired_relation_type = self.pair_relation_types[pair_idx]
                if paired_relation_type not in {"positive", "negative"}:
                    raise ValueError("pair_relation_types must be either 'positive' or 'negative'")
                if self.pair_weights[pair_idx] <= 0:
                    raise ValueError("pair_weights must be positive for all paired samples")
                is_reciprocal_pair = self.pair_indices[pair_idx] == idx
                if relation_type == "positive":
                    if is_reciprocal_pair and paired_relation_type != relation_type:
                        raise ValueError("pair_relation_types must match across paired samples")
                    if is_reciprocal_pair and self.pair_group_ids[pair_idx] != self.pair_group_ids[idx]:
                        raise ValueError("positive pair_indices must stay within the same cross-domain pair group")
                else:
                    if is_reciprocal_pair and self.pair_group_ids[pair_idx] == self.pair_group_ids[idx]:
                        raise ValueError("negative pairs must not reuse the same pair group")
                    if is_reciprocal_pair and paired_relation_type != relation_type:
                        raise ValueError("pair_relation_types must match across paired samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "text": self.samples[idx][0],
            "encoding": self._encodings[idx].float(),
            "domain_id": torch.tensor(self._labels[idx], dtype=torch.long),
        }

        if self.pair_indices is not None:
            pair_idx = self.pair_indices[idx]
            pair_relation_type = self.pair_relation_types[idx]
            item["pair_text"] = self.samples[pair_idx][0]
            item["pair_encoding"] = self._encodings[pair_idx].float()
            item["pair_domain_id"] = torch.tensor(self._labels[pair_idx], dtype=torch.long)
            item["pair_index"] = torch.tensor(pair_idx, dtype=torch.long)
            item["pair_group_id"] = torch.tensor(self.pair_group_ids[idx], dtype=torch.long)
            item["pair_family_id"] = torch.tensor(self.pair_group_ids[idx], dtype=torch.long)
            item["pair_relation_type"] = pair_relation_type
            item["pair_relation_label"] = torch.tensor(
                1.0 if pair_relation_type == "positive" else 0.0,
                dtype=torch.float32,
            )
            item["pair_weight"] = torch.tensor(self.pair_weights[idx], dtype=torch.float32)

        return item


# ---------------------------------------------------------------------------
# Demo factory
# ---------------------------------------------------------------------------

_DOMAIN_TEMPLATES: Dict[int, List[str]] = {
    0: [
        "How to reduce cavitation damage in pump impellers?",
        "Minimise thermal stress in turbine blades under cyclic loading.",
        "Prevent fatigue crack propagation in aluminium alloy frames.",
        "Improve heat transfer in compact heat exchangers.",
        "Reduce vibration amplitude in rotating shafts at resonance.",
    ],
    1: [
        "How do plant vacuoles prevent membrane rupture during osmotic shock?",
        "Mechanism of DNA repair after double-strand breaks in mammalian cells.",
        "How do tardigrades survive extreme desiccation conditions?",
        "Role of microtubules in maintaining cell shape under compression.",
        "How does the immune system distinguish self from non-self antigens?",
    ],
    2: [
        "Prove convergence of gradient descent for strongly convex functions.",
        "Characterise eigenvalue distributions of random Hermitian matrices.",
        "Construct a bijection between the reals and the power set of naturals.",
        "Find the minimal surface spanning a given closed boundary curve.",
        "Analyse stability of fixed points in a nonlinear ODE system.",
    ],
    3: [
        "Explain energy quantisation in a particle confined to a box.",
        "Describe the mechanism of superconductivity in BCS theory.",
        "How does Hawking radiation emerge from the event horizon?",
        "Model electromagnetic wave propagation in anisotropic media.",
        "Analyse decoherence in open quantum systems coupled to a thermal bath.",
    ],
}


def _generate_demo_samples(
    n_samples: int,
    num_domains: int,
) -> Tuple[List[Tuple[str, int]], List[int]]:
    samples: List[Tuple[str, int]] = []
    variant_ids: List[int] = []
    per_domain = n_samples // num_domains
    remainder = n_samples % num_domains

    for domain_id in range(num_domains):
        templates = _DOMAIN_TEMPLATES[domain_id]
        count = per_domain + (1 if domain_id < remainder else 0)
        for i in range(count):
            variant_id = i % len(templates)
            base = templates[variant_id]
            text = f"{base} [variant {i // len(templates) + 1}]"
            samples.append((text, domain_id))
            variant_ids.append(variant_id)

    return samples, variant_ids


def create_demo_dataset(
    n_samples: int = 100,
    num_domains: int = 4,
    embed_dim: int = 64,
    seed: int = 42,
) -> DomainProblemDataset:
    """Create a synthetic demo dataset for baseline HDIM training."""
    assert num_domains <= 4, "Demo dataset supports at most 4 domains."
    random.seed(seed)

    samples, _ = _generate_demo_samples(n_samples, num_domains)
    random.shuffle(samples)
    return DomainProblemDataset(samples, embed_dim=embed_dim)


def create_paired_demo_dataset(
    n_samples: int = 100,
    num_domains: int = 4,
    embed_dim: int = 64,
    seed: int = 42,
    negative_ratio: float = 0.0,
) -> DomainProblemDataset:
    """Create a synthetic demo dataset with explicit positive and negative cross-domain pairs."""
    assert num_domains >= 2, "Paired demo dataset requires at least 2 domains."
    assert num_domains <= 4, "Demo dataset supports at most 4 domains."
    random.seed(seed)

    samples, variant_ids = _generate_demo_samples(n_samples, num_domains)

    variant_to_indices: Dict[int, List[int]] = {}
    for idx, variant_id in enumerate(variant_ids):
        variant_to_indices.setdefault(variant_id, []).append(idx)

    pair_indices: List[int] = []
    pair_relation_types: List[str] = []
    pair_group_ids: List[int] = []
    pair_weights: List[float] = []
    next_negative_group_id = max(variant_ids, default=-1) + 1

    for idx, (_, domain_id) in enumerate(samples):
        positive_candidates = [
            candidate_idx
            for candidate_idx in variant_to_indices[variant_ids[idx]]
            if candidate_idx != idx and samples[candidate_idx][1] != domain_id
        ]
        if not positive_candidates:
            raise ValueError("Could not construct cross-domain pair for sample")

        should_use_negative = negative_ratio > 0.0 and random.random() < negative_ratio
        if should_use_negative:
            negative_candidates = [
                candidate_idx
                for candidate_idx, (_, candidate_domain_id) in enumerate(samples)
                if candidate_idx != idx
                and candidate_domain_id != domain_id
                and variant_ids[candidate_idx] != variant_ids[idx]
            ]
            if negative_candidates:
                pair_idx = random.choice(negative_candidates)
                pair_indices.append(pair_idx)
                pair_relation_types.append("negative")
                pair_group_ids.append(next_negative_group_id)
                pair_weights.append(1.0)
                next_negative_group_id += 1
                continue

        pair_idx = random.choice(positive_candidates)
        pair_indices.append(pair_idx)
        pair_relation_types.append("positive")
        pair_group_ids.append(variant_ids[idx])
        pair_weights.append(1.0)

    order = list(range(len(samples)))
    random.shuffle(order)
    shuffled_samples = [samples[i] for i in order]
    inverse_order = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
    shuffled_pairs = [inverse_order[pair_indices[old_idx]] for old_idx in order]
    shuffled_pair_groups = [pair_group_ids[old_idx] for old_idx in order]
    shuffled_pair_relations = [pair_relation_types[old_idx] for old_idx in order]
    shuffled_pair_weights = [pair_weights[old_idx] for old_idx in order]

    return DomainProblemDataset(
        shuffled_samples,
        embed_dim=embed_dim,
        pair_indices=shuffled_pairs,
        pair_group_ids=shuffled_pair_groups,
        pair_relation_types=shuffled_pair_relations,
        pair_weights=shuffled_pair_weights,
    )


def create_group_aware_split(
    dataset: DomainProblemDataset,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """Split dataset by pair groups/families to avoid template leakage across splits."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    if dataset.pair_group_ids is None:
        generator = torch.Generator().manual_seed(seed)
        train_size = int(train_fraction * len(dataset))
        val_size = len(dataset) - train_size
        return torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    groups: Dict[int, List[int]] = defaultdict(list)
    for idx, group_id in enumerate(dataset.pair_group_ids):
        groups[group_id].append(idx)

    group_ids = list(groups.keys())
    random.Random(seed).shuffle(group_ids)
    target_train_size = int(round(train_fraction * len(dataset)))
    train_indices: List[int] = []
    val_indices: List[int] = []

    for position, group_id in enumerate(group_ids):
        group_indices = groups[group_id]
        remaining_groups = len(group_ids) - position - 1
        train_needed = target_train_size - len(train_indices)

        if not train_indices:
            destination = train_indices
        elif not val_indices and remaining_groups == 0:
            destination = val_indices
        elif len(train_indices) >= target_train_size:
            destination = val_indices
        elif train_needed <= 0:
            destination = val_indices
        elif not val_indices and len(group_indices) > train_needed and remaining_groups > 0:
            destination = val_indices
        else:
            destination = train_indices

        destination.extend(group_indices)

    if not train_indices or not val_indices:
        raise ValueError("group-aware split requires both train and validation groups")

    return (
        torch.utils.data.Subset(dataset, train_indices),
        torch.utils.data.Subset(dataset, val_indices),
    )
