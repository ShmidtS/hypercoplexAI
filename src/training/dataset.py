"""DomainProblemDataset and demo dataset factories for HDIM training."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

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
    rng.manual_seed(abs(hash(text)) % (2**32) if seed is None else seed)
    proj = torch.randn(dim, 256, generator=rng)
    embedding = proj @ char_vec

    norm = embedding.norm()
    if norm > 0:
        embedding = embedding / norm

    return embedding


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DomainProblemDataset(Dataset):
    """Dataset of domain-labelled problem samples for HDIM training.

    Optionally exposes aligned cross-domain pairs for pair-supervised
    isomorphism training.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        embed_dim: int = 64,
        pair_indices: Optional[List[int]] = None,
    ) -> None:
        assert embed_dim % 4 == 0, f"embed_dim must be divisible by 4, got {embed_dim}"
        self.samples = samples
        self.embed_dim = embed_dim
        self.pair_indices = pair_indices
        self._encodings: List[torch.Tensor] = [
            _text_to_embedding(text, embed_dim) for text, _ in samples
        ]
        self._labels: List[int] = [label for _, label in samples]

        if self.pair_indices is not None:
            if len(self.pair_indices) != len(self.samples):
                raise ValueError("pair_indices must have the same length as samples")
            for idx, pair_idx in enumerate(self.pair_indices):
                if pair_idx < 0 or pair_idx >= len(self.samples):
                    raise IndexError("pair_indices contains an out-of-range sample index")
                if pair_idx == idx:
                    raise ValueError("pair_indices must reference a different sample")
                if self._labels[pair_idx] == self._labels[idx]:
                    raise ValueError("pair_indices must point to a sample from a different domain")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "encoding": self._encodings[idx].float(),
            "domain_id": torch.tensor(self._labels[idx], dtype=torch.long),
        }

        if self.pair_indices is not None:
            pair_idx = self.pair_indices[idx]
            item["pair_encoding"] = self._encodings[pair_idx].float()
            item["pair_domain_id"] = torch.tensor(self._labels[pair_idx], dtype=torch.long)

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
) -> DomainProblemDataset:
    """Create a synthetic demo dataset with aligned cross-domain pairs."""
    assert num_domains >= 2, "Paired demo dataset requires at least 2 domains."
    assert num_domains <= 4, "Demo dataset supports at most 4 domains."
    random.seed(seed)

    samples, variant_ids = _generate_demo_samples(n_samples, num_domains)

    variant_to_indices: Dict[int, List[int]] = {}
    for idx, variant_id in enumerate(variant_ids):
        variant_to_indices.setdefault(variant_id, []).append(idx)

    pair_indices: List[int] = []
    for idx, (_, domain_id) in enumerate(samples):
        candidates = [
            candidate_idx
            for candidate_idx in variant_to_indices[variant_ids[idx]]
            if candidate_idx != idx and samples[candidate_idx][1] != domain_id
        ]
        if not candidates:
            raise ValueError("Could not construct cross-domain pair for sample")
        pair_indices.append(random.choice(candidates))

    order = list(range(len(samples)))
    random.shuffle(order)
    shuffled_samples = [samples[i] for i in order]
    inverse_order = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
    shuffled_pairs = [inverse_order[pair_indices[old_idx]] for old_idx in order]

    return DomainProblemDataset(
        shuffled_samples,
        embed_dim=embed_dim,
        pair_indices=shuffled_pairs,
    )
