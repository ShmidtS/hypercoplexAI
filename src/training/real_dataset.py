"""Real cross-domain pairs dataset for HDIM training.

Загружает JSON-файл с размеченными позитивными/негативными парами
из разных доменов для обучения HDIM на реальных данных.

Формат JSON:
[
  {
    "source_text": str,
    "source_domain": int,
    "target_text": str,
    "target_domain": int,
    "relation": "positive" | "negative",
    "group_id": int,
    "family": str
  },
  ...
]
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class RealPairsDataset(Dataset):
    """Dataset of real cross-domain analogy pairs.

    Каждый элемент содержит:
    - text: исходный текст
    - pair_text: парный текст
    - domain_id: домен источника
    - pair_domain_id: домен цели
    - pair_relation_label: 1.0 (positive) / 0.0 (negative)
    - pair_group_id: id группы для leakage-aware split
    - pair_weight: вес пары

    Encoder (SBERT/AdvancedTextEncoder) вызывается в trainer,
    здесь хранятся только raw texts.
    """

    def __init__(
        self,
        pairs: List[Dict],
        *,
        augment_factor: int = 1,
        seed: int = 42,
        add_negatives: bool = True,
        negative_ratio: float = 1.0,
    ) -> None:
        self.pairs = pairs
        self.augment_factor = augment_factor
        self.seed = seed
        self.add_negatives = add_negatives
        self.negative_ratio = negative_ratio
        self._items = self._build_items(seed)

    def _build_items(self, seed: int) -> List[Dict]:
        """Build flat list of items from pairs.

        Каждая пара порождает:
        - source → target (forward, positive)
        - target → source (backward, positive)
        - synthetic negatives: src из группы A paired с tgt из группы B (cross-group)
        """
        rng = random.Random(seed)
        items = []
        for pair in self.pairs:
            item_fwd = {
                "text": pair["source_text"],
                "pair_text": pair["target_text"],
                "domain_id": pair["source_domain"],
                "pair_domain_id": pair["target_domain"],
                "relation": pair["relation"],
                "group_id": pair["group_id"],
            }
            items.append(item_fwd)

            # Backward direction for positive pairs
            if pair["relation"] == "positive":
                item_bwd = {
                    "text": pair["target_text"],
                    "pair_text": pair["source_text"],
                    "domain_id": pair["target_domain"],
                    "pair_domain_id": pair["source_domain"],
                    "relation": "positive",
                    "group_id": pair["group_id"],
                }
                items.append(item_bwd)

        # Generate hard synthetic negatives: cross-group pairs
        if self.add_negatives and len(self.pairs) > 1:
            pos_pairs = [p for p in self.pairs if p["relation"] == "positive"]
            n_neg = int(len(pos_pairs) * self.negative_ratio)
            shuffled_pairs = list(pos_pairs)
            rng.shuffle(shuffled_pairs)
            neg_targets = list(pos_pairs)
            rng.shuffle(neg_targets)
            added = 0
            for src_p, tgt_p in zip(shuffled_pairs, neg_targets):
                if added >= n_neg:
                    break
                # Ensure different group to create a true negative
                if src_p["group_id"] == tgt_p["group_id"]:
                    continue
                item_neg = {
                    "text": src_p["source_text"],
                    "pair_text": tgt_p["target_text"],
                    "domain_id": src_p["source_domain"],
                    "pair_domain_id": tgt_p["target_domain"],
                    "relation": "negative",
                    "group_id": src_p["group_id"],
                }
                items.append(item_neg)
                added += 1

        # Augmentation: repeat with shuffled order
        base_items = list(items)
        for _ in range(self.augment_factor - 1):
            shuffled = list(base_items)
            rng.shuffle(shuffled)
            items.extend(shuffled)

        rng.shuffle(items)
        return items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        item = self._items[idx]
        label = 1.0 if item["relation"] == "positive" else 0.0
        return {
            "text": item["text"],
            "pair_text": item["pair_text"],
            "domain_id": torch.tensor(item["domain_id"], dtype=torch.long),
            "pair_domain_id": torch.tensor(item["pair_domain_id"], dtype=torch.long),
            "pair_relation_label": torch.tensor(label, dtype=torch.float32),
            "pair_group_id": torch.tensor(item["group_id"], dtype=torch.long),
            "pair_family_id": torch.tensor(item["group_id"], dtype=torch.long),
            "pair_weight": torch.tensor(1.0, dtype=torch.float32),
        }


def load_real_pairs_dataset(
    json_path: str | Path,
    *,
    augment_factor: int = 5,
    seed: int = 42,
    add_negatives: bool = True,
    negative_ratio: float = 1.0,
) -> RealPairsDataset:
    """Load real pairs from JSON file.

    Args:
        add_negatives: генерировать синтетические негативные пары (cross-group)
        negative_ratio: отношение negatives к positives (1.0 = 1:1)
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Real pairs JSON not found: {path}")
    pairs = json.loads(path.read_text(encoding="utf-8"))
    return RealPairsDataset(
        pairs,
        augment_factor=augment_factor,
        seed=seed,
        add_negatives=add_negatives,
        negative_ratio=negative_ratio,
    )


def split_real_pairs(
    dataset: RealPairsDataset,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """Split by group_id to avoid leakage."""
    from collections import defaultdict

    groups: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        item = dataset._items[idx]
        groups[item["group_id"]].append(idx)

    group_ids = list(groups.keys())
    random.Random(seed).shuffle(group_ids)

    target_train = int(round(train_fraction * len(dataset)))
    train_indices: List[int] = []
    val_indices: List[int] = []

    for gid in group_ids:
        if len(train_indices) < target_train:
            train_indices.extend(groups[gid])
        else:
            val_indices.extend(groups[gid])

    if not val_indices:
        # Ensure at least one group in val
        last_group = group_ids[-1]
        moved = groups[last_group]
        train_indices = [i for i in train_indices if i not in moved]
        val_indices = moved

    return (
        torch.utils.data.Subset(dataset, train_indices),
        torch.utils.data.Subset(dataset, val_indices),
    )
