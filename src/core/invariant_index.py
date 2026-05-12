from typing import List, Optional

import torch
import torch.nn.functional as F

from .types import AnalogyMatch, InvariantRecord


class InvariantIndex:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.records: dict[str, InvariantRecord] = {}
        self._keys: list[str] = []
        self._vectors: Optional[torch.Tensor] = None
        self._dirty = False

    def add(self, key: str, invariant: torch.Tensor, metadata: Optional[dict] = None) -> None:
        """Store an invariant vector and mark the search cache dirty.

        Args:
            key: Unique record key for replacement or insertion.
            invariant: Tensor with shape (dim,) copied onto the index device.
            metadata: Optional metadata stored with returned matches.
        """
        invariant = invariant.to(self.device).detach().clone()
        self.records[key] = InvariantRecord(
            key=key,
            invariant=invariant,
            metadata=metadata or {},
        )
        self._dirty = True

    def _rebuild(self) -> None:
        self._keys = list(self.records.keys())
        if not self._keys:
            self._vectors = None
            self._dirty = False
            return

        vectors = torch.stack([self.records[key].invariant for key in self._keys])
        self._vectors = F.normalize(vectors, dim=-1)
        self._dirty = False

    def search(self, query: torch.Tensor, top_k: int = 5) -> List[List[AnalogyMatch]]:
        if query.dim() == 1:
            query = query.unsqueeze(0)

        batch_size = query.shape[0]
        if not self.records or top_k <= 0:
            return [[] for _ in range(batch_size)]

        if self._dirty or self._vectors is None:
            self._rebuild()

        if self._vectors is None:
            return [[] for _ in range(batch_size)]

        query = F.normalize(query.to(self.device), dim=-1)
        scores = F.cosine_similarity(query[:, None, :], self._vectors[None, :, :], dim=-1)
        k = min(top_k, len(self._keys), scores.size(-1))
        if k <= 0:
            return [[] for _ in range(batch_size)]
        top_scores, top_indices = torch.topk(scores, k=k, dim=-1)

        results: List[List[AnalogyMatch]] = []
        for batch_scores, batch_indices in zip(top_scores, top_indices):
            matches = []
            for score, index in zip(batch_scores, batch_indices):
                key = self._keys[int(index.item())]
                record = self.records[key]
                matches.append(
                    AnalogyMatch(
                        key=key,
                        score=float(score.item()),
                        invariant=record.invariant,
                        metadata=record.metadata,
                    )
                )
            results.append(matches)
        return results

    def clear(self) -> None:
        self.records.clear()
        self._keys = []
        self._vectors = None
        self._dirty = False

    def __len__(self) -> int:
        return len(self.records)
