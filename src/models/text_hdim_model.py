"""Minimal text-facing wrapper for HDIM retrieval and transfer workflows."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hdim_model import HDIMAuxState, HDIMModel
from src.training.dataset import texts_to_tensor


class TextHDIMModel(nn.Module):
    """Thin text-entry wrapper around HDIMModel.

    This scaffold intentionally keeps text handling minimal: raw texts are
    deterministically projected into fixed-size embeddings, then passed into the
    existing HDIM core. The wrapper is suitable for retrieval/ranking and
    transfer experiments, not for generative language modeling.
    """

    def __init__(self, core_model: HDIMModel) -> None:
        super().__init__()
        self.core_model = core_model

    @property
    def config(self):
        return self.core_model.config

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        embeddings = texts_to_tensor(texts, self.config.hidden_dim)
        if dtype is None:
            dtype = next(self.core_model.parameters()).dtype
        if device is None:
            device = next(self.core_model.parameters()).device
        return embeddings.to(device=device, dtype=dtype)

    def forward_texts(
        self,
        texts: Sequence[str],
        domain_id: torch.Tensor,
        *,
        return_state: bool = False,
        update_memory: bool = True,
        memory_mode: str = "update",
    ):
        encodings = self.encode_texts(texts, device=domain_id.device)
        return self.core_model(
            encodings,
            domain_id,
            return_state=return_state,
            update_memory=update_memory,
            memory_mode=memory_mode,
        )

    def transfer_texts(
        self,
        texts: Sequence[str],
        *,
        source_domain: int,
        target_domain: int,
        return_state: bool = False,
        update_memory: bool = True,
        memory_mode: str = "update",
    ):
        encodings = self.encode_texts(texts)
        return self.core_model.transfer(
            encodings,
            source_domain=source_domain,
            target_domain=target_domain,
            return_state=return_state,
            update_memory=update_memory,
            memory_mode=memory_mode,
        )

    def transfer_text_pairs(
        self,
        source_texts: Sequence[str],
        source_domain_id: torch.Tensor,
        target_domain_id: torch.Tensor,
        *,
        update_memory: bool = True,
        memory_mode: str = "update",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, HDIMAuxState]:
        encodings = self.encode_texts(source_texts, device=source_domain_id.device)
        return self.core_model.transfer_pairs(
            encodings,
            source_domain_id,
            target_domain_id,
            update_memory=update_memory,
            memory_mode=memory_mode,
        )

    def score_text_pairs(
        self,
        source_texts: Sequence[str],
        target_texts: Sequence[str],
        source_domain_id: torch.Tensor,
        target_domain_id: torch.Tensor,
    ) -> torch.Tensor:
        _, _, _, src_state = self.transfer_text_pairs(
            source_texts,
            source_domain_id,
            target_domain_id,
            update_memory=False,
            memory_mode="retrieve",
        )
        _, _, _, tgt_state = self.forward_texts(
            target_texts,
            target_domain_id,
            return_state=True,
            update_memory=False,
            memory_mode="retrieve",
        )
        return F.cosine_similarity(
            src_state.exported_invariant,
            tgt_state.exported_invariant,
            dim=-1,
        )
