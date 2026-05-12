"""Deprecated text-facing wrapper for HDIM retrieval and transfer workflows."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.adapters.text import SimpleTextEncoder
from src.models.text_encoder_protocol import TextEncoder
from src.models.hdim_model import HDIMAuxState, HDIMModel, HDIMTextConfig
from src.models.results import ForwardResult


@dataclass(frozen=True)
class TextPairScoreResult:
    """Structured text-pair scoring artifact for retrieval/ranking workflows."""

    scores: torch.Tensor
    source_state: HDIMAuxState
    target_state: HDIMAuxState

    def to_dict(self) -> dict[str, torch.Tensor | bool | str]:
        result: dict[str, torch.Tensor | bool | str] = {"scores": self.scores}
        for prefix, state in (
            ("source", self.source_state),
            ("target", self.target_state),
        ):
            for key, value in state.to_dict().items():
                result[f"{prefix}_{key}"] = value
        return result


class TextHDIMModel(nn.Module):
    """Trainable text-entry wrapper around HDIMModel."""

    def __init__(
        self,
        core_model: HDIMModel,
        *,
        max_length: int | None = None,
        text_embedding_dim: int | None = None,
        text_hidden_dim: int | None = None,
        text_dropout: float | None = None,
    ) -> None:
        warnings.warn(
            "TextHDIMModel is deprecated; use TextAdapter + HDIMCoreEngine",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        self.core_model = core_model
        text_config = core_model.config.text
        if any(
            value is not None
            for value in (max_length, text_embedding_dim, text_hidden_dim, text_dropout)
        ):
            text_config = HDIMTextConfig(
                vocab_size=text_config.vocab_size,
                max_length=text_config.max_length if max_length is None else max_length,
                embedding_dim=text_config.embedding_dim
                if text_embedding_dim is None
                else text_embedding_dim,
                hidden_dim=text_config.hidden_dim
                if text_hidden_dim is None
                else text_hidden_dim,
                dropout=text_config.dropout if text_dropout is None else text_dropout,
                vocab_path=text_config.vocab_path,
                tokenizer_name=text_config.tokenizer_name,
            )

        self.text_config = text_config
        self.text_encoder: TextEncoder = SimpleTextEncoder.from_text_config(
            output_dim=core_model.config.hidden_dim,
            text_config=text_config,
            fallback_dropout=core_model.config.dropout,
        )

    @property
    def config(self):
        return self.core_model.config

    @property
    def pipeline(self):
        return self.core_model.pipeline

    @property
    def training_inv_head(self):
        return self.core_model.training_inv_head

    def forward(self, *args, **kwargs):
        return self.core_model(*args, **kwargs)

    def transfer(self, *args, **kwargs):
        return self.core_model.transfer(*args, **kwargs)

    def transfer_pairs(self, *args, **kwargs):
        return self.core_model.transfer_pairs(*args, **kwargs)

    def reset_memory(self, strategy: str = 'geometric') -> None:
        self.core_model.reset_memory(strategy=strategy)

    def enable_gradient_checkpointing(self) -> None:
        self.core_model.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self) -> None:
        self.core_model.disable_gradient_checkpointing()

    def enable_learnable_metric(self) -> None:
        self.core_model.enable_learnable_metric()

    def compute_expert_ortho_loss(self) -> "torch.Tensor":
        return self.core_model.pipeline.moe.expert_orthogonalization_loss()

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = next(self.core_model.parameters()).dtype
        if device is None:
            device = next(self.core_model.parameters()).device
        return self.text_encoder(texts, device=device, dtype=dtype)

    def encode_texts_matryoshka(
        self,
        texts: Sequence[str],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor] | None]:
        """Encode texts with Matryoshka multi-scale output.

        Returns:
            (full_encoding, scales_dict) where scales_dict maps dim→embedding
            or (full_encoding, None) if encoder doesn't support Matryoshka.
        """
        if dtype is None:
            dtype = next(self.core_model.parameters()).dtype
        if device is None:
            device = next(self.core_model.parameters()).device

        raw = self.text_encoder(texts, device=device, dtype=dtype)

        # Determine matryoshka dimensions: prefer encoder config, else standard defaults
        matryoshka_dims = getattr(self.text_encoder, "matryoshka_dims", None)
        if matryoshka_dims is None:
            proj = getattr(self.text_encoder, "projection", None)
            if proj is not None and hasattr(proj, "output_dims"):
                matryoshka_dims = proj.output_dims
            else:
                matryoshka_dims = [128, 256, 512]

        # Only keep dims smaller than the full embedding dimension
        embedding_dim = raw.shape[-1]
        matryoshka_dims = [d for d in matryoshka_dims if d < embedding_dim]

        if not matryoshka_dims:
            return raw, None

        scales: dict[int, torch.Tensor] = {
            dim: raw[..., :dim].clone() for dim in matryoshka_dims
        }
        return raw, scales

    def forward_texts(
        self,
        texts: Sequence[str],
        domain_id: torch.Tensor,
        *,
        return_state: bool = False,
        update_memory: bool = True,
        memory_mode: str = "update",
        return_encoding: bool = False,
    ) -> ForwardResult:
        encodings = self.encode_texts(texts, device=domain_id.device)
        result = HDIMModel.forward(
            self.core_model,
            encodings,
            domain_id,
            return_state=return_state,
            update_memory=update_memory,
            memory_mode=memory_mode,
        )
        if return_encoding:
            result.encodings = encodings
        return result

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
    ) -> ForwardResult:
        encodings = self.encode_texts(source_texts, device=source_domain_id.device)
        return HDIMModel.transfer_pairs(
            self.core_model,
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
        return self.score_text_pairs_with_state(
            source_texts,
            target_texts,
            source_domain_id,
            target_domain_id,
        ).scores

    def score_text_pairs_with_state(
        self,
        source_texts: Sequence[str],
        target_texts: Sequence[str],
        source_domain_id: torch.Tensor,
        target_domain_id: torch.Tensor,
    ) -> TextPairScoreResult:
        src_result = self.transfer_text_pairs(
            source_texts,
            source_domain_id,
            target_domain_id,
            update_memory=False,
            memory_mode="retrieve",
        )
        tgt_result = self.forward_texts(
            target_texts,
            target_domain_id,
            return_state=True,
            update_memory=False,
            memory_mode="retrieve",
        )
        src_state = src_result.aux_state
        tgt_state = tgt_result.aux_state
        scores = F.cosine_similarity(
            src_state.exported_invariant,
            tgt_state.exported_invariant,
            dim=-1,
        )
        return TextPairScoreResult(
            scores=scores,
            source_state=src_state,
            target_state=tgt_state,
        )
