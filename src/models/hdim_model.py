"""Thin HDIM model wrappers over the core engine."""

from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn

from src.core.domain_operators import DomainIndexEmbedding
from src.core.engine import CoreEngineConfig, HDIMCoreEngine
from src.models.config import HDIMConfig, HDIMRuntimeConfig, HDIMTextConfig
from src.models.results import CoreResult, ForwardResult, HDIMAuxState

__all__ = ["CoreResult", "ForwardResult", "HDIMAuxState", "HDIMModel", "HDIMModelCore", "HDIMTextConfig"]


class HDIMModelCore(nn.Module):
    """Thin adapter around HDIMCoreEngine."""

    def __init__(self, config: HDIMConfig) -> None:
        super().__init__()
        self.config = config
        self._domain_names: List[str] = config.get_domain_names()
        self.engine = HDIMCoreEngine(
            CoreEngineConfig(
                input_dim=config.hidden_dim,
                clifford_p=config.clifford_p,
                clifford_q=config.clifford_q,
                clifford_r=config.clifford_r,
                domain_names=tuple(self._domain_names),
                dropout=config.dropout,
            )
        )
        self.decoder = nn.Linear(self.engine.algebra.dim, config.hidden_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.engine.encode(x)

    def extract(self, g: torch.Tensor, domain: str) -> torch.Tensor:
        return self.engine.extract(g, domain)

    def match(self, invariant: torch.Tensor, expert_base: Any = None) -> list[list[Any]]:
        return self.engine.match(invariant, expert_base=expert_base)

    def transfer(self, invariant: torch.Tensor, target_domain: str) -> torch.Tensor:
        return self.engine.transfer(invariant, target_domain)

    def forward(
        self,
        x: torch.Tensor,
        source_domain: str = "source",
        target_domain: str = "target",
    ) -> CoreResult:
        if not isinstance(source_domain, str):
            raise TypeError("HDIMModelCore.forward expects source_domain as a domain name")
        g_source = self.encode(x)
        raw_invariant = self.extract(g_source, source_domain)
        matches = self.match(raw_invariant)
        exported_invariant = self.transfer(raw_invariant, target_domain)
        return CoreResult(
            output=self.decoder(exported_invariant),
            raw_invariant=raw_invariant,
            exported_invariant=exported_invariant,
            matches=matches,
            routing_weights=x.new_zeros(x.shape[0], self.config.num_experts),
            slot_outputs=None,
        )


class HDIMModel(nn.Module):
    """Compatibility wrapper that delegates structural work to HDIMCoreEngine."""

    def __init__(self, config: HDIMConfig) -> None:
        super().__init__()
        self._core = HDIMModelCore(config)
        self.config = config
        self.training_inv_head = nn.Linear(self.engine.algebra.dim, config.hidden_dim)
        self.domain_embedding: Optional[DomainIndexEmbedding] = None
        if config.use_domain_embedding:
            self.domain_embedding = DomainIndexEmbedding(dim=config.hidden_dim, max_domains=config.num_domains)
        self.extension_flags = {
            "online_learning": bool(config.runtime.online_learning),
            "hallucination_detection": bool(config.runtime.hallucination_detection),
            "hallucination_feedback": bool(config.runtime.hallucination_feedback),
            "domain_lora": bool(config.use_domain_lora),
        }
        self.online_learner = None
        self.hallucination_detector = None
        self.hallucination_feedback_loop = None
        self.domain_lora = None
        self._use_gradient_checkpointing = False

    @property
    def engine(self) -> HDIMCoreEngine:
        return self._core.engine

    @property
    def decoder(self) -> nn.Linear:
        return self._core.decoder

    @property
    def _domain_names(self) -> List[str]:
        return self._core._domain_names

    @property
    def pipeline(self) -> "HDIMModel":
        return self

    @property
    def clifford_dim(self) -> int:
        return self.engine.algebra.dim

    @property
    def algebra(self):
        return self.engine.algebra

    @property
    def domain_rotors(self):
        return self.engine.domain_rotors

    @property
    def invariant_extractor(self):
        return self.engine.extractor

    @property
    def invariant_index(self):
        return self.engine.index

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._core.encode(x)

    def extract(self, g: torch.Tensor, domain: str) -> torch.Tensor:
        return self._core.extract(g, domain)

    def match(self, invariant: torch.Tensor, expert_base: Any = None) -> list[list[Any]]:
        return self._core.match(invariant, expert_base=expert_base)

    def transfer_invariant(self, invariant: torch.Tensor, target_domain: str) -> torch.Tensor:
        return self._core.transfer(invariant, target_domain)

    def _domain_idx_to_name(self, domain_idx: int) -> str:
        if domain_idx < 0 or domain_idx >= len(self._domain_names):
            raise IndexError(f"domain_idx {domain_idx} out of range [0, {len(self._domain_names)}).")
        return self._domain_names[domain_idx]

    def _resolve_runtime_config(self, *, update_memory: bool, memory_mode: str) -> HDIMRuntimeConfig:
        if memory_mode not in {"none", "retrieve", "update"}:
            raise ValueError(f"Unsupported memory_mode: {memory_mode}")
        return HDIMRuntimeConfig(update_memory=False, memory_mode="none" if memory_mode != "retrieve" else "retrieve")

    def _validate_domain_ids(self, domain_id: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
        if domain_id.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape {tuple(domain_id.shape)}")
        if domain_id.numel() != batch_size:
            raise ValueError(f"{name} length {domain_id.numel()} must match batch size {batch_size}")
        domain_id = domain_id.to(dtype=torch.long)
        if domain_id.numel() > 0 and ((domain_id < 0).any() or (domain_id >= self.config.num_domains).any()):
            raise IndexError(f"{name} values must be in [0, {self.config.num_domains})")
        return domain_id

    def _forward_core(
        self,
        x: torch.Tensor,
        source_domain: str = "source",
        target_domain: str = "target",
    ) -> CoreResult:
        g_source = self.encode(x)
        raw_invariant = self.extract(g_source, source_domain)
        matches = self.match(raw_invariant)
        exported_invariant = self.transfer_invariant(raw_invariant, target_domain)
        output = self.decoder(exported_invariant)
        return CoreResult(
            output=output,
            raw_invariant=raw_invariant,
            exported_invariant=exported_invariant,
            matches=matches,
            routing_weights=x.new_zeros(x.shape[0], self.config.num_experts),
            slot_outputs=None,
        )

    def _forward_pairs_core(
        self,
        x: torch.Tensor,
        source_domain_id: torch.Tensor,
        target_domain_id: torch.Tensor,
    ) -> CoreResult:
        output = x.new_empty(x.shape)
        raw = x.new_empty(x.shape[0], self.engine.algebra.dim)
        exported = x.new_empty(x.shape[0], self.engine.algebra.dim)
        routing = x.new_zeros(x.shape[0], self.config.num_experts)
        matches: list[list[Any]] = [[] for _ in range(x.shape[0])]

        pair_keys = torch.stack((source_domain_id, target_domain_id), dim=1)
        for pair in pair_keys.unique(dim=0):
            src_idx = int(pair[0].item())
            tgt_idx = int(pair[1].item())
            mask = (source_domain_id == src_idx) & (target_domain_id == tgt_idx)
            core = self._forward_core(
                x[mask],
                self._domain_idx_to_name(src_idx),
                self._domain_idx_to_name(tgt_idx),
            )
            output[mask] = core.output.to(dtype=output.dtype)
            raw[mask] = core.raw_invariant.to(dtype=raw.dtype)
            exported[mask] = core.exported_invariant.to(dtype=exported.dtype)
            routing[mask] = core.routing_weights.to(dtype=routing.dtype)
            mask_indices = mask.nonzero(as_tuple=False).flatten().tolist()
            for row, row_matches in zip(mask_indices, core.matches):
                matches[row] = row_matches

        return CoreResult(
            output=output,
            raw_invariant=raw,
            exported_invariant=exported,
            matches=matches,
            routing_weights=routing,
            slot_outputs=None,
        )

    def _build_aux_state(self, core: CoreResult, runtime: HDIMRuntimeConfig) -> HDIMAuxState:
        return HDIMAuxState(
            raw_invariant=core.raw_invariant,
            exported_invariant=core.exported_invariant,
            matches=core.matches,
            training_invariant=self.training_inv_head.forward(core.exported_invariant),
            memory_mode=runtime.memory_mode,
            update_memory=runtime.update_memory,
            memory_updated=False,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        x = args[0]
        domain_id = args[1] if len(args) > 1 else kwargs["domain_id"]
        return_state = kwargs.get("return_state", False)
        update_memory = kwargs.get("update_memory", True)
        memory_mode = kwargs.get("memory_mode", "update")
        domain_id = self._validate_domain_ids(domain_id.to(device=x.device), x.shape[0], "domain_id")
        runtime = self._resolve_runtime_config(update_memory=update_memory, memory_mode=memory_mode)
        core = self._forward_pairs_core(x, domain_id, domain_id)
        invariant = self.training_inv_head.forward(core.exported_invariant).to(dtype=x.dtype)
        domain_embedding = self.domain_embedding
        if domain_embedding is not None:
            invariant = invariant + domain_embedding.forward(domain_id).to(dtype=x.dtype)
        aux_state = self._build_aux_state(core, runtime) if return_state else None
        return ForwardResult(core.output, core.routing_weights, invariant, core.slot_outputs, aux_state)

    def transfer(self, *args: Any, **kwargs: Any) -> Any:
        problem_encoding = args[0]
        source_domain = args[1] if len(args) > 1 else kwargs["source_domain"]
        target_domain = args[2] if len(args) > 2 else kwargs["target_domain"]
        source_domain = int(source_domain.item()) if isinstance(source_domain, torch.Tensor) else int(source_domain)
        target_domain = int(target_domain.item()) if isinstance(target_domain, torch.Tensor) else int(target_domain)
        return_state = kwargs.get("return_state", False)
        update_memory = kwargs.get("update_memory", True)
        memory_mode = kwargs.get("memory_mode", "update")
        runtime = self._resolve_runtime_config(update_memory=update_memory, memory_mode=memory_mode)
        core = self._forward_core(
            problem_encoding,
            self._domain_idx_to_name(source_domain),
            self._domain_idx_to_name(target_domain),
        )
        if return_state:
            return core.output, self._build_aux_state(core, runtime)
        return core.output

    def transfer_pairs(
        self,
        source_encoding: torch.Tensor,
        source_domain_id: torch.Tensor,
        target_domain_id: torch.Tensor,
        *,
        update_memory: bool = True,
        memory_mode: str = "update",
    ) -> ForwardResult:
        source_domain_id = self._validate_domain_ids(
            source_domain_id.to(device=source_encoding.device), source_encoding.shape[0], "source_domain_id"
        )
        target_domain_id = self._validate_domain_ids(
            target_domain_id.to(device=source_encoding.device), source_encoding.shape[0], "target_domain_id"
        )
        runtime = self._resolve_runtime_config(update_memory=update_memory, memory_mode=memory_mode)
        core = self._forward_pairs_core(source_encoding, source_domain_id, target_domain_id)
        invariant = self.training_inv_head.forward(core.exported_invariant).to(dtype=source_encoding.dtype)
        domain_embedding = self.domain_embedding
        if domain_embedding is not None:
            invariant = invariant + domain_embedding.forward(source_domain_id).to(dtype=source_encoding.dtype)
        aux_state = self._build_aux_state(core, runtime)
        return ForwardResult(core.output, core.routing_weights, invariant, core.slot_outputs, aux_state)

    def reset_memory(self, strategy: str = "geometric") -> None:
        return None

    def enable_gradient_checkpointing(self) -> None:
        self._use_gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self._use_gradient_checkpointing = False
