"""HDIM model adapter over the core engine."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from src.core.domain_operators import DomainIndexEmbedding
from src.core.engine import CoreEngineConfig, HDIMCoreEngine
from src.core.rotors import DomainRotationOperator
from src.models.config import HDIMConfig, HDIMRuntimeConfig, HDIMTextConfig
from src.models.results import CoreResult, ForwardResult, HDIMAuxState

__all__ = ["CoreResult", "ForwardResult", "HDIMAuxState", "HDIMModel", "HDIMModelCore", "HDIMTextConfig"]


class _CoreMoEShim(nn.Module):
    """Minimal legacy MoE surface for compatibility only."""

    def __init__(self, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.expert_names = [f"expert_{i}" for i in range(num_experts)]
        self.slots_per_expert = 1
        self.register_buffer("train_scores", torch.full((num_experts,), 1.0 / num_experts))

    def forward(self, u_inv: torch.Tensor) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gate_weights = u_inv.new_full((u_inv.shape[0], self.num_experts), 1.0 / self.num_experts)
        topk_idx = torch.arange(self.top_k, device=u_inv.device, dtype=torch.long).expand(u_inv.shape[0], -1)
        topk_gate_weights = u_inv.new_full((u_inv.shape[0], self.top_k), 1.0 / self.top_k)
        zero = u_inv.new_tensor(0.0)
        return u_inv, {
            "gate_weights": gate_weights,
            "topk_idx": topk_idx,
            "topk_gate_weights": topk_gate_weights,
            "router_loss": zero,
            "z_loss": zero,
            "routing_entropy": zero,
            "train_scores_snapshot": self.train_scores.to(device=u_inv.device, dtype=u_inv.dtype),
            "expert_usage": self.train_scores.to(device=u_inv.device, dtype=u_inv.dtype),
        }

    def enable_aux_loss_free(self, aux_lr: float = 0.01) -> None:
        return None

    def enable_expert_ortho(self) -> None:
        return None

    def expert_orthogonalization_loss(self) -> torch.Tensor:
        return self.train_scores.new_tensor(0.0)


class _CoreMemoryShim(nn.Module):
    """Minimal legacy memory surface for compatibility only."""

    def __init__(self, clifford_dim: int) -> None:
        super().__init__()
        self.memory = nn.Embedding(1, clifford_dim)
        nn.init.zeros_(self.memory.weight)
        self.register_buffer("momentum_S", torch.zeros(1, clifford_dim))
        self.use_gradient_surprise = False
        self.use_adaptive_forgetting = False

    def reset(self, strategy: str = "geometric") -> None:
        self.memory.weight.data.zero_()
        self.momentum_S.zero_()


class _HDIMPipelineFacade(nn.Module):
    """Lightweight pipeline compatibility facade backed by HDIMCoreEngine."""

    def __init__(
        self,
        *,
        engine: HDIMCoreEngine,
        decoder: nn.Module,
        domain_names: list[str],
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.decoder = decoder
        self.domain_names = domain_names
        self.algebra = engine.algebra
        self.clifford_dim = engine.algebra.dim
        self.encoder = engine.encoder
        self.domain_rotors = engine.domain_rotors
        self.invariant_extractor = engine.extractor
        self.invariant_norm = nn.Identity()
        self.invariant_index = engine.index
        self.memory_type = "none"
        self.moe = _CoreMoEShim(num_experts=num_experts, top_k=top_k)
        self.memory = _CoreMemoryShim(self.clifford_dim)
        self._use_gradient_checkpointing = False

    def transfer(
        self,
        x: torch.Tensor,
        source_domain: str,
        target_domain: str,
        *,
        update_memory: bool = True,
        memory_mode: str = "update",
        input_is_invariant: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Transfer a batch from source to target domain.

        Args:
            x: Input encodings or invariants with shape (batch, hidden_dim|clifford_dim).
            source_domain: Domain name used to extract invariants when x is encoded input.
            target_domain: Domain name used to export invariants before decoding.
        """
        if memory_mode not in {"none", "retrieve", "update"}:
            raise ValueError(f"Unsupported memory_mode: {memory_mode}")
        if input_is_invariant:
            g_source = None
            u_inv = x
        else:
            g_source = self.engine.encode(x)
            u_inv = self.engine.extract(g_source, source_domain)
        g_target = self.engine.transfer(u_inv, target_domain)
        output = self.decoder(g_target)
        zero = u_inv.new_tensor(0.0)
        return output, {
            "g_source": g_source,
            "u_inv": u_inv,
            "u_mem": u_inv,
            "u_route": g_target,
            "g_target": g_target,
            "output": output,
            "memory_loss": zero,
            "memory_updated": False,
            "memory_mode": "none",
            "update_memory": False,
            "input_is_invariant": input_is_invariant,
            "raw_invariant": u_inv,
            "memory_augmented_invariant": u_inv,
            "exported_invariant": g_target,
            "invariant": g_target,
            "router_state": {
                "router_loss": zero,
                "z_loss": zero,
                "routing_entropy": zero,
                "topk_idx": torch.empty(u_inv.shape[0], 0, device=u_inv.device, dtype=torch.long),
                "topk_gate_weights": u_inv.new_zeros(u_inv.shape[0], 0),
                "train_scores_snapshot": u_inv.new_zeros(0),
                "expert_usage": u_inv.new_zeros(0),
            },
        }

    def forward(self, x: torch.Tensor, source_domain: str = "source", target_domain: str = "target", **kwargs: Any):
        return self.transfer(x, source_domain, target_domain, **kwargs)

    def encode_domain(self, x: torch.Tensor, domain_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        g_source = self.engine.encode(x)
        return g_source, self.engine.extract(g_source, domain_name)

    def add_domain(self, domain_name: str) -> None:
        if domain_name in self.engine.domain_rotors:
            raise ValueError(f"Domain {domain_name!r} already exists")
        self.engine.domain_rotors[domain_name] = DomainRotationOperator(self.algebra, domain_name=domain_name)
        self.domain_names.append(domain_name)

    def reset_memory(self, strategy: str = "geometric") -> None:
        self.memory.reset(strategy=strategy)

    def enable_gradient_checkpointing(self) -> None:
        self._use_gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self._use_gradient_checkpointing = False


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
        clifford_dim = self.engine.algebra.dim
        self.decoder = nn.Linear(clifford_dim, config.hidden_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.engine.encode(x)

    def extract(self, g: torch.Tensor, domain: str) -> torch.Tensor:
        return self.engine.extract(g, domain)

    def match(self, invariant: torch.Tensor, expert_base: Any = None) -> list[list[Any]]:
        return self.engine.match(invariant, expert_base=expert_base)

    def transfer(self, *args: Any, **kwargs: Any) -> Any:
        invariant, target_domain = args[:2]
        return self.engine.transfer(invariant, target_domain)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Encode and transfer input encodings.

        Args:
            x: Input tensor with shape (batch, hidden_dim).
            source_domain: Domain name used for invariant extraction.
            target_domain: Domain name used for invariant export.
        """
        x = args[0]
        source_domain = args[1] if len(args) > 1 else kwargs.get("source_domain", "source")
        target_domain = args[2] if len(args) > 2 else kwargs.get("target_domain", "target")
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
    """Compatibility HDIMModel with deprecated facade helpers."""

    def __init__(self, config: HDIMConfig) -> None:
        super().__init__()
        self._core = HDIMModelCore(config)
        self.config = config
        clifford_dim = self.engine.algebra.dim
        self.training_inv_head = nn.Linear(clifford_dim, config.hidden_dim)
        self.domain_embedding: Optional[DomainIndexEmbedding] = None
        if config.use_domain_embedding:
            self.domain_embedding = DomainIndexEmbedding(dim=config.hidden_dim, max_domains=config.num_domains)

        runtime = config.runtime
        self.extension_flags = {
            "online_learning": bool(runtime.online_learning),
            "hallucination_detection": bool(runtime.hallucination_detection),
            "hallucination_feedback": bool(runtime.hallucination_feedback),
            "domain_lora": bool(config.use_domain_lora),
        }
        self.online_learner = None
        self.hallucination_detector = None
        self.hallucination_feedback_loop = None
        if runtime.hallucination_feedback:
            from src.extensions.hallucination.feedback import HallucinationFeedbackConfig

            feedback_config = runtime.hallucination_feedback_config or {}
            expert_names = config.expert_names or [f"expert_{i}" for i in range(config.num_experts)]
            self.hallucination_feedback_loop = HallucinationFeedbackConfig(**feedback_config).create_feedback_loop(expert_names)
        self.domain_lora = None
        self._pipeline: _HDIMPipelineFacade | None = None

    @property
    def engine(self) -> HDIMCoreEngine:
        return self._core.engine

    @property
    def decoder(self) -> nn.Linear:
        return self._core.decoder

    @property
    def _domain_names(self) -> List[str]:
        return self._core._domain_names

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._core.encode(x)

    def extract(self, g: torch.Tensor, domain: str) -> torch.Tensor:
        return self._core.extract(g, domain)

    def match(self, invariant: torch.Tensor, expert_base: Any = None) -> list[list[Any]]:
        return self._core.match(invariant, expert_base=expert_base)

    @property
    def pipeline(self) -> _HDIMPipelineFacade:
        if self._pipeline is None:
            self._pipeline = _HDIMPipelineFacade(
                engine=self.engine,
                decoder=self.decoder,
                domain_names=self._domain_names,
                num_experts=self.config.num_experts,
                top_k=self.config.top_k,
            )
        return self._pipeline

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
        g_source = self._core.encode(x)
        raw_invariant = self._core.extract(g_source, source_domain)
        matches = self._core.match(raw_invariant)
        exported_invariant = self._core.transfer(raw_invariant, target_domain)
        output = self.decoder(exported_invariant)
        if "memory" in self.config.extensions:
            routing_weights = x.new_full((x.shape[0], self.config.num_experts), 1.0 / self.config.num_experts)
        else:
            routing_weights = x.new_zeros(x.shape[0], self.config.num_experts)
        return CoreResult(
            output=output,
            raw_invariant=raw_invariant,
            exported_invariant=exported_invariant,
            matches=matches,
            routing_weights=routing_weights,
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
            training_invariant=self.training_inv_head(core.exported_invariant),
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
        invariant = self.training_inv_head(core.exported_invariant).to(dtype=x.dtype)
        if self.domain_embedding is not None:
            invariant = invariant + self.domain_embedding(domain_id).to(dtype=x.dtype)
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
        invariant = self.training_inv_head(core.exported_invariant).to(dtype=source_encoding.dtype)
        if self.domain_embedding is not None:
            invariant = invariant + self.domain_embedding(source_domain_id).to(dtype=source_encoding.dtype)
        aux_state = self._build_aux_state(core, runtime)
        return ForwardResult(core.output, core.routing_weights, invariant, core.slot_outputs, aux_state)

    def add_domain(self, domain_name: str) -> None:
        self.pipeline.add_domain(domain_name)

    def reset_memory(self, strategy: str = "geometric") -> None:
        self.pipeline.reset_memory(strategy=strategy)
        with torch.no_grad():
            n = self.pipeline.moe.num_experts
            self.pipeline.moe.train_scores.fill_(1.0 / n)

    def enable_learnable_metric(self) -> None:
        if hasattr(self.engine.algebra, "enable_learnable_metric"):
            self.engine.algebra.enable_learnable_metric()

    def compute_expert_ortho_loss(self) -> torch.Tensor:
        return next(self.parameters()).new_tensor(0.0)

    def enable_gradient_checkpointing(self) -> None:
        self.pipeline.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self) -> None:
        self.pipeline.disable_gradient_checkpointing()
