"""HDIM (Hypercomplex Domain-Invariant Model) — full PyTorch model definition.

This model wraps the full HDIMPipeline from src.core with a dataclass-based
configuration interface and adds integer-indexed domain routing for use in
batch training scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.core.hdim_pipeline import HDIMPipeline


@dataclass(frozen=True)
class HDIMRuntimeConfig:
    """Runtime controls for memory lifecycle during HDIM execution."""

    update_memory: bool = True
    memory_mode: Literal["none", "retrieve", "update"] = "update"


@dataclass(frozen=True)
class HDIMAuxState:
    """Typed lifecycle state exported by HDIMModel public paths."""

    memory_loss: torch.Tensor
    router_loss: torch.Tensor
    raw_invariant: torch.Tensor
    memory_augmented_invariant: torch.Tensor
    exported_invariant: torch.Tensor
    training_invariant: torch.Tensor
    routing_weights: torch.Tensor
    topk_idx: torch.Tensor
    topk_gate_weights: torch.Tensor
    train_scores_snapshot: torch.Tensor
    expert_usage: torch.Tensor
    routing_entropy: torch.Tensor
    memory_updated: bool
    memory_mode: str
    update_memory: bool

    def to_dict(self) -> Dict[str, Union[torch.Tensor, bool, str]]:
        return {
            "memory_loss": self.memory_loss,
            "router_loss": self.router_loss,
            "raw_invariant": self.raw_invariant,
            "memory_augmented_invariant": self.memory_augmented_invariant,
            "exported_invariant": self.exported_invariant,
            "training_invariant": self.training_invariant,
            "routing_weights": self.routing_weights,
            "topk_idx": self.topk_idx,
            "topk_gate_weights": self.topk_gate_weights,
            "train_scores_snapshot": self.train_scores_snapshot,
            "expert_usage": self.expert_usage,
            "routing_entropy": self.routing_entropy,
            "memory_updated": self.memory_updated,
            "memory_mode": self.memory_mode,
            "update_memory": self.update_memory,
        }


@dataclass(frozen=True)
class HDIMTextConfig:
    """Configuration for the minimal HDIM text encoder path."""

    vocab_size: int = 257
    max_length: int = 128
    embedding_dim: Optional[int] = None
    hidden_dim: Optional[int] = None
    dropout: Optional[float] = None
    vocab_path: Optional[str] = None
    tokenizer_name: Optional[str] = None


@dataclass
class HDIMConfig:
    """Configuration dataclass for HDIMModel.

    Attributes:
        hidden_dim: Input/output feature dimensionality (must be divisible by 4).
        num_domains: Number of named domains to register domain rotors for.
        num_experts: Number of MoE experts in the R3MoERouter.
        dropout: Dropout probability applied after the encoder.
        clifford_p: Positive basis vectors for Cl_{p,q,r} algebra.
        clifford_q: Negative basis vectors.
        clifford_r: Nilpotent basis vectors.
        top_k: Number of active experts per token.
        memory_key_dim: Dimensionality of Titans memory keys.
        domain_names: Explicit domain name list. If None, auto-generates
            ['domain_0', 'domain_1', ...] up to num_domains.
    """

    hidden_dim: int = 64
    num_domains: int = 4
    num_experts: int = 4
    dropout: float = 0.1
    clifford_p: int = 3
    clifford_q: int = 1
    clifford_r: int = 0
    top_k: int = 2
    memory_key_dim: int = 32
    domain_names: Optional[List[str]] = None
    text: HDIMTextConfig = field(default_factory=HDIMTextConfig)
    def get_domain_names(self) -> List[str]:
        """Return the resolved list of domain names."""
        if self.domain_names is not None:
            return list(self.domain_names)
        return [f"domain_{i}" for i in range(self.num_domains)]

class HDIMModel(nn.Module):
    """Hypercomplex Domain-Invariant Model (HDIM).

    Wraps HDIMPipeline from src.core and exposes an integer-indexed domain
    interface suitable for batch training. Domain indices map to the named
    domain rotors registered inside the pipeline.

    Forward returns (output, routing_weights, invariant) where invariant is
    the hidden-dim projection of the canonical training invariant.
    """

    def __init__(self, config: HDIMConfig) -> None:
        super().__init__()
        self.config = config
        self._domain_names: List[str] = config.get_domain_names()

        self.pipeline = HDIMPipeline(
            input_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            clifford_p=config.clifford_p,
            clifford_q=config.clifford_q,
            clifford_r=config.clifford_r,
            domain_names=self._domain_names,
            num_experts=config.num_experts,
            top_k=config.top_k,
            memory_key_dim=config.memory_key_dim,
        )

        self.dropout = nn.Dropout(config.dropout)
        clifford_dim = self.pipeline.clifford_dim
        self.training_inv_head = nn.Linear(clifford_dim, config.hidden_dim)

    def _domain_idx_to_name(self, domain_idx: int) -> str:
        """Convert an integer domain index to its registered name."""
        if domain_idx < 0 or domain_idx >= len(self._domain_names):
            raise IndexError(
                f"domain_idx {domain_idx} out of range [0, {len(self._domain_names)})."
            )
        return self._domain_names[domain_idx]

    def _resolve_runtime_config(
        self,
        *,
        update_memory: bool,
        memory_mode: str,
    ) -> HDIMRuntimeConfig:
        if memory_mode not in {"none", "retrieve", "update"}:
            raise ValueError(f"Unsupported memory_mode: {memory_mode}")
        if memory_mode != "update":
            update_memory = False
        return HDIMRuntimeConfig(
            update_memory=update_memory,
            memory_mode=memory_mode,
        )

    def _build_aux_state(
        self,
        *,
        raw_invariant: torch.Tensor,
        memory_augmented_invariant: torch.Tensor,
        exported_invariant: torch.Tensor,
        routing_weights: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_gate_weights: torch.Tensor,
        train_scores_snapshot: torch.Tensor,
        expert_usage: torch.Tensor,
        routing_entropy: torch.Tensor,
        memory_loss: torch.Tensor,
        router_loss: torch.Tensor,
        memory_updated: bool,
        runtime: HDIMRuntimeConfig,
    ) -> HDIMAuxState:
        return HDIMAuxState(
            memory_loss=memory_loss,
            router_loss=router_loss,
            raw_invariant=raw_invariant,
            memory_augmented_invariant=memory_augmented_invariant,
            exported_invariant=exported_invariant,
            training_invariant=self.training_inv_head(exported_invariant),
            routing_weights=routing_weights,
            topk_idx=topk_idx,
            topk_gate_weights=topk_gate_weights,
            train_scores_snapshot=train_scores_snapshot,
            expert_usage=expert_usage,
            routing_entropy=routing_entropy,
            memory_updated=memory_updated,
            memory_mode=runtime.memory_mode,
            update_memory=runtime.update_memory,
        )

    def _build_aux_state_from_transfer_state(
        self,
        transfer_state: Dict[str, Union[torch.Tensor, bool, str, Dict[str, torch.Tensor]]],
        *,
        runtime: HDIMRuntimeConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> HDIMAuxState:
        router_state = transfer_state["router_state"]
        return self._build_aux_state(
            raw_invariant=transfer_state["raw_invariant"].to(device=device, dtype=dtype),
            memory_augmented_invariant=transfer_state["memory_augmented_invariant"].to(device=device, dtype=dtype),
            exported_invariant=transfer_state["exported_invariant"].to(device=device, dtype=dtype),
            routing_weights=transfer_state["routing_weights"].to(device=device, dtype=dtype),
            topk_idx=router_state["topk_idx"].to(device=device),
            topk_gate_weights=router_state["topk_gate_weights"].to(device=device, dtype=dtype),
            train_scores_snapshot=router_state["train_scores_snapshot"].to(device=device, dtype=dtype),
            expert_usage=router_state["expert_usage"].to(device=device, dtype=dtype),
            routing_entropy=router_state["routing_entropy"].to(device=device, dtype=dtype),
            memory_loss=transfer_state["memory_loss"].to(device=device, dtype=dtype),
            router_loss=router_state["router_loss"].to(device=device, dtype=dtype),
            memory_updated=bool(transfer_state["memory_updated"]),
            runtime=runtime,
        )

    def _allocate_state_tensors(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_invariant = torch.empty(
            batch_size,
            self.pipeline.clifford_dim,
            device=device,
            dtype=dtype,
        )
        memory_augmented_invariant = torch.empty(
            batch_size,
            self.pipeline.clifford_dim,
            device=device,
            dtype=dtype,
        )
        exported_invariant = torch.empty(
            batch_size,
            self.pipeline.clifford_dim,
            device=device,
            dtype=dtype,
        )
        routing_weights = torch.empty(
            batch_size,
            self.config.num_experts,
            device=device,
            dtype=dtype,
        )
        topk_idx = torch.empty(
            batch_size,
            self.config.top_k,
            device=device,
            dtype=torch.long,
        )
        topk_gate_weights = torch.empty(
            batch_size,
            self.config.top_k,
            device=device,
            dtype=dtype,
        )
        return (
            raw_invariant,
            memory_augmented_invariant,
            exported_invariant,
            routing_weights,
            topk_idx,
            topk_gate_weights,
        )

    def forward(
        self,
        x: torch.Tensor,
        domain_id: torch.Tensor,
        *,
        return_state: bool = False,
        update_memory: bool = True,
        memory_mode: str = "update",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        HDIMAuxState,
    ]:
        """Run the HDIM forward pass for same-domain reconstruction batches."""
        x = self.dropout(x)
        domain_id = domain_id.to(device=x.device, dtype=torch.long)
        runtime = self._resolve_runtime_config(
            update_memory=update_memory,
            memory_mode=memory_mode,
        )

        batch_size = x.shape[0]
        output = torch.empty_like(x)
        (
            raw_invariant,
            memory_augmented_invariant,
            exported_invariant,
            routing_weights,
            topk_idx,
            topk_gate_weights,
        ) = self._allocate_state_tensors(
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )
        train_scores_snapshot = torch.empty(
            self.config.num_experts,
            device=x.device,
            dtype=x.dtype,
        )
        expert_usage = torch.empty(
            self.config.num_experts,
            device=x.device,
            dtype=x.dtype,
        )
        memory_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        router_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        routing_entropy = torch.zeros((), device=x.device, dtype=x.dtype)
        memory_updated = False

        for batch_domain_idx in domain_id.unique(sorted=True):
            domain_idx = int(batch_domain_idx.item())
            domain_name = self._domain_idx_to_name(domain_idx)
            mask = domain_id == batch_domain_idx
            group_x = x[mask]
            group_output, transfer_state = self.pipeline.transfer(
                group_x,
                domain_name,
                domain_name,
                update_memory=runtime.update_memory,
                memory_mode=runtime.memory_mode,
            )

            output[mask] = group_output
            routing_weights[mask] = transfer_state["routing_weights"].to(dtype=x.dtype)
            topk_idx[mask] = transfer_state["router_state"]["topk_idx"].to(device=x.device)
            topk_gate_weights[mask] = transfer_state["router_state"]["topk_gate_weights"].to(dtype=x.dtype)
            exported_invariant[mask] = transfer_state["exported_invariant"].to(dtype=x.dtype)
            raw_invariant[mask] = transfer_state["raw_invariant"].to(dtype=x.dtype)
            memory_augmented_invariant[mask] = transfer_state["memory_augmented_invariant"].to(dtype=x.dtype)
            memory_loss = memory_loss + transfer_state["memory_loss"].to(dtype=x.dtype)
            router_loss = router_loss + transfer_state["router_state"]["router_loss"].to(dtype=x.dtype)
            routing_entropy = routing_entropy + transfer_state["router_state"]["routing_entropy"].to(dtype=x.dtype)
            train_scores_snapshot.copy_(transfer_state["router_state"]["train_scores_snapshot"].to(dtype=x.dtype))
            expert_usage.copy_(transfer_state["router_state"]["expert_usage"].to(dtype=x.dtype))
            memory_updated = memory_updated or bool(transfer_state["memory_updated"])

        invariant = self.training_inv_head(exported_invariant).to(dtype=x.dtype)

        if return_state:
            aux_state = self._build_aux_state(
                raw_invariant=raw_invariant,
                memory_augmented_invariant=memory_augmented_invariant,
                exported_invariant=exported_invariant,
                routing_weights=routing_weights,
                topk_idx=topk_idx,
                topk_gate_weights=topk_gate_weights,
                train_scores_snapshot=train_scores_snapshot,
                expert_usage=expert_usage,
                routing_entropy=routing_entropy,
                memory_loss=memory_loss,
                router_loss=router_loss,
                memory_updated=memory_updated,
                runtime=runtime,
            )
            return output, routing_weights, invariant, aux_state
        return output, routing_weights, invariant

    def transfer(
        self,
        problem_encoding: torch.Tensor,
        source_domain: int,
        target_domain: int,
        *,
        return_state: bool = False,
        update_memory: bool = True,
        memory_mode: str = "update",
    ) -> torch.Tensor | Tuple[torch.Tensor, HDIMAuxState]:
        """Transfer a problem encoding from source to target domain."""
        runtime = self._resolve_runtime_config(
            update_memory=update_memory,
            memory_mode=memory_mode,
        )
        src_name = self._domain_idx_to_name(source_domain)
        tgt_name = self._domain_idx_to_name(target_domain)
        output, transfer_state = self.pipeline.transfer(
            problem_encoding,
            src_name,
            tgt_name,
            update_memory=runtime.update_memory,
            memory_mode=runtime.memory_mode,
        )
        if return_state:
            aux_state = self._build_aux_state_from_transfer_state(
                transfer_state,
                runtime=runtime,
                dtype=problem_encoding.dtype,
                device=problem_encoding.device,
            )
            return output, aux_state
        return output

    def reset_memory(self) -> None:
        """Reset stateful HDIM memory and router replay state."""
        self.pipeline.reset_memory()
        with torch.no_grad():
            self.pipeline.moe.train_scores.fill_(1.0 / self.config.num_experts)

    def transfer_pairs(
        self,
        source_encoding: torch.Tensor,
        source_domain_id: torch.Tensor,
        target_domain_id: torch.Tensor,
        *,
        update_memory: bool = True,
        memory_mode: str = "update",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, HDIMAuxState]:
        """Run explicit paired transfer for mixed-domain batches."""
        source_domain_id = source_domain_id.to(device=source_encoding.device, dtype=torch.long)
        target_domain_id = target_domain_id.to(device=source_encoding.device, dtype=torch.long)
        runtime = self._resolve_runtime_config(
            update_memory=update_memory,
            memory_mode=memory_mode,
        )

        batch_size = source_encoding.shape[0]
        (
            raw_invariant,
            memory_augmented_invariant,
            exported_invariant,
            routing_weights,
            topk_idx,
            topk_gate_weights,
        ) = self._allocate_state_tensors(
            batch_size=batch_size,
            device=source_encoding.device,
            dtype=source_encoding.dtype,
        )
        train_scores_snapshot = torch.empty(
            self.config.num_experts,
            device=source_encoding.device,
            dtype=source_encoding.dtype,
        )
        expert_usage = torch.empty(
            self.config.num_experts,
            device=source_encoding.device,
            dtype=source_encoding.dtype,
        )
        output = torch.empty_like(source_encoding)
        memory_loss = torch.zeros((), device=source_encoding.device, dtype=source_encoding.dtype)
        router_loss = torch.zeros((), device=source_encoding.device, dtype=source_encoding.dtype)
        routing_entropy = torch.zeros((), device=source_encoding.device, dtype=source_encoding.dtype)
        memory_updated = False

        pair_keys = torch.stack((source_domain_id, target_domain_id), dim=1)
        unique_pairs = pair_keys.unique(dim=0)
        for pair in unique_pairs:
            src_idx, tgt_idx = int(pair[0].item()), int(pair[1].item())
            src_name = self._domain_idx_to_name(src_idx)
            tgt_name = self._domain_idx_to_name(tgt_idx)
            mask = (source_domain_id == src_idx) & (target_domain_id == tgt_idx)
            group_x = source_encoding[mask]
            group_output, transfer_state = self.pipeline.transfer(
                group_x,
                src_name,
                tgt_name,
                update_memory=runtime.update_memory,
                memory_mode=runtime.memory_mode,
            )
            output[mask] = group_output
            routing_weights[mask] = transfer_state["routing_weights"].to(dtype=source_encoding.dtype)
            topk_idx[mask] = transfer_state["router_state"]["topk_idx"].to(device=source_encoding.device)
            topk_gate_weights[mask] = transfer_state["router_state"]["topk_gate_weights"].to(dtype=source_encoding.dtype)
            exported_invariant[mask] = transfer_state["exported_invariant"].to(dtype=source_encoding.dtype)
            raw_invariant[mask] = transfer_state["raw_invariant"].to(dtype=source_encoding.dtype)
            memory_augmented_invariant[mask] = transfer_state["memory_augmented_invariant"].to(dtype=source_encoding.dtype)
            memory_loss = memory_loss + transfer_state["memory_loss"].to(dtype=source_encoding.dtype)
            router_loss = router_loss + transfer_state["router_state"]["router_loss"].to(dtype=source_encoding.dtype)
            routing_entropy = routing_entropy + transfer_state["router_state"]["routing_entropy"].to(dtype=source_encoding.dtype)
            train_scores_snapshot.copy_(transfer_state["router_state"]["train_scores_snapshot"].to(dtype=source_encoding.dtype))
            expert_usage.copy_(transfer_state["router_state"]["expert_usage"].to(dtype=source_encoding.dtype))
            memory_updated = memory_updated or bool(transfer_state["memory_updated"])

        invariant = self.training_inv_head(exported_invariant).to(dtype=source_encoding.dtype)
        aux_state = self._build_aux_state(
            raw_invariant=raw_invariant,
            memory_augmented_invariant=memory_augmented_invariant,
            exported_invariant=exported_invariant,
            routing_weights=routing_weights,
            topk_idx=topk_idx,
            topk_gate_weights=topk_gate_weights,
            train_scores_snapshot=train_scores_snapshot,
            expert_usage=expert_usage,
            routing_entropy=routing_entropy,
            memory_loss=memory_loss,
            router_loss=router_loss,
            memory_updated=memory_updated,
            runtime=runtime,
        )
        return output, routing_weights, invariant, aux_state
