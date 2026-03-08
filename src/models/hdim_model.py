"""HDIM (Hypercomplex Domain-Invariant Model) — full PyTorch model definition.

This model wraps the full HDIMPipeline from src.core with a dataclass-based
configuration interface and adds integer-indexed domain routing for use in
batch training scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.core.hdim_pipeline import HDIMPipeline


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
        self.raw_inv_head = nn.Linear(clifford_dim, config.hidden_dim)
        self.training_inv_head = nn.Linear(clifford_dim, config.hidden_dim)

    def _domain_idx_to_name(self, domain_idx: int) -> str:
        """Convert an integer domain index to its registered name."""
        if domain_idx < 0 or domain_idx >= len(self._domain_names):
            raise IndexError(
                f"domain_idx {domain_idx} out of range [0, {len(self._domain_names)})."
            )
        return self._domain_names[domain_idx]

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
        Dict[str, torch.Tensor],
    ]:
        """Run the HDIM forward pass for same-domain reconstruction batches."""
        x = self.dropout(x)
        domain_id = domain_id.to(device=x.device, dtype=torch.long)

        batch_size = x.shape[0]
        output = torch.empty_like(x)
        routing_weights = torch.empty(
            batch_size,
            self.config.num_experts,
            device=x.device,
            dtype=x.dtype,
        )
        invariant = torch.empty(
            batch_size,
            self.config.hidden_dim,
            device=x.device,
            dtype=x.dtype,
        )
        exported_invariant = torch.empty(
            batch_size,
            self.pipeline.clifford_dim,
            device=x.device,
            dtype=x.dtype,
        )
        raw_invariant = torch.empty(
            batch_size,
            self.pipeline.clifford_dim,
            device=x.device,
            dtype=x.dtype,
        )
        memory_augmented_invariant = torch.empty(
            batch_size,
            self.pipeline.clifford_dim,
            device=x.device,
            dtype=x.dtype,
        )
        memory_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        router_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        for batch_domain_idx in domain_id.unique(sorted=True):
            domain_idx = int(batch_domain_idx.item())
            domain_name = self._domain_idx_to_name(domain_idx)
            mask = domain_id == batch_domain_idx
            group_x = x[mask]
            group_output, transfer_state = self.pipeline.transfer(
                group_x,
                domain_name,
                domain_name,
                update_memory=update_memory,
                memory_mode=memory_mode,
            )

            output[mask] = group_output
            routing_weights[mask] = transfer_state["routing_weights"].to(dtype=x.dtype)
            invariant[mask] = self.training_inv_head(transfer_state["training_invariant"]).to(dtype=x.dtype)
            processed_invariant[mask] = transfer_state["processed_invariant"].to(dtype=x.dtype)
            raw_invariant[mask] = transfer_state["raw_invariant"].to(dtype=x.dtype)
            training_invariant[mask] = transfer_state["training_invariant"].to(dtype=x.dtype)
            memory_loss = memory_loss + transfer_state["memory_loss"].to(dtype=x.dtype)
            router_loss = router_loss + transfer_state["router_state"]["router_loss"].to(dtype=x.dtype)

        if return_state:
            aux_state: Dict[str, torch.Tensor] = {
                "memory_loss": memory_loss,
                "router_loss": router_loss,
                "processed_invariant": processed_invariant,
                "raw_invariant": raw_invariant,
                "training_invariant": training_invariant,
            }
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
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor | Dict[str, torch.Tensor]]]:
        """Transfer a problem encoding from source to target domain."""
        src_name = self._domain_idx_to_name(source_domain)
        tgt_name = self._domain_idx_to_name(target_domain)
        output, transfer_state = self.pipeline.transfer(
            problem_encoding,
            src_name,
            tgt_name,
            update_memory=update_memory,
            memory_mode=memory_mode,
        )
        if return_state:
            return output, transfer_state
        return output

    def transfer_pairs(
        self,
        source_encoding: torch.Tensor,
        source_domain_id: torch.Tensor,
        target_domain_id: torch.Tensor,
        *,
        update_memory: bool = True,
        memory_mode: str = "update",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Run explicit paired transfer for mixed-domain batches."""
        source_domain_id = source_domain_id.to(device=source_encoding.device, dtype=torch.long)
        target_domain_id = target_domain_id.to(device=source_encoding.device, dtype=torch.long)

        batch_size = source_encoding.shape[0]
        output = torch.empty_like(source_encoding)
        routing_weights = torch.empty(
            batch_size,
            self.config.num_experts,
            device=source_encoding.device,
            dtype=source_encoding.dtype,
        )
        invariant = torch.empty(
            batch_size,
            self.config.hidden_dim,
            device=source_encoding.device,
            dtype=source_encoding.dtype,
        )
        processed_invariant = torch.empty(
            batch_size,
            self.pipeline.clifford_dim,
            device=source_encoding.device,
            dtype=source_encoding.dtype,
        )
        raw_invariant = torch.empty(
            batch_size,
            self.pipeline.clifford_dim,
            device=source_encoding.device,
            dtype=source_encoding.dtype,
        )
        training_invariant = torch.empty(
            batch_size,
            self.pipeline.clifford_dim,
            device=source_encoding.device,
            dtype=source_encoding.dtype,
        )
        memory_loss = torch.zeros((), device=source_encoding.device, dtype=source_encoding.dtype)
        router_loss = torch.zeros((), device=source_encoding.device, dtype=source_encoding.dtype)

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
                update_memory=update_memory,
                memory_mode=memory_mode,
            )
            output[mask] = group_output
            routing_weights[mask] = transfer_state["routing_weights"].to(dtype=source_encoding.dtype)
            invariant[mask] = self.training_inv_head(transfer_state["training_invariant"]).to(dtype=source_encoding.dtype)
            processed_invariant[mask] = transfer_state["processed_invariant"].to(dtype=source_encoding.dtype)
            raw_invariant[mask] = transfer_state["raw_invariant"].to(dtype=source_encoding.dtype)
            training_invariant[mask] = transfer_state["training_invariant"].to(dtype=source_encoding.dtype)
            memory_loss = memory_loss + transfer_state["memory_loss"].to(dtype=source_encoding.dtype)
            router_loss = router_loss + transfer_state["router_state"]["router_loss"].to(dtype=source_encoding.dtype)

        aux_state: Dict[str, torch.Tensor] = {
            "memory_loss": memory_loss,
            "router_loss": router_loss,
            "processed_invariant": processed_invariant,
            "raw_invariant": raw_invariant,
            "training_invariant": training_invariant,
        }
        return output, routing_weights, invariant, aux_state
