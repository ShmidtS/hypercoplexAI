"""HDIM (Hypercomplex Domain-Invariant Model) — full PyTorch model definition.

This model wraps the full HDIMPipeline from src.core with a dataclass-based
configuration interface and adds integer-indexed domain routing for use in
batch training scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

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
    z_loss: torch.Tensor
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
            "z_loss": self.z_loss,
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
    clifford_p: int = 4  # Phase 25: Cl(4,1,0) dim=32 vs old Cl(3,1,0) dim=16
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

    def _moe_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run MoE forward with optional gradient checkpointing.

        checkpoint() requires tensor-only outputs. We compute the output tensor
        under checkpoint (for gradient savings), then re-run the forward without
        checkpoint to get the full router_state dict, but skip the EMA update
        on the second pass to avoid double-mutation.
        """
        if self.pipeline._use_gradient_checkpointing and self.training:
            moe = self.pipeline.moe

            def _moe_output_only(inp: torch.Tensor) -> torch.Tensor:
                """Forward returning only the output tensor (for checkpoint)."""
                orig_shape = inp.shape
                x_flat = inp.reshape(-1, inp.shape[-1])
                T = x_flat.shape[0]

                dispatch, combine = moe._compute_dispatch_combine(x_flat)
                slot_inputs = dispatch.T @ x_flat

                x_exp = slot_inputs.view(moe.num_experts, moe.slots_per_expert, -1)
                h = torch.einsum('esd,ehd->esh', x_exp, moe.W1_stack) + moe.b1_stack.unsqueeze(1)
                h = torch.nn.functional.gelu(h)
                h = torch.nn.functional.dropout(h, p=0.1, training=moe.training)
                out = torch.einsum('esh,edh->esd', h, moe.W2_stack) + moe.b2_stack.unsqueeze(1)
                slot_outputs = out.reshape(-1, slot_inputs.shape[-1])
                output = combine @ slot_outputs
                return output.reshape(orig_shape)

            output = checkpoint(_moe_output_only, x, use_reentrant=False)

            # Second pass: get router_state dict but skip EMA update
            # Temporarily disable training flag to prevent EMA mutation
            was_training = moe.training
            moe.eval()
            with torch.no_grad():
                _, router_state = moe(x)
            moe.train(was_training)
            # Recompute forward for the dict fields we need (cheap)
            _, router_state = moe(x)
            return output, router_state
        else:
            return self.pipeline.moe(x)

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
        z_loss: torch.Tensor,
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
            z_loss=z_loss,
            memory_updated=memory_updated,
            memory_mode=runtime.memory_mode,
            update_memory=runtime.update_memory,
        )

    def _build_aux_state_from_transfer_state(
        self,
        transfer_state: Dict[
            str, Union[torch.Tensor, bool, str, Dict[str, torch.Tensor]]
        ],
        *,
        runtime: HDIMRuntimeConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> HDIMAuxState:
        router_state = transfer_state["router_state"]
        return self._build_aux_state(
            raw_invariant=transfer_state["raw_invariant"].to(
                device=device, dtype=dtype
            ),
            memory_augmented_invariant=transfer_state["memory_augmented_invariant"].to(
                device=device, dtype=dtype
            ),
            exported_invariant=transfer_state["exported_invariant"].to(
                device=device, dtype=dtype
            ),
            routing_weights=transfer_state["routing_weights"].to(
                device=device, dtype=dtype
            ),
            topk_idx=router_state["topk_idx"].to(device=device),
            topk_gate_weights=router_state["topk_gate_weights"].to(
                device=device, dtype=dtype
            ),
            train_scores_snapshot=router_state["train_scores_snapshot"].to(
                device=device, dtype=dtype
            ),
            expert_usage=router_state["expert_usage"].to(device=device, dtype=dtype),
            routing_entropy=router_state["routing_entropy"].to(
                device=device, dtype=dtype
            ),
            memory_loss=transfer_state["memory_loss"].to(device=device, dtype=dtype),
            router_loss=router_state["router_loss"].to(device=device, dtype=dtype),
            z_loss=router_state.get(
                "z_loss", torch.zeros((), device=device, dtype=dtype)
            ).to(device=device, dtype=dtype),
            memory_updated=bool(transfer_state["memory_updated"]),
            runtime=runtime,
        )

    def _allocate_state_tensors(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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
        _num_experts = self.pipeline.moe.num_experts
        _top_k = self.pipeline.moe.top_k
        routing_weights = torch.empty(
            batch_size,
            _num_experts,
            device=device,
            dtype=dtype,
        )
        topk_idx = torch.empty(
            batch_size,
            _top_k,
            device=device,
            dtype=torch.long,
        )
        topk_gate_weights = torch.empty(
            batch_size,
            _top_k,
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
    ) -> (
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            HDIMAuxState,
        ]
    ):
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
        _num_experts_fwd = self.pipeline.moe.num_experts
        train_scores_snapshot = torch.empty(
            _num_experts_fwd,
            device=x.device,
            dtype=x.dtype,
        )
        expert_usage = torch.empty(
            _num_experts_fwd,
            device=x.device,
            dtype=x.dtype,
        )
        memory_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        router_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        z_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        routing_entropy = torch.zeros((), device=x.device, dtype=x.dtype)
        memory_updated = False

        # ── Batched domain processing ──────────────────────────────────
        # Encoder, memory, and decoder are domain-agnostic → run on full batch.
        # Invariant extraction and transfer use per-sample rotors → batch with
        # index_select into domain rotor tensors.
        # MoE softmax(dim=0) is group-dependent → iterate per domain group.
        pipeline = self.pipeline

        # 1) Encode full batch once (domain-agnostic encoder)
        g_source = pipeline.encoder(x)

        # 2) Build per-sample rotors via index_select
        rotors_n = torch.stack(
            [
                pipeline.domain_rotors[self._domain_names[i]]._normalized_R()
                for i in range(len(self._domain_names))
            ]
        )  # (num_domains, clifford_dim)
        rotors_inv = torch.stack(
            [
                pipeline.domain_rotors[self._domain_names[i]].get_inverse()
                for i in range(len(self._domain_names))
            ]
        )  # (num_domains, clifford_dim)
        R_per_sample = rotors_n[domain_id]  # (batch_size, clifford_dim)
        R_inv_per_sample = rotors_inv[domain_id]  # (batch_size, clifford_dim)

        # 3) Batched invariant extraction: u_inv = R^{-1} @ g_source @ R
        step1 = pipeline.algebra.geometric_product(R_inv_per_sample, g_source)
        u_inv = pipeline.algebra.geometric_product(step1, R_per_sample)
        u_inv = pipeline.invariant_norm(u_inv)

        # 4) Apply memory (domain-agnostic — operates on full batch)
        u_mem, memory_state = pipeline._apply_memory(
            u_inv,
            update_memory=runtime.update_memory,
            memory_mode=runtime.memory_mode,
        )

        # 5) MoE routing — must run per domain group because
        #    SoftMoERouter uses softmax(dim=0) across all input tokens,
        #    so mixing domains would contaminate dispatch weights.
        u_route = torch.empty_like(u_mem)
        for batch_domain_idx in domain_id.unique(sorted=True):
            mask = domain_id == batch_domain_idx
            group_route, group_router_state = pipeline.moe(u_mem[mask])
            u_route[mask] = group_route.to(dtype=u_route.dtype)

            # Scatter per-group router state into batch tensors
            routing_weights[mask] = group_router_state["gate_weights"].to(dtype=x.dtype)
            topk_idx[mask] = group_router_state["topk_idx"].to(device=x.device)
            topk_gate_weights[mask] = group_router_state["topk_gate_weights"].to(
                dtype=x.dtype
            )
            router_loss = router_loss + group_router_state["router_loss"].to(
                dtype=x.dtype
            )
            z_loss = z_loss + group_router_state.get(
                "z_loss", torch.zeros((), device=x.device, dtype=x.dtype)
            ).to(dtype=x.dtype)
            routing_entropy = routing_entropy + group_router_state[
                "routing_entropy"
            ].to(dtype=x.dtype)
            train_scores_snapshot.copy_(
                group_router_state["train_scores_snapshot"].to(dtype=x.dtype)
            )
            expert_usage.copy_(group_router_state["expert_usage"].to(dtype=x.dtype))

        # 6) Batched domain transfer: g_target = R @ u_route @ R^{-1}
        step2 = pipeline.algebra.geometric_product(R_per_sample, u_route)
        g_target = pipeline.algebra.geometric_product(step2, R_inv_per_sample)

        # 7) Decode full batch (domain-agnostic)
        output[:] = pipeline.decoder(g_target)

        # Collect remaining state
        exported_invariant[:] = u_route.to(dtype=x.dtype)
        raw_invariant[:] = u_inv.to(dtype=x.dtype)
        memory_augmented_invariant[:] = u_mem.to(dtype=x.dtype)
        memory_loss = memory_state.loss.to(dtype=x.dtype)
        memory_updated = bool(memory_state.updated)

        # B17 FIX: normalize accumulated losses by total samples
        # prevents bias when group sizes are uneven across domains
        total_samples = max(batch_size, 1)
        memory_loss = memory_loss / total_samples
        router_loss = router_loss / total_samples
        z_loss = z_loss / total_samples

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
                z_loss=z_loss,
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

    def add_domain(self, domain_name: str) -> None:
        """Add a new domain rotor to the pipeline in runtime.

        pipeline.add_domain appends to pipeline.domain_names which is the
        same list object as self._domain_names (shared reference from __init__).

        Args:
            domain_name: unique name for the new domain.
        """
        self.pipeline.add_domain(domain_name)
        # _domain_names is the same list object as pipeline.domain_names
        # so no separate append needed

    def remove_domain(self, domain_name: str) -> None:
        """Remove a domain from the pipeline.

        Args:
            domain_name: name of the domain to remove.
        """
        if domain_name not in self._domain_names:
            raise KeyError(f"Domain '{domain_name}' not found.")
        self.pipeline.remove_domain(domain_name)
        # pipeline.remove_domain already removes from pipeline.domain_names
        # which is the same list as self._domain_names

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on the pipeline's memory and MoE paths."""
        self.pipeline.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing on the pipeline."""
        self.pipeline.disable_gradient_checkpointing()

    # ── Phase 22 feature flags ──────────────────────────────────────
    def enable_gradient_surprise(self) -> None:
        """Enable gradient-based surprise metric in Titans memory (Titans, NeurIPS 2025)."""
        self.pipeline.memory.use_gradient_surprise = True

    def enable_adaptive_forgetting(self) -> None:
        """Enable adaptive forgetting based on surprise (high surprise = less forgetting)."""
        self.pipeline.memory.use_adaptive_forgetting = True

    def enable_router_calibration(self) -> None:
        """No-op: router calibration was removed (Phase 26 refactor)."""
        pass

    def enable_adaptive_expert_dropout(self) -> None:
        """No-op: adaptive expert dropout was removed (Phase 26 refactor)."""
        pass

    def enable_learnable_metric(self) -> None:
        """Enable learnable per-blade metric scaling in Clifford algebra (CliffordNet, 2026)."""
        self.pipeline.clifford.use_learnable_metric = True

    # ── Phase 26: MoE-Expert features ──────────────────────────────────
    def enable_shared_expert(self) -> None:
        """Enable DeepSeek-V3 always-on shared expert in SoftMoERouter."""
        self.pipeline.moe.enable_shared_expert()

    def enable_aux_loss_free(self, aux_lr: float = 0.001) -> None:
        """Enable Auxiliary-Loss-Free load balancing (DeepSeek-V3)."""
        self.pipeline.moe.enable_aux_loss_free(aux_lr=aux_lr)

    def enable_expert_ortho(self) -> None:
        """Enable expert orthogonalization loss (arXiv:2505.22323)."""
        self.pipeline.moe.enable_expert_ortho()

    def compute_expert_ortho_loss(self) -> "torch.Tensor":
        """Compute expert orthogonalization loss if enabled."""
        if self.pipeline.moe.use_expert_ortho:
            return self.pipeline.moe.expert_orthogonalization_loss()
        return torch.zeros((), device=next(self.parameters()).device)

    def reset_memory(self, strategy: str = "geometric") -> None:
        """Reset stateful HDIM memory and router replay state.

        strategy:
            'hard'      — полный сброс (только epoch=1 или новый trial)
            'geometric' — мягкое затухание (per-epoch, сохраняет паттерны)
            'stabilize' — только нормировка momentum (LR restart)
        """
        self.pipeline.reset_memory(strategy=strategy)
        with torch.no_grad():
            if hasattr(self.pipeline.moe, "train_scores"):
                n = self.pipeline.moe.num_experts
                if strategy == "hard":
                    # Полный сброс EMA весов роутера
                    self.pipeline.moe.train_scores.fill_(1.0 / n)
                elif strategy == "geometric":
                    # Мягкое затухание к равномерному (не теряем всю историю)
                    uniform = torch.full(
                        (n,),
                        1.0 / n,
                        device=self.pipeline.moe.train_scores.device,
                        dtype=self.pipeline.moe.train_scores.dtype,
                    )
                    self.pipeline.moe.train_scores.mul_(0.7).add_(uniform * 0.3)
                # 'stabilize': не трогаем train_scores

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
        source_domain_id = source_domain_id.to(
            device=source_encoding.device, dtype=torch.long
        )
        target_domain_id = target_domain_id.to(
            device=source_encoding.device, dtype=torch.long
        )
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
        _num_experts_tp = self.pipeline.moe.num_experts
        train_scores_snapshot = torch.empty(
            _num_experts_tp,
            device=source_encoding.device,
            dtype=source_encoding.dtype,
        )
        expert_usage = torch.empty(
            _num_experts_tp,
            device=source_encoding.device,
            dtype=source_encoding.dtype,
        )
        output = torch.empty_like(source_encoding)
        memory_loss = torch.zeros(
            (), device=source_encoding.device, dtype=source_encoding.dtype
        )
        router_loss = torch.zeros(
            (), device=source_encoding.device, dtype=source_encoding.dtype
        )
        z_loss = torch.zeros(
            (), device=source_encoding.device, dtype=source_encoding.dtype
        )
        routing_entropy = torch.zeros(
            (), device=source_encoding.device, dtype=source_encoding.dtype
        )
        memory_updated = False

        # ── Batched domain transfer processing ────────────────────────
        # Encoder, memory, and decoder are domain-agnostic → run on full batch.
        # Invariant extraction uses source rotors, transfer uses target rotors.
        # MoE softmax(dim=0) is group-dependent → iterate per (src, tgt) pair.
        pipeline = self.pipeline

        # 1) Encode full batch once (domain-agnostic encoder)
        g_source = pipeline.encoder(source_encoding)

        # 2) Build per-sample source/target rotors
        rotors_n = torch.stack(
            [
                pipeline.domain_rotors[self._domain_names[i]]._normalized_R()
                for i in range(len(self._domain_names))
            ]
        )
        rotors_inv = torch.stack(
            [
                pipeline.domain_rotors[self._domain_names[i]].get_inverse()
                for i in range(len(self._domain_names))
            ]
        )
        R_src_per_sample = rotors_n[source_domain_id]
        R_src_inv_per_sample = rotors_inv[source_domain_id]
        R_tgt_per_sample = rotors_n[target_domain_id]
        R_tgt_inv_per_sample = rotors_inv[target_domain_id]

        # 3) Batched invariant extraction: u_inv = R_src^{-1} @ g_source @ R_src
        step1 = pipeline.algebra.geometric_product(R_src_inv_per_sample, g_source)
        u_inv = pipeline.algebra.geometric_product(step1, R_src_per_sample)
        u_inv = pipeline.invariant_norm(u_inv)

        # 4) Apply memory (domain-agnostic)
        u_mem, memory_state = pipeline._apply_memory(
            u_inv,
            update_memory=runtime.update_memory,
            memory_mode=runtime.memory_mode,
        )

        # 5) MoE routing — per (src, tgt) pair group (softmax(dim=0) dependency)
        u_route = torch.empty_like(u_mem)
        pair_keys = torch.stack((source_domain_id, target_domain_id), dim=1)
        unique_pairs = pair_keys.unique(dim=0)
        for pair in unique_pairs:
            src_idx, tgt_idx = int(pair[0].item()), int(pair[1].item())
            mask = (source_domain_id == src_idx) & (target_domain_id == tgt_idx)
            group_route, group_router_state = pipeline.moe(u_mem[mask])
            u_route[mask] = group_route.to(dtype=u_route.dtype)

            routing_weights[mask] = group_router_state["gate_weights"].to(
                dtype=source_encoding.dtype
            )
            topk_idx[mask] = group_router_state["topk_idx"].to(
                device=source_encoding.device
            )
            topk_gate_weights[mask] = group_router_state["topk_gate_weights"].to(
                dtype=source_encoding.dtype
            )
            router_loss = router_loss + group_router_state["router_loss"].to(
                dtype=source_encoding.dtype
            )
            z_loss = z_loss + group_router_state.get(
                "z_loss",
                torch.zeros(
                    (), device=source_encoding.device, dtype=source_encoding.dtype
                ),
            ).to(dtype=source_encoding.dtype)
            routing_entropy = routing_entropy + group_router_state[
                "routing_entropy"
            ].to(dtype=source_encoding.dtype)
            train_scores_snapshot.copy_(
                group_router_state["train_scores_snapshot"].to(
                    dtype=source_encoding.dtype
                )
            )
            expert_usage.copy_(
                group_router_state["expert_usage"].to(dtype=source_encoding.dtype)
            )

        # 6) Batched domain transfer: g_target = R_tgt @ u_route @ R_tgt^{-1}
        step2 = pipeline.algebra.geometric_product(R_tgt_per_sample, u_route)
        g_target = pipeline.algebra.geometric_product(step2, R_tgt_inv_per_sample)

        # 7) Decode full batch (domain-agnostic)
        output[:] = pipeline.decoder(g_target)

        # Collect remaining state
        exported_invariant[:] = u_route.to(dtype=source_encoding.dtype)
        raw_invariant[:] = u_inv.to(dtype=source_encoding.dtype)
        memory_augmented_invariant[:] = u_mem.to(dtype=source_encoding.dtype)
        memory_loss = memory_state.loss.to(dtype=source_encoding.dtype)
        memory_updated = bool(memory_state.updated)

        # B17 FIX: normalize accumulated losses by total samples
        # prevents bias when group sizes are uneven across domain pairs
        total_samples = max(batch_size, 1)
        memory_loss = memory_loss / total_samples
        router_loss = router_loss / total_samples
        z_loss = z_loss / total_samples

        invariant = self.training_inv_head(exported_invariant).to(
            dtype=source_encoding.dtype
        )
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
            z_loss=z_loss,
            memory_updated=memory_updated,
            runtime=runtime,
        )
        return output, routing_weights, invariant, aux_state
