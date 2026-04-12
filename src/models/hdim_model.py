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
from src.models.results import CoreResult, ForwardResult


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
    hallucination_risk: float = 0.0
    memory_surprise: float | None = None
    feedback_action: str | None = None  # Phase 33: Hallucination feedback action
    online_loss: torch.Tensor = torch.tensor(0.0)  # Phase 31: Online learning loss
    online_updated: bool = False  # Phase 31: Whether online update fired

    def to_dict(self) -> Dict[str, Union[torch.Tensor, bool, str, float, None]]:
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
            "hallucination_risk": self.hallucination_risk,
            "memory_surprise": self.memory_surprise,
            "online_loss": self.online_loss,
            "online_updated": self.online_updated,
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
        num_experts: Number of MoE experts in the R3MoERouter. If None, computed from expert_names.
        dropout: Dropout probability applied after the encoder.
        clifford_p: Positive basis vectors for Cl_{p,q,r} algebra.
        clifford_q: Negative basis vectors.
        clifford_r: Nilpotent basis vectors.
        top_k: Number of active experts per token.
        memory_key_dim: Dimensionality of Titans memory keys.
        memory_type: Memory module type: titans | hippocampus | neocortex | cls | hbma | msa.
        domain_names: Explicit domain name list. If None, auto-generates
            ['domain_0', 'domain_1', ...] up to num_domains.
        expert_names: Explicit expert name list. If provided, num_experts is computed from it.
    """

    hidden_dim: int = 64
    num_domains: int = 4
    num_experts: Optional[int] = None  # None -> computed from expert_names or default
    dropout: float = 0.1
    clifford_p: int = 3  # Cl(3,1,0) dim=16, matches CliffordInteractionLayer default
    clifford_q: int = 1
    clifford_r: int = 0
    top_k: int = 2
    memory_key_dim: int = 32
    memory_type: str = "titans"  # titans | hippocampus | neocortex | cls | hbma | msa
    domain_names: Optional[List[str]] = None
    expert_names: Optional[List[str]] = None  # New field for dynamic expert names
    text: HDIMTextConfig = field(default_factory=HDIMTextConfig)
    # Phase 31: Self-Evolution (Online Learning)
    online_learning: bool = False  # Enable online TTT learning
    online_replay_size: int = 10000  # Experience replay buffer size
    online_surprise_threshold: float = 0.3  # Surprise threshold for updates
    online_ttt_lr: float = 1e-5  # TTT learning rate
    online_gradient_mode: str = "detached" # Gradient mode: detached | selective | full
    online_gradient_scale: float = 0.1 # Gradient scaling factor for selective/full modes
    # Phase 32: Hallucination Detection
    hallucination_detection: bool = False  # Enable hallucination detection
    hallucination_risk_threshold: float = 0.5  # Risk threshold for hallucination flag
    # Phase 33: Hallucination Feedback Loop (Self-Correction)
    hallucination_feedback: bool = False # Enable hallucination feedback loop
    hallucination_feedback_config: Optional[dict] = None # Override default feedback config
    # MSA memory configuration
    msa_num_prototypes: int = 256
    msa_top_k: int = 16
    msa_chunk_size: int = 64
    msa_num_heads: int = 4
    msa_temperature: float = 0.07
    msa_ema_momentum: float = 0.995
    msa_overflow_capacity: int = 10000
    msa_max_hops: int = 3
    msa_interleave_threshold: float = 0.5
    msa_compression_threshold: int = 128
    msa_diversity_loss_weight: float = 1.0

    def __post_init__(self):
        # Compute num_experts from expert_names if provided
        if self.expert_names is not None:
            computed = len(self.expert_names)
            if self.num_experts is not None and self.num_experts != computed:
                raise ValueError(
                    f"num_experts={self.num_experts} conflicts with "
                    f"len(expert_names)={computed}"
                )
            self.num_experts = computed
        elif self.num_experts is None:
            self.num_experts = 4  # sensible default

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

    Phase 31: Supports self-evolution via OnlineLearner when config.online_learning=True.
    """

    def __init__(self, config: HDIMConfig) -> None:
        super().__init__()
        self.config = config
        self._domain_names: List[str] = config.get_domain_names()

        msa_config = {
            'num_prototypes': config.msa_num_prototypes,
            'top_k': config.msa_top_k,
            'chunk_size': config.msa_chunk_size,
            'num_heads': config.msa_num_heads,
            'temperature': config.msa_temperature,
            'ema_momentum': config.msa_ema_momentum,
            'overflow_capacity': config.msa_overflow_capacity,
            'max_hops': config.msa_max_hops,
            'interleave_threshold': config.msa_interleave_threshold,
            'compression_threshold': config.msa_compression_threshold,
            'diversity_loss_weight': config.msa_diversity_loss_weight,
        }

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
            memory_type=config.memory_type,
            msa_config=msa_config,
        )

        self.dropout = nn.Dropout(config.dropout)
        clifford_dim = self.pipeline.clifford_dim
        self.training_inv_head = nn.Linear(clifford_dim, config.hidden_dim)

        # Phase 31: Online Learner for self-evolution (uses moe output dimension)
        self.online_learner = None
        if config.online_learning:
            from src.core.online_learner import OnlineLearner, GradientMode
            # Parse gradient mode from string
            gradient_mode = GradientMode(config.online_gradient_mode)
            self.online_learner = OnlineLearner(
                hidden_dim=clifford_dim,
                num_experts=config.num_experts or 4,
                replay_buffer_size=config.online_replay_size,
                surprise_threshold=config.online_surprise_threshold,
                ttt_lr=config.online_ttt_lr,
                gradient_mode=gradient_mode,
                gradient_scale=config.online_gradient_scale,
            )

        # Phase 32: Hallucination Detector (optional)
        self.hallucination_detector = None
        if config.hallucination_detection:
            from src.core.hallucination_detector import HallucinationDetector
            self.hallucination_detector = HallucinationDetector(
                num_experts=config.num_experts or 4,
                risk_threshold=config.hallucination_risk_threshold,
            )


        # Phase 33: Hallucination Feedback Loop (optional)
        self.hallucination_feedback_loop = None
        if config.hallucination_feedback:
            from src.core.hallucination_feedback import HallucinationFeedbackLoop
            # Get expert names from pipeline MoE
            expert_names = config.expert_names or [f"expert_{i}" for i in range(config.num_experts or 4)]
            self.hallucination_feedback_loop = HallucinationFeedbackLoop(
                expert_names=expert_names,
                enabled=True,
            )
        # Rotor stacks rebuild each forward (requires_grad rotors need fresh graph)

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
        z_loss: torch.Tensor,
        memory_updated: bool,
        runtime: HDIMRuntimeConfig,
        hallucination_risk: float = 0.0,
        memory_surprise: float | None = None,
        feedback_action: str | None = None,
        online_loss: torch.Tensor | None = None,
        online_updated: bool = False,
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
            hallucination_risk=hallucination_risk,
            memory_surprise=memory_surprise,
            feedback_action=feedback_action,
            online_loss=online_loss if online_loss is not None else torch.tensor(0.0),
            online_updated=online_updated,
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
        raw_invariant = torch.zeros(
            batch_size, self.pipeline.clifford_dim, device=device, dtype=dtype,
        )
        memory_augmented_invariant = torch.zeros(
            batch_size, self.pipeline.clifford_dim, device=device, dtype=dtype,
        )
        exported_invariant = torch.zeros(
            batch_size, self.pipeline.clifford_dim, device=device, dtype=dtype,
        )
        _num_experts = self.pipeline.moe.num_experts
        _top_k = self.pipeline.moe.top_k
        routing_weights = torch.zeros(
            batch_size, _num_experts, device=device, dtype=dtype,
        )
        topk_idx = torch.zeros(
            batch_size, _top_k, device=device, dtype=torch.long,
        )
        topk_gate_weights = torch.zeros(
            batch_size, _top_k, device=device, dtype=dtype,
        )
        return (
            raw_invariant,
            memory_augmented_invariant,
            exported_invariant,
            routing_weights,
            topk_idx,
            topk_gate_weights,
        )

    def _build_rotor_stacks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build stacked normalized and inverse rotor tensors for all domains.

        Rebuilt each forward pass to ensure correct autograd flow into
        domain rotor parameters.
        """
        pipeline = self.pipeline
        rotors_n = torch.stack(
            [pipeline.domain_rotors[self._domain_names[i]]._normalized_R()
             for i in range(len(self._domain_names))]
        )
        rotors_inv = torch.stack(
            [pipeline.domain_rotors[self._domain_names[i]].get_inverse()
             for i in range(len(self._domain_names))]
        )
        return rotors_n, rotors_inv

    def _forward_core(
        self,
        x: torch.Tensor,
        R_inv_extractor: torch.Tensor,
        R_extractor: torch.Tensor,
        R_transfer: torch.Tensor,
        R_transfer_inv: torch.Tensor,
        group_masks: List[torch.Tensor],
        runtime: HDIMRuntimeConfig,
    ) -> CoreResult:
        """Shared core for forward and transfer_pairs.

        Runs encode, invariant extraction, memory, MoE routing, transfer, decode.

        Args:
            x: Input tensor (batch_size, hidden_dim).
            R_inv_extractor: Inverse rotor for invariant extraction (per-sample).
            R_extractor: Forward rotor for invariant extraction (per-sample).
            R_transfer: Forward rotor for domain transfer (per-sample).
            R_transfer_inv: Inverse rotor for domain transfer (per-sample).
            group_masks: Boolean masks defining MoE routing groups.
            runtime: Memory lifecycle configuration.
        """
        batch_size = x.shape[0]
        device, dtype = x.device, x.dtype
        pipeline = self.pipeline

        output = torch.empty_like(x)
        (
            raw_invariant, memory_augmented_invariant, exported_invariant,
            routing_weights, topk_idx, topk_gate_weights,
        ) = self._allocate_state_tensors(batch_size=batch_size, device=device, dtype=dtype)

        _num_experts = pipeline.moe.num_experts
        train_scores_snapshot = torch.zeros(_num_experts, device=device, dtype=dtype)
        expert_usage = torch.zeros(_num_experts, device=device, dtype=dtype)

        # 1) Encode
        g_source = pipeline.encoder(x)

        # 2) Invariant extraction
        step1 = pipeline.algebra.geometric_product(R_inv_extractor, g_source)
        u_inv = pipeline.algebra.geometric_product(step1, R_extractor)
        u_inv = pipeline.invariant_norm(u_inv)

        # 3) Memory
        u_mem, memory_state = pipeline._apply_memory(
            u_inv,
            update_memory=runtime.update_memory,
            memory_mode=runtime.memory_mode,
        )

        # Phase 31: Online Learning (Self-Evolution)
        # Uses gradient mode from config: DETACHED (safe), SELECTIVE (replay only), FULL (experimental)
        online_loss = torch.tensor(0.0, device=device, dtype=dtype)
        online_updated = False
        if self.online_learner is not None and self.training:
            with torch.no_grad():
                u_mean = u_mem.mean(dim=0, keepdim=True).detach()
                moe = pipeline.moe
                if hasattr(moe, 'dispatch_proj'):
                    gate_logits = moe.dispatch_proj(u_mean)
                    slot_idx = int(gate_logits.argmax(dim=-1).item())
                    dominant_expert = slot_idx // getattr(moe, 'slots_per_expert', 1)
                elif hasattr(moe, 'router_proj'):
                    gate_logits = moe.router_proj(u_mean)
                    dominant_expert = int(gate_logits.argmax(dim=-1).item() % (self.config.num_experts or 4))
                elif hasattr(moe, 'kernel') and hasattr(moe.kernel, 'router_proj'):
                    # MoEKernelAdapter wraps MoEKernel as self.kernel
                    gate_logits = moe.kernel.router_proj(u_mean)
                    dominant_expert = int(gate_logits.argmax(dim=-1).item() % (self.config.num_experts or 4))
                else:
                    dominant_expert = 0

            # Use gradient-mode-aware update method
            _loss, online_updated, _ = self.online_learner.online_update_with_mode(
                x=u_mem,
                expert_idx=dominant_expert,
                model=self,
            )
            online_loss = _loss

        # 4) MoE routing per group
        u_route = torch.zeros_like(u_mem)
        router_loss = torch.zeros((), device=device, dtype=dtype)
        z_loss = torch.zeros((), device=device, dtype=dtype)
        routing_entropy = torch.zeros((), device=device, dtype=dtype)

        all_slot_outputs: List[torch.Tensor] = []
        for mask in group_masks:
            # Guard: skip groups with 0 elements to avoid empty MoE input
            if mask.sum().item() == 0:
                continue
            group_route, group_router_state = pipeline.moe(u_mem[mask])
            u_route[mask] = group_route.to(dtype=u_route.dtype)

            routing_weights[mask] = group_router_state["gate_weights"].to(dtype=dtype)
            topk_idx[mask] = group_router_state["topk_idx"].to(device=device)
            topk_gate_weights[mask] = group_router_state["topk_gate_weights"].to(dtype=dtype)
            router_loss = router_loss + group_router_state["router_loss"].to(dtype=dtype)
            z_loss = z_loss + group_router_state.get(
                "z_loss", torch.zeros((), device=device, dtype=dtype)
            ).to(dtype=dtype)
            routing_entropy = routing_entropy + group_router_state["routing_entropy"].to(dtype=dtype)
            train_scores_snapshot = train_scores_snapshot + group_router_state["train_scores_snapshot"].to(dtype=dtype)
            expert_usage = expert_usage + group_router_state["expert_usage"].to(dtype=dtype)

            # Collect slot_outputs from MoEKernel if present
            if "slot_outputs" in group_router_state and group_router_state["slot_outputs"] is not None:
                all_slot_outputs.append(group_router_state["slot_outputs"])

        # Normalize per-group accumulations by number of groups
        num_groups = len(group_masks)
        if num_groups > 1:
            routing_entropy = routing_entropy / num_groups
            train_scores_snapshot = train_scores_snapshot / num_groups
            expert_usage = expert_usage / num_groups

        # 5) Transfer
        step2 = pipeline.algebra.geometric_product(R_transfer, u_route)
        g_target = pipeline.algebra.geometric_product(step2, R_transfer_inv)

        # 6) Decode
        output[:] = pipeline.decoder(g_target)

        exported_invariant[:] = u_route.to(dtype=dtype)
        raw_invariant[:] = u_inv.to(dtype=dtype)
        memory_augmented_invariant[:] = u_mem.to(dtype=dtype)

        memory_loss = memory_state.loss.to(dtype=dtype)
        memory_updated = bool(memory_state.updated)

        # Note: memory_loss, router_loss, z_loss are already batch-meaned
        # by their respective modules — do NOT divide by total_samples again

        # Phase 32: Hallucination detection
        hallucination_risk = 0.0
        memory_surprise_val = None
        if self.hallucination_detector is not None:
            memory_surprise_val = memory_state.surprise.mean().item() if hasattr(memory_state, 'surprise') and memory_state.surprise is not None else None
            result = self.hallucination_detector.from_router_state(
            router_state={
            "routing_entropy": routing_entropy,
            "gate_weights": routing_weights,
            "topk_gate_weights": topk_gate_weights,
            },
            memory_mismatch=torch.tensor(memory_surprise_val) if memory_surprise_val is not None else None,
            memory_loss=memory_loss,
            )
            hallucination_risk = result.hallucination_risk

        # Phase 33: Hallucination Feedback Loop (Self-Correction)
        feedback_action = None
        if self.hallucination_feedback_loop is not None:
            # Get current dominant expert from topk_idx (batch mode)
            current_expert_idx = topk_idx[:, 0].mode().values.item() if topk_idx.numel() > 0 else 0
            expert_names = self.pipeline.moe.expert_names if hasattr(self.pipeline.moe, 'expert_names') else [f'expert_{i}' for i in range(self.config.num_experts or 4)]
            current_expert = expert_names[current_expert_idx] if current_expert_idx < len(expert_names) else expert_names[0]
            
            feedback_result = self.hallucination_feedback_loop.check_and_respond(
                risk_score=hallucination_risk,
                routing_info={
                    'current_expert': current_expert,
                    'expert_weights': routing_weights.mean(dim=0) if routing_weights.numel() > 0 else None,
                },
                base_confidence=1.0,
            )
            feedback_action = feedback_result.action.value
            
            # Update expert hallucination history
            self.hallucination_feedback_loop.update_expert_hallucination_history(
                current_expert, 
                hallucination_risk > 0.5
            )

        # Concatenate slot_outputs if available
        slot_outputs_tensor = torch.cat(all_slot_outputs, dim=0) if all_slot_outputs else None

        return CoreResult(
            output=output,
            routing_weights=routing_weights,
            raw_invariant=raw_invariant,
            memory_augmented_invariant=memory_augmented_invariant,
            exported_invariant=exported_invariant,
            topk_idx=topk_idx,
            topk_gate_weights=topk_gate_weights,
            train_scores_snapshot=train_scores_snapshot,
            expert_usage=expert_usage,
            routing_entropy=routing_entropy,
            memory_loss=memory_loss,
            router_loss=router_loss,
            z_loss=z_loss,
            memory_updated=memory_updated,
            slot_outputs=slot_outputs_tensor,
            hallucination_risk=hallucination_risk,
            memory_surprise=memory_surprise_val,
            feedback_action=feedback_action,
            online_loss=online_loss,
            online_updated=online_updated,
        )

    def forward(
        self,
        x: torch.Tensor,
        domain_id: torch.Tensor,
        *,
        return_state: bool = False,
        update_memory: bool = True,
        memory_mode: str = "update",
    ) -> ForwardResult:
        """Run the HDIM forward pass for same-domain reconstruction batches."""
        x = self.dropout(x)
        domain_id = domain_id.to(device=x.device, dtype=torch.long)
        runtime = self._resolve_runtime_config(
            update_memory=update_memory,
            memory_mode=memory_mode,
        )

        rotors_n, rotors_inv = self._build_rotor_stacks()
        R_per_sample = rotors_n[domain_id]
        R_inv_per_sample = rotors_inv[domain_id]

        group_masks = [domain_id == idx for idx in domain_id.unique(sorted=True)]

        core = self._forward_core(
            x, R_inv_per_sample, R_per_sample, R_per_sample, R_inv_per_sample,
            group_masks, runtime,
        )

        invariant = self.training_inv_head(core.exported_invariant).to(dtype=x.dtype)

        aux_state = None
        if return_state:
            aux_state = self._build_aux_state(
                raw_invariant=core.raw_invariant,
                memory_augmented_invariant=core.memory_augmented_invariant,
                exported_invariant=core.exported_invariant,
                routing_weights=core.routing_weights,
                topk_idx=core.topk_idx,
                topk_gate_weights=core.topk_gate_weights,
                train_scores_snapshot=core.train_scores_snapshot,
                expert_usage=core.expert_usage,
                routing_entropy=core.routing_entropy,
                memory_loss=core.memory_loss,
                router_loss=core.router_loss,
                z_loss=core.z_loss,
                memory_updated=core.memory_updated,
                runtime=runtime,
                hallucination_risk=core.hallucination_risk,
                memory_surprise=core.memory_surprise,
                feedback_action=core.feedback_action,
                online_loss=core.online_loss,
                online_updated=core.online_updated,
            )
        return ForwardResult(
            output=core.output,
            routing_weights=core.routing_weights,
            invariant=invariant,
            slot_outputs=core.slot_outputs,
            aux_state=aux_state,
        )

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
        """Add a new domain rotor to the pipeline in runtime."""
        self.pipeline.add_domain(domain_name)

    def reset_memory(self, strategy: str = "geometric") -> None:
        """Reset stateful HDIM memory and router replay state.

        strategy:
            'hard'      -- full reset (only epoch=1 or new trial)
            'geometric' -- soft decay (per-epoch, preserves patterns)
            'stabilize' -- only momentum normalization (LR restart)
        """
        self.pipeline.reset_memory(strategy=strategy)
        with torch.no_grad():
            if hasattr(self.pipeline.moe, "train_scores"):
                n = self.pipeline.moe.num_experts
                if strategy == "hard":
                    self.pipeline.moe.train_scores.fill_(1.0 / n)
                elif strategy == "geometric":
                    uniform = torch.full(
                        (n,),
                        1.0 / n,
                        device=self.pipeline.moe.train_scores.device,
                        dtype=self.pipeline.moe.train_scores.dtype,
                    )
                    self.pipeline.moe.train_scores.mul_(0.7).add_(uniform * 0.3)
                # 'stabilize': leave train_scores unchanged

    def transfer_pairs(
        self,
        source_encoding: torch.Tensor,
        source_domain_id: torch.Tensor,
        target_domain_id: torch.Tensor,
        *,
        update_memory: bool = True,
        memory_mode: str = "update",
    ) -> ForwardResult:
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

        rotors_n, rotors_inv = self._build_rotor_stacks()
        R_src_per_sample = rotors_n[source_domain_id]
        R_src_inv_per_sample = rotors_inv[source_domain_id]
        R_tgt_per_sample = rotors_n[target_domain_id]
        R_tgt_inv_per_sample = rotors_inv[target_domain_id]

        pair_keys = torch.stack((source_domain_id, target_domain_id), dim=1)
        unique_pairs = pair_keys.unique(dim=0)
        group_masks = [
            (source_domain_id == int(p[0].item())) & (target_domain_id == int(p[1].item()))
            for p in unique_pairs
        ]

        core = self._forward_core(
            source_encoding,
            R_src_inv_per_sample, R_src_per_sample,
            R_tgt_per_sample, R_tgt_inv_per_sample,
            group_masks, runtime,
        )

        invariant = self.training_inv_head(core.exported_invariant).to(
            dtype=source_encoding.dtype
        )

        aux_state = self._build_aux_state(
            raw_invariant=core.raw_invariant,
            memory_augmented_invariant=core.memory_augmented_invariant,
            exported_invariant=core.exported_invariant,
            routing_weights=core.routing_weights,
            topk_idx=core.topk_idx,
            topk_gate_weights=core.topk_gate_weights,
            train_scores_snapshot=core.train_scores_snapshot,
            expert_usage=core.expert_usage,
            routing_entropy=core.routing_entropy,
            memory_loss=core.memory_loss,
            router_loss=core.router_loss,
            z_loss=core.z_loss,
            memory_updated=core.memory_updated,
            runtime=runtime,
            hallucination_risk=core.hallucination_risk,
            memory_surprise=core.memory_surprise,
            feedback_action=core.feedback_action,
            online_loss=core.online_loss,
            online_updated=core.online_updated,
        )
        return ForwardResult(
            output=core.output,
            routing_weights=core.routing_weights,
            invariant=invariant,
            slot_outputs=core.slot_outputs,
            aux_state=aux_state,
        )

    # Phase 22 feature flags

    def enable_gradient_surprise(self) -> None:
        """Enable gradient-based surprise metric in Titans memory."""
        if hasattr(self.pipeline.memory, 'use_gradient_surprise'):
            self.pipeline.memory.use_gradient_surprise = True

    def enable_adaptive_forgetting(self) -> None:
        """Enable adaptive forgetting based on surprise."""
        if hasattr(self.pipeline.memory, 'use_adaptive_forgetting'):
            self.pipeline.memory.use_adaptive_forgetting = True

    def enable_learnable_metric(self) -> None:
        """Enable learnable per-blade metric scaling in Clifford algebra."""
        if hasattr(self.pipeline.algebra, 'enable_learnable_metric'):
            self.pipeline.algebra.enable_learnable_metric()

    # Phase 26 feature flags

    def enable_shared_expert(self) -> None:
        """Enable DeepSeek-V3 always-on shared expert in SoftMoERouter."""
        if hasattr(self.pipeline.moe, 'enable_shared_expert'):
            self.pipeline.moe.enable_shared_expert()

    def enable_aux_loss_free(self, aux_lr: float = 0.001) -> None:
        """Enable Auxiliary-Loss-Free load balancing (DeepSeek-V3)."""
        if hasattr(self.pipeline.moe, 'enable_aux_loss_free'):
            self.pipeline.moe.enable_aux_loss_free(aux_lr=aux_lr)

    def enable_expert_ortho(self) -> None:
        """Enable expert orthogonalization loss."""
        if hasattr(self.pipeline.moe, 'enable_expert_ortho'):
            self.pipeline.moe.enable_expert_ortho()

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on pipeline."""
        self.pipeline.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing on pipeline."""
        self.pipeline.disable_gradient_checkpointing()
