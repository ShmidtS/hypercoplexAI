"""Hallucination Detector — 5-signal weighted risk for artifact detection.

Concept: LLM hallucinations are compression artifacts. HDIM detects them via:
1. Routing Entropy — high entropy = MoE uncertain about expert assignment
2. MoE Confidence — low max weight = no expert strongly "owns" this input
3. Memory Mismatch — large distance to stored patterns = unseen/unknown input
4. EigenScore — high singular value variance = high uncertainty (INSIDE paper)
5. Semantic Entropy — diversity of generated meanings from hidden states

High hallucination risk: high entropy + low confidence + high mismatch + high eigen + high semantic
Low hallucination risk: low entropy + high confidence + low mismatch + low eigen + low semantic
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .semantic_entropy_probe import SemanticEntropyProbe


@dataclass
class HallucinationDetectionResult:
    """Per-sample hallucination detection output."""

    routing_entropy: float  # Shannon entropy of MoE gate distribution
    moe_confidence: float  # Max expert weight (softmax prob)
    memory_mismatch: float  # Surprise: ||d(loss)/d(key)|| norm
    memory_loss: float  # MSE between memory prediction and target
    eigen_score: float  # Inverse singular value sum (high = uncertain)
    semantic_entropy: float  # Diversity of semantic generations
    hallucination_risk: float  # Combined score [0, 1]
    is_potential_hallucination: bool  # Thresholded decision
    evidence_count: int = 0  # How many of 5 signals exceed their thresholds

    def to_dict(self) -> dict:
        return {
            "routing_entropy": self.routing_entropy,
            "moe_confidence": self.moe_confidence,
            "memory_mismatch": self.memory_mismatch,
            "memory_loss": self.memory_loss,
            "eigen_score": self.eigen_score,
            "semantic_entropy": self.semantic_entropy,
            "hallucination_risk": self.hallucination_risk,
            "is_potential_hallucination": self.is_potential_hallucination,
            "evidence_count": self.evidence_count,
        }


class HallucinationDetector(nn.Module):
    """Detects potential hallucinations via MoE routing entropy and memory signals.

    Uses five independent signals from the HDIM pipeline:
    - routing_entropy: from MoE router (soft_moe_router.py / moe_kernel.py)
    - moe_confidence: max gate weight from MoE routing
    - memory_mismatch: surprise from Titans memory gradient
    - memory_loss: MSE from memory retrieval
    - eigen_score: singular value analysis of routing representations
    - semantic_entropy: from SemanticEntropyProbe on hidden states

    Combined via learnable weighted sum into hallucination_risk [0, 1].
    """

    def __init__(
        self,
        num_experts: int = 4,
        hidden_dim: int = 512,
        entropy_threshold: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        mismatch_threshold: Optional[float] = None,
        risk_threshold: float = 0.5,
        learnable_weights: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.entropy_threshold = entropy_threshold or (0.7 * (num_experts ** 0.5 / 4))
        self.confidence_threshold = confidence_threshold or 0.4
        self.mismatch_threshold = mismatch_threshold or 0.5
        self.risk_threshold = risk_threshold

        # Semantic entropy probe submodule
        self.semantic_probe = SemanticEntropyProbe(hidden_dim=hidden_dim)

        # Learnable combination weights (optional)
        self.learnable_weights = learnable_weights
        if learnable_weights:
            self.weight_entropy = nn.Parameter(torch.tensor(0.25))
            self.weight_confidence = nn.Parameter(torch.tensor(0.20))
            self.weight_mismatch = nn.Parameter(torch.tensor(0.20))
            self.weight_semantic = nn.Parameter(torch.tensor(0.20))
            self.weight_eigen = nn.Parameter(torch.tensor(0.15))
        else:
            # Default weights: 5 signals combined
            # entropy: 25%, confidence: 20%, mismatch: 20%, semantic: 20%, eigen: 15%
            self.register_buffer("weight_entropy", torch.tensor(0.25))
            self.register_buffer("weight_confidence", torch.tensor(0.20))
            self.register_buffer("weight_mismatch", torch.tensor(0.20))
            self.register_buffer("weight_semantic", torch.tensor(0.20))
            self.register_buffer("weight_eigen", torch.tensor(0.15))

    @property
    def max_entropy(self) -> float:
        """Maximum possible Shannon entropy for num_experts."""
        return math.log(self.num_experts)

    @torch.no_grad()
    def compute_eigen_score(
        self,
        routing_repr: torch.Tensor,
    ) -> torch.Tensor:
        """Compute EigenScore for hallucination detection.

        EigenScore = 1 / (sum of top-50% singular values)
        High S concentration = low uncertainty = low hallucination risk.

        Based on INSIDE paper (Chen et al., ICLR 2024): variance in embedding
        space indicates uncertainty. Low singular value sum = high variance =
        potential hallucination.

        Uses batched svdvals with float() cast for fp16 stability.

        Args:
            routing_repr: (batch, clifford_dim) — routing representations

        Returns:
            eigen_score: (batch,) — per-sample inverse singular value sum
        """
        if routing_repr.dim() == 1:
            routing_repr = routing_repr.unsqueeze(0)

        # Batched eigen score via torch.linalg.svdvals (single GPU kernel)
        if routing_repr.dim() == 2:
            routing_repr = routing_repr.unsqueeze(1)  # (B, 1, D) for batched svdvals
        s_vals = torch.linalg.svdvals(routing_repr.float())  # (B, min(M, N))
        k = max(1, int(s_vals.shape[-1] * 0.5))
        eigen_score = 1.0 / (s_vals[:, :k].sum(dim=-1) + 1e-8)  # (B,)

        return eigen_score

    @staticmethod
    def _simplex_projection(w: torch.Tensor) -> torch.Tensor:
        """Project w onto probability simplex: sum=1, all>=0."""
        K = w.shape[0]
        sorted_w, _ = torch.sort(w, descending=True)
        cumsum = torch.cumsum(sorted_w, dim=0)
        rho = (1 + torch.arange(1, K + 1, device=w.device, dtype=w.dtype) * sorted_w - cumsum) > 0
        rho_idx = rho.nonzero()[-1].item() if rho.any() else 0
        theta = (cumsum[rho_idx] - 1) / (rho_idx + 1)
        return torch.clamp(w - theta, min=0)

    @torch.no_grad()
    def compute_hallucination_risk(
        self,
        routing_entropy: torch.Tensor,
        moe_confidence: torch.Tensor,
        memory_mismatch: Optional[torch.Tensor] = None,
        memory_loss: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        routing_repr: Optional[torch.Tensor] = None,
    ) -> HallucinationDetectionResult:
        """Compute hallucination risk from MoE and memory signals.

        Args:
            routing_entropy: Per-batch Shannon entropy [batch]
            moe_confidence: Max expert weight (softmax) [batch]
            memory_mismatch: Surprise score [batch] or None
            memory_loss: Memory MSE [batch] or None
            hidden_states: Hidden states for semantic entropy [batch, seq, hidden] or [batch, hidden]
            routing_repr: Routing representation for eigen score [batch, routing_dim]

        Returns:
            HallucinationDetectionResult with combined risk score
        """
        # Normalize routing entropy to [0, 1]
        norm_entropy = routing_entropy / self.max_entropy

        # Invert confidence so high confidence = low risk
        inv_confidence = 1.0 - moe_confidence

        # Memory mismatch (fallback to 0 if unavailable)
        mismatch_val = (
            memory_mismatch
            if memory_mismatch is not None
            else torch.zeros_like(routing_entropy)
        )
        # Normalize mismatch to [0, 1] via sigmoid
        norm_mismatch = torch.sigmoid(mismatch_val - 2.0)

        # Semantic entropy from hidden states (fallback to 0)
        if hidden_states is not None:
            semantic_entropy = self.semantic_probe(hidden_states)
        else:
            semantic_entropy = torch.zeros_like(routing_entropy)

        # Eigen score from routing representation (fallback to 0)
        if routing_repr is not None:
            eigen_score = self.compute_eigen_score(routing_repr)
        else:
            eigen_score = torch.zeros_like(routing_entropy)

        # Normalize eigen_score to [0, 1] via sigmoid (centered at 1.0 — expected mean)
        norm_eigen = torch.sigmoid(eigen_score - 1.0)

        # Combined risk via 5-signal weighted sum with simplex projection
        weights = self._simplex_projection(torch.stack([
            self.weight_entropy, self.weight_confidence, self.weight_mismatch,
            self.weight_semantic, self.weight_eigen,
        ]))
        risk = (
            weights[0] * norm_entropy
            + weights[1] * inv_confidence
            + weights[2] * norm_mismatch
            + weights[3] * semantic_entropy
            + weights[4] * norm_eigen
        )

        # Ensure risk is in [0, 1]
        risk_clamped = torch.clamp(risk, 0.0, 1.0)

        # Compute all means on GPU first
        risk_mean = risk_clamped.mean()
        entropy_mean = routing_entropy.mean()
        confidence_mean = moe_confidence.mean()
        mismatch_mean = memory_mismatch.mean() if memory_mismatch is not None else torch.zeros_like(risk_mean)
        loss_mean = memory_loss.mean() if memory_loss is not None else torch.zeros_like(risk_mean)
        semantic_mean = semantic_entropy.mean()
        eigen_mean = eigen_score.mean() if routing_repr is not None else torch.zeros_like(risk_mean)

        # Tensor-based threshold checks — no GPU sync
        evidence_count = int((
            (entropy_mean > self.entropy_threshold)
            + (confidence_mean < self.confidence_threshold)
            + (mismatch_mean > self.mismatch_threshold)
            + (eigen_mean > 1.0)
            + (semantic_mean > 0.5)
        ).sum().item())

        # Single GPU->CPU transfer: stack all scalars, .item() once
        _batch = torch.stack([
            entropy_mean, confidence_mean, mismatch_mean,
            loss_mean, eigen_mean, semantic_mean, risk_mean,
        ]).cpu()
        _entropy_v, _conf_v, _mismatch_v, _loss_v, _eigen_v, _semantic_v, _risk_v = (
            float(v) for v in _batch
        )

        return HallucinationDetectionResult(
            routing_entropy=round(_entropy_v, 6),
            moe_confidence=round(_conf_v, 6),
            memory_mismatch=round(_mismatch_v, 6),
            memory_loss=round(_loss_v, 6),
            eigen_score=round(_eigen_v, 6),
            semantic_entropy=round(_semantic_v, 6),
            hallucination_risk=round(_risk_v, 6),
            is_potential_hallucination=_risk_v > self.risk_threshold,
            evidence_count=evidence_count,
        )

    def from_router_state(
        self, router_state: dict, routing_repr: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None, **memory_kwargs
    ) -> HallucinationDetectionResult:
        """Extract signals from MoE router_state dict and compute risk.

        Args:
            router_state: Dict from SoftMoERouter.forward() or MoEKernel forward state
            routing_repr: Optional routing representations for eigen_score computation
            device: Target device for fallback tensors. If None, derived from router_state.
            **memory_kwargs: memory_mismatch, memory_loss, hidden_states from TitansMemory

        Returns:
            HallucinationDetectionResult
        """
        # BUG-12 FIX: derive device from router_state or use explicit device param
        _ref = router_state.get("gate_weights", router_state.get("scores", None))
        _device = device if device is not None else (
            _ref.device if _ref is not None else torch.device('cpu')
        )
        _dtype = _ref.dtype if _ref is not None else torch.float32

        routing_entropy = router_state.get("routing_entropy", None)
        if routing_entropy is None:
            routing_entropy = torch.tensor(0.0, device=_device, dtype=_dtype)
        gate_weights = router_state.get("gate_weights", router_state.get("scores", None))
        topk_gate = router_state.get("topk_gate_weights", None)

        # MoE confidence from top-k gate weights, weighted by expert accuracy
        expert_accuracy = router_state.get("expert_accuracy", None)
        if topk_gate is not None:
            moe_confidence = topk_gate.max(dim=-1).values
        elif gate_weights is not None:
            moe_confidence = gate_weights.max(dim=-1).values
        else:
            moe_confidence = torch.tensor(0.0, device=routing_entropy.device, dtype=routing_entropy.dtype)
        # Weight confidence by expert reliability if available (L1.3)
        if expert_accuracy is not None and gate_weights is not None:
            top_expert = gate_weights.argmax(dim=-1)
            reliability = expert_accuracy[top_expert]
            moe_confidence = moe_confidence * reliability

        return self.compute_hallucination_risk(
            routing_entropy=routing_entropy,
            moe_confidence=moe_confidence,
            routing_repr=routing_repr,
            **memory_kwargs,
        )
