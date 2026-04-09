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
            self.register_buffer("weight_confidence", torch.tensor(-0.20))
            self.register_buffer("weight_mismatch", torch.tensor(0.20))
            self.register_buffer("weight_semantic", torch.tensor(0.20))
            self.register_buffer("weight_eigen", torch.tensor(0.15))

    @property
    def max_entropy(self) -> float:
        """Maximum possible Shannon entropy for num_experts."""
        return math.log(self.num_experts)

    def compute_eigen_score(
        self,
        routing_repr: torch.Tensor,
        top_k: int = 5,
    ) -> torch.Tensor:
        """Compute EigenScore for hallucination detection.

        EigenScore = 1 / (sum of top-k singular values)
        High S concentration = low uncertainty = low hallucination risk.

        Based on INSIDE paper (Chen et al., ICLR 2024): variance in embedding
        space indicates uncertainty. Low singular value sum = high variance =
        potential hallucination.

        Processes each sample independently (batch dimension preserved).

        Args:
            routing_repr: (batch, clifford_dim) — routing representations
            top_k: Number of singular values to consider

        Returns:
            eigen_score: (batch,) — per-sample inverse singular value sum
        """
        if routing_repr.dim() == 1:
            routing_repr = routing_repr.unsqueeze(0)

        batch_size = routing_repr.shape[0]

        # Per-sample eigen score computation
        scores = []
        for i in range(batch_size):
            sample = routing_repr[i : i + 1]  # (1, clifford_dim)
            # Center single sample (degenerate: centering a single vector gives 0,
            # so use the vector itself as the representation)
            _, S, _ = torch.linalg.svd(sample, full_matrices=False)
            k = min(top_k, S.shape[0])
            eigen_score_i = 1.0 / (S[:k].sum() + 1e-8)
            scores.append(eigen_score_i)

        return torch.stack(scores)  # (batch,)

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
        loss_val = (
            memory_loss
            if memory_loss is not None
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

        # Combined risk via 5-signal weighted sum
        risk = (
            self.weight_entropy * norm_entropy
            + self.weight_confidence * inv_confidence
            + self.weight_mismatch * norm_mismatch
            + self.weight_semantic * semantic_entropy
            + self.weight_eigen * norm_eigen
        )

        # Ensure risk is in [0, 1]
        risk_clamped = torch.clamp(risk, 0.0, 1.0)

        # Per-batch result, return mean for batch-level
        risk_val = risk_clamped.mean().item()
        entropy_val = routing_entropy.mean().item()
        confidence_val = moe_confidence.mean().item()
        mismatch_v = memory_mismatch.mean().item() if memory_mismatch is not None else 0.0
        loss_v = memory_loss.mean().item() if memory_loss is not None else 0.0
        semantic_v = semantic_entropy.mean().item()
        eigen_v = eigen_score.mean().item() if routing_repr is not None else 0.0

        return HallucinationDetectionResult(
            routing_entropy=round(entropy_val, 6),
            moe_confidence=round(confidence_val, 6),
            memory_mismatch=round(mismatch_v, 6),
            memory_loss=round(loss_v, 6),
            eigen_score=round(eigen_v, 6),
            semantic_entropy=round(semantic_v, 6),
            hallucination_risk=round(risk_val, 6),
            is_potential_hallucination=risk_val > self.risk_threshold,
        )

    def from_router_state(
        self, router_state: dict, routing_repr: Optional[torch.Tensor] = None, **memory_kwargs
    ) -> HallucinationDetectionResult:
        """Extract signals from MoE router_state dict and compute risk.

        Args:
            router_state: Dict from SoftMoERouter.forward() or MoEKernel forward state
            routing_repr: Optional routing representations for eigen_score computation
            **memory_kwargs: memory_mismatch, memory_loss, hidden_states from TitansMemory

        Returns:
            HallucinationDetectionResult
        """
        routing_entropy = router_state.get("routing_entropy", torch.tensor(0.0))
        gate_weights = router_state.get("gate_weights", router_state.get("scores", None))
        topk_gate = router_state.get("topk_gate_weights", None)

        # MoE confidence from top-k gate weights
        if topk_gate is not None:
            moe_confidence = topk_gate.max(dim=-1).values
        elif gate_weights is not None:
            moe_confidence = gate_weights.max(dim=-1).values
        else:
            moe_confidence = torch.tensor(0.0)

        return self.compute_hallucination_risk(
            routing_entropy=routing_entropy,
            moe_confidence=moe_confidence,
            routing_repr=routing_repr,
            **memory_kwargs,
        )
