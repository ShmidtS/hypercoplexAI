"""Hallucination Detector — routing entropy + memory mismatch for artifact detection.

Concept: LLM hallucinations are compression artifacts. HDIM detects them via:
  1. Routing Entropy — high entropy = MoE uncertain about expert assignment
  2. MoE Confidence — low max weight = no expert strongly "owns" this input
  3. Memory Mismatch — large distance to stored patterns = unseen/unknown input

High hallucination risk: high entropy + low confidence + high mismatch
Low hallucination risk: low entropy + high confidence + low mismatch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class HallucinationDetectionResult:
    """Per-sample hallucination detection output."""

    routing_entropy: float        # Shannon entropy of MoE gate distribution
    moe_confidence: float         # Max expert weight (softmax prob)
    memory_mismatch: float        # Surprise: ||d(loss)/d(key)|| norm
    memory_loss: float            # MSE between memory prediction and target
    hallucination_risk: float     # Combined score [0, 1]
    is_potential_hallucination: bool  # Thresholded decision

    def to_dict(self) -> dict:
        return {
            "routing_entropy": self.routing_entropy,
            "moe_confidence": self.moe_confidence,
            "memory_mismatch": self.memory_mismatch,
            "memory_loss": self.memory_loss,
            "hallucination_risk": self.hallucination_risk,
            "is_potential_hallucination": self.is_potential_hallucination,
        }


class HallucinationDetector(nn.Module):
    """Detects potential hallucinations via MoE routing entropy and memory signals.

    Uses three independent signals from the HDIM pipeline:
    - routing_entropy: from MoE router (soft_moe_router.py / moe_kernel.py)
    - moe_confidence: max gate weight from MoE routing
    - memory_mismatch: surprise from Titans memory gradient
    - memory_loss: MSE from memory retrieval

    Combined via learnable weighted sum into hallucination_risk [0, 1].
    """

    def __init__(
        self,
        num_experts: int = 4,
        entropy_threshold: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        mismatch_threshold: Optional[float] = None,
        risk_threshold: float = 0.5,
        learnable_weights: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.entropy_threshold = entropy_threshold or (0.7 * (num_experts ** 0.5 / 4))
        self.confidence_threshold = confidence_threshold or 0.4
        self.mismatch_threshold = mismatch_threshold or 0.5
        self.risk_threshold = risk_threshold

        # Learnable combination weights (optional)
        self.learnable_weights = learnable_weights
        if learnable_weights:
            self.weight_entropy = nn.Parameter(torch.tensor(0.4))
            self.weight_confidence = nn.Parameter(torch.tensor(-0.3))
            self.weight_mismatch = nn.Parameter(torch.tensor(0.3))
        else:
            # Default: entropy and mismatch increase risk, confidence decreases it
            self.register_buffer("weight_entropy", torch.tensor(0.50))
            self.register_buffer("weight_confidence", torch.tensor(-0.30))
            self.register_buffer("weight_mismatch", torch.tensor(0.20))

    @property
    def max_entropy(self) -> float:
        """Maximum possible Shannon entropy for num_experts."""
        import math
        return math.log(self.num_experts)

    def compute_hallucination_risk(
        self,
        routing_entropy: torch.Tensor,
        moe_confidence: torch.Tensor,
        memory_mismatch: Optional[torch.Tensor] = None,
        memory_loss: Optional[torch.Tensor] = None,
    ) -> HallucinationDetectionResult:
        """Compute hallucination risk from MoE and memory signals.

        Args:
            routing_entropy: Per-batch Shannon entropy [batch]
            moe_confidence: Max expert weight (softmax) [batch]
            memory_mismatch: Surprise score [batch] or None
            memory_loss: Memory MSE [batch] or None

        Returns:
            HallucinationDetectionResult with combined risk score
        """
        # Normalize routing entropy to [0, 1]
        norm_entropy = routing_entropy / self.max_entropy

        # Invert confidence so high confidence = low risk
        inv_confidence = 1.0 - moe_confidence

        # Memory mismatch (fallback to 0 if unavailable)
        mismatch_val = memory_mismatch if memory_mismatch is not None else torch.zeros_like(routing_entropy)
        loss_val = memory_loss if memory_loss is not None else torch.zeros_like(routing_entropy)

        # Normalize mismatch to [0, 1] via sigmoid
        norm_mismatch = torch.sigmoid(mismatch_val - 2.0)

        # Combined risk via weighted sum + sigmoid
        # Baseline offset shifts neutral point toward "safe"
        risk = (
            self.weight_entropy * norm_entropy
            + self.weight_confidence * inv_confidence
            + self.weight_mismatch * norm_mismatch
            - 0.2  # baseline bias toward safety
        )
        risk_sigmoid = torch.sigmoid(risk * 3.0)

        # Per-batch result, return mean for batch-level
        risk_val = risk_sigmoid.mean().item()
        entropy_val = routing_entropy.mean().item()
        confidence_val = moe_confidence.mean().item()
        mismatch_v = memory_mismatch.mean().item() if memory_mismatch is not None else 0.0
        loss_v = memory_loss.mean().item() if memory_loss is not None else 0.0

        return HallucinationDetectionResult(
            routing_entropy=round(entropy_val, 6),
            moe_confidence=round(confidence_val, 6),
            memory_mismatch=round(mismatch_v, 6),
            memory_loss=round(loss_v, 6),
            hallucination_risk=round(risk_val, 6),
            is_potential_hallucination=risk_val > self.risk_threshold,
        )

    def from_router_state(self, router_state: dict, **memory_kwargs) -> HallucinationDetectionResult:
        """Extract signals from MoE router_state dict and compute risk.

        Args:
            router_state: Dict from SoftMoERouter.forward() or MoEKernel forward state
            **memory_kwargs: memory_mismatch, memory_loss from TitansMemory

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
            **memory_kwargs,
        )
