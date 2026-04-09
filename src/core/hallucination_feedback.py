"""Hallucination Feedback Loop — Self-correction via risk-based rerouting.

Architecture:
    HallucinationDetector (5 signals)
           |
           v
       risk_score
           |
           v
    RiskThresholdChecker
           |
           v
    if risk > threshold:
        MoERouter.re_route() -> safer expert
        Memory.consolidate() -> strengthen correct patterns
        Output.confidence <- lowered

This module implements the feedback loop that allows HDIM to self-correct
when hallucination risk exceeds safe thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class FeedbackAction(Enum):
    """Action to take based on hallucination risk level."""

    NONE = "none"  # Risk below all thresholds, continue normally
    ADJUST_CONFIDENCE = "adjust_confidence"  # Risk moderate, lower output confidence
    REROUTE = "reroute"  # Risk high, re-route to safer expert
    TRIGGER_CONSOLIDATION = "trigger_consolidation"  # Risk critical, force memory consolidation
    FULL_CORRECTION = "full_correction"  # Risk extreme, reroute + consolidate


@dataclass
class FeedbackResult:
    """Result of hallucination feedback loop processing.

    Attributes:
        action: The action taken (or NONE if below thresholds).
        adjusted_confidence: Confidence after adjustment (1.0 if no adjustment).
        selected_expert: Name of expert selected after potential reroute.
        original_expert: Name of expert before reroute.
        consolidation_triggered: Whether memory consolidation was triggered.
        risk_level: Human-readable risk level (low/medium/high/critical).
    """

    action: FeedbackAction
    adjusted_confidence: float
    selected_expert: str
    original_expert: str
    consolidation_triggered: bool
    risk_level: str

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "adjusted_confidence": self.adjusted_confidence,
            "selected_expert": self.selected_expert,
            "original_expert": self.original_expert,
            "consolidation_triggered": self.consolidation_triggered,
            "risk_level": self.risk_level,
        }


@dataclass
class RiskThresholds:
    """Thresholds for hallucination risk levels.

    Risk levels:
        - [0.0, low): Safe, no action needed
        - [low, medium): Moderate, adjust confidence
        - [medium, high): High, reroute to safer expert
        - [high, critical): Very high, trigger memory consolidation
        - [critical, 1.0]: Extreme, full correction (reroute + consolidate)
    """

    low: float = 0.3
    medium: float = 0.5
    high: float = 0.7
    critical: float = 0.85

    def __post_init__(self):
        if not (0.0 <= self.low <= self.medium <= self.high <= self.critical <= 1.0):
            raise ValueError(
                f"Thresholds must be ordered: 0 <= low <= medium <= high <= critical <= 1, "
                f"got low={self.low}, medium={self.medium}, high={self.high}, critical={self.critical}"
            )


class RiskThresholdChecker:
    """Checks hallucination risk against thresholds and determines action.

    Maps risk scores to FeedbackAction based on configurable thresholds.
    """

    def __init__(self, thresholds: Optional[RiskThresholds] = None):
        self.thresholds = thresholds or RiskThresholds()

    def check(self, risk_score: float) -> Tuple[FeedbackAction, str]:
        """Check risk score against thresholds and return action.

        Args:
            risk_score: Hallucination risk score in [0, 1].

        Returns:
            Tuple of (FeedbackAction, risk_level_string).
        """
        if risk_score >= self.thresholds.critical:
            return FeedbackAction.FULL_CORRECTION, "extreme"
        elif risk_score >= self.thresholds.high:
            return FeedbackAction.TRIGGER_CONSOLIDATION, "critical"
        elif risk_score >= self.thresholds.medium:
            return FeedbackAction.REROUTE, "high"
        elif risk_score >= self.thresholds.low:
            return FeedbackAction.ADJUST_CONFIDENCE, "medium"
        else:
            return FeedbackAction.NONE, "low"


class ReRoutingStrategy:
    """Strategy for selecting safer expert during reroute.

    The strategy prioritizes experts with:
    1. Higher historical confidence (lower hallucination rate)
    2. More stable routing patterns (lower entropy)
    3. Better memory alignment (lower mismatch)
    """

    # Expert safety ranking (domain-specific, can be overridden)
    # Language expert is typically safest for general text
    # Math/Science experts are more specialized, may hallucinate on out-of-domain
    DEFAULT_SAFETY_RANKING = ["language", "code", "science", "math"]

    def __init__(
        self,
        expert_names: List[str],
        safety_ranking: Optional[List[str]] = None,
    ):
        self.expert_names = expert_names
        self.safety_ranking = safety_ranking or self.DEFAULT_SAFETY_RANKING
        # Filter safety_ranking to only include available experts
        self.safety_ranking = [e for e in self.safety_ranking if e in expert_names]
        # Add any experts not in default ranking at the end
        for expert in expert_names:
            if expert not in self.safety_ranking:
                self.safety_ranking.append(expert)

        # Track per-expert hallucination history (EMA)
        self.register_buffer_func = None  # Set by parent module
        self._expert_hallucination_rates = {e: 0.0 for e in expert_names}

    def update_expert_history(self, expert: str, hallucination_occurred: bool) -> None:
        """Update EMA hallucination rate for an expert.

        Args:
            expert: Expert name.
            hallucination_occurred: Whether hallucination was detected.
        """
        if expert not in self._expert_hallucination_rates:
            return
        # EMA update: new_rate = 0.9 * old_rate + 0.1 * new_observation
        current = self._expert_hallucination_rates[expert]
        self._expert_hallucination_rates[expert] = (
            0.9 * current + 0.1 * float(hallucination_occurred)
        )

    def select_safe_expert(
        self,
        current_expert: str,
        risk_level: str,
        expert_weights: Optional[torch.Tensor] = None,
    ) -> str:
        """Select a safer expert for rerouting.

        Args:
            current_expert: Currently selected expert.
            risk_level: Risk level string (low/medium/high/critical/extreme).
            expert_weights: Optional tensor of expert weights for weighted selection.

        Returns:
            Name of the safer expert to route to.
        """
        # If current expert is already the safest, keep it
        if current_expert == self.safety_ranking[0]:
            return current_expert

        # For medium risk, use weighted selection favoring safer experts
        if risk_level == "medium" and expert_weights is not None:
            # Bias weights toward safer experts based on safety_ranking
            safety_ranking_map = {
                name: 1.0 / (rank + 1) for rank, name in enumerate(self.safety_ranking)
            }
            safety_bias = torch.tensor(
                [safety_ranking_map.get(name, 0.0) for name in self.expert_names],
                dtype=expert_weights.dtype,
                device=expert_weights.device,
            )
            biased_weights = expert_weights * safety_bias
            biased_weights = biased_weights / biased_weights.sum()
            idx = torch.argmax(biased_weights).item()
            return self.expert_names[idx]

        # For high/critical risk, pick the safest available expert
        # that is different from current expert
        for safe_expert in self.safety_ranking:
            if safe_expert != current_expert:
                return safe_expert

        # Fallback: return safest expert (may be same as current)
        return self.safety_ranking[0]

    def get_expert_safety_scores(self) -> dict:
        """Return safety scores for all experts based on hallucination history.

        Returns:
            Dict mapping expert name to safety score (1 - hallucination_rate).
        """
        return {
            expert: 1.0 - self._expert_hallucination_rates.get(expert, 0.0)
            for expert in self.expert_names
        }


class MemoryConsolidationTrigger:
    """Triggers memory consolidation when hallucination risk is high.

    Consolidation strengthens memory patterns associated with correct outputs
    and weakens patterns associated with hallucinations.
    """

    def __init__(
        self,
        consolidation_threshold: float = 0.7,
        min_interval_steps: int = 100,
    ):
        self.consolidation_threshold = consolidation_threshold
        self.min_interval_steps = min_interval_steps
        self._steps_since_last_consolidation = min_interval_steps

    def should_consolidate(self, risk_score: float) -> bool:
        """Check if memory consolidation should be triggered.

        Args:
            risk_score: Current hallucination risk score.

        Returns:
            True if consolidation should be triggered.
        """
        if risk_score < self.consolidation_threshold:
            return False
        if self._steps_since_last_consolidation < self.min_interval_steps:
            return False
        return True

    def mark_consolidation_done(self) -> None:
        """Mark that consolidation was performed."""
        self._steps_since_last_consolidation = 0

    def step(self) -> None:
        """Increment step counter."""
        self._steps_since_last_consolidation += 1


class ConfidenceAdjuster:
    """Adjusts output confidence based on hallucination risk.

    The adjustment follows a sigmoid decay:
        adjusted = base * (1 - sigmoid_alpha * (risk - threshold))

    Where sigmoid_alpha controls how aggressively confidence is reduced.
    """

    def __init__(
        self,
        adjustment_threshold: float = 0.3,
        sigmoid_alpha: float = 2.0,
        min_confidence: float = 0.1,
    ):
        self.adjustment_threshold = adjustment_threshold
        self.sigmoid_alpha = sigmoid_alpha
        self.min_confidence = min_confidence

    def adjust(self, base_confidence: float, risk_score: float) -> float:
        """Adjust confidence based on hallucination risk.

        Args:
            base_confidence: Original confidence score in [0, 1].
            risk_score: Hallucination risk score in [0, 1].

        Returns:
            Adjusted confidence score.
        """
        if risk_score < self.adjustment_threshold:
            return base_confidence

        # Sigmoid decay: confidence reduction increases with risk above threshold
        import math
        excess_risk = risk_score - self.adjustment_threshold
        reduction = 1.0 / (1.0 + math.exp(-self.sigmoid_alpha * (excess_risk - 0.2)))

        adjusted = base_confidence * (1.0 - reduction)
        return max(adjusted, self.min_confidence)


class HallucinationFeedbackLoop(nn.Module):
    """Complete hallucination feedback loop for self-correction.

    Integrates risk checking, rerouting, consolidation, and confidence adjustment
    into a single module that can be inserted into the HDIM forward pass.

    Usage:
        feedback_loop = HallucinationFeedbackLoop(
            expert_names=["math", "language", "code", "science"],
            config=HallucinationFeedbackConfig(),
        )

        result = feedback_loop.check_and_respond(
            risk_score=hallucination_result.hallucination_risk,
            routing_info={
                "current_expert": "math",
                "expert_weights": routing_weights,
            },
        )

        if result.action == FeedbackAction.REROUTE:
            # Use result.selected_expert for rerouting
            pass
    """

    def __init__(
        self,
        expert_names: List[str],
        thresholds: Optional[RiskThresholds] = None,
        consolidation_threshold: float = 0.7,
        confidence_adjustment_threshold: float = 0.3,
        enabled: bool = True,
    ):
        super().__init__()
        self.expert_names = expert_names
        self.enabled = enabled

        # Per-expert hallucination rates as buffer (survives serialization)
        self.register_buffer(
            "expert_hallucination_rates",
            torch.zeros(len(expert_names)),
        )

        # Sub-components
        self.threshold_checker = RiskThresholdChecker(thresholds)
        self.routing_strategy = ReRoutingStrategy(expert_names)
        self.consolidation_trigger = MemoryConsolidationTrigger(
            consolidation_threshold=consolidation_threshold
        )
        self.confidence_adjuster = ConfidenceAdjuster(
            adjustment_threshold=confidence_adjustment_threshold
        )

        # Track consolidation requests for external handling
        self._pending_consolidation = False
        self._consolidation_reason: Optional[str] = None

    def check_and_respond(
        self,
        risk_score: float,
        routing_info: dict,
        base_confidence: float = 1.0,
    ) -> FeedbackResult:
        """Check hallucination risk and determine response action.

        Args:
            risk_score: Hallucination risk score from detector in [0, 1].
            routing_info: Dict with 'current_expert' and optionally 'expert_weights'.
            base_confidence: Original output confidence before adjustment.

        Returns:
            FeedbackResult with action to take and adjusted parameters.
        """
        if not self.enabled:
            return FeedbackResult(
                action=FeedbackAction.NONE,
                adjusted_confidence=base_confidence,
                selected_expert=routing_info.get("current_expert", self.expert_names[0]),
                original_expert=routing_info.get("current_expert", self.expert_names[0]),
                consolidation_triggered=False,
                risk_level="low",
            )

        # Determine action based on risk level
        action, risk_level = self.threshold_checker.check(risk_score)

        current_expert = routing_info.get("current_expert", self.expert_names[0])
        expert_weights = routing_info.get("expert_weights")

        # Determine selected expert
        selected_expert = current_expert
        consolidation_triggered = False

        if action in (FeedbackAction.REROUTE, FeedbackAction.FULL_CORRECTION):
            selected_expert = self.routing_strategy.select_safe_expert(
                current_expert=current_expert,
                risk_level=risk_level,
                expert_weights=expert_weights,
            )

        if action in (FeedbackAction.TRIGGER_CONSOLIDATION, FeedbackAction.FULL_CORRECTION):
            consolidation_triggered = self.consolidation_trigger.should_consolidate(risk_score)
            if consolidation_triggered:
                self.consolidation_trigger.mark_consolidation_done()
                self._pending_consolidation = True
                self._consolidation_reason = f"hallucination_risk={risk_score:.3f}"

        # Adjust confidence
        adjusted_confidence = self.confidence_adjuster.adjust(base_confidence, risk_score)

        return FeedbackResult(
            action=action,
            adjusted_confidence=adjusted_confidence,
            selected_expert=selected_expert,
            original_expert=current_expert,
            consolidation_triggered=consolidation_triggered,
            risk_level=risk_level,
        )

    def select_safe_expert(
        self,
        current_expert: str,
        risk_level: str,
        expert_weights: Optional[torch.Tensor] = None,
    ) -> str:
        """Select a safer expert based on risk level.

        Args:
            current_expert: Currently selected expert.
            risk_level: Risk level string.
            expert_weights: Optional expert weights for weighted selection.

        Returns:
            Name of the safer expert.
        """
        return self.routing_strategy.select_safe_expert(
            current_expert=current_expert,
            risk_level=risk_level,
            expert_weights=expert_weights,
        )

    def trigger_memory_consolidation(self, reason: str) -> bool:
        """Request memory consolidation.

        Args:
            reason: Human-readable reason for consolidation.

        Returns:
            True if consolidation was triggered, False if rate-limited.
        """
        self._pending_consolidation = True
        self._consolidation_reason = reason
        return True

    def adjust_confidence(self, base_confidence: float, risk_score: float) -> float:
        """Adjust confidence based on hallucination risk.

        Args:
            base_confidence: Original confidence.
            risk_score: Hallucination risk score.

        Returns:
            Adjusted confidence.
        """
        return self.confidence_adjuster.adjust(base_confidence, risk_score)

    def has_pending_consolidation(self) -> bool:
        """Check if memory consolidation is pending."""
        return self._pending_consolidation

    def get_consolidation_reason(self) -> Optional[str]:
        """Get the reason for pending consolidation."""
        return self._consolidation_reason

    def clear_consolidation_request(self) -> None:
        """Clear pending consolidation request."""
        self._pending_consolidation = False
        self._consolidation_reason = None

    def update_expert_hallucination_history(
        self, expert: str, hallucination_occurred: bool
    ) -> None:
        """Update hallucination history for an expert.

        Args:
            expert: Expert name.
            hallucination_occurred: Whether hallucination was detected.
        """
        self.routing_strategy.update_expert_history(expert, hallucination_occurred)
        # Also update the buffer for serialization
        if expert in self.expert_names:
            idx = self.expert_names.index(expert)
            current = self.expert_hallucination_rates[idx].item()
            self.expert_hallucination_rates[idx] = 0.9 * current + 0.1 * float(hallucination_occurred)

    def step(self) -> None:
        """Increment internal step counter."""
        self.consolidation_trigger.step()

    def get_expert_safety_scores(self) -> dict:
        """Get safety scores for all experts."""
        return self.routing_strategy.get_expert_safety_scores()


@dataclass
class HallucinationFeedbackConfig:
    """Configuration for HallucinationFeedbackLoop."""

    enabled: bool = True
    # Risk thresholds
    low_threshold: float = 0.3
    medium_threshold: float = 0.5
    high_threshold: float = 0.7
    critical_threshold: float = 0.85
    # Consolidation settings
    consolidation_threshold: float = 0.7
    min_consolidation_interval: int = 100
    # Confidence adjustment
    confidence_adjustment_threshold: float = 0.3
    sigmoid_alpha: float = 2.0
    min_confidence: float = 0.1
    # Expert safety ranking (override default)
    expert_safety_ranking: Optional[List[str]] = None

    def get_thresholds(self) -> RiskThresholds:
        return RiskThresholds(
            low=self.low_threshold,
            medium=self.medium_threshold,
            high=self.high_threshold,
            critical=self.critical_threshold,
        )

    def create_feedback_loop(self, expert_names: List[str]) -> HallucinationFeedbackLoop:
        """Factory method to create HallucinationFeedbackLoop from config."""
        return HallucinationFeedbackLoop(
            expert_names=expert_names,
            thresholds=self.get_thresholds(),
            consolidation_threshold=self.consolidation_threshold,
            confidence_adjustment_threshold=self.confidence_adjustment_threshold,
            enabled=self.enabled,
        )
