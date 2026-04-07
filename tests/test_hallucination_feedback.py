"""Tests for HallucinationFeedbackLoop — Self-correction via risk-based rerouting."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from src.core.hallucination_feedback import (
    FeedbackAction,
    FeedbackResult,
    RiskThresholds,
    RiskThresholdChecker,
    ReRoutingStrategy,
    MemoryConsolidationTrigger,
    ConfidenceAdjuster,
    HallucinationFeedbackLoop,
    HallucinationFeedbackConfig,
)


class TestRiskThresholds:
    """Test risk threshold configuration."""

    def test_default_thresholds_ordered(self):
        thresholds = RiskThresholds()
        assert 0.0 <= thresholds.low <= thresholds.medium <= thresholds.high <= thresholds.critical <= 1.0

    def test_custom_thresholds(self):
        thresholds = RiskThresholds(low=0.2, medium=0.4, high=0.6, critical=0.8)
        assert thresholds.low == 0.2
        assert thresholds.medium == 0.4
        assert thresholds.high == 0.6
        assert thresholds.critical == 0.8

    def test_invalid_thresholds_raises(self):
        with pytest.raises(ValueError):
            RiskThresholds(low=0.5, medium=0.3)  # low > medium

    def test_extreme_valid_thresholds(self):
        thresholds = RiskThresholds(low=0.0, medium=0.25, high=0.5, critical=1.0)
        assert thresholds.low == 0.0
        assert thresholds.critical == 1.0


class TestRiskThresholdChecker:
    """Test risk level classification."""

    def test_low_risk_returns_none_action(self):
        checker = RiskThresholdChecker()
        action, level = checker.check(0.1)
        assert action == FeedbackAction.NONE
        assert level == "low"

    def test_medium_risk_returns_adjust_confidence(self):
        checker = RiskThresholdChecker()
        action, level = checker.check(0.4)
        assert action == FeedbackAction.ADJUST_CONFIDENCE
        assert level == "medium"

    def test_high_risk_returns_reroute(self):
        checker = RiskThresholdChecker()
        action, level = checker.check(0.6)
        assert action == FeedbackAction.REROUTE
        assert level == "high"

    def test_critical_risk_returns_trigger_consolidation(self):
        checker = RiskThresholdChecker()
        action, level = checker.check(0.75)
        assert action == FeedbackAction.TRIGGER_CONSOLIDATION
        assert level == "critical"

    def test_extreme_risk_returns_full_correction(self):
        checker = RiskThresholdChecker()
        action, level = checker.check(0.9)
        assert action == FeedbackAction.FULL_CORRECTION
        assert level == "extreme"

    def test_custom_thresholds(self):
        thresholds = RiskThresholds(low=0.2, medium=0.4, high=0.6, critical=0.8)
        checker = RiskThresholdChecker(thresholds)

        action, _ = checker.check(0.15)
        assert action == FeedbackAction.NONE

        action, _ = checker.check(0.3)
        assert action == FeedbackAction.ADJUST_CONFIDENCE

        action, _ = checker.check(0.5)
        assert action == FeedbackAction.REROUTE

        action, _ = checker.check(0.7)
        assert action == FeedbackAction.TRIGGER_CONSOLIDATION


class TestReRoutingStrategy:
    """Test safe expert selection."""

    def test_default_safety_ranking(self):
        strategy = ReRoutingStrategy(["math", "language", "code", "science"])
        # Language should be first (safest)
        assert strategy.safety_ranking[0] == "language"

    def test_select_safe_expert_different_from_current(self):
        strategy = ReRoutingStrategy(["math", "language", "code", "science"])

        # When current is math (least safe), should pick language (safest)
        safe = strategy.select_safe_expert("math", "high")
        assert safe != "math"
        assert safe == "language"

    def test_select_safe_expert_when_already_safest(self):
        strategy = ReRoutingStrategy(["math", "language", "code", "science"])

        # When already on safest expert
        safe = strategy.select_safe_expert("language", "high")
        assert safe == "language"

    def test_select_with_expert_weights(self):
        strategy = ReRoutingStrategy(["math", "language", "code", "science"])
        expert_weights = torch.tensor([0.1, 0.2, 0.3, 0.4])

        safe = strategy.select_safe_expert("math", "medium", expert_weights)
        # Should still prefer safer expert
        assert safe in strategy.safety_ranking

    def test_expert_history_update(self):
        strategy = ReRoutingStrategy(["math", "language"])

        # Initially no hallucination history
        scores = strategy.get_expert_safety_scores()
        assert scores["math"] == 1.0

        # Update with hallucination
        strategy.update_expert_history("math", True)
        scores = strategy.get_expert_safety_scores()
        assert scores["math"] < 1.0

    def test_ema_update(self):
        strategy = ReRoutingStrategy(["math"])

        # Multiple updates
        strategy.update_expert_history("math", True)
        strategy.update_expert_history("math", False)

        scores = strategy.get_expert_safety_scores()
        # EMA: 0.9 * 0.1 + 0.1 * 0 = 0.09, then 0.9 * 0.09 + 0.1 * 1 = 0.181
        # Safety score = 1 - hallucination_rate
        assert 0.8 < scores["math"] < 1.0


class TestMemoryConsolidationTrigger:
    """Test memory consolidation triggering."""

    def test_no_consolidation_below_threshold(self):
        trigger = MemoryConsolidationTrigger(consolidation_threshold=0.7)
        assert not trigger.should_consolidate(0.5)

    def test_consolidation_above_threshold(self):
        trigger = MemoryConsolidationTrigger(consolidation_threshold=0.7, min_interval_steps=0)
        assert trigger.should_consolidate(0.8)

    def test_rate_limiting(self):
        trigger = MemoryConsolidationTrigger(consolidation_threshold=0.7, min_interval_steps=10)

        # First trigger should work
        trigger._steps_since_last_consolidation = 100
        assert trigger.should_consolidate(0.8)

        # Mark consolidation and reset counter
        trigger.mark_consolidation_done()

        # Immediate second attempt should be rate-limited
        assert not trigger.should_consolidate(0.8)

    def test_step_counter(self):
        trigger = MemoryConsolidationTrigger(min_interval_steps=10)
        trigger.mark_consolidation_done()

        for _ in range(5):
            trigger.step()

        assert trigger._steps_since_last_consolidation == 5


class TestConfidenceAdjuster:
    """Test confidence adjustment based on risk."""

    def test_no_adjustment_below_threshold(self):
        adjuster = ConfidenceAdjuster(adjustment_threshold=0.3)
        adjusted = adjuster.adjust(0.9, 0.2)
        assert adjusted == pytest.approx(0.9, rel=1e-6)

    def test_adjustment_above_threshold(self):
        adjuster = ConfidenceAdjuster(adjustment_threshold=0.3)
        adjusted = adjuster.adjust(0.9, 0.5)
        assert adjusted < 0.9

    def test_higher_risk_lower_confidence(self):
        adjuster = ConfidenceAdjuster(adjustment_threshold=0.3)

        adjusted_low_risk = adjuster.adjust(0.9, 0.35)
        adjusted_high_risk = adjuster.adjust(0.9, 0.7)

        assert adjusted_high_risk < adjusted_low_risk

    def test_min_confidence_floor(self):
        adjuster = ConfidenceAdjuster(adjustment_threshold=0.3, min_confidence=0.1)

        # Even with extreme risk
        adjusted = adjuster.adjust(0.9, 0.95)
        assert adjusted >= 0.1

    def test_zero_risk_no_change(self):
        adjuster = ConfidenceAdjuster(adjustment_threshold=0.3)
        adjusted = adjuster.adjust(0.8, 0.0)
        assert adjusted == pytest.approx(0.8, rel=1e-6)


class TestHallucinationFeedbackLoop:
    """Test complete feedback loop."""

    def test_disabled_returns_none_action(self):
        loop = HallucinationFeedbackLoop(
            expert_names=["math", "language", "code", "science"],
            enabled=False,
        )

        result = loop.check_and_respond(
            risk_score=0.9,
            routing_info={"current_expert": "math"},
        )

        assert result.action == FeedbackAction.NONE
        assert result.adjusted_confidence == 1.0

    def test_low_risk_no_action(self):
        loop = HallucinationFeedbackLoop(
            expert_names=["math", "language", "code", "science"],
        )

        result = loop.check_and_respond(
            risk_score=0.1,
            routing_info={"current_expert": "math"},
        )

        assert result.action == FeedbackAction.NONE
        assert result.risk_level == "low"

    def test_high_risk_triggers_reroute(self):
        loop = HallucinationFeedbackLoop(
            expert_names=["math", "language", "code", "science"],
        )

        result = loop.check_and_respond(
            risk_score=0.6,
            routing_info={"current_expert": "math"},
        )

        assert result.action == FeedbackAction.REROUTE
        assert result.selected_expert != "math"

    def test_critical_risk_triggers_consolidation(self):
        loop = HallucinationFeedbackLoop(
            expert_names=["math", "language", "code", "science"],
        )
        # Clear rate limit
        loop.consolidation_trigger._steps_since_last_consolidation = 100

        result = loop.check_and_respond(
            risk_score=0.75,
            routing_info={"current_expert": "math"},
        )

        assert result.action == FeedbackAction.TRIGGER_CONSOLIDATION
        assert result.consolidation_triggered

    def test_full_correction_at_extreme_risk(self):
        loop = HallucinationFeedbackLoop(
            expert_names=["math", "language", "code", "science"],
        )
        loop.consolidation_trigger._steps_since_last_consolidation = 100

        result = loop.check_and_respond(
            risk_score=0.9,
            routing_info={"current_expert": "math"},
        )

        assert result.action == FeedbackAction.FULL_CORRECTION
        assert result.selected_expert != "math"
        assert result.consolidation_triggered

    def test_confidence_adjustment(self):
        loop = HallucinationFeedbackLoop(
            expert_names=["math", "language", "code", "science"],
        )

        result = loop.check_and_respond(
            risk_score=0.4,
            routing_info={"current_expert": "math"},
            base_confidence=0.95,
        )

        assert result.adjusted_confidence < 0.95

    def test_to_dict(self):
        loop = HallucinationFeedbackLoop(
            expert_names=["math", "language"],
        )

        result = loop.check_and_respond(
            risk_score=0.1,
            routing_info={"current_expert": "math"},
        )

        d = result.to_dict()
        assert "action" in d
        assert "adjusted_confidence" in d
        assert "selected_expert" in d
        assert "risk_level" in d

    def test_pending_consolidation(self):
        loop = HallucinationFeedbackLoop(
            expert_names=["math", "language"],
        )

        assert not loop.has_pending_consolidation()

        loop.trigger_memory_consolidation("test")
        assert loop.has_pending_consolidation()
        assert loop.get_consolidation_reason() == "test"

        loop.clear_consolidation_request()
        assert not loop.has_pending_consolidation()

    def test_expert_safety_tracking(self):
        loop = HallucinationFeedbackLoop(
            expert_names=["math", "language"],
        )

        # Update history
        loop.update_expert_hallucination_history("math", True)

        scores = loop.get_expert_safety_scores()
        assert scores["math"] < 1.0

    def test_step_increments_consolidation_counter(self):
        loop = HallucinationFeedbackLoop(
            expert_names=["math", "language"],
        )
        loop.consolidation_trigger.mark_consolidation_done()

        loop.step()
        loop.step()

        assert loop.consolidation_trigger._steps_since_last_consolidation == 2


class TestHallucinationFeedbackConfig:
    """Test config factory."""

    def test_default_config(self):
        config = HallucinationFeedbackConfig()
        assert config.enabled is True
        assert config.low_threshold == 0.3
        assert config.medium_threshold == 0.5
        assert config.high_threshold == 0.7
        assert config.critical_threshold == 0.85

    def test_get_thresholds(self):
        config = HallucinationFeedbackConfig(
            low_threshold=0.2,
            medium_threshold=0.4,
            high_threshold=0.6,
            critical_threshold=0.8,
        )
        thresholds = config.get_thresholds()
        assert thresholds.low == 0.2
        assert thresholds.critical == 0.8

    def test_create_feedback_loop(self):
        config = HallucinationFeedbackConfig(enabled=True)
        loop = config.create_feedback_loop(["math", "language"])

        assert isinstance(loop, HallucinationFeedbackLoop)
        assert loop.enabled is True


class TestHDIMIntegration:
    """Test integration with HDIMModel."""

    def test_feedback_disabled_by_default(self):
        from src.models.hdim_model import HDIMConfig, HDIMModel

        config = HDIMConfig(hidden_dim=64, num_domains=2)
        model = HDIMModel(config)

        assert model.hallucination_feedback_loop is None

    def test_feedback_enabled_in_config(self):
        from src.models.hdim_model import HDIMConfig, HDIMModel

        config = HDIMConfig(
            hidden_dim=64,
            num_domains=2,
            hallucination_feedback=True,
        )
        model = HDIMModel(config)

        assert model.hallucination_feedback_loop is not None

    def test_feedback_action_in_aux_state(self):
        from src.models.hdim_model import HDIMConfig, HDIMModel

        config = HDIMConfig(
            hidden_dim=64,
            num_domains=2,
            hallucination_detection=True,
            hallucination_feedback=True,
        )
        model = HDIMModel(config)
        model.eval()

        x = torch.randn(2, 64)
        domain_id = torch.tensor([0, 1])

        with torch.no_grad():
            output, rw, inv, so, aux = model.forward(x, domain_id, return_state=True)

        assert hasattr(aux, 'feedback_action')
        # feedback_action may be None if risk is low

    def test_feedback_flow_through_forward(self):
        from src.models.hdim_model import HDIMConfig, HDIMModel

        config = HDIMConfig(
            hidden_dim=64,
            num_domains=2,
            hallucination_detection=True,
            hallucination_feedback=True,
        )
        model = HDIMModel(config)
        model.eval()

        x = torch.randn(2, 64)
        domain_id = torch.tensor([0, 1])

        with torch.no_grad():
            output, rw, inv, so, aux = model.forward(x, domain_id, return_state=True)

        # Should have hallucination_risk computed
        assert hasattr(aux, 'hallucination_risk')
        assert aux.hallucination_risk >= 0.0

        # Should have feedback_action (may be None or a string)
        assert aux.feedback_action is None or isinstance(aux.feedback_action, str)
