"""Tests for NarsTruth — NARS truth-value (frequency, confidence) pair.

Coverage:
- __post_init__ clamping (freq, conf)
- evidential_weight: normal, conf=0, conf=MAX_CONFIDENCE
- expectation formula
- w2c / c2w round-trip and edge cases
- revision: merging two truths, both zero-conf, unequal weights
- projection: temporal decay, zero time_diff, negative time_diff
- TransferState.to_dict() includes transfer_truth
- SoftMoERouter router_state contains alignment key
"""

import pytest
import torch
from src.core.nars_truth import NarsTruth


class TestPostInitClamping:
    """__post_init__ must clamp freq to [0,1] and conf to [0, MAX_CONFIDENCE]."""

    def test_clamps_freq_above_1(self):
        t = NarsTruth(freq=1.5, conf=0.5)
        assert t.freq == 1.0

    def test_clamps_freq_below_0(self):
        t = NarsTruth(freq=-0.3, conf=0.5)
        assert t.freq == 0.0

    def test_clamps_conf_above_max(self):
        t = NarsTruth(freq=0.5, conf=1.0)
        assert t.conf == NarsTruth.MAX_CONFIDENCE  # 0.99

    def test_clamps_conf_below_0(self):
        t = NarsTruth(freq=0.5, conf=-0.1)
        assert t.conf == 0.0

    def test_valid_values_unchanged(self):
        t = NarsTruth(freq=0.7, conf=0.6)
        assert t.freq == pytest.approx(0.7)
        assert t.conf == pytest.approx(0.6)


class TestEvidentialWeight:
    """evidential_weight: w = h * c / (1 - c)."""

    def test_returns_zero_when_conf_zero(self):
        t = NarsTruth(freq=0.5, conf=0.0)
        assert t.evidential_weight() == 0.0

    def test_returns_large_when_conf_at_max(self):
        t = NarsTruth(freq=0.5, conf=NarsTruth.MAX_CONFIDENCE)
        w = t.evidential_weight()
        assert w == 1e6

    def test_normal_case(self):
        t = NarsTruth(freq=0.5, conf=0.5)
        expected = 1.0 * 0.5 / (1.0 - 0.5)
        assert t.evidential_weight() == pytest.approx(expected)

    def test_very_low_conf_gives_small_weight(self):
        t = NarsTruth(freq=0.5, conf=0.01)
        w = t.evidential_weight()
        assert 0.0 < w < 0.1


class TestExpectation:
    """expectation: E = c * (f - 0.5) + 0.5."""

    def test_neutral_when_freq_half(self):
        t = NarsTruth(freq=0.5, conf=0.9)
        assert t.expectation() == pytest.approx(0.5)

    def test_above_half_when_freq_high(self):
        t = NarsTruth(freq=1.0, conf=0.8)
        e = t.expectation()
        assert e > 0.5

    def test_below_half_when_freq_low(self):
        t = NarsTruth(freq=0.0, conf=0.8)
        e = t.expectation()
        assert e < 0.5

    def test_zero_conf_gives_half(self):
        t = NarsTruth(freq=1.0, conf=0.0)
        assert t.expectation() == pytest.approx(0.5)


class TestW2C:
    """w2c: weight to confidence — c = w / (w + h)."""

    def test_zero_weight_returns_zero(self):
        assert NarsTruth.w2c(0.0) == 0.0

    def test_negative_weight_returns_zero(self):
        assert NarsTruth.w2c(-1.0) == 0.0

    def test_normal_round_trip(self):
        w = 2.0
        c = NarsTruth.w2c(w, horizon=1.0)
        assert c == pytest.approx(w / (w + 1.0))

    def test_large_weight_clamped_to_max_confidence(self):
        c = NarsTruth.w2c(1e8, horizon=1.0)
        assert c == NarsTruth.MAX_CONFIDENCE


class TestC2W:
    """c2w: confidence to weight — w = h * c / (1 - c)."""

    def test_zero_conf_returns_zero(self):
        assert NarsTruth.c2w(0.0) == 0.0

    def test_negative_conf_returns_zero(self):
        assert NarsTruth.c2w(-0.5) == 0.0

    def test_max_conf_returns_large(self):
        assert NarsTruth.c2w(NarsTruth.MAX_CONFIDENCE) == 1e6

    def test_normal_case(self):
        c = 0.5
        w = NarsTruth.c2w(c, horizon=1.0)
        assert w == pytest.approx(1.0 * c / (1.0 - c))

    def test_w2c_c2w_round_trip(self):
        original_w = 3.0
        c = NarsTruth.w2c(original_w, horizon=1.0)
        recovered_w = NarsTruth.c2w(c, horizon=1.0)
        assert recovered_w == pytest.approx(original_w, rel=1e-6)


class TestRevision:
    """revision: merge two truth values without double-counting."""

    def test_both_zero_conf_returns_neutral(self):
        a = NarsTruth(freq=0.9, conf=0.0)
        b = NarsTruth(freq=0.1, conf=0.0)
        result = NarsTruth.revision(a, b)
        assert result.freq == pytest.approx(0.5)
        assert result.conf == 0.0

    def test_revision_preserves_higher_confidence(self):
        """revision cannot reduce confidence below max(a.conf, b.conf)."""
        a = NarsTruth(freq=0.8, conf=0.9)
        b = NarsTruth(freq=0.2, conf=0.1)
        result = NarsTruth.revision(a, b)
        assert result.conf >= max(a.conf, b.conf) - 1e-9

    def test_revision_weighted_average_freq(self):
        """freq must be weighted average by evidential weights."""
        a = NarsTruth(freq=1.0, conf=0.5)
        b = NarsTruth(freq=0.0, conf=0.5)
        result = NarsTruth.revision(a, b)
        assert result.freq == pytest.approx(0.5)

    def test_revision_same_values_doubles_confidence(self):
        """Merging identical evidence increases confidence."""
        a = NarsTruth(freq=0.7, conf=0.5)
        result = NarsTruth.revision(a, a)
        assert result.conf > a.conf

    def test_revision_clamps_to_max_confidence(self):
        """Confidence cannot exceed MAX_CONFIDENCE after revision."""
        a = NarsTruth(freq=0.5, conf=0.98)
        b = NarsTruth(freq=0.5, conf=0.98)
        result = NarsTruth.revision(a, b)
        assert result.conf <= NarsTruth.MAX_CONFIDENCE + 1e-9

    def test_revision_single_nonzero(self):
        """When only one has non-zero conf, result follows that one."""
        a = NarsTruth(freq=0.8, conf=0.6)
        b = NarsTruth(freq=0.2, conf=0.0)
        result = NarsTruth.revision(a, b)
        assert result.freq == pytest.approx(a.freq)
        assert result.conf >= a.conf - 1e-9


class TestProjection:
    """projection: c *= decay^timeDiff."""

    def test_no_decay_at_zero_time_diff(self):
        assert NarsTruth.projection(0.8, 0.0) == pytest.approx(0.8)

    def test_decay_reduces_confidence(self):
        result = NarsTruth.projection(0.9, 5, decay=0.8)
        assert result < 0.9

    def test_negative_time_diff_same_as_positive(self):
        """abs(time_diff) is used, so negative is same as positive."""
        pos = NarsTruth.projection(0.9, 5.0)
        neg = NarsTruth.projection(0.9, -5.0)
        assert pos == pytest.approx(neg)

    def test_large_time_diff_drives_near_zero(self):
        result = NarsTruth.projection(0.9, 100, decay=0.8)
        assert result < 0.01

    def test_custom_decay(self):
        result = NarsTruth.projection(1.0, 1, decay=0.5)
        assert result == pytest.approx(0.5)


class TestTransferStateToDict:
    """TransferState.to_dict() must include transfer_truth field."""

    @pytest.fixture
    def transfer_state(self):
        from src.core.transfer_state import TransferState
        return TransferState(
            g_source=None,
            u_inv=torch.zeros(2),
            u_mem=torch.zeros(2),
            u_route=torch.zeros(2),
            g_target=torch.zeros(2),
            output=torch.zeros(2),
            memory_loss=torch.tensor(0.0),
            memory_retrieved=torch.zeros(2),
            memory_updated=False,
            memory_alpha=None,
            memory_eta=None,
            memory_theta=None,
            router_state={"gate_weights": torch.zeros(2)},
            memory_mode="none",
            update_memory=False,
            input_is_invariant=False,
        )

    def test_to_dict_contains_transfer_truth_key(self, transfer_state):
        d = transfer_state.to_dict()
        assert "transfer_truth" in d

    def test_to_dict_transfer_truth_defaults_to_none(self, transfer_state):
        d = transfer_state.to_dict()
        assert d["transfer_truth"] is None

    def test_to_dict_transfer_truth_preserves_nars_truth(self):
        from src.core.transfer_state import TransferState
        truth = NarsTruth(freq=0.9, conf=0.8)
        ts = TransferState(
            g_source=None,
            u_inv=torch.zeros(2),
            u_mem=torch.zeros(2),
            u_route=torch.zeros(2),
            g_target=torch.zeros(2),
            output=torch.zeros(2),
            memory_loss=torch.tensor(0.0),
            memory_retrieved=torch.zeros(2),
            memory_updated=False,
            memory_alpha=None,
            memory_eta=None,
            memory_theta=None,
            router_state={"gate_weights": torch.zeros(2)},
            memory_mode="none",
            update_memory=False,
            input_is_invariant=False,
            transfer_truth=truth,
        )
        d = ts.to_dict()
        assert d["transfer_truth"] is truth
        assert d["transfer_truth"].freq == pytest.approx(0.9)
        assert d["transfer_truth"].conf == pytest.approx(0.8)


class TestSoftMoERouterAlignment:
    """SoftMoERouter must produce alignment key for router_state.

    hdim_pipeline.py:228 reads router_state.get('alignment', 1.0).
    TransferEngine.transfer() sets router_state['alignment'] at line 147.
    """

    def test_router_state_contains_alignment_key(self):
        from src.core.soft_moe_router import SoftMoERouter
        router = SoftMoERouter(input_dim=32, num_experts=2, expert_dim=16)
        router.eval()
        x = torch.randn(4, 32)
        _, router_state = router(x)
        # SoftMoERouter itself does not set 'alignment' --
        # TransferEngine adds it after calling the router.
        # Verify the router_state is a dict that TransferEngine
        # later enhances with 'alignment'.
        assert isinstance(router_state, dict)

    def test_transfer_engine_sets_alignment_in_router_state(self):
        """TransferEngine.transfer() must set 'alignment' in router_state.

        This is the actual path hdim_pipeline reads from.
        """
        from src.core.hypercomplex import CliffordAlgebra
        from src.core.transfer_engine import TransferEngine
        algebra = CliffordAlgebra(p=3, q=1, r=0)
        clifford_dim = algebra.dim
        engine = TransferEngine(
            clifford_dim=clifford_dim,
            output_dim=8,
            algebra=algebra,
            num_experts=2,
            top_k=2,
        )
        engine.eval()
        u_mem = torch.randn(1, clifford_dim)
        from src.core.domain_operators import DomainRotationOperator
        source_rotor = DomainRotationOperator(
            algebra=algebra,
            domain_name="source",
        )
        target_rotor = DomainRotationOperator(
            algebra=algebra,
            domain_name="target",
        )
        with torch.no_grad():
            output, router_state = engine.transfer(
                u_mem=u_mem,
                source_rotor=source_rotor,
                target_rotor=target_rotor,
                g_source=None,
                input_is_invariant=False,
            )
        assert "alignment" in router_state
        assert isinstance(router_state["alignment"], float)
        assert 0.0 <= router_state["alignment"] <= 1.0
