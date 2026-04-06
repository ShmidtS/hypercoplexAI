"""Tests for MaxScoreRouter — Wang et al. ACL 2025 routing."""

import pytest
import torch
import torch.nn as nn

from src.core.maxscore_router import (
    MaxScoreRouter,
    RouterResult,
    RouterCheckpoint,
)
from src.core.moe_interface import MoERouter


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def router():
    """Basic MaxScoreRouter instance in eval mode."""
    r = MaxScoreRouter(
        hidden_dim=64,
        num_experts=8,
        top_k=2,
        temperature=10.0,
    )
    r.eval()
    return r


@pytest.fixture
def router_train():
    """MaxScoreRouter in training mode."""
    r = MaxScoreRouter(
        hidden_dim=64,
        num_experts=8,
        top_k=2,
        temperature=10.0,
    )
    r.train()
    return r


@pytest.fixture
def small_router():
    """Small router for quick tests."""
    return MaxScoreRouter(
        hidden_dim=32,
        num_experts=4,
        top_k=2,
    )


# ============================================================
# Basic Construction
# ============================================================

class TestMaxScoreRouterConstruction:
    """Test router initialization."""

    def test_basic_construction(self):
        """Router initializes with correct parameters."""
        router = MaxScoreRouter(
            hidden_dim=128,
            num_experts=16,
            top_k=4,
            temperature=5.0,
        )
        assert router.hidden_dim == 128
        assert router.num_experts == 16
        assert router.top_k == 4
        assert router.temperature == 5.0
        assert router.num_slots == 16

    def test_top_k_capped_at_num_experts(self):
        """top_k is capped at num_experts."""
        router = MaxScoreRouter(
            hidden_dim=64,
            num_experts=4,
            top_k=10,  # exceeds num_experts
        )
        assert router.top_k == 4

    def test_gate_weight_shape(self, router):
        """Gate has correct weight shape."""
        assert router.gate.weight.shape == (router.num_experts, router.hidden_dim)

    def test_train_scores_initialized_uniform(self, router):
        """train_scores initialized to uniform distribution."""
        expected = torch.ones(router.num_experts) / router.num_experts
        assert torch.allclose(router.train_scores, expected)

    def test_inherits_from_moe_router(self, router):
        """MaxScoreRouter inherits from MoERouter."""
        assert isinstance(router, MoERouter)

    def test_extra_repr(self, router):
        """extra_repr returns correct string."""
        repr_str = router.extra_repr()
        assert "hidden_dim=64" in repr_str
        assert "num_experts=8" in repr_str
        assert "top_k=2" in repr_str


# ============================================================
# Forward Pass
# ============================================================

class TestMaxScoreRouterForward:
    """Test forward pass."""

    def test_output_shape_2d(self, router):
        """Forward pass preserves 2D input shape."""
        x = torch.randn(16, 64)
        output, info = router(x)
        assert output.shape == x.shape

    def test_output_shape_3d(self, router):
        """Forward pass preserves 3D input shape."""
        x = torch.randn(4, 8, 64)
        output, info = router(x)
        assert output.shape == x.shape

    def test_output_no_nan(self, router):
        """Output has no NaN or Inf."""
        x = torch.randn(16, 64)
        output, _ = router(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_topk_indices_shape_2d(self, router):
        """topk_idx has correct shape for 2D input."""
        x = torch.randn(16, 64)
        _, info = router(x)
        assert info["topk_idx"].shape == (16, router.top_k)

    def test_topk_indices_shape_3d(self, router):
        """topk_idx has correct shape for 3D input."""
        x = torch.randn(4, 8, 64)
        _, info = router(x)
        assert info["topk_idx"].shape == (4, 8, router.top_k)

    def test_topk_indices_valid(self, router):
        """topk_idx values are valid expert indices."""
        x = torch.randn(16, 64)
        _, info = router(x)
        idx = info["topk_idx"]
        assert idx.min() >= 0
        assert idx.max() < router.num_experts

    def test_topk_weights_sum_to_one(self, router):
        """topk_weights sum to 1 per token."""
        x = torch.randn(16, 64)
        _, info = router(x)
        weights = info["topk_gate_weights"]
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gate_weights_sum_to_one(self, router):
        """gate_weights (full scores) sum to 1 per token."""
        x = torch.randn(16, 64)
        _, info = router(x)
        weights = info["gate_weights"]
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_expert_usage_shape(self, router):
        """expert_usage has correct shape."""
        x = torch.randn(16, 64)
        _, info = router(x)
        assert info["expert_usage"].shape == (router.num_experts,)

    def test_routing_entropy_positive(self, router):
        """routing_entropy is positive and finite."""
        x = torch.randn(16, 64)
        _, info = router(x)
        entropy = info["routing_entropy"]
        assert entropy.item() > 0
        assert torch.isfinite(entropy)

    def test_router_loss_non_negative(self, router):
        """router_loss is non-negative."""
        x = torch.randn(16, 64)
        _, info = router(x)
        assert info["router_loss"].item() >= 0

    def test_batch_size_1(self, router):
        """Forward works with batch_size=1."""
        x = torch.randn(1, 64)
        output, info = router(x)
        assert output.shape == (1, 64)
        assert info["topk_idx"].shape == (1, router.top_k)


# ============================================================
# SoftTopk Operator
# ============================================================

class TestSoftTopk:
    """Test SoftTopk operator."""

    def test_soft_topk_returns_correct_k(self, router):
        """_soft_topk returns exactly k elements."""
        scores = torch.softmax(torch.randn(10, 8), dim=-1)
        weights, indices = router._soft_topk(scores)
        assert weights.shape == (10, router.top_k)
        assert indices.shape == (10, router.top_k)

    def test_soft_topk_weights_sum_to_one(self, router):
        """SoftTopk weights sum to 1."""
        scores = torch.softmax(torch.randn(10, 8), dim=-1)
        weights, _ = router._soft_topk(scores)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_soft_topk_indices_valid(self, router):
        """SoftTopk indices are valid."""
        scores = torch.softmax(torch.randn(10, 8), dim=-1)
        _, indices = router._soft_topk(scores)
        assert indices.min() >= 0
        assert indices.max() < router.num_experts

    def test_soft_topk_no_nan(self, router):
        """SoftTopk produces no NaN."""
        scores = torch.softmax(torch.randn(10, 8), dim=-1)
        weights, indices = router._soft_topk(scores)
        assert not torch.isnan(weights).any()
        assert not torch.isnan(indices).any()

    def test_temperature_affects_sharpness(self):
        """Higher temperature = softer selection, lower = sharper."""
        scores = torch.softmax(torch.randn(10, 8), dim=-1)

        router_hot = MaxScoreRouter(64, 8, top_k=2, temperature=100.0)
        router_cold = MaxScoreRouter(64, 8, top_k=2, temperature=0.1)

        weights_hot, _ = router_hot._soft_topk(scores)
        weights_cold, _ = router_cold._soft_topk(scores)

        # Cold temperature should produce sharper (more peaked) weights
        # Hot temperature should produce softer (more uniform) weights
        # Measure sharpness as std deviation - higher std = sharper
        std_hot = weights_hot.std(dim=-1).mean()
        std_cold = weights_cold.std(dim=-1).mean()

        # Cold should have higher std (more peaked/sharp distribution)
        assert std_cold > std_hot


# ============================================================
# Entropy and Load Balance
# ============================================================

class TestEntropyAndLoadBalance:
    """Test entropy computation and load balancing."""

    def test_entropy_uniform_distribution(self, router):
        """Entropy is maximum for uniform distribution."""
        # Uniform distribution: p = 1/E for all experts
        uniform = torch.ones(100, router.num_experts) / router.num_experts
        entropy = router._compute_entropy(uniform)
        # Max entropy for uniform: log(E)
        max_entropy = torch.log(torch.tensor(router.num_experts, dtype=torch.float))
        assert torch.isclose(entropy, max_entropy, atol=1e-5)

    def test_entropy_deterministic_distribution(self, router):
        """Entropy is zero for deterministic (one-hot) distribution."""
        # One-hot: all mass on one expert
        one_hot = torch.zeros(100, router.num_experts)
        one_hot[:, 0] = 1.0
        entropy = router._compute_entropy(one_hot)
        assert torch.isclose(entropy, torch.tensor(0.0), atol=1e-5)

    def test_load_balance_loss_positive(self, router):
        """Load balance loss is positive."""
        scores = torch.softmax(torch.randn(100, router.num_experts), dim=-1)
        loss = router._compute_load_balance_loss(scores)
        assert loss.item() >= 0

    def test_load_balance_lower_for_uniform(self, router):
        """Load balance loss is lower for uniform distribution."""
        uniform = torch.ones(100, router.num_experts) / router.num_experts

        # Skewed distribution
        skewed = torch.zeros(100, router.num_experts)
        skewed[:, 0] = 0.9
        skewed[:, 1:] = 0.1 / (router.num_experts - 1)

        loss_uniform = router._compute_load_balance_loss(uniform)
        loss_skewed = router._compute_load_balance_loss(skewed)

        # Uniform should have lower loss (more balanced)
        assert loss_uniform < loss_skewed


# ============================================================
# Gradient Flow
# ============================================================

class TestGradientFlow:
    """Test gradient flow through router."""

    def test_input_gradient(self, router_train):
        """Gradients flow back to input."""
        x = torch.randn(16, 64, requires_grad=True)
        output, info = router_train(x)
        loss = output.mean() + info["router_loss"]
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gate_gradient(self, router_train):
        """Gradients flow to gate weights."""
        x = torch.randn(16, 64)
        output, info = router_train(x)
        loss = output.mean() + info["router_loss"]
        loss.backward()
        assert router_train.gate.weight.grad is not None
        assert not torch.isnan(router_train.gate.weight.grad).any()

    def test_gradient_no_nan_inf(self, router_train):
        """Gradients contain no NaN or Inf."""
        x = torch.randn(16, 64, requires_grad=True)
        output, info = router_train(x)
        loss = output.mean() + info["router_loss"]
        loss.backward()
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(router_train.gate.weight.grad).all()

    def test_entropy_weight_affects_gradient(self):
        """entropy_weight affects gradient magnitude."""
        router_high = MaxScoreRouter(64, 8, top_k=2, entropy_weight=1.0)
        router_low = MaxScoreRouter(64, 8, top_k=2, entropy_weight=0.001)

        router_high.train()
        router_low.train()

        x = torch.randn(16, 64, requires_grad=True)

        # High entropy weight
        _, info_high = router_high(x.clone())
        loss_high = info_high["router_loss"]
        loss_high.backward()
        grad_high = router_high.gate.weight.grad.clone().abs().mean()

        # Low entropy weight
        router_low.gate.weight.grad = None
        _, info_low = router_low(x.clone())
        loss_low = info_low["router_loss"]
        loss_low.backward()
        grad_low = router_low.gate.weight.grad.clone().abs().mean()

        # Different entropy weights should produce different gradients
        # (not testing magnitude direction, just that they differ)
        assert not torch.isclose(grad_high, grad_low, rtol=0.1)


# ============================================================
# Train Scores (EMA)
# ============================================================

class TestTrainScores:
    """Test EMA train_scores tracking."""

    def test_train_scores_updated_in_training(self, router_train):
        """train_scores updated during training."""
        initial = router_train.train_scores.clone()
        x = torch.randn(100, 64)
        router_train(x)
        assert not torch.equal(router_train.train_scores, initial)

    def test_train_scores_static_in_eval(self, router):
        """train_scores not updated in eval mode."""
        initial = router.train_scores.clone()
        x = torch.randn(100, 64)
        router(x)
        # Use allclose because floating point arithmetic
        assert torch.allclose(router.train_scores, initial, atol=1e-6)

    def test_reset_training_state(self, router_train):
        """reset_training_state resets to uniform."""
        # Modify train_scores
        router_train.train_scores.fill_(0.5)
        router_train.reset_training_state()
        expected = torch.ones(router_train.num_experts) / router_train.num_experts
        assert torch.allclose(router_train.train_scores, expected)

    def test_get_expert_load(self, router):
        """get_expert_load returns train_scores clone."""
        load = router.get_expert_load()
        assert torch.equal(load, router.train_scores)
        # Verify it's a clone, not the same tensor
        load.fill_(0.5)
        assert not torch.equal(load, router.train_scores)


# ============================================================
# Checkpoint/Rollback (MAJOR-3)
# ============================================================

class TestCheckpointRollback:
    """Test checkpoint/rollback mechanism."""

    def test_save_checkpoint(self, router):
        """save_checkpoint stores state."""
        router.save_checkpoint(step=100, loss_value=0.5)
        assert router.has_checkpoint()
        assert router._checkpoint_step == 100

    def test_checkpoint_content(self, router):
        """Checkpoint contains correct state."""
        router.gate.weight.data.fill_(0.123)
        router.train_scores.fill_(0.05)
        router.save_checkpoint(step=10)

        cp = router.get_checkpoint()
        assert cp is not None
        assert cp.step == 10
        assert torch.allclose(cp.gate_weight, torch.full_like(cp.gate_weight, 0.123))
        assert torch.allclose(cp.train_scores, torch.full_like(cp.train_scores, 0.05))

    def test_load_checkpoint(self, router):
        """load_checkpoint restores state."""
        # Save initial state as checkpoint
        original_weight = router.gate.weight.data.clone()
        original_scores = router.train_scores.clone()
        router.save_checkpoint(step=0)

        # Modify state
        router.gate.weight.data.fill_(0.999)
        router.train_scores.fill_(0.999)

        # Load checkpoint - should restore to original
        success = router.load_checkpoint()
        assert success
        assert torch.allclose(router.gate.weight.data, original_weight)
        assert torch.allclose(router.train_scores, original_scores)

    def test_rollback(self, router):
        """rollback reverts to checkpoint."""
        # Save initial state
        original_weight = router.gate.weight.data.clone()
        router.save_checkpoint(step=0)

        # Modify
        router.gate.weight.data.fill_(0.5)
        router.train_scores.fill_(0.5)

        # Rollback
        success = router.rollback()
        assert success

        # Should be back to initial state
        assert torch.allclose(router.gate.weight.data, original_weight)

    def test_rollback_no_checkpoint(self, router):
        """rollback returns False if no checkpoint."""
        router.clear_checkpoint()
        success = router.rollback()
        assert not success

    def test_load_checkpoint_no_checkpoint(self, router):
        """load_checkpoint returns False if no checkpoint."""
        router.clear_checkpoint()
        success = router.load_checkpoint()
        assert not success

    def test_clear_checkpoint(self, router):
        """clear_checkpoint removes checkpoint."""
        router.save_checkpoint(step=1)
        assert router.has_checkpoint()
        router.clear_checkpoint()
        assert not router.has_checkpoint()

    def test_checkpoint_metadata(self, router):
        """Checkpoint stores metadata."""
        router.save_checkpoint(step=42, loss_value=0.123)
        cp = router.get_checkpoint()
        assert cp.metadata["top_k"] == router.top_k
        assert cp.metadata["temperature"] == router.temperature
        assert cp.loss_value == 0.123

    def test_checkpoint_step_tracking(self, router):
        """Checkpoint step is tracked correctly."""
        router.save_checkpoint(step=100)
        assert router._checkpoint_step == 100

        router.save_checkpoint(step=200)
        assert router._checkpoint_step == 200


# ============================================================
# MoERouter Interface
# ============================================================

class TestMoERouterInterface:
    """Test MoERouter interface compliance."""

    def test_forward_returns_tuple(self, router):
        """forward returns (Tensor, Dict)."""
        x = torch.randn(8, 64)
        result = router.forward(x)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], dict)

    def test_forward_dict_required_keys(self, router):
        """forward info dict contains required keys."""
        x = torch.randn(8, 64)
        _, info = router.forward(x)

        required_keys = [
            "loss",
            "router_loss",
            "topk_idx",
            "gate_weights",
            "routing_entropy",
            "expert_usage",
        ]

        for key in required_keys:
            assert key in info, f"Missing key: {key}"

    def test_get_expert_load_returns_tensor(self, router):
        """get_expert_load returns Tensor[num_experts]."""
        load = router.get_expert_load()
        assert isinstance(load, torch.Tensor)
        assert load.shape == (router.num_experts,)

    def test_expert_orthogonalization_loss_returns_scalar(self, router):
        """expert_orthogonalization_loss returns scalar."""
        loss = router.expert_orthogonalization_loss()
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_expert_orthogonalization_loss_is_zero(self, router):
        """expert_orthogonalization_loss is zero for MaxScoreRouter."""
        loss = router.expert_orthogonalization_loss()
        assert loss.item() == 0.0

    def test_reset_training_state_exists(self, router):
        """reset_training_state method exists."""
        assert hasattr(router, "reset_training_state")
        router.reset_training_state()  # Should not raise


# ============================================================
# Z-Loss
# ============================================================

class TestZLoss:
    """Test router z-loss for stability."""

    def test_z_loss_zero_when_disabled(self):
        """z_loss is zero when z_loss_weight=0."""
        router = MaxScoreRouter(64, 8, top_k=2, z_loss_weight=0.0)
        x = torch.randn(16, 64)
        _, info = router(x)
        assert info["z_loss"].item() == 0.0

    def test_z_loss_positive_when_enabled(self):
        """z_loss is positive when enabled."""
        router = MaxScoreRouter(64, 8, top_k=2, z_loss_weight=0.01)
        x = torch.randn(16, 64)
        _, info = router(x)
        assert info["z_loss"].item() >= 0

    def test_z_loss_affects_total_loss(self):
        """z_loss affects total router_loss."""
        router_no_z = MaxScoreRouter(64, 8, top_k=2, z_loss_weight=0.0)
        router_with_z = MaxScoreRouter(64, 8, top_k=2, z_loss_weight=1.0)

        x = torch.randn(16, 64)

        _, info_no_z = router_no_z(x)
        _, info_with_z = router_with_z(x)

        # Router loss should be different when z_loss is included
        # (not testing magnitude, just that they differ)
        assert info_no_z["router_loss"] != info_with_z["router_loss"]


# ============================================================
# Routing Statistics
# ============================================================

class TestRoutingStats:
    """Test get_routing_stats utility."""

    def test_routing_stats_returns_dict(self, router):
        """get_routing_stats returns dict."""
        x = torch.randn(100, 64)
        stats = router.get_routing_stats(x)
        assert isinstance(stats, dict)

    def test_routing_stats_keys(self, router):
        """get_routing_stats contains expected keys."""
        x = torch.randn(100, 64)
        stats = router.get_routing_stats(x)

        expected_keys = ["entropy", "max_usage", "min_usage", "usage_std"]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_routing_stats_values_positive(self, router):
        """get_routing_stats values are valid."""
        x = torch.randn(100, 64)
        stats = router.get_routing_stats(x)

        assert stats["entropy"] > 0
        assert stats["max_usage"] >= 0
        assert stats["min_usage"] >= 0
        assert stats["max_usage"] >= stats["min_usage"]

    def test_routing_stats_no_gradients(self, router):
        """get_routing_stats doesn't compute gradients."""
        x = torch.randn(100, 64, requires_grad=True)
        stats = router.get_routing_stats(x)
        # Should not have computed gradients
        assert x.grad is None


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_expert(self):
        """Router works with single expert."""
        router = MaxScoreRouter(32, 1, top_k=1)
        x = torch.randn(8, 32)
        output, info = router(x)
        assert output.shape == (8, 32)
        assert info["topk_idx"].shape == (8, 1)
        assert info["expert_usage"].shape == (1,)

    def test_top_k_equals_num_experts(self):
        """Router works when top_k == num_experts."""
        router = MaxScoreRouter(32, 4, top_k=4)
        x = torch.randn(8, 32)
        output, info = router(x)
        assert info["topk_idx"].shape == (8, 4)

    def test_large_batch(self, router):
        """Router handles large batches."""
        x = torch.randn(1024, 64)
        output, info = router(x)
        assert output.shape == (1024, 64)
        assert info["topk_idx"].shape == (1024, router.top_k)

    def test_very_small_input(self):
        """Router handles very small input dimensions."""
        router = MaxScoreRouter(4, 2, top_k=1)
        x = torch.randn(8, 4)
        output, info = router(x)
        assert output.shape == (8, 4)

    def test_all_zeros_input(self, router):
        """Router handles zero input."""
        x = torch.zeros(8, 64)
        output, info = router(x)
        assert torch.isfinite(output).all()
        assert torch.isfinite(info["router_loss"]).all()

    def test_large_input_values(self, router):
        """Router handles large input values."""
        x = torch.randn(8, 64) * 100
        output, info = router(x)
        assert torch.isfinite(output).all()


# ============================================================
# Comparison with SoftMoERouter
# ============================================================

class TestVsSoftMoERouter:
    """Compare MaxScoreRouter with SoftMoERouter."""

    def test_both_implement_moe_router(self):
        """Both routers implement MoERouter interface."""
        from src.core.soft_moe_router import SoftMoERouter

        max_score = MaxScoreRouter(64, 8, top_k=2)
        soft_moe = SoftMoERouter(input_dim=64, num_experts=8, expert_dim=128)

        assert isinstance(max_score, MoERouter)
        assert isinstance(soft_moe, MoERouter)

    def test_different_routing_strategies(self):
        """MaxScore and SoftMoE use different routing."""
        from src.core.soft_moe_router import SoftMoERouter

        max_score = MaxScoreRouter(64, 8, top_k=2)
        soft_moe = SoftMoERouter(input_dim=64, num_experts=8, expert_dim=128)

        x = torch.randn(16, 64)

        _, info_max = max_score(x)
        _, info_soft = soft_moe(x)

        # MaxScore: top_k indices (sparse)
        # SoftMoE: dispatch/combine matrices (dense)
        assert "topk_idx" in info_max
        assert "dispatch_weights" in info_soft


# ============================================================
# Device Tests
# ============================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCuda:
    """Test router on CUDA device."""

    def test_cuda_forward(self):
        """Router works on CUDA."""
        router = MaxScoreRouter(64, 8, top_k=2).cuda()
        x = torch.randn(16, 64).cuda()
        output, info = router(x)
        assert output.device.type == "cuda"
        assert info["topk_idx"].device.type == "cuda"

    def test_cuda_gradient(self):
        """Gradients work on CUDA."""
        router = MaxScoreRouter(64, 8, top_k=2).cuda().train()
        x = torch.randn(16, 64, device="cuda", requires_grad=True)
        output, info = router(x)
        loss = output.mean() + info["router_loss"]
        loss.backward()
        assert x.grad is not None
        assert x.grad.device.type == "cuda"

    def test_cuda_checkpoint(self):
        """Checkpoint works on CUDA."""
        router = MaxScoreRouter(64, 8, top_k=2).cuda()
        router.save_checkpoint(step=1)
        assert router.has_checkpoint()

        cp = router.get_checkpoint()
        assert cp.gate_weight.device.type == "cuda"
