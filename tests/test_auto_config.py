"""
Unit tests for AutoConfig — automatic parameter derivation.

Tests cover:
- compute_* functions (pure, unit-testable)
- AutoConfig derivation chain
- Validation rules
- Conversion methods
- Factory methods
"""

import pytest

from src.core.auto_config import (
    AutoConfig,
    compute_clifford_dim,
    compute_expert_hidden_dim,
    compute_memory_key_dim,
    compute_num_experts,
    get_encoder_dim,
    validate_quaternion_dim,
    ENCODER_DIMS,
)


# ============================================================
# Test compute_* functions
# ============================================================

class TestComputeCliffordDim:
    """Tests for compute_clifford_dim."""
    
    def test_basic_signatures(self):
        """Test standard Clifford algebra signatures."""
        assert compute_clifford_dim(3, 1, 0) == 16  # Spacetime algebra
        assert compute_clifford_dim(4, 1, 0) == 32  # Phase 25 signature
        assert compute_clifford_dim(0, 0, 0) == 1   # Trivial case
        assert compute_clifford_dim(1, 0, 0) == 2   # Complex numbers
        assert compute_clifford_dim(0, 1, 0) == 2   # Dual numbers
        assert compute_clifford_dim(0, 2, 0) == 4   # Quaternions
    
    def test_nilpotent_basis(self):
        """Test signatures with nilpotent basis vectors."""
        assert compute_clifford_dim(2, 1, 1) == 16  # With one nilpotent
        assert compute_clifford_dim(2, 2, 2) == 64  # Multiple nilpotents
    
    def test_negative_signature_raises(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError, match="all must be >= 0"):
            compute_clifford_dim(-1, 1, 0)
        with pytest.raises(ValueError, match="all must be >= 0"):
            compute_clifford_dim(1, -1, 0)


class TestComputeExpertHiddenDim:
    """Tests for compute_expert_hidden_dim."""
    
    def test_default_multiplier(self):
        """Test default 2x expansion."""
        assert compute_expert_hidden_dim(16) == 32
        assert compute_expert_hidden_dim(32) == 64
        assert compute_expert_hidden_dim(64) == 128
    
    def test_custom_multiplier(self):
        """Test custom expansion factors."""
        assert compute_expert_hidden_dim(32, 4) == 128
        assert compute_expert_hidden_dim(32, 1) == 32
    
    def test_invalid_inputs(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="clifford_dim must be positive"):
            compute_expert_hidden_dim(0)
        with pytest.raises(ValueError, match="multiplier must be positive"):
            compute_expert_hidden_dim(32, 0)


class TestComputeMemoryKeyDim:
    """Tests for compute_memory_key_dim."""
    
    def test_default_divisor(self):
        """Test default halving."""
        assert compute_memory_key_dim(32) == 16
        assert compute_memory_key_dim(64) == 32
    
    def test_custom_divisor(self):
        """Test custom compression factors."""
        assert compute_memory_key_dim(32, 4) == 8
        assert compute_memory_key_dim(32, 1) == 32
    
    def test_minimum_one(self):
        """Test that minimum is 1."""
        assert compute_memory_key_dim(1) == 1
        assert compute_memory_key_dim(2, 4) == 1
    
    def test_invalid_inputs(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="clifford_dim must be positive"):
            compute_memory_key_dim(0)
        with pytest.raises(ValueError, match="divisor must be positive"):
            compute_memory_key_dim(32, 0)


class TestComputeNumExperts:
    """Tests for compute_num_experts."""
    
    def test_from_expert_names(self):
        """Test deriving from expert_names."""
        assert compute_num_experts(["math", "code"], None) == 2
        assert compute_num_experts(["a", "b", "c"], None) == 3
    
    def test_from_explicit_num(self):
        """Test using explicit number."""
        assert compute_num_experts(None, 8) == 8
        assert compute_num_experts(None, 1) == 1
    
    def test_default(self):
        """Test default when neither provided."""
        assert compute_num_experts(None, None) == 4
    
    def test_conflict_raises(self):
        """Test that conflicting values raise."""
        with pytest.raises(ValueError, match="conflicts with"):
            compute_num_experts(["a", "b"], 3)
    
    def test_matching_values(self):
        """Test that matching values are accepted."""
        assert compute_num_experts(["a", "b"], 2) == 2


class TestGetEncoderDim:
    """Tests for get_encoder_dim."""
    
    def test_encoder_types(self):
        """Test encoder type lookup."""
        assert get_encoder_dim("sbert") == 768
        assert get_encoder_dim("modernbert") == 768
        assert get_encoder_dim("custom") == 128
    
    def test_encoder_names_exact_match(self):
        """Test exact encoder name matches."""
        assert get_encoder_dim("sbert", "all-minilm-l6-v2") == 384
        assert get_encoder_dim("modernbert", "answerdotai/modernbert-large") == 1024
        assert get_encoder_dim("sbert", "answerdotai/modernbert-base") == 768
    
    def test_unknown_encoder_name(self):
        """Test error for unknown encoder."""
        with pytest.raises(ValueError, match="Unknown encoder_name"):
            get_encoder_dim("sbert", "unknown-model")


class TestValidateQuaternionDim:
    """Tests for validate_quaternion_dim."""
    
    def test_valid_dimensions(self):
        """Test dimensions divisible by 4."""
        assert validate_quaternion_dim(4) == []
        assert validate_quaternion_dim(768) == []
        assert validate_quaternion_dim(1024) == []
    
    def test_invalid_dimensions(self):
        """Test dimensions not divisible by 4."""
        # 101 is NOT divisible by 4 (101 % 4 = 1)
        errors = validate_quaternion_dim(101)
        assert len(errors) == 1
        assert "must be divisible by 4" in errors[0]
        assert "101" in errors[0]


# ============================================================
# Test AutoConfig class
# ============================================================

class TestAutoConfigBasic:
    """Basic AutoConfig tests."""
    
    def test_default_initialization(self):
        """Test default values with relaxed validation."""
        cfg = AutoConfig(strict_validation=False)
        assert cfg.encoder_type == "sbert"
        assert cfg.encoder_output_dim == 768
        assert cfg.hidden_dim_resolved == 768
        assert cfg.clifford_dim == 32  # Cl(4,1,0)
        assert cfg.num_experts_resolved == 4
    
    def test_modernbert_encoder(self):
        """Test ModernBERT encoder type."""
        cfg = AutoConfig(encoder_type="modernbert", strict_validation=False)
        assert cfg.encoder_output_dim == 768
        assert cfg.hidden_dim_resolved == 768
    
    def test_custom_encoder_name(self):
        """Test custom encoder name."""
        cfg = AutoConfig(encoder_name="answerdotai/ModernBERT-large", strict_validation=False)
        assert cfg.encoder_output_dim == 1024
        assert cfg.hidden_dim_resolved == 1024
    
    def test_explicit_hidden_dim(self):
        """Test explicit hidden_dim override."""
        cfg = AutoConfig(hidden_dim=256, strict_validation=False)
        assert cfg.hidden_dim_resolved == 256
    
    def test_expert_names_derivation(self):
        """Test num_experts from expert_names."""
        cfg = AutoConfig(expert_names=["math", "code", "science"], strict_validation=False)
        assert cfg.num_experts_resolved == 3


class TestAutoConfigDerivation:
    """Test derivation chain."""
    
    def test_clifford_dim_derivation(self):
        """Test clifford_dim from p,q,r."""
        cfg = AutoConfig(clifford_p=3, clifford_q=1, clifford_r=0, strict_validation=False)
        assert cfg.clifford_dim == 16
        
        cfg2 = AutoConfig(clifford_p=5, clifford_q=2, clifford_r=1, strict_validation=False)
        assert cfg2.clifford_dim == 256  # 2^8
    
    def test_expert_hidden_dim_derivation(self):
        """Test expert_hidden_dim from clifford_dim."""
        cfg = AutoConfig(strict_validation=False)  # clifford_dim=32
        assert cfg.expert_hidden_dim_resolved == 64  # 32 * 2
        
        # Explicit override
        cfg2 = AutoConfig(expert_hidden_dim=128, strict_validation=False)
        assert cfg2.expert_hidden_dim_resolved == 128
    
    def test_memory_key_dim_derivation(self):
        """Test memory_key_dim from clifford_dim."""
        cfg = AutoConfig(strict_validation=False)  # clifford_dim=32
        assert cfg.memory_key_dim_resolved == 16  # 32 // 2
        
        # Explicit override
        cfg2 = AutoConfig(memory_key_dim=64, strict_validation=False)
        assert cfg2.memory_key_dim_resolved == 64


class TestAutoConfigValidation:
    """Test validation rules."""
    
    def test_quaternion_validation_pass(self):
        """Test quaternion divisibility validation passes for valid dims."""
        # Valid
        cfg = AutoConfig(hidden_dim=768, clifford_p=7, clifford_q=1, strict_validation=False)
        assert cfg.hidden_dim_resolved == 768
    
    def test_quaternion_validation_fail(self):
        """Test quaternion divisibility validation fails for invalid dims."""
        # Invalid - 101 is not divisible by 4 (101 % 4 = 1)
        with pytest.raises(ValueError, match="must be divisible by 4"):
            AutoConfig(hidden_dim=101)
    
    def test_small_clifford_dim_warning(self):
        """Test warning for small clifford_dim."""
        with pytest.raises(ValueError, match="clifford_dim=8 is very small"):
            AutoConfig(clifford_p=2, clifford_q=1, clifford_r=0, hidden_dim=64)
    
    def test_hidden_dim_large_vs_clifford(self):
        """Test warning for hidden_dim >> clifford_dim."""
        # 768 > 32 * 16 = 512, should fail with strict validation
        with pytest.raises(ValueError, match="aggressive projection"):
            AutoConfig(hidden_dim=768)  # clifford_dim=32
    
    def test_expert_names_conflict(self):
        """Test num_experts/expert_names conflict."""
        with pytest.raises(ValueError, match="conflicts with"):
            AutoConfig(expert_names=["a", "b"], num_experts=3, strict_validation=False)


class TestAutoConfigConversion:
    """Test conversion methods."""
    
    def test_to_hdim_config(self):
        """Test HDIMConfig conversion."""
        cfg = AutoConfig(
            encoder_type="modernbert",
            expert_names=["math", "code"],
            hidden_dim=256,
            strict_validation=False,
        )
        hdim_cfg = cfg.to_hdim_config(dropout=0.2)
        
        assert hdim_cfg.hidden_dim == 256
        assert hdim_cfg.num_experts == 2
        assert hdim_cfg.expert_names == ["math", "code"]
        assert hdim_cfg.dropout == 0.2
    
    def test_to_hdim_config_with_num_domains(self):
        """Test HDIMConfig conversion with num_domains."""
        cfg = AutoConfig(hidden_dim=256, strict_validation=False)
        hdim_cfg = cfg.to_hdim_config(num_domains=8)
        assert hdim_cfg.num_domains == 8
    
    def test_to_moe_kernel_config(self):
        """Test MoEKernelConfig conversion."""
        cfg = AutoConfig(expert_names=["math", "language"], strict_validation=False)
        moe_cfg = cfg.to_moe_kernel_config(use_shared_expert=False)
        
        assert moe_cfg.input_dim == 32  # clifford_dim
        assert moe_cfg.num_experts == 2
        assert moe_cfg.use_shared_expert is False
    
    def test_to_experiment_config(self):
        """Test ExperimentConfig conversion."""
        cfg = AutoConfig(hidden_dim=128, strict_validation=False)
        exp_cfg = cfg.to_experiment_config(epochs=10, batch_size=32)
        
        assert exp_cfg.hidden_dim == 128
        assert exp_cfg.epochs == 10
        assert exp_cfg.batch_size == 32


class TestAutoConfigFactoryMethods:
    """Test factory methods."""
    
    def test_from_encoder(self):
        """Test from_encoder factory."""
        cfg = AutoConfig.from_encoder("modernbert", expert_names=["math"], strict_validation=False)
        assert cfg.encoder_type == "modernbert"
        assert cfg.num_experts_resolved == 1
    
    def test_from_clifford_signature(self):
        """Test from_clifford_signature factory."""
        cfg = AutoConfig.from_clifford_signature(3, 1, 0)
        assert cfg.clifford_dim == 16
        assert cfg.hidden_dim_resolved == 16  # defaults to clifford_dim
        
        cfg2 = AutoConfig.from_clifford_signature(4, 1, 0, hidden_dim=64)
        assert cfg2.clifford_dim == 32
        assert cfg2.hidden_dim_resolved == 64
    
    def test_from_hdim_config(self):
        """Test from_hdim_config factory."""
        from src.models.hdim_model import HDIMConfig
        
        hdim_cfg = HDIMConfig(
            hidden_dim=128,
            num_experts=8,
            clifford_p=3,
            clifford_q=1,
        )
        auto_cfg = AutoConfig.from_hdim_config(hdim_cfg)
        
        assert auto_cfg.hidden_dim_resolved == 128
        assert auto_cfg.num_experts_resolved == 8
        assert auto_cfg.clifford_dim == 16


class TestAutoConfigRepr:
    """Test representation methods."""
    
    def test_repr(self):
        """Test __repr__."""
        cfg = AutoConfig(strict_validation=False)
        repr_str = repr(cfg)
        assert "AutoConfig" in repr_str
        assert "hidden_dim=" in repr_str
        assert "clifford_dim=" in repr_str
    
    def test_summary(self):
        """Test summary method."""
        cfg = AutoConfig(encoder_type="modernbert", expert_names=["math"], strict_validation=False)
        summary = cfg.summary()
        assert "AutoConfig Summary" in summary
        assert "modernbert" in summary
        assert "math" in summary


# ============================================================
# Integration tests
# ============================================================

class TestAutoConfigIntegration:
    """Integration tests with real HDIM components."""
    
    def test_full_pipeline_config(self):
        """Test creating a full pipeline configuration."""
        cfg = AutoConfig(
            encoder_type="modernbert",
            encoder_name="answerdotai/ModernBERT-base",
            expert_names=["math", "language", "code", "science"],
            clifford_p=4,
            clifford_q=1,
            strict_validation=False,
        )
        
        # Verify all derived values
        assert cfg.encoder_output_dim == 768
        assert cfg.hidden_dim_resolved == 768
        assert cfg.clifford_dim == 32
        assert cfg.num_experts_resolved == 4
        assert cfg.expert_hidden_dim_resolved == 64
        assert cfg.memory_key_dim_resolved == 16
        
        # Convert to all config types
        hdim_cfg = cfg.to_hdim_config()
        moe_cfg = cfg.to_moe_kernel_config()
        exp_cfg = cfg.to_experiment_config()
        
        # Verify consistency
        assert hdim_cfg.hidden_dim == exp_cfg.hidden_dim
        assert hdim_cfg.num_experts == moe_cfg.num_experts
    
    def test_unknown_encoder_with_hidden_dim(self):
        """Test unknown encoder with explicit hidden_dim works."""
        # This should work with explicit hidden_dim (encoder_type='custom' is set implicitly)
        cfg = AutoConfig(encoder_name="custom-encoder-xyz", hidden_dim=256, strict_validation=False)
        assert cfg.hidden_dim_resolved == 256
    
    def test_unknown_encoder_without_hidden_dim(self):
        """Test unknown encoder without hidden_dim raises."""
        # This should fail without hidden_dim
        with pytest.raises(ValueError, match="specify hidden_dim explicitly"):
            AutoConfig(encoder_name="completely-unknown-model")


# ============================================================
# Edge cases
# ============================================================

class TestAutoConfigEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_expert(self):
        """Test single expert configuration."""
        cfg = AutoConfig(expert_names=["math"], strict_validation=False)
        assert cfg.num_experts_resolved == 1
    
    def test_large_clifford_dim(self):
        """Test large Clifford dimensions."""
        cfg = AutoConfig(clifford_p=8, clifford_q=0, clifford_r=0, hidden_dim=1024, strict_validation=False)
        assert cfg.clifford_dim == 256
        assert cfg.expert_hidden_dim_resolved == 512
    
    def test_empty_expert_names(self):
        """Test empty expert_names list."""
        # Empty list should work (0 experts, though unusual)
        cfg = AutoConfig(expert_names=[], strict_validation=False)
        assert cfg.num_experts_resolved == 0
    
    def test_very_small_hidden_dim(self):
        """Test minimal hidden dimensions."""
        # Should work if divisible by 4
        cfg = AutoConfig(hidden_dim=4, clifford_p=2, clifford_q=0, clifford_r=0, strict_validation=False)
        assert cfg.hidden_dim_resolved == 4
        assert cfg.clifford_dim == 4
    
    def test_strict_validation_disable(self):
        """Test that strict_validation=False bypasses checks."""
        # This would normally fail quaternion check
        cfg = AutoConfig(hidden_dim=100, strict_validation=False)
        assert cfg.hidden_dim_resolved == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
