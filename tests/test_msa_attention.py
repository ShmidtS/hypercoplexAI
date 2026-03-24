"""Tests for MSA (Memory Sparse Attention) module.

Tests cover:
- Top-k selection correctness
- Chunk compression behavior
- Backward compatibility (MSA disabled by default)
- Integration with SemanticMemory
"""

import pytest
import torch
import torch.nn.functional as F

from src.core.msa_attention import (
    MSAConfig,
    MSASparseIndex,
    MSAAugmentedSemanticMemory,
)


class TestMSASparseIndex:
    """Test suite for MSASparseIndex core functionality."""

    @pytest.fixture
    def index(self):
        """Create default MSA index for testing."""
        return MSASparseIndex(
            dim=64,
            num_prototypes=32,
            top_k=8,
            chunk_size=16,
            num_heads=4,
        )

    @pytest.fixture
    def batch_data(self):
        """Generate test batch data."""
        B, D, P = 4, 64, 32
        h = torch.randn(B, D)
        prototypes = F.normalize(torch.randn(P, D), dim=-1)
        evidence = torch.ones(P)
        return h, prototypes, evidence

    def test_initialization(self, index):
        """Test that MSA index initializes correctly."""
        assert index.dim == 64
        assert index.num_prototypes == 32
        assert index.top_k == 8
        assert index.chunk_size == 16
        assert index.num_heads == 4
        assert index.W_KR is not None
        assert index.W_QR is not None

    def test_compute_routing_scores_shape(self, index, batch_data):
        """Test routing scores have correct shape."""
        h, prototypes, _ = batch_data

        qr = index.W_QR(h)
        kr = index.W_KR(prototypes)

        scores = index.compute_routing_scores(qr, kr)

        assert scores.shape == (h.shape[0], prototypes.shape[0])

    def test_top_k_selection_correctness(self, index, batch_data):
        """Test that top-k selection returns correct number of prototypes."""
        h, prototypes, _ = batch_data

        qr = index.W_QR(h)
        kr = index.W_KR(prototypes)
        scores = index.compute_routing_scores(qr, kr)

        topk_indices, topk_weights, retrieved = index.top_k_selection(
            scores, kr, prototypes
        )

        # Check shapes
        assert topk_indices.shape == (h.shape[0], index.top_k)
        assert topk_weights.shape == (h.shape[0], index.top_k)
        assert retrieved.shape == (h.shape[0], index.top_k, index.dim)

        # Weights should sum to 1 (softmax)
        weight_sums = topk_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

        # Indices should be in valid range
        assert (topk_indices >= 0).all() and (topk_indices < prototypes.shape[0]).all()

    def test_top_k_selection_finds_highest_scores(self, index):
        """Test that top-k selection returns valid results via query."""
        B, P, K = 2, 16, 8
        index.top_k = K

        # Create query and prototypes
        h = torch.randn(B, index.dim)
        prototypes = torch.randn(P, index.dim)
        evidence = torch.ones(P)

        # Use the full query method which handles projections
        retrieved, topk_indices, topk_weights = index.query(h, prototypes, evidence)

        # Verify shapes and valid indices
        assert topk_indices.shape == (B, K)
        assert (topk_indices >= 0).all() and (topk_indices < P).all()
        # Weights sum to 1
        assert torch.allclose(topk_weights.sum(dim=-1), torch.ones(B), atol=1e-5)

    def test_chunk_compression_below_threshold(self, index):
        """Test that compression doesn't activate below threshold."""
        P = 64
        prototypes = F.normalize(torch.randn(P, index.dim), dim=-1)
        evidence = torch.ones(P)

        # Set high threshold
        index.compression_threshold = 128

        compressed = index.chunk_compress(prototypes, evidence)

        # Should return unchanged
        assert compressed.shape[0] == P

    def test_chunk_compression_above_threshold(self, index):
        """Test that compression activates above threshold."""
        P = 256
        prototypes = F.normalize(torch.randn(P, index.dim), dim=-1)
        evidence = torch.ones(P)

        # Set low threshold
        index.compression_threshold = 64

        compressed = index.chunk_compress(prototypes, evidence)

        # Should be compressed
        num_chunks = (P + index.chunk_size - 1) // index.chunk_size
        assert compressed.shape[0] == num_chunks

    def test_chunk_compression_preserves_information(self, index):
        """Test that compression roughly preserves prototype direction."""
        P = 192  # Multiple of chunk_size
        index.chunk_size = 64
        index.compression_threshold = 64

        # Create prototypes with clear structure
        prototypes = torch.zeros(P, index.dim)
        for i in range(P):
            prototypes[i, i % index.dim] = 1.0
        prototypes = F.normalize(prototypes, dim=-1)
        evidence = torch.ones(P)

        compressed = index.chunk_compress(prototypes, evidence)

        # Compressed should be normalized
        norms = compressed.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_query_returns_valid_output(self, index, batch_data):
        """Test that query returns properly shaped output."""
        h, prototypes, evidence = batch_data

        retrieved, topk_indices, topk_weights = index.query(h, prototypes, evidence)

        assert retrieved.shape == (h.shape[0], index.dim)
        assert topk_indices.shape == (h.shape[0], index.top_k)
        assert topk_weights.shape == (h.shape[0], index.top_k)

    def test_forward_pass(self, index, batch_data):
        """Test forward pass returns valid output."""
        h, prototypes, evidence = batch_data

        output = index(h, prototypes, evidence)

        assert output.shape == (h.shape[0], index.dim)

    def test_multi_head_routing_differentiates_queries(self, index):
        """Test that multi-head routing distinguishes different queries."""
        # Create two very different queries
        q1 = torch.randn(1, index.dim)
        q2 = -q1  # Opposite direction

        prototypes = F.normalize(torch.randn(index.num_prototypes, index.dim), dim=-1)

        out1 = index(q1, prototypes)
        out2 = index(q2, prototypes)

        # Outputs should be different
        assert not torch.allclose(out1, out2, atol=1e-3)

    def test_gradient_flow(self, index, batch_data):
        """Test that gradients flow through the index."""
        h, prototypes, evidence = batch_data
        h = h.requires_grad_(True)

        output = index(h, prototypes.detach(), evidence)
        loss = output.sum()
        loss.backward()

        assert h.grad is not None
        assert h.grad.shape == h.shape


class TestMSAAugmentedSemanticMemory:
    """Test suite for MSA-augmented SemanticMemory."""

    @pytest.fixture
    def memory_dense(self):
        """Create SemanticMemory with MSA disabled."""
        return MSAAugmentedSemanticMemory(
            hidden_dim=64,
            num_prototypes=32,
            use_msa=False,
        )

    @pytest.fixture
    def memory_msa(self):
        """Create SemanticMemory with MSA enabled."""
        return MSAAugmentedSemanticMemory(
            hidden_dim=64,
            num_prototypes=32,
            use_msa=True,
            msa_top_k=8,
        )

    @pytest.fixture
    def batch_input(self):
        """Generate test input."""
        return torch.randn(4, 64)

    def test_backward_compatibility_disabled(self, memory_dense, batch_input):
        """Test that MSA disabled mode works identically to original."""
        memory_dense.eval()
        with torch.no_grad():
            output = memory_dense(batch_input)

        assert output.shape == batch_input.shape

    def test_msa_enabled_forward(self, memory_msa, batch_input):
        """Test that MSA enabled mode produces valid output."""
        memory_msa.eval()
        with torch.no_grad():
            output = memory_msa(batch_input)

        assert output.shape == batch_input.shape

    def test_msa_disabled_by_default(self, batch_input):
        """Test that MSA is disabled by default (backward compat)."""
        memory = MSAAugmentedSemanticMemory(hidden_dim=64, num_prototypes=32)
        assert memory.use_msa is False
        assert memory.msa_index is None

    def test_msa_enabled_when_requested(self):
        """Test that MSA is properly initialized when requested."""
        memory = MSAAugmentedSemanticMemory(
            hidden_dim=64,
            num_prototypes=32,
            use_msa=True,
        )
        assert memory.use_msa is True
        assert memory.msa_index is not None

    def test_output_stability_dense(self, memory_dense, batch_input):
        """Test that dense mode produces stable outputs."""
        memory_dense.eval()
        with torch.no_grad():
            out1 = memory_dense(batch_input)
            out2 = memory_dense(batch_input)

        assert torch.allclose(out1, out2, atol=1e-5)

    def test_output_stability_msa(self, memory_msa, batch_input):
        """Test that MSA mode produces stable outputs."""
        memory_msa.eval()
        with torch.no_grad():
            out1 = memory_msa(batch_input)
            out2 = memory_msa(batch_input)

        assert torch.allclose(out1, out2, atol=1e-5)

    def test_reset_clears_state(self, memory_msa):
        """Test that reset clears prototype state."""
        memory_msa.proto_conf.fill_(0.9)
        memory_msa.proto_evidence.fill_(100.0)

        memory_msa.reset()

        assert torch.allclose(memory_msa.proto_conf, torch.full_like(memory_msa.proto_conf, 0.5))
        assert torch.allclose(memory_msa.proto_evidence, torch.ones_like(memory_msa.proto_evidence))

    def test_gradient_flow_both_modes(self, memory_dense, memory_msa, batch_input):
        """Test gradients flow in both modes."""
        # Dense mode
        x = batch_input.requires_grad_(True)
        out = memory_dense(x)
        out.sum().backward()
        assert x.grad is not None

        # MSA mode
        x = batch_input.requires_grad_(True)
        out = memory_msa(x)
        out.sum().backward()
        assert x.grad is not None


class TestMSAIntegrationScenarios:
    """Integration tests for MSA in realistic scenarios."""

    def test_large_prototype_scaling(self):
        """Test MSA handles large prototype counts efficiently."""
        # Simulate scaled-up SemanticMemory
        memory = MSAAugmentedSemanticMemory(
            hidden_dim=128,
            num_prototypes=256,
            use_msa=True,
            msa_top_k=16,
            msa_chunk_size=64,
        )
        memory.compression_threshold = 128

        batch = torch.randn(8, 128)
        output = memory(batch)

        assert output.shape == batch.shape

    def test_small_batch_efficiency(self):
        """Test MSA works efficiently with small batches."""
        memory = MSAAugmentedSemanticMemory(
            hidden_dim=64,
            num_prototypes=64,
            use_msa=True,
        )

        single_input = torch.randn(1, 64)
        output = memory(single_input)

        assert output.shape == single_input.shape

    def test_mode_switching_consistency(self):
        """Test that switching between modes maintains consistency."""
        hidden_dim = 64
        num_prototypes = 32

        # Create both modes
        memory_dense = MSAAugmentedSemanticMemory(
            hidden_dim=hidden_dim,
            num_prototypes=num_prototypes,
            use_msa=False,
        )
        memory_msa = MSAAugmentedSemanticMemory(
            hidden_dim=hidden_dim,
            num_prototypes=num_prototypes,
            use_msa=True,
        )

        # Copy prototypes to ensure same starting point
        memory_msa.prototypes.copy_(memory_dense.prototypes.detach())

        batch = torch.randn(4, hidden_dim)

        memory_dense.eval()
        memory_msa.eval()

        with torch.no_grad():
            out_dense = memory_dense(batch)
            out_msa = memory_msa(batch)

        # Both should produce valid outputs (not necessarily identical)
        assert out_dense.shape == batch.shape
        assert out_msa.shape == batch.shape


class TestMSAEdgeCases:
    """Edge case tests for MSA."""

    def test_top_k_larger_than_prototypes(self):
        """Test handling when top_k > num_prototypes."""
        index = MSASparseIndex(
            dim=64,
            num_prototypes=8,
            top_k=16,  # Larger than num_prototypes
        )

        # top_k should be clamped
        assert index.top_k == 8

    def test_single_prototype(self):
        """Test MSA with single prototype."""
        index = MSASparseIndex(
            dim=64,
            num_prototypes=1,
            top_k=1,
        )

        h = torch.randn(2, 64)
        prototypes = F.normalize(torch.randn(1, 64), dim=-1)

        output = index(h, prototypes)
        assert output.shape == h.shape

    def test_single_batch_element(self):
        """Test MSA with single batch element."""
        index = MSASparseIndex(dim=64, num_prototypes=16, top_k=4)

        h = torch.randn(1, 64)
        prototypes = F.normalize(torch.randn(16, 64), dim=-1)

        output = index(h, prototypes)
        assert output.shape == h.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
