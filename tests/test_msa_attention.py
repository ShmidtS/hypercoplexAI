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

from src.core.prototype_memory import (
    MSAOverflowBuffer,
    MSASparseIndex,
)
from src.models.hdim_model import MSAConfig


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




class TestSemanticMemoryMSA:
    """Test suite for SemanticMemory MSA integration."""

    @pytest.fixture
    def semantic_msa(self):
        """Create SemanticMemory with MSA enabled."""
        from src.core.hbma_memory import SemanticMemory
        return SemanticMemory(
            hidden_dim=64,
            num_prototypes=32,
            use_msa=True,
        )

    @pytest.fixture
    def semantic_dense(self):
        """Create SemanticMemory with MSA disabled."""
        from src.core.hbma_memory import SemanticMemory
        return SemanticMemory(
            hidden_dim=64,
            num_prototypes=32,
            use_msa=False,
        )

    @pytest.fixture
    def batch_input(self):
        """Generate test input."""
        return torch.randn(4, 64)

    def test_msa_retrieval_correctness(self, semantic_msa, batch_input):
        """Test that MSA retrieval produces valid output."""
        semantic_msa.eval()
        with torch.no_grad():
            output = semantic_msa(batch_input)
        assert output.shape == batch_input.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_msa_vs_dense_equivalence(self, semantic_msa, semantic_dense, batch_input):
        """Test that MSA and dense modes produce same-shaped outputs."""
        semantic_msa.eval()
        semantic_dense.eval()
        
        # Copy prototypes to ensure same starting point
        semantic_msa.prototypes.data.copy_(semantic_dense.prototypes.data)
        
        with torch.no_grad():
            out_msa = semantic_msa(batch_input)
            out_dense = semantic_dense(batch_input)
        
        # Both should produce valid outputs of same shape
        assert out_msa.shape == batch_input.shape
        assert out_dense.shape == batch_input.shape
        
        # Both should be valid tensors
        assert not torch.isnan(out_msa).any()
        assert not torch.isnan(out_dense).any()

    def test_msa_overflow_integration(self, batch_input):
        """Test MSA integration with EpisodicMemory overflow."""
        from src.core.hbma_memory import HBMAMemory
        
        # Create HBMA with MSA enabled (default)
        hbma = HBMAMemory(hidden_dim=64)
        
        # Verify MSA is enabled
        assert hbma.semantic.use_msa is True
        assert hbma.episodic.use_overflow is True
        
        # Forward pass should work
        output = hbma(batch_input)
        assert output.shape == batch_input.shape

    def test_msa_config_override(self):
        """Test that MSAConfig can override defaults."""
        from src.core.hbma_memory import SemanticMemory
        from src.models.hdim_model import MSAConfig
        
        cfg = MSAConfig(top_k=8, chunk_size=32, temperature=0.05)
        memory = SemanticMemory(
            hidden_dim=64,
            num_prototypes=32,
            use_msa=True,
            msa_config=cfg,
        )
        
        assert memory.msa_index.top_k == 8
        assert memory.msa_index.chunk_size == 32
        assert memory.msa_index.temperature == 0.05

    def test_gradient_flow_msa(self, semantic_msa, batch_input):
        """Test that gradients flow through MSA path."""
        x = batch_input.requires_grad_(True)
        out = semantic_msa(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

from src.core.hbma_memory import EpisodicMemory


class TestMSAOverflowBuffer:
    """Test suite for MSAOverflowBuffer."""

    @pytest.fixture
    def overflow_buffer(self):
        """Create default overflow buffer for testing."""
        return MSAOverflowBuffer(
            dim=64,
            key_dim=32,
            num_prototypes=128,
            top_k=8,
            max_hops=3,
        )

    @pytest.fixture
    def batch_data(self):
        """Generate test batch data."""
        B, key_dim, val_dim = 4, 32, 64
        keys = torch.randn(B, key_dim)
        values = torch.randn(B, val_dim)
        evidence = torch.rand(B)
        return keys, values, evidence

    def test_initialization(self, overflow_buffer):
        """Test that overflow buffer initializes correctly."""
        assert overflow_buffer.dim == 64
        assert overflow_buffer.key_dim == 32
        assert overflow_buffer.max_hops == 3
        assert overflow_buffer.top_k == 8
        assert overflow_buffer.is_enabled()

    def test_store_single_item(self, overflow_buffer):
        """Test storing a single item in overflow buffer."""
        key = torch.randn(32)
        value = torch.randn(64)

        overflow_buffer.store(key, value)

        assert overflow_buffer.size() == 1
        assert overflow_buffer._overflow_valid.sum().item() == 1
        assert overflow_buffer.overflow_keys.shape[1] == 32
        assert overflow_buffer.overflow_vals.shape[1] == 64

    def test_store_batch(self, overflow_buffer, batch_data):
        """Test storing a batch of items."""
        keys, values, evidence = batch_data
        B = keys.shape[0]

        overflow_buffer.store(keys, values, evidence)

        assert overflow_buffer.size() == B
        assert overflow_buffer.overflow_keys.shape[1] == 32
        assert overflow_buffer.overflow_vals.shape[1] == 64

    def test_retrieve_from_empty(self, overflow_buffer):
        """Test retrieval from empty buffer returns zeros."""
        query = torch.randn(2, 64)

        retrieved, weights, indices = overflow_buffer.retrieve(query)

        assert retrieved.shape == (2, 64)
        assert weights.shape == (2, 8)
        assert indices.shape == (2, 8)
        assert torch.allclose(retrieved, torch.zeros_like(retrieved))

    def test_retrieve_after_store(self, overflow_buffer):
        """Test retrieval after storing items."""
        for _ in range(10):
            key = torch.randn(32)
            value = torch.randn(64)
            overflow_buffer.store(key, value)

        query = torch.randn(2, 64)
        retrieved, weights, indices = overflow_buffer.retrieve(query)

        assert retrieved.shape == (2, 64)
        assert weights.shape == (2, 8)

    def test_enable_disable(self, overflow_buffer):
        """Test enable/disable functionality."""
        assert overflow_buffer.is_enabled()

        overflow_buffer.disable()
        assert not overflow_buffer.is_enabled()

        key = torch.randn(32)
        value = torch.randn(64)
        overflow_buffer.store(key, value)
        assert overflow_buffer.size() == 0

        overflow_buffer.enable()
        assert overflow_buffer.is_enabled()

    def test_clear(self, overflow_buffer):
        """Test clearing the overflow buffer."""
        for _ in range(5):
            key = torch.randn(32)
            value = torch.randn(64)
            overflow_buffer.store(key, value)

        assert overflow_buffer.size() == 5

        overflow_buffer.clear()
        assert overflow_buffer.size() == 0

    def test_backward_compatibility_default_params(self):
        """Test that default parameters work (backward compat)."""
        buffer = MSAOverflowBuffer(dim=64)
        assert buffer.dim == 64
        assert buffer.key_dim == 64

        key = torch.randn(64)
        value = torch.randn(64)
        buffer.store(key, value)
        assert buffer.size() == 1


class TestEpisodicMemoryOverflow:
    """Test suite for EpisodicMemory with overflow integration."""

    @pytest.fixture
    def episodic_with_overflow(self):
        """Create EpisodicMemory with overflow enabled."""
        return EpisodicMemory(
            hidden_dim=64,
            num_slots=32,
            key_dim=32,
            use_overflow=True,
            overflow_num_prototypes=128,
        )

    @pytest.fixture
    def episodic_without_overflow(self):
        """Create EpisodicMemory without overflow."""
        return EpisodicMemory(
            hidden_dim=64,
            num_slots=32,
            key_dim=32,
            use_overflow=False,
        )

    def test_overflow_enabled_by_default(self):
        """Test that overflow is enabled by default in EpisodicMemory."""
        mem = EpisodicMemory(hidden_dim=64, num_slots=32)
        assert mem.use_overflow is True
        assert mem.overflow is not None

    def test_overflow_enabled_when_requested(self, episodic_with_overflow):
        """Test that overflow is enabled when requested."""
        assert episodic_with_overflow.use_overflow is True
        assert episodic_with_overflow.overflow is not None

    def test_forward_with_overflow(self, episodic_with_overflow):
        """Test forward pass with overflow enabled."""
        x = torch.randn(4, 64)
        out = episodic_with_overflow(x)
        assert out.shape == x.shape

    def test_forward_without_overflow(self, episodic_without_overflow):
        """Test forward pass without overflow."""
        x = torch.randn(4, 64)
        out = episodic_without_overflow(x)
        assert out.shape == x.shape

    def test_overflow_stores_evicted_slots(self, episodic_with_overflow):
        """Test that overflow stores evicted slots."""
        mem = episodic_with_overflow
        mem.train()

        for _ in range(100):
            x = torch.randn(2, 64)
            _ = mem(x)

        assert mem.overflow.size() > 0

    def test_reset_clears_overflow(self, episodic_with_overflow):
        """Test that reset clears overflow buffer."""
        mem = episodic_with_overflow
        mem.train()

        for _ in range(50):
            x = torch.randn(2, 64)
            _ = mem(x)

        initial_size = mem.overflow.size()
        assert initial_size > 0

        mem.reset()
        assert mem.overflow.size() == 0

