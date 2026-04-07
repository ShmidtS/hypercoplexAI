"""Tests for MemoryPersistence — save/load/checkpoint for HDIM memory systems.

Coverage:
- TitansMemoryModule save/load
- HBMAMemory save/load
- Checkpoint atomicity with backup
- JSON export
- Version validation
- Compression for large buffers
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch

from src.core.memory_persistence import (
    MemoryPersistence,
    PersistenceMetadata,
    PERSISTENCE_VERSION,
)
from src.core.titans_memory import TitansMemoryModule
from src.core.hbma_memory import HBMAMemory, CLSMemory
from src.core.memory_interface import TitansAdapter, HBMAMemoryAdapter


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def persistence():
    """Create MemoryPersistence instance."""
    return MemoryPersistence(compress_threshold_mb=1.0)


@pytest.fixture
def titans_memory():
    """Create TitansMemoryModule for testing."""
    return TitansMemoryModule(key_dim=32, val_dim=64, hidden_dim=64)


@pytest.fixture
def hbma_memory():
    """Create HBMAMemory for testing."""
    return HBMAMemory(hidden_dim=64)


@pytest.fixture
def cls_memory():
    """Create CLSMemory for testing."""
    return CLSMemory(hidden_dim=64)


@pytest.fixture
def titans_adapter(titans_memory):
    """Create TitansAdapter wrapping TitansMemoryModule."""
    return TitansAdapter(titans_memory, clifford_dim=64, memory_key_dim=32)


@pytest.fixture
def hbma_adapter(hbma_memory):
    """Create HBMAMemoryAdapter wrapping HBMAMemory."""
    return HBMAMemoryAdapter(hbma_memory)


# =============================================================================
# Test TitansMemoryModule Save/Load
# =============================================================================

class TestTitansSaveLoad:
    """Tests for TitansMemoryModule save/load."""

    def test_save_creates_file(self, persistence, titans_memory, temp_dir):
        """Save creates a file."""
        path = temp_dir / "titans.pt"
        metadata = persistence.save(titans_memory, path)

        assert path.exists()
        assert metadata.memory_type == "titans"
        assert metadata.version == PERSISTENCE_VERSION

    def test_load_restores_state(self, persistence, titans_memory, temp_dir):
        """Load restores memory state."""
        # Train memory to get non-zero weights
        titans_memory.train()
        key = torch.randn(4, 32)
        value = torch.randn(4, 64)
        titans_memory.update(key, value)

        weight_before = titans_memory.memory.weight.clone().detach()
        momentum_before = titans_memory.momentum_S.clone().detach()

        # Save
        path = temp_dir / "titans.pt"
        persistence.save(titans_memory, path)

        # Modify memory
        titans_memory.memory.weight.data.fill_(0.5)
        titans_memory.momentum_S.fill_(0.5)

        # Load
        persistence.load(titans_memory, path)

        # Verify restored
        assert torch.allclose(titans_memory.memory.weight, weight_before)
        assert torch.allclose(titans_memory.momentum_S, momentum_before)

    def test_save_load_roundtrip(self, persistence, titans_memory, temp_dir):
        """Save followed by load produces identical state."""
        titans_memory.train()
        key = torch.randn(4, 32)
        value = torch.randn(4, 64)
        titans_memory.update(key, value)

        path = temp_dir / "titans_roundtrip.pt"
        persistence.save(titans_memory, path)

        # Create new module with same config
        new_memory = TitansMemoryModule(key_dim=32, val_dim=64, hidden_dim=64)
        persistence.load(new_memory, path)

        # States should match
        assert torch.allclose(
            titans_memory.memory.weight,
            new_memory.memory.weight,
        )
        assert torch.allclose(
            titans_memory.momentum_S,
            new_memory.momentum_S,
        )

    def test_load_wrong_type_raises(self, persistence, titans_memory, hbma_memory, temp_dir):
        """Loading wrong memory type raises ValueError."""
        path = temp_dir / "titans.pt"
        persistence.save(titans_memory, path)

        with pytest.raises(ValueError, match="Memory type mismatch"):
            persistence.load(hbma_memory, path)

    def test_load_missing_file_raises(self, persistence, titans_memory, temp_dir):
        """Loading non-existent file raises FileNotFoundError."""
        path = temp_dir / "nonexistent.pt"

        with pytest.raises(FileNotFoundError):
            persistence.load(titans_memory, path)


# =============================================================================
# Test HBMAMemory Save/Load
# =============================================================================

class TestHBMASaveLoad:
    """Tests for HBMAMemory save/load."""

    def test_save_creates_file(self, persistence, hbma_memory, temp_dir):
        """Save creates a file."""
        path = temp_dir / "hbma.pt"
        metadata = persistence.save(hbma_memory, path)

        assert path.exists()
        assert metadata.memory_type == "hbma"
        assert metadata.hidden_dim == 64

    def test_load_restores_working_memory(self, persistence, hbma_memory, temp_dir):
        """Load restores working memory buffers."""
        hbma_memory.train()
        x = torch.randn(4, 64)
        hbma_memory(x)  # Populate working memory

        buf_before = hbma_memory.working.buf.clone().detach()

        path = temp_dir / "hbma.pt"
        persistence.save(hbma_memory, path)

        # Clear buffers
        hbma_memory.reset()

        # Load
        persistence.load(hbma_memory, path)

        # Verify working memory restored
        assert torch.allclose(hbma_memory.working.buf, buf_before)

    def test_load_restores_episodic_memory(self, persistence, hbma_memory, temp_dir):
        """Load restores episodic memory keys/values."""
        hbma_memory.train()
        x = torch.randn(4, 64)
        for _ in range(10):
            hbma_memory(x)

        keys_before = hbma_memory.episodic.mem_keys.clone().detach()
        vals_before = hbma_memory.episodic.mem_vals.clone().detach()

        path = temp_dir / "hbma.pt"
        persistence.save(hbma_memory, path)

        hbma_memory.reset()
        persistence.load(hbma_memory, path)

        assert torch.allclose(hbma_memory.episodic.mem_keys, keys_before)
        assert torch.allclose(hbma_memory.episodic.mem_vals, vals_before)

    def test_load_restores_semantic_prototypes(self, persistence, hbma_memory, temp_dir):
        """Load restores semantic memory prototypes."""
        hbma_memory.train()
        x = torch.randn(4, 64)
        for _ in range(10):
            hbma_memory(x)

        protos_before = hbma_memory.semantic.prototypes.clone().detach()
        conf_before = hbma_memory.semantic.proto_conf.clone().detach()

        path = temp_dir / "hbma.pt"
        persistence.save(hbma_memory, path)

        hbma_memory.reset()
        persistence.load(hbma_memory, path)

        assert torch.allclose(hbma_memory.semantic.prototypes, protos_before)
        assert torch.allclose(hbma_memory.semantic.proto_conf, conf_before)

    def test_cls_memory_save_load(self, persistence, cls_memory, temp_dir):
        """CLSMemory (alias) save/load works."""
        cls_memory.train()
        x = torch.randn(4, 64)
        cls_memory(x)

        path = temp_dir / "cls.pt"
        metadata = persistence.save(cls_memory, path)

        assert metadata.memory_type == "hbma"

        new_cls = CLSMemory(hidden_dim=64)
        persistence.load(new_cls, path)

        assert torch.allclose(cls_memory.working.buf, new_cls.working.buf)


# =============================================================================
# Test Adapter Save/Load
# =============================================================================

class TestAdapterSaveLoad:
    """Tests for adapter save/load."""

    def test_titans_adapter_unwrap(self, persistence, titans_adapter, temp_dir):
        """TitansAdapter saves underlying Titans memory."""
        path = temp_dir / "adapter.pt"
        metadata = persistence.save(titans_adapter, path)

        assert metadata.memory_type == "titans"

    def test_titans_adapter_load(self, persistence, titans_adapter, temp_dir):
        """TitansAdapter load restores titans memory."""
        titans_adapter.train()
        x = torch.randn(4, 64)
        titans_adapter(x, update_memory=True)

        weight_before = titans_adapter.titans.memory.weight.clone().detach()

        path = temp_dir / "adapter.pt"
        persistence.save(titans_adapter, path)

        # Modify
        titans_adapter.titans.memory.weight.data.fill_(0.0)

        # Load
        persistence.load(titans_adapter, path)

        assert torch.allclose(titans_adapter.titans.memory.weight, weight_before)

    def test_hbma_adapter_save_load(self, persistence, hbma_adapter, temp_dir):
        """HBMAMemoryAdapter save/load works."""
        hbma_adapter.train()
        x = torch.randn(4, 64)
        hbma_adapter(x, update_memory=True)

        buf_before = hbma_adapter.hbma.working.buf.clone().detach()

        path = temp_dir / "hbma_adapter.pt"
        metadata = persistence.save(hbma_adapter, path)

        assert metadata.memory_type == "hbma"

        hbma_adapter.hbma.reset()
        persistence.load(hbma_adapter, path)

        assert torch.allclose(hbma_adapter.hbma.working.buf, buf_before)


# =============================================================================
# Test Checkpoint Atomicity
# =============================================================================

class TestCheckpoint:
    """Tests for atomic checkpoint with backup."""

    def test_checkpoint_creates_file(self, persistence, titans_memory, temp_dir):
        """Checkpoint creates main file."""
        path = temp_dir / "checkpoint.pt"
        persistence.checkpoint(titans_memory, path)

        assert path.exists()

    def test_checkpoint_creates_backup(self, persistence, titans_memory, temp_dir):
        """Checkpoint creates backup of existing file."""
        path = temp_dir / "checkpoint.pt"

        # First checkpoint
        persistence.checkpoint(titans_memory, path)

        # Modify and checkpoint again
        titans_memory.train()
        key = torch.randn(4, 32)
        value = torch.randn(4, 64)
        titans_memory.update(key, value)
        persistence.checkpoint(titans_memory, path)

        # Backup should exist
        backup = path.with_suffix(".pt.1")
        assert backup.exists()

    def test_checkpoint_rotates_backups(self, persistence, titans_memory, temp_dir):
        """Checkpoint rotates backup files."""
        path = temp_dir / "checkpoint.pt"

        for i in range(4):
            titans_memory.train()
            key = torch.randn(4, 32)
            value = torch.randn(4, 64)
            titans_memory.update(key, value)
            persistence.checkpoint(titans_memory, path)

        # Should have .1, .2, .3 backups
        assert (path.with_suffix(".pt.1")).exists()
        assert (path.with_suffix(".pt.2")).exists()
        assert (path.with_suffix(".pt.3")).exists()

    def test_checkpoint_max_backups(self, persistence, titans_memory, temp_dir):
        """Checkpoint respects max_backups parameter."""
        path = temp_dir / "checkpoint.pt"
        max_backups = 2

        for i in range(5):
            titans_memory.train()
            key = torch.randn(4, 32)
            value = torch.randn(4, 64)
            titans_memory.update(key, value)
            persistence.checkpoint(titans_memory, path, max_backups=max_backups)

        # Only max_backups files should exist
        existing_backups = list(temp_dir.glob("checkpoint.pt.*"))
        assert len(existing_backups) <= max_backups

    def test_checkpoint_atomic_on_interrupt(self, persistence, titans_memory, temp_dir):
        """Checkpoint is atomic (temp file cleaned on failure)."""
        path = temp_dir / "atomic.pt"

        # Should not leave temp files on success
        persistence.checkpoint(titans_memory, path)

        temp_files = list(temp_dir.glob(".tmp_checkpoint_*"))
        assert len(temp_files) == 0


# =============================================================================
# Test JSON Export
# =============================================================================

class TestJSONExport:
    """Tests for JSON export functionality."""

    def test_export_creates_json(self, persistence, titans_memory, temp_dir):
        """Export creates JSON file."""
        path = temp_dir / "export.json"
        persistence.export_json(titans_memory, path)

        assert path.exists()

        # Valid JSON
        with open(path) as f:
            data = json.load(f)
        assert "version" in data
        assert "memory_type" in data

    def test_export_includes_tensor_info(self, persistence, titans_memory, temp_dir):
        """Export includes tensor shape/dtype/stats."""
        path = temp_dir / "export.json"
        persistence.export_json(titans_memory, path)

        with open(path) as f:
            data = json.load(f)

        state = data["state"]
        # Should have memory.weight tensor info
        assert "memory.weight" in state
        tensor_info = state["memory.weight"]
        assert "shape" in tensor_info
        assert "dtype" in tensor_info
        assert "min" in tensor_info
        assert "max" in tensor_info
        assert "mean" in tensor_info

    def test_export_includes_small_tensors(self, persistence, titans_memory, temp_dir):
        """Export includes values for small tensors."""
        path = temp_dir / "export_small.json"
        # Use buffer_samples larger than momentum_S size (64*32=2048)
        persistence.export_json(titans_memory, path, buffer_samples=3000)

        with open(path) as f:
            data = json.load(f)

        # momentum_S (2048 elements) should now be included with values
        state = data["state"]
        assert "momentum_S" in state
        assert "values" in state["momentum_S"]

    def test_export_excludes_large_tensors(self, persistence, titans_memory, temp_dir):
        """Export excludes values for large tensors."""
        path = temp_dir / "export_large.json"
        persistence.export_json(titans_memory, path, buffer_samples=10)

        with open(path) as f:
            data = json.load(f)

        # memory.weight has 64*32=2048 elements, too large for buffer_samples=10
        state = data["state"]
        if "memory.weight" in state:
            assert "values" not in state["memory.weight"]

    def test_export_hbma_includes_subsystems(self, persistence, hbma_memory, temp_dir):
        """Export includes all HBMA subsystems."""
        path = temp_dir / "hbma_export.json"
        persistence.export_json(hbma_memory, path)

        with open(path) as f:
            data = json.load(f)

        state = data["state"]
        # Should have working/episodic/semantic/procedural buffers
        assert any("working.buf" in k for k in state.keys())


# =============================================================================
# Test Version Validation
# =============================================================================

class TestVersionValidation:
    """Tests for version validation."""

    def test_current_version_saves(self, persistence, titans_memory, temp_dir):
        """Current version saves correctly."""
        path = temp_dir / "versioned.pt"
        metadata = persistence.save(titans_memory, path)

        assert metadata.version == PERSISTENCE_VERSION

    def test_load_validates_version(self, persistence, titans_memory, temp_dir):
        """Load validates version on load."""
        path = temp_dir / "valid_version.pt"
        persistence.save(titans_memory, path)

        # Should not raise
        persistence.load(titans_memory, path)

    def test_load_rejects_wrong_version(self, titans_memory, temp_dir):
        """Load rejects file with wrong version."""
        # Manually create file with wrong version
        path = temp_dir / "wrong_version.pt"
        save_dict = {
            "version": "0.0.0",
            "metadata": {
                "version": "0.0.0",
                "timestamp": "2024-01-01T00:00:00",
                "memory_type": "titans",
                "hidden_dim": 64,
                "compressed": False,
                "checksum": "abcd",
            },
            "state": titans_memory.state_dict(),
        }
        import torch
        torch.save(save_dict, path)

        persistence = MemoryPersistence()

        with pytest.raises(ValueError, match="Version mismatch"):
            persistence.load(titans_memory, path)


# =============================================================================
# Test Compression
# =============================================================================

class TestCompression:
    """Tests for compression of large buffers."""

    def test_small_buffers_not_compressed(self, persistence, titans_memory, temp_dir):
        """Small buffers are not compressed by default."""
        path = temp_dir / "small.pt"
        metadata = persistence.save(titans_memory, path)

        assert metadata.compressed is False

    def test_force_compression(self, persistence, titans_memory, temp_dir):
        """Can force compression."""
        path = temp_dir / "forced.pt"
        metadata = persistence.save(titans_memory, path, compress=True)

        assert metadata.compressed is True

    def test_force_no_compression(self, persistence, titans_memory, temp_dir):
        """Can disable compression."""
        path = temp_dir / "no_compress.pt"
        metadata = persistence.save(titans_memory, path, compress=False)

        assert metadata.compressed is False

    def test_large_buffers_auto_compressed(self, temp_dir):
        """Large buffers trigger auto-compression."""
        # Set very low threshold
        persistence = MemoryPersistence(compress_threshold_mb=0.001)
        titans = TitansMemoryModule(key_dim=32, val_dim=64, hidden_dim=64)

        path = temp_dir / "large.pt"
        metadata = persistence.save(titans, path)

        assert metadata.compressed is True


# =============================================================================
# Test Checksum Validation
# =============================================================================

class TestChecksum:
    """Tests for checksum validation."""

    def test_checksum_computed(self, persistence, titans_memory, temp_dir):
        """Checksum is computed on save."""
        path = temp_dir / "checksum.pt"
        metadata = persistence.save(titans_memory, path)

        assert len(metadata.checksum) == 16  # MD5 truncated to 16 chars

    def test_checksum_consistent(self, persistence, titans_memory, temp_dir):
        """Same state produces same checksum."""
        path1 = temp_dir / "check1.pt"
        path2 = temp_dir / "check2.pt"

        meta1 = persistence.save(titans_memory, path1)
        meta2 = persistence.save(titans_memory, path2)

        # Same state should produce same checksum
        assert meta1.checksum == meta2.checksum

    def test_checksum_differs_for_different_state(self, persistence, titans_memory, temp_dir):
        """Different state produces different checksum."""
        path1 = temp_dir / "state1.pt"
        path2 = temp_dir / "state2.pt"

        meta1 = persistence.save(titans_memory, path1)

        # Modify state
        titans_memory.train()
        key = torch.randn(4, 32)
        value = torch.randn(4, 64)
        titans_memory.update(key, value)

        meta2 = persistence.save(titans_memory, path2)

        assert meta1.checksum != meta2.checksum


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_state(self, persistence, titans_memory, temp_dir):
        """Handles empty/uninitialized state."""
        # Fresh memory with zero momentum
        path = temp_dir / "empty.pt"
        persistence.save(titans_memory, path)

        new_memory = TitansMemoryModule(key_dim=32, val_dim=64, hidden_dim=64)
        persistence.load(new_memory, path)

        # Should not raise

    def test_path_with_parent_dirs(self, persistence, titans_memory, temp_dir):
        """Creates parent directories if needed."""
        path = temp_dir / "deep" / "nested" / "dir" / "memory.pt"
        persistence.save(titans_memory, path)

        assert path.exists()

    def test_path_string_vs_pathlib(self, persistence, titans_memory, temp_dir):
        """Accepts both string and Path objects."""
        str_path = str(temp_dir / "string.pt")
        path_path = temp_dir / "path.pt"

        persistence.save(titans_memory, str_path)
        persistence.save(titans_memory, path_path)

        assert Path(str_path).exists()
        assert path_path.exists()

    def test_device_handling(self, persistence, titans_memory, temp_dir):
        """Loads to correct device."""
        path = temp_dir / "device.pt"
        persistence.save(titans_memory, path)

        # Load on CPU
        new_memory = TitansMemoryModule(key_dim=32, val_dim=64, hidden_dim=64)
        persistence.load(new_memory, path)

        # Should be on CPU
        assert new_memory.memory.weight.device.type == "cpu"
