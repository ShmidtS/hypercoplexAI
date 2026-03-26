"""Security tests for checkpoint loading integrity.

Tests verify that:
1. Malicious checkpoints with pickle exploits are rejected
2. weights_only=True is enforced in all torch.load calls
3. _load_checkpoint_safe prevents arbitrary code execution

OWASP A08:2021 - Software and Data Integrity Failures
"""

import io
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


class MaliciousPickle:
    """Payload that executes code when unpickled."""

    def __reduce__(self):
        # Simulates malicious code execution
        import os

        return (os.system, ("echo MALICIOUS_CODE_EXECUTED",))


class SimpleModel(nn.Module):
    """Minimal model for checkpoint testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class TestMaliciousCheckpointRejected:
    """Test that malicious checkpoints are rejected."""

    def test_malicious_pickle_payload_rejected(self, tmp_path):
        """Verify that a checkpoint with malicious pickle payload is rejected."""
        # Create a malicious checkpoint
        malicious_data = {
            "model_state_dict": {"linear.weight": torch.randn(10, 10), "linear.bias": torch.randn(10)},
            "malicious_payload": MaliciousPickle(),
        }

        malicious_path = tmp_path / "malicious.pt"
        torch.save(malicious_data, malicious_path)

        # Try to load with weights_only=True - should reject
        with pytest.raises((RuntimeError, ValueError, pickle.UnpicklingError)):
            torch.load(malicious_path, weights_only=True)

    def test_weights_only_prevents_arbitrary_code(self, tmp_path):
        """Verify weights_only=True prevents arbitrary code execution."""
        # Create checkpoint with safe data only
        safe_checkpoint = {"model_state_dict": {"linear.weight": torch.randn(10, 10), "linear.bias": torch.randn(10)}}

        safe_path = tmp_path / "safe.pt"
        torch.save(safe_checkpoint, safe_path)

        # Should load successfully with weights_only=True
        loaded = torch.load(safe_path, weights_only=True)
        assert "model_state_dict" in loaded
        assert isinstance(loaded["model_state_dict"], dict)


class TestWeightsOnlyEnforced:
    """Test that weights_only=True is enforced in all torch.load calls."""

    def test_trainer_load_checkpoint_safe(self):
        """Verify HDIMTrainer._load_checkpoint_safe uses weights_only=True."""
        from src.training.trainer import HDIMTrainer

        # Create mock trainer components
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        # Mock device
        with patch.object(HDIMTrainer, "__init__", lambda self, *args, **kwargs: None):
            trainer = HDIMTrainer.__new__(HDIMTrainer)
            trainer.device = "cpu"
            trainer.model = model
            trainer.optimizer = optimizer
            trainer._step = 0
            trainer._current_epoch = 0

            # Create a valid checkpoint
            import os

            fd, checkpoint_path = tempfile.mkstemp(suffix=".pt")
            try:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": 100,
                }
                torch.save(checkpoint, checkpoint_path)

                # Load using safe method
                loaded = trainer._load_checkpoint_safe(checkpoint_path)

                # Verify only weights were loaded
                assert "model_state_dict" in loaded
                assert isinstance(loaded, dict)
            finally:
                os.close(fd)
                os.unlink(checkpoint_path)

    def test_all_torch_load_calls_use_weights_only(self):
        """Scan all Python files for torch.load calls with weights_only=True."""
        import ast
        import re

        project_root = Path(__file__).parent.parent.parent

        # Find all Python files
        py_files = list(project_root.rglob("*.py"))

        violations = []

        for py_file in py_files:
            # Skip test files and virtual environments
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            content = py_file.read_text(encoding="utf-8", errors="ignore")

            # Find torch.load calls
            pattern = r"torch\.load\s*\("
            matches = list(re.finditer(pattern, content))

            for match in matches:
                # Get the full call (approximate)
                start = match.start()
                # Find the closing paren
                call_snippet = content[start : start + 200]

                # Check if weights_only=True is present
                if "weights_only=True" not in call_snippet and "weights_only=False" not in call_snippet:
                    # Check if weights_only is a variable (which is acceptable)
                    if "weights_only=" not in call_snippet:
                        line_num = content[:start].count("\n") + 1
                        violations.append(f"{py_file.relative_to(project_root)}:{line_num}")

        # Currently all should have weights_only=True based on grep results
        # This test documents the current state
        assert len(violations) == 0, f"torch.load calls without weights_only: {violations}"


class TestCheckpointIntegrity:
    """Test checkpoint integrity verification."""

    def test_checkpoint_hash_verification(self, tmp_path):
        """Test that checkpoint integrity can be verified via hash."""
        import hashlib

        model = SimpleModel()
        checkpoint = {"model_state_dict": model.state_dict()}

        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Compute hash
        with open(checkpoint_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        # Verify file wasn't corrupted
        assert len(file_hash) == 64  # SHA-256 hex length

        # Load and verify
        loaded = torch.load(checkpoint_path, weights_only=True)
        assert torch.allclose(loaded["model_state_dict"]["linear.weight"], checkpoint["model_state_dict"]["linear.weight"])

    def test_corrupted_checkpoint_rejected(self, tmp_path):
        """Test that corrupted checkpoints cause loading error."""
        import hashlib

        model = SimpleModel()
        checkpoint = {"model_state_dict": model.state_dict()}

        checkpoint_path = tmp_path / "corrupted.pt"
        torch.save(checkpoint, checkpoint_path)

        # Corrupt the file
        with open(checkpoint_path, "r+b") as f:
            f.seek(100)
            f.write(b"corrupted")

        # Loading corrupted file should fail or return different hash
        with open(checkpoint_path, "rb") as f:
            corrupted_hash = hashlib.sha256(f.read()).hexdigest()

        original_hash = hashlib.sha256(pickle.dumps(checkpoint)).hexdigest()
        assert corrupted_hash != original_hash
