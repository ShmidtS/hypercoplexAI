"""Memory Persistence — Save/Load/Checkpoint for HDIM memory systems.

Supports:
- TitansMemoryModule (k, v memory matrix + momentum buffer)
- HBMAMemory (Working, Episodic, Semantic, Procedural subsystems)
- MemoryInterface adapters (TitansAdapter, HBMAMemoryAdapter)

Features:
- Versioned torch.save format with metadata
- Atomic checkpoint with backup
- Human-readable JSON export
- Compression for large buffers
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import torch
import torch.nn as nn

from src.core.titans_memory import TitansMemoryModule
from src.core.hbma_memory import HBMAMemory, CLSMemory
from src.core.memory_interface import TitansAdapter, HBMAMemoryAdapter


# Version for persistence format compatibility
PERSISTENCE_VERSION = "1.0.0"


@dataclass
class PersistenceMetadata:
    """Metadata for persisted memory state."""
    version: str = PERSISTENCE_VERSION
    timestamp: str = ""
    memory_type: str = ""  # "titans" | "hbma" | "adapter"
    hidden_dim: int = 0
    compressed: bool = False
    checksum: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class MemoryPersistence:
    """
    Persistence layer for HDIM memory systems.

    Handles save/load/checkpoint for:
    - TitansMemoryModule
    - HBMAMemory / CLSMemory
    - TitansAdapter / HBMAMemoryAdapter

    Usage:
        persistence = MemoryPersistence()
        persistence.save(memory, "memory.pt")
        persistence.load(memory, "memory.pt")
        persistence.checkpoint(memory, "checkpoint.pt")
        persistence.export_json(memory, "memory_export.json")
    """

    def __init__(self, compress_threshold_mb: float = 10.0):
        """
        Initialize persistence handler.

        Args:
            compress_threshold_mb: Compress buffers larger than this (MB).
        """
        self.compress_threshold_bytes = int(compress_threshold_mb * 1024 * 1024)

    def save(
        self,
        memory: nn.Module,
        path: str | Path,
        compress: bool | None = None,
    ) -> PersistenceMetadata:
        """
        Save memory state to file.

        Args:
            memory: Memory module (TitansMemoryModule, HBMAMemory, or adapter).
            path: Output file path.
            compress: Override auto-compression. None = auto-detect.

        Returns:
            Metadata about saved state.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Extract actual memory module if wrapped in adapter
        actual_memory, memory_type = self._unwrap_memory(memory)

        # Get state dict
        state = self._extract_state(actual_memory)

        # Determine compression
        state_size_bytes = self._estimate_size(state)
        should_compress = compress if compress is not None else state_size_bytes > self.compress_threshold_bytes

        # Build metadata
        metadata = PersistenceMetadata(
            memory_type=memory_type,
            hidden_dim=self._get_hidden_dim(actual_memory),
            compressed=should_compress,
            checksum=self._compute_checksum(state),
        )

        save_dict = {
            "version": PERSISTENCE_VERSION,
            "metadata": asdict(metadata),
            "state": state,
        }

        # Save with torch
        if should_compress:
            torch.save(save_dict, path, _use_new_zipfile_serialization=True)
        else:
            torch.save(save_dict, path)

        return metadata

    def load(
        self,
        memory: nn.Module,
        path: str | Path,
        strict: bool = True,
    ) -> PersistenceMetadata:
        """
        Load memory state from file.

        Args:
            memory: Memory module to load into.
            path: Input file path.
            strict: Raise error on missing/unexpected keys.

        Returns:
            Metadata from saved state.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Memory file not found: {path}")

        # Load saved dict
        save_dict = torch.load(path, map_location="cpu", weights_only=False)

        # Validate version
        saved_version = save_dict.get("version", "unknown")
        if saved_version != PERSISTENCE_VERSION:
            raise ValueError(
                f"Version mismatch: file has {saved_version}, expected {PERSISTENCE_VERSION}"
            )

        metadata = PersistenceMetadata(**save_dict["metadata"])

        # Extract actual memory module if wrapped in adapter
        actual_memory, memory_type = self._unwrap_memory(memory)

        # Validate memory type
        if metadata.memory_type != memory_type:
            raise ValueError(
                f"Memory type mismatch: file has {metadata.memory_type}, expected {memory_type}"
            )

        # Load state
        state = save_dict["state"]
        self._load_state(actual_memory, state, strict)

        return metadata

    def checkpoint(
        self,
        memory: nn.Module,
        path: str | Path,
        max_backups: int = 5,
    ) -> PersistenceMetadata:
        """
        Atomic checkpoint with backup rotation.

        Creates backup files: path.1, path.2, ..., path.N
        Uses temp file + rename for atomicity.

        Args:
            memory: Memory module to checkpoint.
            path: Checkpoint file path.
            max_backups: Maximum number of backup files to keep.

        Returns:
            Metadata about checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first (atomic on POSIX, near-atomic on Windows)
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=".tmp_checkpoint_",
            suffix=".pt",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Save to temp
            metadata = self.save(memory, tmp_path)

            # Rotate backups
            self._rotate_backups(path, max_backups)

            # Atomic rename
            if path.exists():
                backup_path = path.with_suffix(path.suffix + ".1")
                shutil.move(str(path), str(backup_path))

            shutil.move(str(tmp_path), str(path))

        except Exception:
            # Clean up temp file on failure
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        return metadata

    def export_json(
        self,
        memory: nn.Module,
        path: str | Path,
        include_buffers: bool = True,
        buffer_samples: int = 100,
    ) -> None:
        """
        Export memory state as human-readable JSON.

        Large tensors are represented as shape/dtype/stats.
        Small tensors (< buffer_samples) can be fully included.

        Args:
            memory: Memory module to export.
            path: Output JSON file path.
            include_buffers: Include buffer samples in output.
            buffer_samples: Max elements to include per tensor.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        actual_memory, memory_type = self._unwrap_memory(memory)
        state = self._extract_state(actual_memory)

        # Convert to JSON-serializable format
        json_state = self._state_to_json(state, include_buffers, buffer_samples)

        export_dict = {
            "version": PERSISTENCE_VERSION,
            "memory_type": memory_type,
            "exported_at": datetime.utcnow().isoformat(),
            "state": json_state,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_dict, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _unwrap_memory(self, memory: nn.Module) -> tuple[nn.Module, str]:
        """Unwrap adapter and return (actual_module, type_string)."""
        if isinstance(memory, TitansAdapter):
            return memory.titans, "titans"
        elif isinstance(memory, HBMAMemoryAdapter):
            return memory.hbma, "hbma"
        elif isinstance(memory, TitansMemoryModule):
            return memory, "titans"
        elif isinstance(memory, (HBMAMemory, CLSMemory)):
            return memory, "hbma"
        else:
            raise TypeError(f"Unsupported memory type: {type(memory).__name__}")

    def _extract_state(self, memory: nn.Module) -> dict[str, Any]:
        """Extract state dict with metadata for persistence."""
        state = memory.state_dict()

        # Add type annotations for reconstruction
        result = {"_type": type(memory).__name__}

        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.cpu().clone()
            else:
                result[key] = value

        return result

    def _load_state(
        self,
        memory: nn.Module,
        state: dict[str, Any],
        strict: bool,
    ) -> None:
        """Load state dict into memory module."""
        # Remove metadata key if present
        state_clean = {k: v for k, v in state.items() if not k.startswith("_")}

        # Convert tensors to correct device
        device = next(memory.parameters()).device if list(memory.parameters()) else "cpu"
        state_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in state_clean.items()
        }

        memory.load_state_dict(state_on_device, strict=strict)

    def _get_hidden_dim(self, memory: nn.Module) -> int:
        """Get hidden dimension for metadata."""
        if isinstance(memory, TitansMemoryModule):
            return memory.val_dim
        elif isinstance(memory, (HBMAMemory, CLSMemory)):
            return memory.hidden_dim
        return 0

    def _estimate_size(self, state: dict[str, Any]) -> int:
        """Estimate state size in bytes."""
        total = 0
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
        return total

    def _compute_checksum(self, state: dict[str, Any]) -> str:
        """Compute simple checksum of state tensors."""
        hasher = __import__("hashlib").md5()
        for key in sorted(state.keys()):
            value = state[key]
            if isinstance(value, torch.Tensor):
                hasher.update(key.encode())
                hasher.update(value.cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]

    def _rotate_backups(self, path: Path, max_backups: int) -> None:
        """Rotate backup files: .N -> .N+1, remove oldest."""
        for i in range(max_backups - 1, 0, -1):
            old_backup = path.with_suffix(path.suffix + f".{i}")
            new_backup = path.with_suffix(path.suffix + f".{i + 1}")
            if old_backup.exists():
                if new_backup.exists():
                    new_backup.unlink()
                shutil.move(str(old_backup), str(new_backup))

        # Remove oldest if exceeds max
        oldest = path.with_suffix(path.suffix + f".{max_backups}")
        if oldest.exists():
            oldest.unlink()

    def _state_to_json(
        self,
        state: dict[str, Any],
        include_buffers: bool,
        buffer_samples: int,
    ) -> dict[str, Any]:
        """Convert state dict to JSON-serializable format."""
        result = {}
        for key, value in state.items():
            if key.startswith("_"):
                continue

            if isinstance(value, torch.Tensor):
                tensor_info = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "numel": value.numel(),
                }

                # Add stats
                if value.numel() > 0:
                    tensor_info["min"] = float(value.min().item())
                    tensor_info["max"] = float(value.max().item())
                    if value.is_floating_point():
                        tensor_info["mean"] = float(value.mean().item())
                        tensor_info["std"] = float(value.std().item() if value.numel() > 1 else 0.0)

                # Optionally include sample values
                if include_buffers and value.numel() <= buffer_samples:
                    tensor_info["values"] = value.flatten().tolist()

                result[key] = tensor_info
            else:
                result[key] = value

        return result
