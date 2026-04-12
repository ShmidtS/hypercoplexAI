"""Lightweight GPU-side metric accumulator. Defers .item() to epoch end."""
import torch
from collections import defaultdict
from typing import Dict


class GPULogger:
    def __init__(self):
        self._buffer: Dict[str, list] = defaultdict(list)

    def log(self, key: str, value: torch.Tensor):
        """Accumulate a scalar tensor on GPU. No .item() call."""
        if value.dim() == 0:
            self._buffer[key].append(value.detach())

    def flush(self) -> Dict[str, float]:
        """Call .item() once per key, return averaged stats."""
        result = {}
        for key, values in self._buffer.items():
            if values:
                stacked = torch.stack(values)
                result[key] = stacked.float().mean().item()
        self._buffer.clear()
        return result


# Module-level singleton for convenience
_default_logger = GPULogger()


def get_logger() -> GPULogger:
    return _default_logger
