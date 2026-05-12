"""Streamlined HDIM model exports."""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import Any

from .config import HDIMConfig
from .hdim_model import HDIMModel
from .model_factory import build_hdim_model, build_text_adapter
from .results import HDIMAuxState

__all__ = [
    "HDIMConfig",
    "HDIMModel",
    "build_hdim_model",
    "build_text_adapter",
    "HDIMAuxState",
]

_DEPRECATED_EXPORTS = {
    "TextHDIMModel": "src.models.text_hdim_model",
    "build_sbert_hdim_model": "src.models.model_factory",
    "build_modernbert_hdim_model": "src.models.model_factory",
}


def __getattr__(name: str) -> Any:
    module_name = _DEPRECATED_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    warnings.warn(
        f"src.models.{name} is deprecated; use HDIMModel with build_text_adapter when text support is needed",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(import_module(module_name), name)
