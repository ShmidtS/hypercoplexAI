"""Semantic Entropy Probe — Linear probe for uncertainty quantification from hidden states.

Reference: Kossen et al., ICLR 2024
"Semantic Entropy Probes: Cheap Uncertainty Quantification for LLMs"

Key insight: Semantic entropy can be predicted from hidden states alone,
avoiding expensive multi-sample generation. 45x-450x faster than full
semantic entropy computation.

Architecture:
- Linear probe: hidden_dim -> 1 (entropy prediction)
- Pool over sequence dimension (mean pooling)
- Sigmoid activation for [0, 1] range

Integration:
- Called in HallucinationDetector with hidden_states from HDIM pipeline
- Probe layer configurable (default: last layer)
- Zero-initialized for stable training start
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class SemanticEntropyResult:
    """Output from SemanticEntropyProbe."""

    entropy_pred: float  # Predicted semantic entropy [0, 1]
    hidden_norm: float  # L2 norm of pooled hidden states
    probe_layer: int  # Layer index used for prediction

    def to_dict(self) -> dict:
        return {
            "entropy_pred": self.entropy_pred,
            "hidden_norm": self.hidden_norm,
            "probe_layer": self.probe_layer,
        }


class SemanticEntropyProbe(nn.Module):
    """Linear probe for semantic entropy prediction from hidden states.

    Predicts semantic entropy directly from model's internal representations,
    avoiding the need for multiple sample generations. Trained to match
    true semantic entropy computed from multi-sample outputs.

    Reference:
        Kossen et al., ICLR 2024
        "Semantic Entropy Probes: Cheap Uncertainty Quantification for LLMs"

    Performance:
        - 45x-450x faster than multi-sample semantic entropy
        - Trained once, reused for inference
        - Minimal computational overhead (single linear layer)

    Example:
        >>> probe = SemanticEntropyProbe(hidden_dim=256)
        >>> hidden = torch.randn(4, 128, 256)  # (batch, seq, hidden)
        >>> entropy = probe(hidden)  # (batch,) in [0, 1]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        probe_layer: int = -1,
        init_scale: float = 0.01,
    ):
        """Initialize the semantic entropy probe.

        Args:
            hidden_dim: Dimension of hidden states (default: 256 for HDIM)
            probe_layer: Which layer to probe (-1 = last, -2 = second-to-last)
            init_scale: Initial weight scale (small for stable start)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.probe_layer = probe_layer

        # Linear probe: hidden_dim -> 1
        self.probe = nn.Linear(hidden_dim, 1)

        # Zero initialization for stable training start
        # This ensures initial predictions are ~0.5 (sigmoid(0))
        nn.init.zeros_(self.probe.weight)
        nn.init.zeros_(self.probe.bias)

        # Optional: small random init for exploration
        if init_scale > 0:
            nn.init.normal_(self.probe.weight, mean=0.0, std=init_scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Predict semantic entropy from hidden states.

        Args:
            hidden_states: Hidden tensor of shape:
                - (batch, seq_len, hidden_dim) — full sequence
                - (batch, hidden_dim) — already pooled
                - (batch, num_layers, seq_len, hidden_dim) — all layers
            layer_index: Override probe_layer if hidden_states has multiple layers

        Returns:
            entropy_pred: (batch,) predicted semantic entropy in [0, 1]
        """
        h = hidden_states

        # Handle multi-layer hidden states
        if h.dim() == 4:
            # (batch, num_layers, seq_len, hidden_dim)
            layer_idx = layer_index if layer_index is not None else self.probe_layer
            if layer_idx < 0:
                layer_idx = h.shape[1] + layer_idx
            h = h[:, layer_idx, :, :]  # (batch, seq_len, hidden_dim)

        # Pool over sequence dimension
        if h.dim() == 3:
            # (batch, seq_len, hidden_dim) -> (batch, hidden_dim)
            h = h.mean(dim=1)

        # Linear projection + sigmoid for [0, 1] range
        entropy_logit = self.probe(h).squeeze(-1)  # (batch,)
        entropy_pred = torch.sigmoid(entropy_logit)

        return entropy_pred

    def predict_with_metadata(
        self,
        hidden_states: torch.Tensor,
        layer_index: Optional[int] = None,
    ) -> SemanticEntropyResult:
        """Predict entropy with additional metadata.

        Args:
            hidden_states: Same as forward()
            layer_index: Override probe_layer

        Returns:
            SemanticEntropyResult with entropy, hidden norm, layer info
        """
        entropy_pred = self.forward(hidden_states, layer_index)

        # Compute hidden norm for quality check
        h = hidden_states
        if h.dim() == 4:
            layer_idx = layer_index if layer_index is not None else self.probe_layer
            if layer_idx < 0:
                layer_idx = h.shape[1] + layer_idx
            h = h[:, layer_idx, :, :]
        if h.dim() == 3:
            h = h.mean(dim=1)

        hidden_norm = torch.norm(h, dim=-1).mean().item()
        layer_used = layer_index if layer_index is not None else self.probe_layer

        return SemanticEntropyResult(
            entropy_pred=round(entropy_pred.mean().item(), 6),
            hidden_norm=round(hidden_norm, 6),
            probe_layer=layer_used,
        )

    def reset_parameters(self):
        """Reset probe to zero initialization."""
        nn.init.zeros_(self.probe.weight)
        nn.init.zeros_(self.probe.bias)
