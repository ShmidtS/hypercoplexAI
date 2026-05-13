#!/usr/bin/env python
"""Simple streaming online learner for HDIM encodings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientMode(Enum):
    """Compatibility gradient mode flag."""

    DETACHED = "detached"
    SELECTIVE = "selective"
    FULL = "full"


@dataclass
class OnlineLearnerConfig:
    """Configuration for streaming prototype updates."""

    hidden_dim: int = 256
    num_experts: int = 4
    replay_buffer_size: int = 10000
    replay_batch_size: int = 32
    ema_decay: float = 0.999
    ttt_lr: float = 1e-5
    surprise_threshold: float = 0.3
    consolidation_interval: int = 1000
    grad_clip: float = 1.0
    gradient_mode: GradientMode = GradientMode.DETACHED
    gradient_scale: float = 0.1


class OnlineLearner(nn.Module):
    """Streaming prototype learner for per-expert adaptation."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 4,
        replay_buffer_size: int = 10000,
        replay_batch_size: int = 32,
        ema_decay: float = 0.999,
        ttt_lr: float = 1e-5,
        surprise_threshold: float = 0.3,
        consolidation_interval: int = 1000,
        grad_clip: float = 1.0,
        device: torch.device | None = None,
        gradient_mode: GradientMode = GradientMode.DETACHED,
        gradient_scale: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.ttt_lr = ttt_lr
        self.surprise_threshold = surprise_threshold
        self.grad_clip = grad_clip
        self.replay_batch_size = replay_batch_size
        self.consolidation_interval = consolidation_interval
        self.gradient_mode = gradient_mode
        self.gradient_scale = gradient_scale
        self.replay_buffer_size = replay_buffer_size
        self.ema_decay = ema_decay

        init_device = device or torch.device("cpu")
        self.register_buffer("prototypes", torch.zeros(num_experts, hidden_dim, device=init_device))
        self.register_buffer("prototype_initialized", torch.zeros(num_experts, dtype=torch.bool, device=init_device))
        self.register_buffer("expert_update_count", torch.zeros(num_experts, dtype=torch.long, device=init_device))
        self.register_buffer("expert_surprise_accum", torch.zeros(num_experts, dtype=torch.float, device=init_device))
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long, device=init_device))

    @property
    def device(self) -> torch.device:
        """Derive device from learner buffers."""
        return self.prototypes.device

    def compute_surprise(self, x: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
        """Compute cosine distance from a reference or nearest initialized prototype."""
        if reference is not None:
            if reference.dim() == 1:
                return 1.0 - F.cosine_similarity(x, reference.unsqueeze(0), dim=-1)
            return 1.0 - F.cosine_similarity(x.unsqueeze(1), reference.unsqueeze(0), dim=-1).max(dim=1)[0]

        initialized = self.prototype_initialized
        if not torch.any(initialized):
            return torch.ones(x.size(0), device=x.device, dtype=x.dtype)

        prototypes = self.prototypes[initialized].to(device=x.device, dtype=x.dtype)
        similarity = F.cosine_similarity(x.unsqueeze(1), prototypes.unsqueeze(0), dim=-1).max(dim=1)[0]
        return 1.0 - similarity

    def online_update(
        self,
        x: torch.Tensor,
        expert_idx: int,
        target: torch.Tensor | None = None,
        force_update: bool = False,
    ) -> tuple[torch.Tensor, bool, float]:
        """Update a prototype and optionally report MSE to an explicit target."""
        loss, updated, surprise_mean = self.online_update_with_mode(x, expert_idx, force_update=force_update)
        if updated and target is not None:
            loss = F.mse_loss(x, target)
        return loss, updated, surprise_mean

    def online_update_with_mode(
        self,
        x: torch.Tensor,
        expert_idx: int,
        model: nn.Module | None = None,
        force_update: bool = False,
    ) -> tuple[torch.Tensor, bool, float]:
        """Perform a streaming prototype update."""
        _ = model
        self.step_count += 1
        expert_idx = int(expert_idx)
        if expert_idx < 0 or expert_idx >= self.num_experts:
            raise ValueError(f"expert_idx must be in [0, {self.num_experts}), got {expert_idx}")

        prototype = self.prototypes[expert_idx].to(device=x.device, dtype=x.dtype)
        reference = prototype if bool(self.prototype_initialized[expert_idx].item()) else None
        surprise = self.compute_surprise(x, reference=reference)
        surprise_mean = float(surprise.mean().item())
        should_update = force_update or (self.training and surprise_mean > self.surprise_threshold)
        if not should_update:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype), False, surprise_mean

        batch_mean = x.detach().mean(dim=0).to(device=self.device, dtype=self.prototypes.dtype)
        with torch.no_grad():
            if bool(self.prototype_initialized[expert_idx].item()):
                self.prototypes[expert_idx].lerp_(batch_mean, self.ttt_lr)
            else:
                self.prototypes[expert_idx].copy_(batch_mean)
                self.prototype_initialized[expert_idx] = True
            self.expert_update_count[expert_idx] += 1
            self.expert_surprise_accum[expert_idx] += surprise_mean

        updated_reference = self.prototypes[expert_idx].to(device=x.device, dtype=x.dtype).unsqueeze(0).expand_as(x)
        loss = F.mse_loss(x * self.gradient_scale, updated_reference.detach()) * self.gradient_scale
        return loss, True, surprise_mean

    def replay_step(self, model: nn.Module) -> torch.Tensor | None:
        """Replay is intentionally disabled in the streaming learner."""
        _ = model
        return None

    def should_consolidate(self) -> bool:
        """Check if consolidation should be triggered."""
        return int(self.step_count.item()) % self.consolidation_interval == 0

    def get_stats(self) -> dict[str, Any]:
        """Get online learning statistics."""
        return {
            "step_count": int(self.step_count.item()),
            "buffer_size": 0,
            "expert_update_count": self.expert_update_count.tolist(),
            "expert_surprise_avg": (self.expert_surprise_accum / (self.expert_update_count + 1)).tolist(),
            "prototype_initialized": self.prototype_initialized.tolist(),
            "gradient_mode": self.gradient_mode.value,
            "gradient_scale": self.gradient_scale,
        }

    def save_state(self) -> dict[str, Any]:
        """Serialize online learner state."""
        return {
            "prototypes": self.prototypes.cpu().clone(),
            "prototype_initialized": self.prototype_initialized.cpu().clone(),
            "expert_update_count": self.expert_update_count.cpu().clone(),
            "expert_surprise_accum": self.expert_surprise_accum.cpu().clone(),
            "step_count": self.step_count.cpu().clone(),
            "gradient_mode": self.gradient_mode.value,
            "gradient_scale": self.gradient_scale,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore online learner state."""
        if "prototypes" in state:
            self.prototypes.copy_(state["prototypes"])
        if "prototype_initialized" in state:
            self.prototype_initialized.copy_(state["prototype_initialized"])
        if "expert_update_count" in state:
            self.expert_update_count.copy_(state["expert_update_count"])
        if "expert_surprise_accum" in state:
            self.expert_surprise_accum.copy_(state["expert_surprise_accum"])
        if "step_count" in state:
            self.step_count.copy_(state["step_count"])
        if "gradient_mode" in state:
            self.gradient_mode = GradientMode(state["gradient_mode"])
        if "gradient_scale" in state:
            self.gradient_scale = state["gradient_scale"]

    def reset_stats(self) -> None:
        """Reset tracking statistics."""
        self.expert_update_count.zero_()
        self.expert_surprise_accum.zero_()
        self.step_count.zero_()
