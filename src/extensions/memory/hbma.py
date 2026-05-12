"""HBMA — Human-Brain-Inspired Memory Architecture facade.

Concrete subsystems live in focused modules under ``src.extensions.memory``.
This module preserves backward-compatible imports for existing callers.
"""
from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .consolidation import ConsolidationEngine
from .context import ConsolidationContext
from .episodic import EpisodicMemory
from .interface import MemoryInterface, MemoryResult
from .plugin import MemorySubsystemPlugin
from .procedural import ProceduralMemory
from .salience import SalienceScorer
from .semantic import SemanticMemory
from .truth import NarsTruth
from .working import WorkingMemory


class HBMAMemory(MemoryInterface):
    """
    Human-Brain-Inspired Memory Architecture (pure PyTorch).

    4-system hierarchy:
      Working Memory    : sliding context buffer (immediate attention)
      Episodic Memory   : surprise-gated fast binding (hippocampus)
      Semantic Memory   : EMA prototype store (neocortex)
      Procedural Memory : learnable pattern store (implicit skills)

    Plus:
      ConsolidationEngine : Working->Episodic->Semantic pipeline
      SalienceScorer      : multi-factor retrieval weighting
      Learned routing gate: decides blend of all four systems

    Implements MemoryInterface directly (adapter logic inlined):
      forward(x, update_memory=False) -> MemoryResult
      memory_loss() -> combined auxiliary loss
      reset() -> clear all buffers
    """

    def __init__(
        self,
        hidden_dim: int,
        wm_capacity: int = 64,
        ep_slots: int = 256,
        ep_key_dim: int = 32,
        ep_forgetting_rate: float = 0.05,
        ep_surprise_threshold: float = 0.4,
        sem_prototypes: int = 256,
        sem_ema_momentum: float = 0.995,
        sem_temperature: float = 0.07,
        proc_patterns: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.working    = WorkingMemory(
            hidden_dim=hidden_dim,
            capacity=wm_capacity,
            dropout=dropout,
        )
        self.episodic   = EpisodicMemory(
            hidden_dim=hidden_dim,
            num_slots=ep_slots,
            key_dim=ep_key_dim,
            forgetting_rate=ep_forgetting_rate,
            surprise_threshold=ep_surprise_threshold,
            dropout=dropout,
        )
        self.semantic   = SemanticMemory(
            hidden_dim=hidden_dim,
            num_prototypes=sem_prototypes,
            ema_momentum=sem_ema_momentum,
            temperature=sem_temperature,
            dropout=dropout,
        )
        self.procedural = ProceduralMemory(
            hidden_dim=hidden_dim,
            num_patterns=proc_patterns,
            dropout=dropout,
        )
        self.consolidation = ConsolidationEngine(hidden_dim=hidden_dim, dropout=dropout)

        self._plugins = nn.ModuleList()
        self._plugin_names: list[str] = []
        self._needs_rebuild = False
        self._global_step = 0

        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),
        )

        self.fusion_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.fusion_gate = nn.Linear(hidden_dim, hidden_dim)
        self.norm        = nn.LayerNorm(hidden_dim)
        self.dropout_    = nn.Dropout(dropout)

    def register_plugin(self, plugin: MemorySubsystemPlugin) -> None:
        """Register a plugin subsystem. Call before first forward()."""
        if plugin.name in self._plugin_names:
            raise ValueError(f"Plugin '{plugin.name}' already registered")
        self._plugins.append(plugin)
        self._plugin_names.append(plugin.name)
        self._needs_rebuild = True

    def _get_all_subsystems(self) -> list[tuple[str, nn.Module]]:
        """Returns ordered list of (name, module) for all subsystems."""
        result = [
            ("working", self.working),
            ("episodic", self.episodic),
            ("semantic", self.semantic),
            ("procedural", self.procedural),
        ]
        indexed = list(zip(self._plugin_names, list(self._plugins)))
        indexed.sort(key=lambda x: x[1].priority)
        for name, mod in indexed:
            result.append((name, mod))
        return cast(list[tuple[str, nn.Module]], result)

    def _maybe_rebuild(self) -> None:
        """Rebuild router/fusion if plugins have been added."""
        if not self._needs_rebuild:
            return
        n = len(self._get_all_subsystems())
        old_router_out = self.router[-1].out_features
        if n == old_router_out:
            self._needs_rebuild = False
            return

        self.router = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, n),
        )
        self.fusion_proj = nn.Linear(self.hidden_dim * n, self.hidden_dim)
        self.fusion_gate = nn.Linear(self.hidden_dim, self.hidden_dim)
        self._needs_rebuild = False

    def _hbma_forward(
        self,
        x: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Internal HBMA forward: returns augmented representation.

        This is the original HBMAMemory.forward logic, preserved exactly.
        """
        self._maybe_rebuild()

        x = self.consolidation.consolidate(
            x, self.working, self.episodic, self.semantic
        )

        if self._plugins:
            ctx = ConsolidationContext(
                hidden=x, working=self.working, episodic=self.episodic,
                semantic=self.semantic, procedural=self.procedural,
                is_training=self.training, step=self._global_step,
            )
            for p in self._plugins:
                p.on_consolidate(ctx)

        all_subs = self._get_all_subsystems()
        outputs = [mod(x) for _, mod in all_subs]

        gate = F.softmax(self.router(x).float(), dim=-1).to(x.dtype)
        blended = sum(
            gate[:, i:i+1] * out for i, out in enumerate(outputs)
        )

        concat = torch.cat(outputs, dim=-1)
        fused  = self.dropout_(F.gelu(self.fusion_proj(concat)))
        fg     = torch.sigmoid(self.fusion_gate(fused))
        out    = fg * blended + (1 - fg) * fused
        out    = self.norm(out)

        self._global_step += 1

        if return_gate:
            return out, gate
        return out

    # ------------------------------------------------------------------
    # MemoryInterface contract
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        update_memory: bool = False,
    ) -> MemoryResult:
        """MemoryInterface: single-input forward returning MemoryResult.

        Inlined from HBMAMemoryAdapter:
        - Calls internal _hbma_forward for actual computation
        - Wraps output in MemoryResult
        - Computes surprise as normalized deviation from input
        - update_memory flag controls whether memory is actually updated
          (HBMA updates internally during training when self.training is True)
        """
        output = cast(torch.Tensor, self._hbma_forward(x))

        loss = self._compute_memory_loss()
        actually_updated = update_memory and self.training

        surprise = (output - x).norm(dim=-1, keepdim=True) / (x.norm(dim=-1, keepdim=True) + 1e-8)

        return MemoryResult(
            output=output,
            loss=loss,
            updated=actually_updated,
            surprise=surprise.detach(),
        )

    def _compute_memory_loss(self) -> torch.Tensor:
        """Combined auxiliary loss from all subsystems."""
        sem_loss  = self.semantic.diversity_loss()
        p_norm    = F.normalize(self.procedural.patterns, dim=-1)
        proc_sim  = p_norm @ p_norm.T
        mask      = ~torch.eye(proc_sim.shape[0], dtype=torch.bool,
                               device=proc_sim.device)
        proc_loss = proc_sim[mask].pow(2).mean()

        base_loss = 0.7 * sem_loss + 0.3 * proc_loss

        if not self._plugins:
            return base_loss
        plugin_losses = torch.tensor(0.0, device=base_loss.device, dtype=base_loss.dtype)
        for p in self._plugins:
            pl = p.auxiliary_loss()
            if isinstance(pl, torch.Tensor):
                plugin_losses = plugin_losses + pl
        n = len(self._plugins) + 1
        return (base_loss + plugin_losses) / n

    def reset(self, strategy: str = 'geometric') -> None:
        """Reset all stateful buffers across all memory systems."""
        self.working.reset()
        self.episodic.reset()
        self.semantic.reset()
        self.procedural.reset()
        for p in self._plugins:
            p.reset()

    def memory_loss(self) -> torch.Tensor:
        """Current auxiliary memory loss."""
        return self._compute_memory_loss()


HippocampusMemory = EpisodicMemory
NeocortexMemory   = SemanticMemory


class CLSMemory(HBMAMemory):
    """Backward-compatible alias: CLSMemory now delegates to HBMAMemory."""
    pass


__all__ = [
    "HBMAMemory",
    "CLSMemory",
    "HippocampusMemory",
    "NeocortexMemory",
    "NarsTruth",
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "ConsolidationEngine",
    "MemorySubsystemPlugin",
    "SalienceScorer",
    "ConsolidationContext",
]
