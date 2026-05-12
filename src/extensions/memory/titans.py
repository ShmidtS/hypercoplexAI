"""HDIM — Titans Memory Module

Neural long-term memory with test-time training (TTT) updates.

Mathematics (from Titans architecture):
  L_memory = || M_{t-1}(k_t) - v_t ||^2
  S_t = eta_t * S_{t-1} - theta_t * grad L_memory   (momentum gradient step)
  M_t = (1 - alpha_t) * M_{t-1} + S_t          (memory update)
  alpha_t, eta_t, theta_t are learnable scalars (sigmoid gates from input)

Implements MemoryInterface directly — adapter logic is inlined:
  - key_proj: projects input x to key space for memory lookup
  - gate: learned blend gate for x + gate * retrieved
  - forward(x, update_memory=False) -> MemoryResult
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interface import MemoryInterface, MemoryResult


@dataclass
class _MemoryState:
    """Internal state from retrieve/update operations."""
    retrieved: torch.Tensor
    loss: torch.Tensor
    updated: bool
    alpha: Optional[torch.Tensor] = None
    eta: Optional[torch.Tensor] = None
    theta: Optional[torch.Tensor] = None


class TitansMemory(MemoryInterface):
    """
    Neural associative memory with TTT update, implementing MemoryInterface.

    Memory M is stored as a linear layer without bias: M: R^{key_dim} -> R^{val_dim}
    Updated online via gradient step on L_memory at each forward.

    Adapter logic inlined:
      - key_proj projects input x to key space
      - gate blends: x + gate * retrieved
      - Returns MemoryResult directly (no adapter wrapper)

    Args:
        clifford_dim: input/output dimension (val_dim for memory)
        memory_key_dim: dimension of memory keys
        hidden_dim: dimension for gate projection computation
        memory_max_norm: max norm constraint for memory weights
        ttt_lr_scale: scale for TTT learning rate
    """

    def __init__(
        self,
        clifford_dim: int,
        memory_key_dim: int = 32,
        hidden_dim: int = 64,
        memory_max_norm: float = 5.0,
        ttt_lr_scale: float = 0.005,
    ):
        super().__init__()
        self.clifford_dim = clifford_dim
        self.memory_key_dim = memory_key_dim
        self.key_dim = memory_key_dim
        self.val_dim = clifford_dim

        # Configuration parameters
        self.memory_max_norm = memory_max_norm
        self.ttt_lr_scale = ttt_lr_scale

        # Key projection: x -> key space (inlined from TitansAdapter)
        self.key_proj = nn.Linear(clifford_dim, memory_key_dim)

        # Blend gate: x -> scalar gate (inlined from TitansAdapter)
        self.gate = nn.Sequential(
            nn.Linear(clifford_dim, clifford_dim // 4),
            nn.ReLU(),
            nn.Linear(clifford_dim // 4, 1),
        )

        # Memory M implemented as linear layer (weight matrix)
        self.memory = nn.Linear(memory_key_dim, clifford_dim, bias=False)
        nn.init.normal_(self.memory.weight, std=0.01)

        # Momentum state S (buffer, not parameter)
        self.register_buffer('momentum_S', torch.zeros(clifford_dim, memory_key_dim))
        self.register_buffer('evidence_strength', torch.zeros(memory_key_dim))

        # Gates alpha (forget), eta (momentum), theta (lr) — two-layer projection
        self.gate_proj = nn.Sequential(
            nn.Linear(memory_key_dim + clifford_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3, bias=True),
        )
        nn.init.zeros_(self.gate_proj[2].weight)
        nn.init.constant_(self.gate_proj[2].bias, 0.0)

        # Phase 22: gradient-based surprise + adaptive forgetting
        self.use_gradient_surprise: bool = False
        self.use_adaptive_forgetting: bool = False
        self._last_surprise: float = 0.0

        # Phase 29: RAG-compatible freeze state
        self._frozen: bool = False

        # Track last loss for memory_loss() method
        self.register_buffer("_last_loss", torch.tensor(0.0))

    # ==================== Phase 29: RAG-compatible API ====================

    def freeze_memory(self) -> None:
        """Freeze memory weights for RAG inference."""
        self.memory.weight.requires_grad_(False)
        self._frozen = True

    def unfreeze_memory(self) -> None:
        """Unfreeze for training."""
        self.memory.weight.requires_grad_(True)
        self._frozen = False

    def is_frozen(self) -> bool:
        """Check if memory is frozen."""
        return self._frozen

    @torch.no_grad()
    def retrieve_only(self, k: torch.Tensor) -> torch.Tensor:
        """RAG-compatible retrieval without memory update.

        Args:
            k: key tensor (..., key_dim)

        Returns:
            Retrieved value tensor (..., val_dim)
        """
        return self.memory(k)

    # Backward compatibility properties
    @property
    def _MEMORY_MAX_NORM(self) -> float:
        return self.memory_max_norm

    @property
    def _TTT_LR_SCALE(self) -> float:
        return self.ttt_lr_scale

    # =========================================================================
    # Internal operations (k, v API)
    # =========================================================================

    def _retrieve(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> _MemoryState:
        retrieved = self.memory(k)
        loss_memory = F.mse_loss(retrieved, v.detach())
        return _MemoryState(retrieved=retrieved, loss=loss_memory, updated=False)

    def _compute_surprise(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Gradient norm as surprise metric (Titans, NeurIPS 2025)."""
        k32 = k.detach().float().requires_grad_(True)
        pred = self.memory(k32)
        loss = F.mse_loss(pred, v.detach().float())
        grad = torch.autograd.grad(loss, k32, retain_graph=False)[0]
        return grad.norm(dim=-1).mean()

    def _adaptive_alpha(self, surprise: torch.Tensor, base_alpha: torch.Tensor) -> torch.Tensor:
        """High surprise -> less forgetting."""
        surprise_norm = torch.sigmoid(surprise - 1.0)
        return base_alpha * (1.0 - 0.5 * surprise_norm)

    def _update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # C4+A7 FIX: explicit fp32 TTT path — avoids AMP fp16 overflow
        k32 = k.detach().float()
        v32 = v.detach().float()
        mem_w = self.memory.weight.detach().float().requires_grad_(True)
        k_agg = k32.reshape(-1, k32.shape[-1]).mean(0) if k32.dim() > 1 else k32
        v_agg = v32.reshape(-1, v32.shape[-1]).mean(0) if v32.dim() > 1 else v32
        kv_agg = torch.cat([k_agg, v_agg], dim=-1)
        gates = torch.sigmoid(self.gate_proj(kv_agg.to(self.gate_proj[0].weight.dtype)))
        alpha = gates[..., 0].float()
        eta   = gates[..., 1].float()
        theta = gates[..., 2].float()
        pred = k32 @ mem_w.T
        loss_ttt = F.mse_loss(pred, v32)
        (grad,) = torch.autograd.grad(
            loss_ttt,
            mem_w,
            retain_graph=False,
            create_graph=False,
        )
        grad_clamped = grad.detach().clamp(-1.0, 1.0)
        mom_fp32 = self.momentum_S.detach().float()
        new_momentum = eta * mom_fp32 - self.ttt_lr_scale * theta * grad_clamped
        momentum_norm = new_momentum.norm()
        if momentum_norm > self.memory_max_norm:
            new_momentum = new_momentum * (self.memory_max_norm / (momentum_norm + 1e-8))
        self.momentum_S.copy_(new_momentum.to(self.momentum_S.dtype))
        ema_beta = 0.99
        self.evidence_strength.mul_(ema_beta).add_(eta.detach().abs().mean() * (1 - ema_beta))
        self.evidence_strength.clamp_(max=10.0)
        effective_alpha = alpha
        if self.use_gradient_surprise:
            surprise = self._compute_surprise(k32, v32)
            self._last_surprise = surprise.item()
            if self.use_adaptive_forgetting:
                effective_alpha = self._adaptive_alpha(surprise, alpha)
        self.evidence_strength.clamp_(min=0.0, max=10.0)
        evidence_factor = 1.0 / (1.0 + self.evidence_strength.mean())
        effective_alpha = effective_alpha * evidence_factor
        new_weight = (1 - effective_alpha) * mem_w.detach() + new_momentum
        weight_norm = new_weight.norm()
        if weight_norm > self._MEMORY_MAX_NORM:
            new_weight = new_weight * (self._MEMORY_MAX_NORM / (weight_norm + 1e-8))
        self.memory.weight.data.copy_(new_weight.to(self.memory.weight.dtype))
        return alpha.detach(), eta.detach(), theta.detach()

    def _retrieve_and_update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        update_memory: bool = True,
    ) -> _MemoryState:
        if self._frozen and update_memory:
            update_memory = False

        state = self._retrieve(k, v)
        if update_memory and self.training:
            alpha, eta, theta = self._update(k, v)
            state.updated = True
            state.alpha = alpha
            state.eta = eta
            state.theta = theta
        return state

    # =========================================================================
    # MemoryInterface contract
    # =========================================================================

    def forward(
        self,
        x: torch.Tensor,
        update_memory: bool = False,
    ) -> MemoryResult:
        """MemoryInterface: single-input forward returning MemoryResult.

        Projects x to key space, retrieves/updates memory, applies gating.

        Args:
            x: input tensor [B, clifford_dim]
            update_memory: whether to update memory

        Returns:
            MemoryResult with gated output, loss, and gate values
        """
        # Key projection (inlined from TitansAdapter)
        k = self.key_proj(x)

        # Retrieve and optionally update (v = x for single-input API)
        mem_state = self._retrieve_and_update(k, x, update_memory=update_memory)
        self._last_loss = mem_state.loss.detach()

        # Gated blend: x + gate * retrieved (inlined from TitansAdapter)
        gate_val = torch.sigmoid(self.gate(x))
        gated_output = x + gate_val * mem_state.retrieved

        # Export surprise signal if available
        surprise = None
        if self.use_gradient_surprise:
            surprise = torch.tensor(self._last_surprise, device=x.device)

        return MemoryResult(
            output=gated_output,
            loss=mem_state.loss,
            updated=mem_state.updated,
            alpha=mem_state.alpha,
            eta=mem_state.eta,
            theta=mem_state.theta,
            surprise=surprise,
        )

    def reset(self, strategy: str = 'geometric', decay_window: float = 50.0) -> None:
        """Smart reset — not full zero, preserves important patterns.

        Strategies:
            'hard'      — full reset to zeros (only at init/epoch=1)
            'geometric' — exponential decay of weights (preserves patterns)
            'stabilize' — momentum normalization without changing memory weights
        """
        with torch.no_grad():
            if strategy == 'hard':
                self.memory.weight.zero_()
                self.momentum_S.zero_()
                self.evidence_strength.zero_()
            elif strategy == 'geometric':
                decay = torch.exp(torch.tensor(-1.0 / max(decay_window, 1.0)))
                self.memory.weight.mul_(decay)
                self.momentum_S.mul_(decay * 0.5)
            elif strategy == 'stabilize':
                norm = self.momentum_S.norm()
                if norm > self._MEMORY_MAX_NORM:
                    self.momentum_S.mul_(self._MEMORY_MAX_NORM / (norm + 1e-8))

    # Backward compat alias
    reset_memory = reset

    def memory_loss(self) -> torch.Tensor:
        """Current auxiliary memory loss."""
        return self._last_loss

    def stats(self) -> dict:
        """Return memory statistics for monitoring."""
        return {
            "memory_size": self.memory.weight.shape,
            "momentum_value": self.momentum_S.norm().item(),
            "is_frozen": self._frozen,
            "last_loss": self.memory(self.memory.weight.new_zeros(1, self.key_dim)).detach().norm().item(),
        }


# Backward compatibility alias
TitansMemoryModule = TitansMemory
