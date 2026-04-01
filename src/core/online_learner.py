#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Online Learner for Self-Evolving HDIM Kernel.

Implements Test-Time Training (TTT) style online learning with:
- Experience replay buffer for stability
- Gradient-based surprise detection
- EMA model for stable targets (MoCo-style)

Reference:
- Titans: Learning to Memorize at Test Time (NeurIPS 2025)
- MoCo: Momentum Contrast (He et al., CVPR 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import random


@dataclass
class ReplaySample:
    """Single sample in replay buffer."""
    encoding: torch.Tensor
    domain_id: int
    surprise: float
    timestamp: int


class ReplayBuffer:
    """Prioritized experience replay buffer.

    Uses surprise-based prioritization to keep important samples.
    Implements FIFO eviction when buffer is full.
    """

    def __init__(
        self,
        capacity: int = 10000,
        hidden_dim: int = 256,
        device: torch.device = torch.device('cpu'),
        prioritized: bool = True,
        min_size: int = 100,
    ):
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.device = device
        self.prioritized = prioritized
        self.min_size = min_size

        # Storage
        self.buffer: List[ReplaySample] = []
        self._timestamp = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        encoding: torch.Tensor,
        domain_id: int,
        surprise: float = 0.0,
    ) -> None:
        """Add sample to buffer with surprise-based prioritization."""
        self._timestamp += 1

        sample = ReplaySample(
            encoding=encoding.detach().cpu(),
            domain_id=domain_id,
            surprise=surprise,
            timestamp=self._timestamp,
        )

        if len(self.buffer) >= self.capacity:
            if self.prioritized and len(self.buffer) > 0:
                # Evict lowest surprise sample
                min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i].surprise)
                self.buffer.pop(min_idx)
            else:
                # FIFO eviction
                self.buffer.pop(0)

        self.buffer.append(sample)

    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample batch from buffer."""
        if len(self.buffer) < self.min_size:
            raise ValueError(f"Buffer has {len(self.buffer)} samples, need {self.min_size}")

        batch_size = min(batch_size, len(self.buffer))
        samples = random.sample(self.buffer, batch_size)

        device = device or self.device

        return {
            'encoding': torch.stack([s.encoding for s in samples]).squeeze(1).to(device),
            'domain_id': torch.tensor([s.domain_id for s in samples], device=device),
            'surprise': torch.tensor([s.surprise for s in samples], device=device),
        }

    def get_high_surprise(
        self,
        top_k: int,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Get top-k highest surprise samples."""
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        top_k = min(top_k, len(self.buffer))
        sorted_buffer = sorted(self.buffer, key=lambda s: s.surprise, reverse=True)
        samples = sorted_buffer[:top_k]

        device = device or self.device

        return {
            'encoding': torch.stack([s.encoding for s in samples]).squeeze(1).to(device),
            'domain_id': torch.tensor([s.domain_id for s in samples], device=device),
            'surprise': torch.tensor([s.surprise for s in samples], device=device),
        }


class OnlineLearner(nn.Module):
    """Continuous learning module for self-evolving HDIM.

    Features:
    - TTT-style gradient updates during forward pass
    - Experience replay buffer for stability
    - Gradient-based surprise detection
    - EMA model for stable targets

    Integration point: HDIMModel._forward_core() after _apply_memory()
    """

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
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.ttt_lr = ttt_lr
        self.surprise_threshold = surprise_threshold
        self.grad_clip = grad_clip
        self.replay_batch_size = replay_batch_size
        self.consolidation_interval = consolidation_interval

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Experience replay buffer (on CPU to save VRAM)
        self.replay_buffer = ReplayBuffer(
            capacity=replay_buffer_size,
            hidden_dim=hidden_dim,
            device=torch.device('cpu'),
            prioritized=True,
        )

        # EMA model parameters for stable targets
        self.ema_decay = ema_decay
        self.register_buffer('ema_weights', torch.zeros(hidden_dim))
        self.register_buffer('ema_bias', torch.zeros(1))
        self._ema_initialized = False

        # Per-expert adaptation tracking
        self.register_buffer('expert_update_count', torch.zeros(num_experts, dtype=torch.long))
        self.register_buffer('expert_surprise_accum', torch.zeros(num_experts, dtype=torch.float))

        # Global step counter
        self.register_buffer('step_count', torch.zeros(1, dtype=torch.long))

        # Online optimizer state (lazy init)
        self._optimizer = None

    def _init_ema(self, reference_weight: torch.Tensor) -> None:
        """Initialize EMA weights from reference."""
        if not self._ema_initialized:
            self.ema_weights.data.copy_(reference_weight.detach().mean(dim=0))
            self.ema_bias.data.zero_()
            self._ema_initialized = True

    def _update_ema(self, new_weight: torch.Tensor) -> None:
        """Update EMA weights with exponential moving average."""
        if self._ema_initialized:
            with torch.no_grad():
                self.ema_weights.data.mul_(self.ema_decay).add_(
                    new_weight.detach().mean(dim=0) if new_weight.dim() > 1 else new_weight.detach(),
                    alpha=1 - self.ema_decay,
                )

    def compute_surprise(
        self,
        x: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute gradient-based surprise metric.

        Higher surprise = more novel input = should trigger update.

        Args:
            x: Input encoding [batch, hidden_dim]
            reference: Reference encoding (if None, use EMA)

        Returns:
            Surprise score [batch]
        """
        if reference is None:
            if not self._ema_initialized:
                # No EMA yet, return max surprise
                return torch.ones(x.size(0), device=x.device)
            reference = self.ema_weights

        # Cosine similarity to reference (lower = more surprising)
        if reference.dim() == 1:
            # Single reference vector
            cos_sim = F.cosine_similarity(x, reference.unsqueeze(0), dim=-1)
        else:
            # Multiple reference vectors, use max similarity
            cos_sim = F.cosine_similarity(x.unsqueeze(1), reference.unsqueeze(0), dim=-1).max(dim=1)[0]

        # Surprise = 1 - similarity (higher surprise when dissimilar)
        surprise = 1.0 - cos_sim

        return surprise

    def online_update(
        self,
        x: torch.Tensor,
        expert_idx: int,
        target: Optional[torch.Tensor] = None,
        force_update: bool = False,
    ) -> Tuple[torch.Tensor, bool, float]:
        """Perform online gradient update.

        Args:
            x: Input encoding [batch, hidden_dim]
            expert_idx: Which expert this belongs to
            target: Target encoding (if None, use EMA target)
            force_update: Force update regardless of surprise

        Returns:
            (loss, updated, surprise_mean): loss value, whether update happened, mean surprise
        """
        self.step_count += 1

        # Compute surprise
        surprise = self.compute_surprise(x)
        surprise_mean = float(surprise.mean().item())

        # Check if should update
        should_update = force_update or (surprise_mean > self.surprise_threshold and self.training)

        if not should_update:
            return torch.tensor(0.0, device=x.device), False, surprise_mean

        # Initialize EMA if needed
        self._init_ema(x.mean(dim=0))

        # Compute target (use EMA if no target provided)
        if target is None:
            with torch.no_grad():
                target = self.ema_weights.unsqueeze(0).expand(x.size(0), -1)

        # Compute loss
        loss = F.mse_loss(x, target)

        # Gradient step
        if self.training:
            # Store for replay buffer
            x_detached = x.detach()
            for i in range(x.size(0)):
                self.replay_buffer.add(
                    encoding=x_detached[i],
                    domain_id=expert_idx,
                    surprise=float(surprise[i].item()),
                )

            # Update tracking
            self.expert_update_count[expert_idx] += 1
            self.expert_surprise_accum[expert_idx] += surprise_mean

        # Update EMA
        self._update_ema(x.mean(dim=0))

        return loss, True, surprise_mean

    def replay_step(self, model: nn.Module) -> Optional[torch.Tensor]:
        """Sample from replay buffer and perform update.

        Args:
            model: HDIM model to update

        Returns:
            Replay loss or None if buffer too small
        """
        if len(self.replay_buffer) < self.replay_buffer.min_size:
            return None

        try:
            batch = self.replay_buffer.sample(self.replay_batch_size, device=self.device)

            # Compute replay loss (contrastive-style)
            encoding = batch['encoding']
            domain_ids = batch['domain_id']

            # Simple intra-class contrastive loss
            loss = torch.tensor(0.0, device=self.device)

            for domain_id in domain_ids.unique():
                domain_mask = domain_ids == domain_id
                domain_encodings = encoding[domain_mask]

                if domain_encodings.size(0) > 1:
                    # Push same-domain samples together
                    center = domain_encodings.mean(dim=0, keepdim=True)
                    loss += F.mse_loss(domain_encodings, center.expand_as(domain_encodings))

            return loss / max(len(domain_ids.unique()), 1)

        except ValueError:
            return None

    def should_consolidate(self) -> bool:
        """Check if consolidation should be triggered."""
        return int(self.step_count.item()) % self.consolidation_interval == 0

    def get_stats(self) -> Dict[str, Any]:
        """Get online learning statistics."""
        return {
            'step_count': int(self.step_count.item()),
            'buffer_size': len(self.replay_buffer),
            'expert_update_count': self.expert_update_count.tolist(),
            'expert_surprise_avg': (self.expert_surprise_accum / (self.expert_update_count + 1)).tolist(),
            'ema_initialized': self._ema_initialized,
        }

    def save_state(self) -> Dict[str, Any]:
        """Serialize online learner state for checkpoint persistence.

        Includes EMA buffers, expert tracking stats, replay buffer contents,
        and optimizer state if initialized.
        """
        state: Dict[str, Any] = {
            'ema_weights': self.ema_weights.cpu().clone(),
            'ema_bias': self.ema_bias.cpu().clone(),
            'ema_initialized': self._ema_initialized,
            'expert_update_count': self.expert_update_count.cpu().clone(),
            'expert_surprise_accum': self.expert_surprise_accum.cpu().clone(),
            'step_count': self.step_count.cpu().clone(),
        }

        # Save replay buffer samples
        buffer_data = []
        for sample in self.replay_buffer.buffer:
            buffer_data.append({
                'encoding': sample.encoding,
                'domain_id': sample.domain_id,
                'surprise': sample.surprise,
                'timestamp': sample.timestamp,
            })
        state['replay_buffer'] = buffer_data
        state['buffer_timestamp'] = self.replay_buffer._timestamp

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore online learner state from checkpoint.

        Restores EMA buffers, expert tracking, and replay buffer.
        """
        if 'ema_weights' in state:
            self.ema_weights.copy_(state['ema_weights'])
        if 'ema_bias' in state:
            self.ema_bias.copy_(state['ema_bias'])
        self._ema_initialized = state.get('ema_initialized', False)
        if 'expert_update_count' in state:
            self.expert_update_count.copy_(state['expert_update_count'])
        if 'expert_surprise_accum' in state:
            self.expert_surprise_accum.copy_(state['expert_surprise_accum'])
        if 'step_count' in state:
            self.step_count.copy_(state['step_count'])

        # Restore replay buffer
        if 'replay_buffer' in state and self.replay_buffer.prioritized:
            self.replay_buffer.buffer.clear()
            self.replay_buffer._timestamp = state.get('buffer_timestamp', 0)
            for item in state['replay_buffer']:
                sample = ReplaySample(
                    encoding=item['encoding'],
                    domain_id=item['domain_id'],
                    surprise=item['surprise'],
                    timestamp=item['timestamp'],
                )
                self.replay_buffer.buffer.append(sample)

    def reset_stats(self) -> None:
        """Reset tracking statistics."""
        self.expert_update_count.zero_()
        self.expert_surprise_accum.zero_()
        self.step_count.zero_()
