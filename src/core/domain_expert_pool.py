"""
HDIM — Domain Expert Pool
Загружает маленькие тематические SBERT-энкодеры как специализированные MoE-эксперты.

Каждый эксперт — это frozen sentence-transformer модель с trainable projection head.
HDIM-роутер выбирает нужного эксперта через soft dispatch, комбинируя их выходы.

Эксперты (lightweight, all frozen + trainable projection):
  0: all-MiniLM-L6-v2 (22M)      — general semantics
  1: paraphrase-MiniLM-L3-v2 (17M) — paraphrase/structural similarity
  2: multi-qa-MiniLM-L6-cos-v1 (22M) — QA/domain-crossing
  3: all-MiniLM-L12-v2 (33M)     — deep semantic analysis

Все эксперты frozen, только projection head (Linear→GELU→Linear) обучается.
Общий footprint: ~94M frozen params + ~100K trainable per expert.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# Expert model registry: name → (hf_model_name, output_dim)
EXPERT_REGISTRY = {
    0: ("sentence-transformers/all-MiniLM-L6-v2", 384),
    1: ("sentence-transformers/paraphrase-MiniLM-L3-v2", 384),
    2: ("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", 384),
    3: ("sentence-transformers/all-MiniLM-L12-v2", 384),
}


class ExpertProjection(nn.Module):
    """Trainable projection head for a frozen domain expert encoder.

    Maps expert's native dimension (typically 384 for MiniLM) to HDIM's
    clifford_dim.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DomainExpert(nn.Module):
    """Single domain expert: frozen SBERT + trainable projection.

    The SBERT encoder is kept on CPU and runs in no_grad mode to save VRAM.
    Only the projection head lives on GPU and receives gradients.
    """

    def __init__(
        self,
        expert_id: int,
        hf_model_name: str,
        native_dim: int,
        output_dim: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.expert_id = expert_id
        self.hf_model_name = hf_model_name
        self.native_dim = native_dim
        self.output_dim = output_dim

        # Load frozen SBERT on CPU
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(hf_model_name, device="cpu")
        for param in self._encoder.parameters():
            param.requires_grad = False
        self._encoder.eval()

        # Trainable projection: native_dim → output_dim
        self.projection = ExpertProjection(native_dim, output_dim)

    def encode_texts(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        """Encode texts with frozen encoder (always on CPU), return on target device."""
        with torch.no_grad():
            embeddings = self._encoder.encode(
                list(texts),
                convert_to_tensor=True,
                device="cpu",
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        # Move to target device and match projection dtype
        proj_dtype = next(self.projection.parameters()).dtype
        return embeddings.to(device=device, dtype=proj_dtype)

    def forward(
        self,
        texts: Sequence[str],
        device: torch.device,
        target_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode and project to output_dim."""
        if len(texts) == 0:
            dim = target_dim or self.output_dim
            return torch.empty(0, dim, device=device)

        embeddings = self.encode_texts(texts, device)
        projected = self.projection(embeddings)

        if target_dim is not None and target_dim != self.output_dim:
            projected = projected[..., :target_dim]

        return projected


class SharedExpert(nn.Module):
    """Always-on shared expert (DeepSeek-V3 pattern).

    Processes ALL inputs regardless of routing. Captures common cross-domain
    patterns that every domain benefits from.

    Architecture: Linear → GELU → Linear (lightweight FFN)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class DomainExpertPool(nn.Module):
    """Pool of specialized domain experts + shared expert.

    Manages loading, caching, and forward pass of multiple frozen domain
    encoders with trainable projections.

    Args:
        output_dim: Output dimension (matches clifford_dim)
        expert_ids: Which expert IDs to load from EXPERT_REGISTRY
        device: Device for trainable projections (encoders stay on CPU)
        use_shared_expert: Enable DeepSeek-V3 always-on shared expert
    """

    def __init__(
        self,
        output_dim: int,
        expert_ids: Optional[List[int]] = None,
        device: torch.device = torch.device("cpu"),
        use_shared_expert: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.expert_ids = expert_ids or [0, 1, 2, 3]

        # Domain-specific experts (frozen SBERT + trainable projection)
        self.experts = nn.ModuleDict()
        for eid in self.expert_ids:
            if eid not in EXPERT_REGISTRY:
                raise ValueError(f"Unknown expert ID {eid}. Available: {list(EXPERT_REGISTRY.keys())}")
            hf_name, native_dim = EXPERT_REGISTRY[eid]
            expert = DomainExpert(
                expert_id=eid,
                hf_model_name=hf_name,
                native_dim=native_dim,
                output_dim=output_dim,
            )
            self.experts[str(eid)] = expert

        # Shared expert (always-on)
        self.use_shared_expert = use_shared_expert
        if use_shared_expert:
            self.shared_expert = SharedExpert(output_dim, hidden_dim=output_dim * 2)

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def precompute_cache(
        self,
        texts: Sequence[str],
        batch_size: int = 128,
    ) -> None:
        """Pre-encode all texts with all domain experts for speed."""
        for eid, expert in self.experts.items():
            if hasattr(expert, '_encoder'):
                print(f"  Expert {eid} ({expert.hf_model_name.split('/')[-1]}): caching {len(texts)} texts...")
                # Each expert has its own SBERT cache
                expert._encoder.encode(
                    list(texts),
                    convert_to_tensor=True,
                    device="cpu",
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )

    def forward_expert(
        self,
        texts: Sequence[str],
        expert_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Run a single expert on texts."""
        expert = self.experts[str(expert_id)]
        return expert(texts, device=device, target_dim=self.output_dim)

    def forward_all_experts(
        self,
        texts: Sequence[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Stack all expert outputs: (num_experts, batch, output_dim)."""
        outputs = []
        for eid in self.expert_ids:
            out = self.forward_expert(texts, eid, device)
            outputs.append(out)
        return torch.stack(outputs)  # (E, B, D)

    def forward_with_routing(
        self,
        texts: Sequence[str],
        routing_weights: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Combine expert outputs using routing weights.

        Args:
            texts: batch of text strings
            routing_weights: (batch, num_experts) — from SoftMoERouter
            device: target device

        Returns:
            (batch, output_dim) — weighted combination of expert outputs
        """
        # Get all expert outputs
        expert_outputs = self.forward_all_experts(texts, device)  # (E, B, D)

        # Weighted combination: (E, B, D) * (B, E, 1) -> (B, D)
        weights = routing_weights.T.unsqueeze(-1)  # (E, B, 1)
        combined = (expert_outputs * weights).sum(dim=0)  # (B, D)

        # Add shared expert (always-on)
        if self.use_shared_expert:
            # shared expert processes combined output
            shared_out = self.shared_expert(combined)
            combined = combined + shared_out

        return combined
