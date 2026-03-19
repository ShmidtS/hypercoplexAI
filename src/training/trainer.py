"""HDIMTrainer — training loop, loss computation, and checkpoint management."""

from __future__ import annotations
from dataclasses import dataclass
import math
import os
from typing import Any, Dict, Literal, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from src.models.hdim_model import HDIMAuxState, HDIMModel
from src.training.augmentations import EmbeddingAugmenter
@dataclass(frozen=True)
class TrainingRegime:
    mode: Literal["reconstruction", "paired"]
    update_memory: bool
    memory_mode: Literal["none", "retrieve", "update"]

class HDIMTrainer:
    """Trainer for the HDIMModel."""

    def __init__(
        self,
        model: HDIMModel,
        optimizer: Optimizer,
        device: str | torch.device = "cpu",
        lambda_iso: float = 0.1,
        lambda_routing: float = 0.05,
        lambda_pair: float = 0.1,
        lambda_memory: float = 0.05,
        negative_margin: float = 1.0,
        ranking_margin: float = 0.2,
        use_infonce: bool = True,
        infonce_temperature: float = 0.15,
        lambda_sts: float = 0.0,
        lambda_angle: float = 0.0,
        lambda_supcon: float = 0.0,
        lambda_z: float = 0.0,
        lambda_expert_ortho: float = 0.0,
        learnable_temperature: bool = False,
        lambda_dcl: float = 0.0,
        lambda_uniformity: float = 0.0,
        lambda_diversity_var: float = 0.01,
        lambda_diversity_ortho: float = 0.005,
        use_sc_temperature: bool = False,
        lambda_matryoshka: float = 0.1,
        # Temperature scheduling parameters (exposed for configurability)
        temp_schedule: str = "none",
        tau_max: float = 0.1,
        tau_min: float = 0.01,
        temp_schedule_T_0: int = 20,
        focal_gamma: float = 1.0,
        # Embedding augmentation
        aug_noise_std: float = 0.0,
        aug_mixup_alpha: float = 0.0,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.lambda_iso = lambda_iso
        self.lambda_routing = lambda_routing
        self.lambda_pair = lambda_pair
        self.lambda_memory = lambda_memory
        self.negative_margin = negative_margin
        self.ranking_margin = ranking_margin
        self.use_infonce = use_infonce
        self.infonce_temperature = infonce_temperature
        self.lambda_sts = lambda_sts
        self.lambda_angle = lambda_angle
        self.lambda_supcon = lambda_supcon
        self.lambda_z = lambda_z
        self.lambda_expert_ortho = lambda_expert_ortho
        self.lambda_dcl = lambda_dcl
        self.lambda_uniformity = lambda_uniformity
        self.lambda_diversity_var = lambda_diversity_var
        self.lambda_diversity_ortho = lambda_diversity_ortho
        self.lambda_matryoshka = lambda_matryoshka
        self.use_sc_temperature = use_sc_temperature

        # Warn about conflicting loss combinations
        if focal_gamma < 1.0 and lambda_dcl > 0:
            print("WARNING: Focal-InfoNCE + DCL simultaneously — DCL already removes positive from denominator, "
                  "focal modulation is redundant. Consider using DCL alone or InfoNCE+Focal alone.")
        if use_infonce and lambda_supcon > 0:
            print("WARNING: InfoNCE + SupCon simultaneously — SupCon generalizes InfoNCE, running both wastes compute. "
                  "SupCon overrides InfoNCE when group_id is present.")
        if lambda_uniformity > lambda_pair * 2:
            print("WARNING: lambda_uniformity >> lambda_pair — uniformity pushes points apart, "
                  "may prevent contrastive clustering. Recommend lambda_uniformity < lambda_pair.")
        if lambda_dcl > 0 and lambda_uniformity > 0:
            print("NOTE: DCL + Uniformity is a strong combo for embedding uniformity — ensure lambda_pair is tuned.")
        self.use_hard_negatives: bool = False
        self._last_cluster_temp: float | None = None
        self._step: int = 0
        # Focal-InfoNCE gamma (Hou & Li, EMNLP 2023)
        self._focal_gamma: float = focal_gamma  # 1.0 = standard InfoNCE, <1 = focal
        # Temperature scheduling (now configurable)
        self._current_epoch: int = 0
        self._temp_schedule: str = temp_schedule  # "none" | "warm_restart"
        self._tau_max: float = tau_max
        self._tau_min: float = tau_min
        self._temp_schedule_T_0: int = temp_schedule_T_0  # matches scheduler T_0
        # Learnable temperature (log-scale for numerical stability)
        # B33 FIX: register as buffer on model so it survives checkpoint save/load
        import math
        if learnable_temperature:
            self._log_temp = nn.Parameter(
                torch.tensor(math.log(infonce_temperature), device=torch.device(device))
            )
            # Register on model so it appears in state_dict
            self.model.register_parameter('_log_temp', self._log_temp)
        else:
            self._log_temp = None
        # Embedding augmenter (training only, no-op in eval)
        self._augmenter = EmbeddingAugmenter(
            noise_std=aug_noise_std,
            dropout_p=0.0,
            mixup_alpha=aug_mixup_alpha,
        ) if (aug_noise_std > 0 or aug_mixup_alpha > 0) else None
    def _resolve_training_regime(self, batch: Dict[str, Any]) -> TrainingRegime:
        is_pair_batch = self._has_pairs(batch)
        return TrainingRegime(
            mode="paired" if is_pair_batch else "reconstruction",
            update_memory=self.model.training,
            memory_mode="update" if self.model.training else "retrieve",
        )

    def _uses_text_model(self) -> bool:
        return hasattr(self.model, "forward_texts") and hasattr(
            self.model, "encode_texts"
        )

    def _has_texts(self, batch: Dict[str, Any]) -> bool:
        texts = batch.get("text")
        return isinstance(texts, Sequence) and not isinstance(texts, (str, bytes))

    def _has_pair_texts(self, batch: Dict[str, Any]) -> bool:
        texts = batch.get("pair_text")
        return isinstance(texts, Sequence) and not isinstance(texts, (str, bytes))

    def _extract_texts(self, batch: Dict[str, Any], key: str) -> list[str]:
        texts = batch.get(key)
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
            raise KeyError(f"Batch key '{key}' must contain a sequence of raw texts")
        return [str(text) for text in texts]

    def _encode_texts(
        self,
        texts: Sequence[str],
        *,
        collect_matryoshka: bool = False,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor] | None] | torch.Tensor:
        if collect_matryoshka and hasattr(self.model, 'encode_texts_matryoshka'):
            full_enc, scales = self.model.encode_texts_matryoshka(
                texts, device=self.device,
            )
            return full_enc, scales
        enc = self.model.encode_texts(texts, device=self.device)
        if collect_matryoshka:
            return enc, None
        return enc

    def _extract_iso_targets(
        self, batch: Dict[str, Any], aux_state: HDIMAuxState
    ) -> torch.Tensor:
        if self._uses_text_model() and self._has_pair_texts(batch):
            pair_texts = self._extract_texts(batch, "pair_text")
            pair_domain_id = batch["pair_domain_id"].to(self.device)
            return self._compute_pair_iso_targets_from_text(pair_texts, pair_domain_id)

        if "pair_encoding" not in batch or "pair_domain_id" not in batch:
            return self._training_invariant(aux_state).detach()

        pair_encoding = batch["pair_encoding"].to(self.device)
        pair_domain_id = batch["pair_domain_id"].to(self.device)
        return self._compute_pair_iso_targets(pair_encoding, pair_domain_id)

    def _training_invariant(self, aux_state: HDIMAuxState) -> torch.Tensor:
        return aux_state.training_invariant

    def _has_pairs(self, batch: Dict[str, Any]) -> bool:
        return ("pair_encoding" in batch and "pair_domain_id" in batch) or (
            self._has_pair_texts(batch) and "pair_domain_id" in batch
        )

    def _compute_pair_iso_targets(
        self, pair_encoding: torch.Tensor, pair_domain_id: torch.Tensor
    ) -> torch.Tensor:
        _, _, _, aux_state = self._forward_batch(
            pair_encoding,
            pair_domain_id,
            TrainingRegime(
                mode="reconstruction",
                update_memory=False,
                memory_mode="retrieve",
            ),
        )
        return self._training_invariant(aux_state).detach()

    def _compute_pair_iso_targets_from_text(
        self, pair_texts: Sequence[str], pair_domain_id: torch.Tensor
    ) -> torch.Tensor:
        _, _, _, aux_state = self._forward_text_batch(
            pair_texts,
            pair_domain_id,
            TrainingRegime(
                mode="reconstruction",
                update_memory=False,
                memory_mode="retrieve",
            ),
        )
        return self._training_invariant(aux_state).detach()

    def _compute_reconstruction_loss(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return nn.functional.mse_loss(output, target)

    def _compute_pair_iso_loss(
        self,
        training_invariant: torch.Tensor,
        iso_target: torch.Tensor,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        if not self._has_pairs(batch):
            return self.compute_iso_loss(training_invariant, iso_target)

        pair_relation_label = batch.get("pair_relation_label")
        pair_weight = batch.get("pair_weight")
        if pair_relation_label is None or pair_weight is None:
            return self.compute_iso_loss(training_invariant, iso_target)

        pair_relation_label = pair_relation_label.to(
            self.device, dtype=training_invariant.dtype
        )
        pair_weight = pair_weight.to(self.device, dtype=training_invariant.dtype)
        per_sample_mse = F.mse_loss(
            training_invariant, iso_target, reduction="none"
        ).mean(dim=-1)
        positive_mask = pair_relation_label > 0.5
        negative_mask = ~positive_mask
        losses = []
        if positive_mask.any():
            positive_loss = (
                per_sample_mse[positive_mask] * pair_weight[positive_mask]
            ).sum() / pair_weight[positive_mask].sum()
            losses.append(positive_loss)
        if negative_mask.any():
            negative_penalty = F.relu(
                self.negative_margin - per_sample_mse[negative_mask]
            )
            weighted_negative = (
                negative_penalty * pair_weight[negative_mask]
            ).sum() / pair_weight[negative_mask].sum()
            losses.append(weighted_negative)
        if not losses:
            return per_sample_mse.mean()
        return torch.stack(losses).mean()

    def _zero_loss(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=reference.device, dtype=reference.dtype)

    def _compute_diversity_loss(self, exported_invariant: torch.Tensor) -> torch.Tensor:
        """Anti-collapse: encourage variance in invariant vectors.

        Prevents the degenerate solution where all exported_invariants
        converge to the same constant vector (causing STS→1.0, margin→0).

        Uses two penalties:
        1. Variance penalty: -mean(‖x_i - mean‖²) — push vectors apart
        2. Orthogonality bonus: encourage non-zero pairwise angles

        Returns zero loss when batch is too small (< 4) to be meaningful.
        """
        B = exported_invariant.shape[0]
        if B < 4:
            return self._zero_loss(exported_invariant)

        # Normalize for consistent scale
        x = F.normalize(exported_invariant, dim=-1)  # (B, D)

        # 1. Variance penalty: encourage vectors to spread out on the hypersphere
        mean_vec = x.mean(dim=0, keepdim=True)  # (1, D)
        variance = ((x - mean_vec) ** 2).sum(dim=-1).mean()  # scalar

        # 2. Cosine spread: penalize high average pairwise cosine similarity
        # When all vectors collapse to one direction, avg_cosine → 1.0
        cos_matrix = x @ x.T  # (B, B)
        eye = torch.eye(B, device=x.device, dtype=torch.bool)
        off_diag = cos_matrix.masked_fill(eye, 0.0)
        avg_cosine = off_diag.abs().sum() / max(2 * (B * B - B), 1)

        # Combined: we want HIGH variance and LOW avg cosine
        # loss_div = -variance + avg_cosine (minimize this)
        loss_div = -self.lambda_diversity_var * variance + self.lambda_diversity_ortho * avg_cosine
        return loss_div

    def _resolve_pair_group_id(self, batch: Dict[str, Any]) -> torch.Tensor | None:
        pair_group_id = batch.get("pair_group_id")
        if pair_group_id is None:
            pair_group_id = batch.get("pair_family_id")
        if pair_group_id is None:
            return None
        return pair_group_id.to(self.device)

    def _extract_pair_target_state(
        self, batch: Dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._uses_text_model() and self._has_pair_texts(batch):
            pair_texts = self._extract_texts(batch, "pair_text")
            pair_domain_id = batch["pair_domain_id"].to(self.device)
            _, _, _, aux_state = self._forward_text_batch(
                pair_texts,
                pair_domain_id,
                TrainingRegime(
                    mode="reconstruction",
                    update_memory=False,
                    memory_mode="retrieve",
                ),
            )
            return (
                self._training_invariant(aux_state).detach(),
                aux_state.exported_invariant.detach(),
            )

        pair_encoding = batch.get("pair_encoding")
        pair_domain_id = batch.get("pair_domain_id")
        if pair_encoding is None or pair_domain_id is None:
            return None
        _, _, _, aux_state = self._forward_batch(
            pair_encoding.to(self.device),
            pair_domain_id.to(self.device),
            TrainingRegime(
                mode="reconstruction",
                update_memory=False,
                memory_mode="retrieve",
            ),
        )
        return (
            self._training_invariant(aux_state).detach(),
            aux_state.exported_invariant.detach(),
        )

    def _compute_negative_pair_indices(
        self,
        pair_group_id: torch.Tensor,
        domain_id: torch.Tensor,
        pair_domain_id: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = pair_group_id.shape[0]
        batch_indices = torch.arange(batch_size, device=self.device)
        negative_indices = torch.full(
            (batch_size,), -1, dtype=torch.long, device=self.device
        )
        for idx in range(batch_size):
            valid_candidates = batch_indices[pair_group_id != pair_group_id[idx]]
            if valid_candidates.numel() == 0:
                continue
            cross_domain_candidates = valid_candidates[
                pair_domain_id[valid_candidates] != domain_id[idx]
            ]
            negative_indices[idx] = (
                cross_domain_candidates[0]
                if cross_domain_candidates.numel() > 0
                else valid_candidates[0]
            )
        return negative_indices
    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for temperature scheduling.

        Memory reset strategy (revised Phase 21):
          epoch=1 -> hard reset (new training run)
          epoch>1 -> NO per-epoch reset (memory accumulates patterns per LR cycle)

        Per-epoch geometric decay was counterproductive:
        score at ep5 dropped from ~0.75 to ~0.25 (patterns lost each epoch).
        LR restart in gpu_train.py calls stabilize() for momentum normalization.
        """
        self._current_epoch = epoch
        if epoch == 1 and hasattr(self.model, "reset_memory"):
            self.model.reset_memory(strategy="hard")

    def _effective_temperature(self) -> float:
        """Возвращает эффективную температуру (обучаемую, scheduled, или фиксированную)."""
        if self._log_temp is not None:
            return float(self._log_temp.exp().clamp(0.01, 0.5).item())
        if self._temp_schedule == "warm_restart":
            # Linear decay within current LR restart cycle
            T_0 = self._temp_schedule_T_0
            epoch_in_cycle = self._current_epoch % T_0
            fraction = epoch_in_cycle / max(T_0, 1)
            return self._tau_max - (self._tau_max - self._tau_min) * fraction
        return self.infonce_temperature

    def _cluster_scaled_temperature(self, embeddings: torch.Tensor, base_temp: float) -> float:
        """SC-InfoNCE: scale temperature based on cluster tightness (Cheng et al., Nov 2025).

        Tighter clusters → lower temperature (sharper distribution).
        Looser clusters → higher temperature (softer distribution).
        """
        if embeddings.numel() == 0:
            return base_temp
        cluster_var = embeddings.var(dim=0, unbiased=False).mean()
        scale = torch.exp(-cluster_var)
        scaled = base_temp * scale.clamp(0.5, 2.0)
        self._last_cluster_temp = float(scaled.item())
        return self._last_cluster_temp

    def _compute_angle_loss(
        self,
        source_inv: torch.Tensor,
        target_inv: torch.Tensor,
        pair_relation_label: torch.Tensor,
        pair_weight: torch.Tensor,
    ) -> torch.Tensor:
        """AnglE Loss: работает в угловом пространстве, избегает насыщения косинуса.

        Li & Li (2023) — AnglE-optimized Text Embeddings.
        Для позитивных пар: angle → 0 (cos → 1).
        Для негативных пар: angle → π/2 (cos → 0).
        """
        import math
        B = source_inv.shape[0]
        if B < 2:
            return self._zero_loss(source_inv)

        src = F.normalize(source_inv, dim=-1)
        tgt = F.normalize(target_inv, dim=-1)
        cos_sim = (src * tgt).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)  # [B]

        # Угол в [0, 1]: 0=идентичные, 1=ортогональные
        angle = torch.acos(cos_sim) * 2 / math.pi  # [B]

        # Позитивные: angle → 0, негативные: angle → 1
        target_angle = 1.0 - pair_relation_label  # [B]

        loss_per_sample = (angle - target_angle).pow(2)
        weights = pair_weight.to(source_inv.device, dtype=source_inv.dtype)
        return (loss_per_sample * weights).sum() / weights.sum().clamp_min(1e-8)

    def _mine_hard_negatives(
        self,
        source_inv: torch.Tensor,
        target_inv: torch.Tensor,
        pair_relation_label: torch.Tensor,
        top_k: int = 4,
    ) -> torch.Tensor:
        """Online hard negative mining: find hardest negatives in-batch.

        Returns indices (B,) — for each anchor, index of hardest negative.
        -1 means no hard negative found.
        """
        B = source_inv.shape[0]
        src = F.normalize(source_inv.detach(), dim=-1)
        tgt = F.normalize(target_inv.detach(), dim=-1)
        sim = src @ tgt.T  # (B, B)

        # Mask diagonal and true positives
        eye = torch.eye(B, dtype=torch.bool, device=self.device)
        pos_mask = pair_relation_label > 0.5
        # Use -1e4 instead of -1e9 to avoid fp16 overflow (max fp16 ~65504)
        _NEG_INF = -1e4
        sim = sim.masked_fill(eye, _NEG_INF)
        # Mask same-positive pairs
        for i in range(B):
            if pos_mask[i]:
                sim[i] = sim[i].masked_fill(pos_mask, _NEG_INF)
                sim[i, i] = _NEG_INF

        # Hardest negative = highest similarity among non-positives
        hard_neg_idx = sim.argmax(dim=-1)  # (B,)
        return hard_neg_idx

    def _compute_infonce_loss(
        self,
        source_inv: torch.Tensor,
        target_inv: torch.Tensor,
        pair_relation_label: torch.Tensor,
        pair_weight: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """InfoNCE loss (NT-Xent) for positive/negative pair discrimination.

        Использует все негативы в батче одновременно (in-batch negatives),
        что значительно эффективнее чем pairwise ranking margin.
        Проверено в SimCLR, CLIP, sentence-transformers.

        Args:
            source_inv: (B, D) source exported_invariant, L2-normalized
            target_inv: (B, D) target exported_invariant, L2-normalized
            pair_relation_label: (B,) 1.0=positive, 0.0=negative
            pair_weight: (B,) per-sample weights
            temperature: softmax temperature (меньше = резче границы)
        Returns:
            scalar InfoNCE loss
        """
        B = source_inv.shape[0]
        if B < 2:
            return self._zero_loss(source_inv)

        # NaN/Inf protection during validation
        if torch.isnan(source_inv).any() or torch.isinf(source_inv).any():
            return self._zero_loss(source_inv)
        if torch.isnan(target_inv).any() or torch.isinf(target_inv).any():
            return self._zero_loss(source_inv)

        positive_mask = pair_relation_label > 0.5
        if not positive_mask.any():
            return self._zero_loss(source_inv)

        src = F.normalize(source_inv, dim=-1)  # (B, D)
        tgt = F.normalize(target_inv, dim=-1)  # (B, D)

        # Similarity matrix: (B, B)
        sim_matrix = src @ tgt.T / temperature

        # InfoNCE: for each positive pair (i,i), treat all other j≠i as negatives
        # Только для положительных пар считаем loss
        pos_indices = positive_mask.nonzero(as_tuple=True)[0]

        # Labels: diagonal = self-match for positives
        labels = torch.arange(B, device=self.device)
        loss_per_sample = F.cross_entropy(sim_matrix[pos_indices], labels[pos_indices], reduction='none')

        weights = pair_weight[pos_indices]
        return (loss_per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


    def _compute_focal_infonce_loss(
        self,
        source_inv: torch.Tensor,
        target_inv: torch.Tensor,
        pair_relation_label: torch.Tensor,
        pair_weight: torch.Tensor,
        temperature: float = 0.07,
        gamma: float = 0.5,
    ) -> torch.Tensor:
        """Focal-InfoNCE loss (Hou & Li, EMNLP Findings 2023).

        Downweights easy negatives via focal exponent scaling.
        When gamma < 1.0, the denominator compresses contribution of
        well-separated negatives, focusing learning on hardest pairs.

        Implementation: apply gamma to the similarity matrix before
        cross-entropy, which is equivalent to focal modulation of
        the softmax probabilities.

        Args:
            source_inv: (B, D) source exported_invariant
            target_inv: (B, D) target exported_invariant
            pair_relation_label: (B,) 1.0=positive, 0.0=negative
            pair_weight: (B,) per-sample weights
            temperature: softmax temperature
            gamma: focal parameter (1.0=standard, 0.5=moderate focus on hard)
        Returns:
            scalar InfoNCE loss (always >= 0)
        """
        B = source_inv.shape[0]
        if B < 2:
            return self._zero_loss(source_inv)

        # NaN/Inf protection during validation
        if torch.isnan(source_inv).any() or torch.isinf(source_inv).any():
            return self._zero_loss(source_inv)
        if torch.isnan(target_inv).any() or torch.isinf(target_inv).any():
            return self._zero_loss(source_inv)

        positive_mask = pair_relation_label > 0.5
        if not positive_mask.any():
            return self._zero_loss(source_inv)

        src = F.normalize(source_inv, dim=-1)
        tgt = F.normalize(target_inv, dim=-1)

        pos_indices = positive_mask.nonzero(as_tuple=True)[0]
        labels = torch.arange(B, device=self.device)

        if gamma >= 0.99:
            # Standard InfoNCE path (no focal modulation)
            sim_matrix = src @ tgt.T / temperature
            loss_per_sample = F.cross_entropy(
                sim_matrix[pos_indices], labels[pos_indices], reduction='none'
            )
        else:
            # Focal-InfoNCE: scale logits by gamma before CE
            # This compresses the softmax denominator toward uniform,
            # downweighting easy negatives that already have low probability.
            sim_matrix = src @ tgt.T / temperature
            # Focal scaling: lower gamma → more aggressive downweighting
            # Equivalent to: (exp(s/τ))^γ / Σ_j (exp(s_j/τ))^γ
            focal_matrix = sim_matrix * gamma
            loss_per_sample = F.cross_entropy(
                focal_matrix[pos_indices], labels[pos_indices], reduction='none'
            )

        weights = pair_weight[pos_indices]
        return (loss_per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


    def _compute_dcl_loss(
        self,
        source_inv: torch.Tensor,
        target_inv: torch.Tensor,
        pair_relation_label: torch.Tensor,
        pair_weight: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """Decoupled Contrastive Loss (DCL) — Yeh et al., NeurIPS 2022.

        Ключевое отличие от InfoNCE: positive убирается из знаменателя.
        L_DCL = -log( exp(s_pos/τ) / Σ_{j: j≠pos} exp(s_j/τ) )

        Это устраняет "positive-negative coupling" — InfoNCE подавляет
        градиент от трудных негативов когда positive слишком похож.
        Математически: gradient для negative = p_j*(1 + β_j) где β_j > 0 для DCL.

        Refs: Decoupled Contrastive Learning (Yeh et al., NeurIPS 2022)
        """
        B = source_inv.shape[0]
        if B < 2:
            return self._zero_loss(source_inv)

        positive_mask = pair_relation_label > 0.5
        if not positive_mask.any():
            return self._zero_loss(source_inv)

        src = F.normalize(source_inv, dim=-1)   # (B, D)
        tgt = F.normalize(target_inv, dim=-1)   # (B, D)

        sim_matrix = src @ tgt.T / temperature  # (B, B)

        pos_indices = positive_mask.nonzero(as_tuple=True)[0]
        diag_mask = torch.eye(B, dtype=torch.bool, device=self.device)

        # Vectorized: mask diagonal to -inf for all rows at once
        # Use float32 to avoid fp16 overflow with large negative values under AMP
        masked_sim = sim_matrix.float()
        masked_sim.masked_fill_(diag_mask, -torch.finfo(torch.float32).max)
        log_denom = torch.logsumexp(masked_sim, dim=1)  # (B,)
        s_pos = sim_matrix.diag()  # (B,)
        loss_per_sample = -(s_pos - log_denom)  # (B,)
        # Apply weights only to positive samples
        weights = pair_weight * positive_mask.float()
        if weights.sum() == 0:
            return self._zero_loss(source_inv)
        return (loss_per_sample * weights).sum() / weights.sum().clamp_min(1e-8)

    def _compute_uniformity_alignment_loss(
        self,
        source_inv: torch.Tensor,
        target_inv: torch.Tensor,
        pair_relation_label: torch.Tensor,
        t_uniform: float = 2.0,
        lambda_align: float = 1.0,
        lambda_uniform: float = 0.5,
    ) -> torch.Tensor:
        """Uniformity + Alignment loss (Wang & Isola, ICML 2020).

        L_align  = E[||f(x)-f(y)||²]  — выравниваем позитивные пары
        L_uniform = log E[exp(-t||f(x)-f(z)||²)] — отталкиваем все равномерно

        Эти два компонента разделены: alignment только для позитивов,
        uniformity для всех. Это более стабильно чем InfoNCE.

        Refs: Understanding Contrastive Representation Learning through
        Alignment and Uniformity on the Hypersphere (Wang & Isola, ICML 2020)
        """
        B = source_inv.shape[0]
        if B < 4:
            return self._zero_loss(source_inv)

        positive_mask = pair_relation_label > 0.5
        src = F.normalize(source_inv, dim=-1)  # (B, D)
        tgt = F.normalize(target_inv, dim=-1)  # (B, D)

        # Alignment: только для positives
        if positive_mask.any():
            src_pos = src[positive_mask]  # (Np, D)
            tgt_pos = tgt[positive_mask]  # (Np, D)
            loss_align = (src_pos - tgt_pos).pow(2).sum(dim=-1).mean()
        else:
            loss_align = self._zero_loss(source_inv)

        # Uniformity: по всем парам (src ∪ tgt)
        all_vecs = torch.cat([src, tgt], dim=0)  # (2B, D)
        diffs = all_vecs.unsqueeze(0) - all_vecs.unsqueeze(1)  # (2B, 2B, D)
        sq_dists = diffs.pow(2).sum(dim=-1)  # (2B, 2B)
        # Маскируем диагональ (нулевое расстояние до себя)
        mask = ~torch.eye(2 * B, dtype=torch.bool, device=self.device)
        sq_dists_off = sq_dists[mask].view(2 * B, 2 * B - 1)
        # Numerically stable log-mean-exp to avoid Inf overflow
        neg_sq = -t_uniform * sq_dists_off
        loss_uniform = torch.logsumexp(neg_sq, dim=-1).mean() - math.log(neg_sq.shape[-1])

        return lambda_align * loss_align + lambda_uniform * loss_uniform

    def _compute_supcon_loss(
        self,
        source_inv: torch.Tensor,
        target_inv: torch.Tensor,
        pair_relation_label: torch.Tensor,
        pair_weight: torch.Tensor,
        pair_group_id: torch.Tensor,
    ) -> torch.Tensor:
        """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

        Для каждого anchor i множество позитивов P(i) = все j из того же
        family (pair_group_id[i]==pair_group_id[j]) с label>0.5.
        L_SupCon_i = -1/|P(i)| * Σ_{p∈P(i)} log[ exp(sim(i,p)/T) / Σ_{a≠i} exp(sim(i,a)/T) ]

        Преимущества перед InfoNCE: использует ВСЕ позитивы из семейства,
        не только диагональный, что эффективнее при аугментации.
        """
        B = source_inv.shape[0]
        if B < 2:
            return self._zero_loss(source_inv)

        positive_mask = pair_relation_label > 0.5
        if not positive_mask.any():
            return self._zero_loss(source_inv)

        temp = self._effective_temperature()
        src = F.normalize(source_inv, dim=-1)   # (B, D)
        tgt = F.normalize(target_inv, dim=-1)   # (B, D)
        sim_matrix = src @ tgt.T / temp          # (B, B)

        # Маска позитивов: same group AND positive label
        group_match = (pair_group_id.unsqueeze(0) == pair_group_id.unsqueeze(1))  # (B, B)
        label_pos   = positive_mask.unsqueeze(1).expand(B, B)                     # (B, B)
        pos_mask    = group_match & label_pos                                       # (B, B)
        # Исключаем диагональ (anchor = себя)
        pos_mask.fill_diagonal_(False)

        # Знаменатель: все a ≠ i
        eye = torch.eye(B, dtype=torch.bool, device=self.device)
        denom_mask = ~eye  # (B, B)

        log_denom = torch.logsumexp(sim_matrix.float().masked_fill(eye, -torch.finfo(torch.float32).max), dim=-1)  # float32 for fp16 overflow safety

        # Vectorized: for each sample, average over positive similarities
        num_pos = pos_mask.sum(dim=1).clamp(min=1)  # (B,) — at least 1 to avoid div by 0
        # Mask non-positives to 0, sum per row, divide by count
        pos_sim_sum = (sim_matrix * pos_mask.float()).sum(dim=1)  # (B,)
        loss_per_sample = -((pos_sim_sum / num_pos) - log_denom)  # (B,)
        # Zero out samples with no positives
        has_pos = pos_mask.any(dim=1).float()
        loss_per_sample = loss_per_sample * has_pos
        weights = pair_weight * has_pos
        if weights.sum() == 0:
            return self._zero_loss(source_inv)
        return (loss_per_sample * weights).sum() / weights.sum().clamp_min(1e-8)

    def _compute_pair_ranking_loss(
        self,
        exported_invariant: torch.Tensor,
        pair_exported_target: torch.Tensor | None,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        if not self._has_pairs(batch) or pair_exported_target is None:
            return self._zero_loss(exported_invariant)

        pair_relation_label = batch.get("pair_relation_label")
        pair_weight = batch.get("pair_weight")
        pair_group_id = self._resolve_pair_group_id(batch)
        if pair_relation_label is None or pair_weight is None:
            return self._zero_loss(exported_invariant)

        pair_relation_label = pair_relation_label.to(
            self.device, dtype=exported_invariant.dtype
        )
        pair_weight = pair_weight.to(self.device, dtype=exported_invariant.dtype)


        # InfoNCE path (use_infonce=True by default via ranking_margin <= 0)
        if getattr(self, 'use_infonce', True):
            temp = self._effective_temperature()
            if getattr(self, 'use_sc_temperature', False):
                temp = self._cluster_scaled_temperature(exported_invariant, temp)
            use_focal = self._focal_gamma < 1.0
            # Hard negative mining: используем hardest negatives в batche
            use_hard_neg = getattr(self, 'use_hard_negatives', False)
            if use_hard_neg and exported_invariant.shape[0] >= 4:
                hard_neg_idx = self._mine_hard_negatives(
                    exported_invariant, pair_exported_target, pair_relation_label
                )
                hard_pair_target = pair_exported_target[hard_neg_idx]
                hard_labels = torch.zeros_like(pair_relation_label)
                hard_weights = pair_weight * 0.5
                aug_src = torch.cat([exported_invariant, exported_invariant[pair_relation_label > 0.5]], dim=0)
                aug_tgt = torch.cat([pair_exported_target, hard_pair_target[pair_relation_label > 0.5]], dim=0)
                aug_lbl = torch.cat([pair_relation_label, hard_labels[pair_relation_label > 0.5]], dim=0)
                aug_w   = torch.cat([pair_weight, hard_weights[pair_relation_label > 0.5]], dim=0)
                if use_focal:
                    infonce = self._compute_focal_infonce_loss(aug_src, aug_tgt, aug_lbl, aug_w, temperature=temp, gamma=self._focal_gamma)
                else:
                    infonce = self._compute_infonce_loss(aug_src, aug_tgt, aug_lbl, aug_w, temperature=temp)
            else:
                if use_focal:
                    infonce = self._compute_focal_infonce_loss(
                        exported_invariant, pair_exported_target,
                        pair_relation_label, pair_weight,
                        temperature=temp, gamma=self._focal_gamma,
                    )
                else:
                    infonce = self._compute_infonce_loss(
                        exported_invariant, pair_exported_target,
                        pair_relation_label, pair_weight,
                        temperature=temp,
                    )
            # AnglE loss поверх InfoNCE (если включён)
            total_loss = infonce
            if getattr(self, 'lambda_angle', 0.0) > 0:
                angle = self._compute_angle_loss(
                    exported_invariant,
                    pair_exported_target,
                    pair_relation_label,
                    pair_weight,
                )
                total_loss = total_loss + self.lambda_angle * angle
            # SupCon loss (Supervised Contrastive, если group_id доступен)
            if getattr(self, 'lambda_supcon', 0.0) > 0 and pair_group_id is not None:
                supcon = self._compute_supcon_loss(
                    exported_invariant,
                    pair_exported_target,
                    pair_relation_label,
                    pair_weight,
                    pair_group_id,
                )
                total_loss = total_loss + self.lambda_supcon * supcon
            # DCL loss (Decoupled Contrastive, Yeh et al. 2022)
            if getattr(self, 'lambda_dcl', 0.0) > 0:
                dcl = self._compute_dcl_loss(
                    exported_invariant,
                    pair_exported_target,
                    pair_relation_label,
                    pair_weight,
                    temperature=temp,
                )
                total_loss = total_loss + self.lambda_dcl * dcl
            # Uniformity + Alignment loss (Wang & Isola, ICML 2020)
            if getattr(self, 'lambda_uniformity', 0.0) > 0:
                uni_align = self._compute_uniformity_alignment_loss(
                    exported_invariant,
                    pair_exported_target,
                    pair_relation_label,
                )
                total_loss = total_loss + self.lambda_uniformity * uni_align
            return total_loss

        # Legacy ranking margin fallback
        if pair_group_id is None:
            return self._zero_loss(exported_invariant)

        pair_domain_id = batch["pair_domain_id"].to(self.device)
        domain_id = batch["domain_id"].to(self.device)

        positive_mask = pair_relation_label > 0.5
        if not positive_mask.any() or exported_invariant.shape[0] < 2:
            return self._zero_loss(exported_invariant)

        source_normalized = F.normalize(exported_invariant, dim=-1)
        target_normalized = F.normalize(pair_exported_target, dim=-1)
        similarities = source_normalized @ target_normalized.transpose(0, 1)
        anchor_indices = torch.arange(exported_invariant.shape[0], device=self.device)
        negative_indices = self._compute_negative_pair_indices(
            pair_group_id,
            domain_id,
            pair_domain_id,
        )
        valid_mask = positive_mask & (negative_indices >= 0)
        if not valid_mask.any():
            return self._zero_loss(exported_invariant)

        positive_scores = similarities[
            anchor_indices[valid_mask], anchor_indices[valid_mask]
        ]
        negative_scores = similarities[
            anchor_indices[valid_mask],
            negative_indices[valid_mask],
        ]
        margin_loss = F.relu(self.ranking_margin - positive_scores + negative_scores)
        weighted_loss = margin_loss * pair_weight[valid_mask]
        return weighted_loss.sum() / pair_weight[valid_mask].sum().clamp_min(1e-8)

    def compute_pair_ranking_loss(
        self,
        exported_invariant: torch.Tensor,
        pair_exported_target: torch.Tensor | None,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        return self._compute_pair_ranking_loss(
            exported_invariant,
            pair_exported_target,
            batch,
        )

    def _compute_matryoshka_loss(
        self,
        matryoshka_embs: Dict[int, torch.Tensor],
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """Matryoshka Representation Learning loss.
        
        Trains the encoder to produce meaningful embeddings at multiple dimensions.
        Each scale contributes equally to the total loss.
        
        Ref: Kusupati et al., NeurIPS 2022
        
        For HDIM: we compute contrastive loss at each Matryoshka scale,
        encouraging all dimensionalities to preserve semantic structure.
        """
        if not matryoshka_embs:
            return self._zero_loss(next(iter(batch.values())))
        
        # Need pair info for contrastive loss at each scale
        pair_relation_label = batch.get("pair_relation_label")
        if pair_relation_label is None:
            # No pairs - can't compute Matryoshka contrastive loss
            return self._zero_loss(next(iter(matryoshka_embs.values())))
        
        pair_weight = batch.get("pair_weight")
        if pair_weight is None:
            pair_weight = torch.ones_like(pair_relation_label)
        
        # For Matryoshka, we need both source and target embeddings at each scale
        # The batch should contain "matryoshka_source" and "matryoshka_target" dicts
        matryoshka_source = batch.get("matryoshka_source", matryoshka_embs)
        matryoshka_target = batch.get("matryoshka_target", matryoshka_embs)
        
        if not matryoshka_source or not matryoshka_target:
            return self._zero_loss(next(iter(matryoshka_embs.values())))
        
        total_loss = self._zero_loss(next(iter(matryoshka_source.values())))
        n_scales = 0
        
        # Compute contrastive loss at each scale
        for dim in matryoshka_source.keys():
            if dim not in matryoshka_target:
                continue
                
            src_emb = matryoshka_source[dim]
            tgt_emb = matryoshka_target[dim]
            
            # Use InfoNCE at each scale
            scale_loss = self._compute_infonce_loss(
                src_emb,
                tgt_emb,
                pair_relation_label,
                pair_weight,
                temperature=self._effective_temperature(),
            )
            total_loss = total_loss + scale_loss
            n_scales += 1
        
        # Average across scales
        if n_scales > 0:
            total_loss = total_loss / n_scales
        
        return total_loss
    
    def _compute_pair_loss_terms(
        self,
        aux_state: HDIMAuxState,
        batch: Dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_target_state = self._extract_pair_target_state(batch)
        if pair_target_state is None:
            iso_target = self._extract_iso_targets(batch, aux_state)
            pair_exported_target = None
        else:
            iso_target, pair_exported_target = pair_target_state
        loss_pair = self._compute_pair_ranking_loss(
            aux_state.exported_invariant,
            pair_exported_target,
            batch,
        )
        return iso_target, loss_pair
    def _compute_router_diagnostics(
        self, aux_state: HDIMAuxState
    ) -> Dict[str, torch.Tensor]:
        return {
            "routing_entropy": aux_state.routing_entropy,
            "expert_usage": aux_state.expert_usage,
            "topk_idx": aux_state.topk_idx,
            "topk_gate_weights": aux_state.topk_gate_weights,
            "train_scores_snapshot": aux_state.train_scores_snapshot,
        }

    def _forward_text_batch(
        self,
        texts: Sequence[str],
        domain_id: torch.Tensor,
        regime: TrainingRegime,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, HDIMAuxState]:
        return self.model.forward_texts(
            texts,
            domain_id,
            return_state=True,
            update_memory=regime.update_memory,
            memory_mode=regime.memory_mode,
        )

    def _forward_batch(
        self,
        encoding: torch.Tensor,
        domain_id: torch.Tensor,
        regime: TrainingRegime,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, HDIMAuxState]:
        return self.model(
            encoding,
            domain_id,
            return_state=True,
            update_memory=regime.update_memory,
            memory_mode=regime.memory_mode,
        )

    def _compute_batch_losses(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        domain_id = batch["domain_id"].to(self.device)
        regime = self._resolve_training_regime(batch)
        use_text_path = self._uses_text_model() and self._has_texts(batch)

        if use_text_path:
            texts = self._extract_texts(batch, "text")
            if regime.mode == "paired":
                pair_texts = self._extract_texts(batch, "pair_text")
                pair_domain_id = batch["pair_domain_id"].to(self.device)
                # Encode source once (was double-encoded by transfer_text_pairs + _encode_texts)
                _src_enc, _src_scales = self._encode_texts(texts, collect_matryoshka=True)
                output, routing_weights, invariant, aux_state = (
                    self.model.core_model.transfer_pairs(
                        _src_enc,
                        domain_id,
                        pair_domain_id,
                        update_memory=regime.update_memory,
                        memory_mode=regime.memory_mode,
                    )
                )
                _prl = batch.get("pair_relation_label")
                _tgt_enc, _tgt_scales = self._encode_texts(pair_texts, collect_matryoshka=True)
                # Apply embedding augmentation to pair (non-anchor) side
                if self._augmenter is not None:
                    _tgt_enc = self._augmenter(_tgt_enc, pairs_only=False)
                if _src_scales is not None and _tgt_scales is not None:
                    batch["matryoshka_source"] = _src_scales
                    batch["matryoshka_target"] = _tgt_scales
                    # Use max-dim dict for matryoshka_embeddings
                    batch["matryoshka_embeddings"] = _src_scales
                if _prl is not None:
                    _pos_mask = (_prl.to(self.device) > 0.5).unsqueeze(-1).expand_as(_src_enc)
                    recon_target = torch.where(_pos_mask, _tgt_enc, _src_enc)
                else:
                    recon_target = _tgt_enc
            else:
                output, routing_weights, invariant, aux_state = self._forward_text_batch(
                    texts, domain_id, regime
                )
                recon_target = self._encode_texts(texts)
        else:
            encoding = batch["encoding"].to(self.device)
            if regime.mode == "paired":
                pair_encoding = batch["pair_encoding"].to(self.device)
                # Apply embedding augmentation to pair (non-anchor) side
                if self._augmenter is not None:
                    pair_encoding = self._augmenter(pair_encoding, pairs_only=False)
                pair_domain_id = batch["pair_domain_id"].to(self.device)
                output, routing_weights, invariant, aux_state = self.model.transfer_pairs(
                    encoding,
                    domain_id,
                    pair_domain_id,
                    update_memory=regime.update_memory,
                    memory_mode=regime.memory_mode,
                )
                # For negative pairs: recon_target = source encoding (no transfer)
                # For positive pairs: recon_target = pair_encoding (cross-domain transfer)
                pair_relation_label = batch.get("pair_relation_label")
                if pair_relation_label is not None:
                    pos_mask = (pair_relation_label > 0.5).unsqueeze(-1).expand_as(encoding)
                    recon_target = torch.where(pos_mask, pair_encoding, encoding)
                else:
                    recon_target = pair_encoding
            else:
                output, routing_weights, invariant, aux_state = self._forward_batch(
                    encoding, domain_id, regime
                )
                recon_target = encoding
        training_invariant = self._training_invariant(aux_state)
        iso_target, loss_pair = self._compute_pair_loss_terms(aux_state, batch)
        loss_recon = self._compute_reconstruction_loss(output, recon_target)
        loss_iso = self._compute_pair_iso_loss(training_invariant, iso_target, batch)
        loss_routing = aux_state.router_loss
        pair_relation_label = batch.get("pair_relation_label")
        pair_weight = batch.get("pair_weight")
        loss_memory = aux_state.memory_loss
        # STS regularization: поддерживаем косинусное сходство позитивных пар
        # iso_target содержит training_invariant пары — используем как proxy
        loss_sts = self._zero_loss(training_invariant)
        if self.lambda_sts > 0 and self._has_pairs(batch):
            _prl = batch.get("pair_relation_label")
            if _prl is not None:
                _pos_mask = (_prl.to(self.device) > 0.5)
                if _pos_mask.any():
                    # training_invariant и iso_target — одинаковая размерность
                    src_norm = F.normalize(training_invariant[_pos_mask], dim=-1)
                    tgt_norm = F.normalize(iso_target[_pos_mask], dim=-1)
                    cos_sim = (src_norm * tgt_norm).sum(dim=-1)
                    loss_sts = (1.0 - cos_sim).mean()

        # Router z-loss (ST-MoE stability)
        loss_z = aux_state.z_loss

        # Anti-collapse diversity loss: encourage variance in exported_invariant
        # Prevents trivial solution where all invariants are identical (STS→1.0, margin→0)
        # Only applied when we have paired data (need diverse samples in batch)
        loss_diversity = self._zero_loss(training_invariant)
        if self._has_pairs(batch) and pair_relation_label is not None:
            loss_diversity = self._compute_diversity_loss(aux_state.exported_invariant)

        # Matryoshka multi-scale loss (if encoder returns dict of scales)
        loss_matryoshka = self._zero_loss(training_invariant)
        # Check if batch has matryoshka embeddings
        matryoshka_embs = batch.get("matryoshka_embeddings")
        if matryoshka_embs is not None and isinstance(matryoshka_embs, dict):
            loss_matryoshka = self._compute_matryoshka_loss(matryoshka_embs, batch)
        
        # Phase 26: Expert orthogonalization loss
        loss_expert_ortho = self._zero_loss(loss_recon)
        if self.lambda_expert_ortho > 0 and hasattr(self.model, 'compute_expert_ortho_loss'):
            loss_expert_ortho = self.model.compute_expert_ortho_loss()

        loss_total = (
            loss_recon
            + self.lambda_iso * loss_iso
            + self.lambda_pair * loss_pair
            + self.lambda_routing * loss_routing
            + self.lambda_memory * loss_memory
            + self.lambda_sts * loss_sts
            + self.lambda_z * loss_z
            + loss_diversity
            + self.lambda_matryoshka * loss_matryoshka
            + self.lambda_expert_ortho * loss_expert_ortho
        )
        batch_losses = {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_iso": loss_iso,
            "loss_pair": loss_pair,
            "loss_routing": loss_routing,
            "loss_memory": loss_memory,
            "loss_diversity": loss_diversity,
            "loss_matryoshka": loss_matryoshka,
            "loss_expert_ortho": loss_expert_ortho,
            "routing_weights": routing_weights,
            "invariant": invariant,
            "raw_invariant": aux_state.raw_invariant,
            "memory_augmented_invariant": aux_state.memory_augmented_invariant,
            "exported_invariant": aux_state.exported_invariant,
            "training_invariant": training_invariant,
            "training_mode": regime.mode,
        }
        if pair_relation_label is not None:
            batch_losses["pair_relation_label"] = pair_relation_label.to(
                self.device, dtype=training_invariant.dtype
            )
        if pair_weight is not None:
            batch_losses["pair_weight"] = pair_weight.to(
                self.device, dtype=training_invariant.dtype
            )
        if self._has_pairs(batch):
            batch_losses["iso_target"] = iso_target
        batch_losses.update(self._compute_router_diagnostics(aux_state))
        return batch_losses
    def evaluate_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            return self._compute_batch_losses(batch)

    def train_step(self, batch: Dict[str, Any], scaler=None) -> torch.Tensor:
        """Execute one training step with optional AMP scaler support."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = self._compute_batch_losses(batch)
        loss_total = losses["loss_total"]
        if scaler is not None:
            # NaN/Inf loss: skip step entirely and force scale reduction.
            # Without this, backward() produces NaN grads → scaler.step()
            # considers the step "successful" (no exception) → scaler.update()
            # keeps the high scale → next normal batch overflows → permanent
            # loss explosion (7 → 2414 → 18M over 3 epochs).
            if torch.isnan(loss_total) or torch.isinf(loss_total):
                # Force scale reduction for stability
                new_scale = max(scaler.get_scale() * 0.5, 1.0)
                scaler.update(new_scale)
                self._last_grad_norm = float('inf')
                self._step += 1
                return loss_total.detach()

            scaler.scale(loss_total).backward()
            scaler.unscale_(self.optimizer)
            # Zero NaN/Inf gradients from uninitialized memory/MoE state
            has_nan_grad = False
            for p in self.model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    p.grad.zero_()
                    has_nan_grad = True
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self._last_grad_norm = grad_norm.item() if grad_norm.numel() == 1 else grad_norm.max().item()
            if has_nan_grad:
                # NaN grads found — skip step and reduce scale to prevent
                # permanent scaler corruption from non-finite gradients.
                scaler.update(max(scaler.get_scale() * 0.5, 1.0))
            else:
                scaler.step(self.optimizer)
                scaler.update()
        else:
            loss_total.backward()
            for p in self.model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    p.grad.zero_()
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self._last_grad_norm = grad_norm.item() if grad_norm.numel() == 1 else grad_norm.max().item()
            self.optimizer.step()
        self._step += 1
        self._last_loss_components = {
            k: v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v
            for k, v in losses.items()
            if not isinstance(v, torch.Tensor) or v.dim() == 0
        }
        self._last_loss_components["grad_norm"] = self._last_grad_norm
        return loss_total.detach()

    def compute_iso_loss(
        self, u_inv_source: torch.Tensor, u_inv_target: torch.Tensor
    ) -> torch.Tensor:
        return nn.functional.mse_loss(u_inv_source, u_inv_target)

    def compute_routing_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        mean_routing = routing_weights.mean(dim=0)
        eps = 1e-8
        entropy = -(mean_routing * (mean_routing + eps).log()).sum()
        return -entropy
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        totals: Dict[str, float] = {
            "loss_recon": 0.0,
            "loss_iso": 0.0,
            "loss_pair": 0.0,
            "loss_routing": 0.0,
            "loss_memory": 0.0,
            "loss_diversity": 0.0,
            "loss_matryoshka": 0.0,
            "loss_total": 0.0,
        }
        n_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                losses = self._compute_batch_losses(batch)
            for key in totals:
                val = losses[key].item()
                # Skip NaN/Inf values to prevent corruption of validation metrics
                if math.isfinite(val):
                    totals[key] += val
            n_batches += 1
        if n_batches > 0:
            for key in totals:
                totals[key] /= n_batches
        return totals
    def save_checkpoint(self, path: str, scaler=None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "step": self._step,
            "current_epoch": self._current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # Temperature scheduling config
            "temp_schedule": self._temp_schedule,
            "tau_max": self._tau_max,
            "tau_min": self._tau_min,
            "temp_schedule_T_0": self._temp_schedule_T_0,
            "focal_gamma": self._focal_gamma,
        }
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        torch.save(checkpoint, path)

    def _load_checkpoint_safe(self, path: str) -> dict:
        """Safely load checkpoint with weights_only=True (OWASP A08 fix).

        Prevents arbitrary code execution via pickle deserialization.
        Only tensor data is loaded, no Python objects.
        """
        import io

        with open(path, "rb") as f:
            data = f.read()

        return torch.load(
            io.BytesIO(data),
            map_location=self.device,
            weights_only=True,
        )

    def load_checkpoint(self, path: str, scaler=None) -> None:
        checkpoint = self._load_checkpoint_safe(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._step = checkpoint.get("step", 0)
        self._current_epoch = checkpoint.get("current_epoch", 0)
        # Restore temperature scheduling config
        self._temp_schedule = checkpoint.get("temp_schedule", "none")
        self._tau_max = checkpoint.get("tau_max", 0.1)
        self._tau_min = checkpoint.get("tau_min", 0.01)
        self._temp_schedule_T_0 = checkpoint.get("temp_schedule_T_0", 20)
        self._focal_gamma = checkpoint.get("focal_gamma", 1.0)
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
