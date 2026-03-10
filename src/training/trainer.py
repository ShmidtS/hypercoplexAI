"""HDIMTrainer — training loop, loss computation, and checkpoint management."""

from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Any, Dict, Literal, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from src.models.hdim_model import HDIMAuxState, HDIMModel
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
        infonce_temperature: float = 0.07,
        lambda_sts: float = 0.0,
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
        self._step: int = 0
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

    def _encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        return self.model.encode_texts(texts, device=self.device)

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

    def _exported_invariant(self, aux_state: HDIMAuxState) -> torch.Tensor:
        return aux_state.exported_invariant
    def _training_invariant(self, aux_state: HDIMAuxState) -> torch.Tensor:
        return aux_state.training_invariant
    def _iso_reference_invariant(self, aux_state: HDIMAuxState) -> torch.Tensor:
        return self._training_invariant(aux_state)

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
            return self._compute_infonce_loss(
                exported_invariant,
                pair_exported_target,
                pair_relation_label,
                pair_weight,
                temperature=getattr(self, 'infonce_temperature', 0.07),
            )

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
                output, routing_weights, invariant, aux_state = (
                    self.model.transfer_text_pairs(
                        texts,
                        domain_id,
                        pair_domain_id,
                        update_memory=regime.update_memory,
                        memory_mode=regime.memory_mode,
                    )
                )
                # For negative pairs: recon_target = source encoding
                _prl = batch.get("pair_relation_label")
                _src_enc = self._encode_texts(texts)
                _tgt_enc = self._encode_texts(pair_texts)
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

        loss_total = (
            loss_recon
            + self.lambda_iso * loss_iso
            + self.lambda_pair * loss_pair
            + self.lambda_routing * loss_routing
            + self.lambda_memory * loss_memory
            + self.lambda_sts * loss_sts
        )
        batch_losses = {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_iso": loss_iso,
            "loss_pair": loss_pair,
            "loss_routing": loss_routing,
            "loss_memory": loss_memory,
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

    def train_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        self.model.train()
        self.optimizer.zero_grad()
        losses = self._compute_batch_losses(batch)
        loss_total = losses["loss_total"]
        loss_total.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self._step += 1
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
            "loss_total": 0.0,
        }
        n_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                losses = self._compute_batch_losses(batch)
                for key in totals:
                    totals[key] += losses[key].item()
                n_batches += 1
        if n_batches > 0:
            for key in totals:
                totals[key] /= n_batches
        return totals
    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "step": self._step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._step = checkpoint.get("step", 0)
