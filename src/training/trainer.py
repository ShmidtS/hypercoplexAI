"""HDIMTrainer — training loop, loss computation, and checkpoint management."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Literal, Tuple

import torch.nn.functional as F

import torch
import torch.nn as nn
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
        negative_margin: float = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.lambda_iso = lambda_iso
        self.lambda_routing = lambda_routing
        self.negative_margin = negative_margin
        self._step: int = 0

    def _resolve_training_regime(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> TrainingRegime:
        is_pair_batch = self._has_pairs(batch)
        return TrainingRegime(
            mode="paired" if is_pair_batch else "reconstruction",
            update_memory=self.model.training,
            memory_mode="update" if self.model.training else "retrieve",
        )

    def _extract_iso_targets(
        self,
        batch: Dict[str, torch.Tensor],
        aux_state: HDIMAuxState,
    ) -> torch.Tensor:
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

    def _has_pairs(self, batch: Dict[str, torch.Tensor]) -> bool:
        return "pair_encoding" in batch and "pair_domain_id" in batch

    def _compute_pair_iso_targets(
        self,
        pair_encoding: torch.Tensor,
        pair_domain_id: torch.Tensor,
    ) -> torch.Tensor:
        _, _, _, aux_state = self.model(
            pair_encoding,
            pair_domain_id,
            return_state=True,
            update_memory=False,
            memory_mode="retrieve",
        )
        return self._training_invariant(aux_state).detach()

    def _compute_reconstruction_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return nn.functional.mse_loss(output, target)

    def _compute_pair_iso_loss(
        self,
        training_invariant: torch.Tensor,
        iso_target: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not self._has_pairs(batch):
            return self.compute_iso_loss(training_invariant, iso_target)

        pair_relation_label = batch.get("pair_relation_label")
        pair_weight = batch.get("pair_weight")
        if pair_relation_label is None or pair_weight is None:
            return self.compute_iso_loss(training_invariant, iso_target)

        pair_relation_label = pair_relation_label.to(self.device, dtype=training_invariant.dtype)
        pair_weight = pair_weight.to(self.device, dtype=training_invariant.dtype)
        per_sample_mse = F.mse_loss(training_invariant, iso_target, reduction="none").mean(dim=-1)
        positive_mask = pair_relation_label > 0.5
        negative_mask = ~positive_mask
        losses = []
        if positive_mask.any():
            positive_loss = (per_sample_mse[positive_mask] * pair_weight[positive_mask]).sum() / pair_weight[positive_mask].sum()
            losses.append(positive_loss)
        if negative_mask.any():
            negative_penalty = F.relu(self.negative_margin - per_sample_mse[negative_mask])
            weighted_negative = (negative_penalty * pair_weight[negative_mask]).sum() / pair_weight[negative_mask].sum()
            losses.append(weighted_negative)
        if not losses:
            return per_sample_mse.mean()
        return torch.stack(losses).mean()

    def _compute_router_diagnostics(
        self,
        aux_state: HDIMAuxState,
    ) -> Dict[str, torch.Tensor]:
        return {
            "routing_entropy": aux_state.routing_entropy,
            "expert_usage": aux_state.expert_usage,
            "topk_idx": aux_state.topk_idx,
            "topk_gate_weights": aux_state.topk_gate_weights,
            "train_scores_snapshot": aux_state.train_scores_snapshot,
        }

    def _compute_batch_losses(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        encoding = batch["encoding"].to(self.device)
        domain_id = batch["domain_id"].to(self.device)
        regime = self._resolve_training_regime(batch)

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
            recon_target = pair_encoding
        else:
            output, routing_weights, invariant, aux_state = self._forward_batch(
                encoding,
                domain_id,
                regime=regime,
            )
            recon_target = encoding

        training_invariant = self._training_invariant(aux_state)
        iso_target = self._extract_iso_targets(batch, aux_state)
        loss_recon = self._compute_reconstruction_loss(output, recon_target)
        loss_iso = self._compute_pair_iso_loss(training_invariant, iso_target, batch)
        loss_routing = aux_state.router_loss
        pair_relation_label = batch.get("pair_relation_label")
        pair_weight = batch.get("pair_weight")
        loss_memory = aux_state.memory_loss
        loss_total = (
            loss_recon
            + self.lambda_iso * loss_iso
            + self.lambda_routing * loss_routing
            + loss_memory
        )

        batch_losses = {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_iso": loss_iso,
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
            batch_losses["pair_relation_label"] = pair_relation_label.to(self.device, dtype=training_invariant.dtype)
        if pair_weight is not None:
            batch_losses["pair_weight"] = pair_weight.to(self.device, dtype=training_invariant.dtype)
        if self._has_pairs(batch):
            batch_losses["iso_target"] = iso_target
        batch_losses.update(self._compute_router_diagnostics(aux_state))
        return batch_losses

    def evaluate_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            return self._compute_batch_losses(batch)

    def _forward_batch(
        self,
        encoding: torch.Tensor,
        domain_id: torch.Tensor,
        regime: TrainingRegime,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, HDIMAuxState]:
        output, routing_weights, invariant, aux_state = self.model(
            encoding,
            domain_id,
            return_state=True,
            update_memory=regime.update_memory,
            memory_mode=regime.memory_mode,
        )
        return output, routing_weights, invariant, aux_state

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
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
        self,
        u_inv_source: torch.Tensor,
        u_inv_target: torch.Tensor,
    ) -> torch.Tensor:
        return nn.functional.mse_loss(u_inv_source, u_inv_target)

    def compute_routing_loss(
        self,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        mean_routing = routing_weights.mean(dim=0)
        eps = 1e-8
        entropy = -(mean_routing * (mean_routing + eps).log()).sum()
        return -entropy

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        totals: Dict[str, float] = {
            "loss_recon": 0.0,
            "loss_iso": 0.0,
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
