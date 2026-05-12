"""Deprecated HDIMTrainer compatibility wrapper over invariant-only training."""

from __future__ import annotations

import logging
import os
import pickle
import warnings
from typing import Any

import torch

from src.training.invariant_losses import (
    compute_infonce_loss,
    compute_iso_loss,
    compute_pair_iso_loss,
)
from src.training.invariant_trainer import InvariantTrainer

logger = logging.getLogger(__name__)


class HDIMTrainer:
    """Deprecated trainer shim; use InvariantTrainer for new code."""

    def __init__(
        self,
        model,
        optimizer,
        device: str | torch.device = "cpu",
        *args,
        config: Any | None = None,
        negative_margin: float = 1.0,
        infonce_temperature: float = 0.10,
        **kwargs,
    ) -> None:
        if args and config is None:
            config = args[0]
        if not isinstance(device, (str, torch.device)):
            config = device
            device = "cpu"
        warnings.warn(
            "HDIMTrainer is deprecated; use InvariantTrainer",
            DeprecationWarning,
            stacklevel=2,
        )
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.config = config
        self.negative_margin = negative_margin
        self.infonce_temperature = infonce_temperature
        self.lambda_routing = kwargs.pop("lambda_routing", 0.01)
        self.lambda_memory = kwargs.pop("lambda_memory", 0.05)
        self.lambda_z = kwargs.pop("lambda_z", 0.01)
        self.lambda_expert_ortho = kwargs.pop("lambda_expert_ortho", 0.01)
        self.lambda_online = kwargs.pop("lambda_online", 0.01)
        self.lambda_matryoshka = kwargs.pop("lambda_matryoshka", 0.15)
        self._step = 0
        self._current_epoch = 0
        self._last_loss_components: dict[str, float] = {}
        self._grad_norm_ema = 0.0
        self._core = InvariantTrainer(model, optimizer, str(self.device))
        self._warn_skipped_extension_losses()

    def _warn_skipped_extension_losses(self) -> None:
        skipped = {
            "routing": self.lambda_routing,
            "memory": self.lambda_memory,
            "z_loss": self.lambda_z,
            "expert_ortho": self.lambda_expert_ortho,
            "online": self.lambda_online,
            "matryoshka": self.lambda_matryoshka,
        }
        enabled = [name for name, value in skipped.items() if value]
        if enabled:
            logger.warning(
                "HDIMTrainer compatibility shim skips extension losses: %s",
                ", ".join(enabled),
            )

    def training_step(self, batch: dict) -> dict:
        losses = self._core.training_step(self._ensure_pair_batch(batch))
        self._step += 1
        self._last_loss_components = self._scalar_losses(losses)
        return losses

    def train_step(self, batch: dict, scaler=None) -> torch.Tensor:
        if scaler is not None:
            logger.warning("HDIMTrainer compatibility shim ignores AMP scaler")
        losses = self.training_step(batch)
        return losses["loss_total"]

    def train_epoch(self, dataloader) -> dict:
        totals: dict[str, float] = {}
        n_batches = 0
        for batch in dataloader:
            losses = self.training_step(batch)
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    totals[key] = totals.get(key, 0.0) + float(value.item())
            n_batches += 1
        metrics = {key: value / n_batches for key, value in totals.items()} if n_batches else {}
        self._last_loss_components = dict(metrics)
        return metrics

    def train(self, dataloader, epochs: int = 1) -> dict:
        metrics: dict = {}
        for _ in range(epochs):
            metrics = self.train_epoch(dataloader)
        return metrics

    def evaluate_batch(self, batch: dict) -> dict:
        return self._core.evaluate_batch(self._ensure_pair_batch(batch))

    def validate(self, dataloader) -> dict:
        totals: dict[str, float] = {}
        n_batches = 0
        for batch in dataloader:
            losses = self.evaluate_batch(batch)
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    totals[key] = totals.get(key, 0.0) + float(value.item())
            n_batches += 1
        metrics = {key: value / n_batches for key, value in totals.items()} if n_batches else {}
        metrics.setdefault("grad_norm_ema", self._grad_norm_ema)
        metrics.setdefault("effective_temperature", 0.0)
        metrics.setdefault("loss_gap", 0.0)
        return metrics

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch

    def compute_iso_loss(
        self, u_inv_source: torch.Tensor, u_inv_target: torch.Tensor
    ) -> torch.Tensor:
        return compute_iso_loss(u_inv_source, u_inv_target)

    def compute_routing_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        logger.warning("HDIMTrainer compatibility shim skips routing loss")
        return routing_weights.new_tensor(0.0)

    def _compute_pair_iso_loss(
        self,
        training_invariant: torch.Tensor,
        iso_target: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        return compute_pair_iso_loss(
            training_invariant,
            iso_target,
            batch,
            negative_margin=self.negative_margin,
            device=self.device,
            has_pairs=self._has_pairs(batch),
        )

    def compute_pair_ranking_loss(
        self,
        exported_invariant: torch.Tensor,
        pair_exported_target: torch.Tensor | None,
        batch: dict,
    ) -> torch.Tensor:
        if pair_exported_target is None:
            return exported_invariant.new_tensor(0.0)
        pair_relation_label = batch.get("pair_relation_label")
        pair_weight = batch.get("pair_weight")
        if pair_relation_label is None or pair_weight is None:
            return compute_iso_loss(exported_invariant, pair_exported_target)
        return compute_infonce_loss(
            exported_invariant,
            pair_exported_target,
            pair_relation_label,
            pair_weight,
            temperature=self.infonce_temperature,
            pair_group_id=batch.get("pair_group_id"),
            device=self.device,
        )

    def save_checkpoint(self, path: str, scaler=None, scheduler=None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "step": self._step,
            "current_epoch": self._current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)

    def _load_checkpoint_safe(self, path: str) -> dict:
        try:
            return torch.load(path, map_location=self.device, weights_only=True)
        except (RuntimeError, pickle.UnpicklingError) as exc:
            raise RuntimeError(
                f"Failed to load checkpoint with weights_only=True: {exc}. "
                f"Checkpoint at '{path}' contains non-tensor objects and cannot be loaded safely."
            ) from exc

    @staticmethod
    def _migrate_checkpoint_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("pipeline.moe.kernel."):
                new_key = "pipeline.moe." + key[len("pipeline.moe.kernel."):]
            new_state_dict[new_key] = value
        return new_state_dict

    def load_checkpoint(self, path: str, scaler=None, scheduler=None) -> None:
        checkpoint = self._load_checkpoint_safe(path)
        state_dict = checkpoint.get("model_state_dict", {})
        self.model.load_state_dict(self._migrate_checkpoint_state_dict(state_dict), strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._step = checkpoint.get("step", 0)
        self._current_epoch = checkpoint.get("current_epoch", 0)
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def _has_pairs(self, batch: dict) -> bool:
        return ("pair_encoding" in batch and "pair_domain_id" in batch) or (
            "pair_text" in batch and "pair_domain_id" in batch
        )

    def _ensure_pair_batch(self, batch: dict) -> dict:
        if self._has_pairs(batch):
            return batch
        paired = dict(batch)
        if "encoding" in batch:
            paired["pair_encoding"] = batch["encoding"]
        elif "text" in batch:
            paired["pair_text"] = batch["text"]
        paired["pair_domain_id"] = batch["domain_id"]
        paired["pair_relation_label"] = torch.ones_like(
            batch["domain_id"], dtype=torch.float32, device=batch["domain_id"].device
        )
        paired["pair_weight"] = torch.ones_like(paired["pair_relation_label"])
        return paired

    def _scalar_losses(self, losses: dict) -> dict[str, float]:
        return {
            key: float(value.item())
            for key, value in losses.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0
        }
