"""Minimal invariant-only trainer for HDIM core learning."""

from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

from src.core.engine import HDIMCoreEngine
from src.training.invariant_losses import compute_pair_iso_loss

if TYPE_CHECKING:
    from src.models.hdim_model import HDIMModel


class InvariantTrainer:
    """Train HDIM invariants from paired source/target domain batches only."""

    def __init__(
        self,
        model: HDIMModel | HDIMCoreEngine,
        optimizer,
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.negative_margin = float(kwargs.pop("negative_margin", 1.0))
        self.infonce_temperature = float(kwargs.pop("infonce_temperature", 0.10))
        self._step = 0
        self._current_epoch = 0
        self._grad_norm_ema = 0.0
        self._log_temp = None

    def training_step(self, batch: dict) -> dict:
        """Run one paired invariant optimization step and return loss metrics."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.compute_losses(self._ensure_pair_batch(batch))
        losses["loss"].backward()
        self.optimizer.step()
        self._step += 1
        return {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in losses.items()}

    def train_step(self, batch: dict, scaler=None) -> torch.Tensor:
        losses = self.training_step(batch)
        return losses["loss_total"]

    def train_epoch(self, dataloader) -> dict:
        """Train over a dataloader and return averaged invariant metrics."""
        totals: dict[str, float] = {}
        n_batches = 0
        for batch in dataloader:
            losses = self.training_step(batch)
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    totals[key] = totals.get(key, 0.0) + float(value.item())
            n_batches += 1
        if n_batches == 0:
            return {}
        return {key: value / n_batches for key, value in totals.items()}

    def train(self, dataloader, epochs: int = 1) -> dict:
        """Train for a fixed number of epochs and return final epoch metrics."""
        metrics: dict = {}
        for _ in range(epochs):
            metrics = self.train_epoch(dataloader)
        return metrics

    def evaluate_batch(self, batch: dict) -> dict:
        """Compute invariant losses without optimizer updates."""
        self.model.eval()
        with torch.no_grad():
            losses = self.compute_losses(self._ensure_pair_batch(batch))
        return {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in losses.items()}

    def validate(self, dataloader) -> dict:
        """Average invariant losses over a dataloader without optimizer updates."""
        totals: dict[str, float] = {}
        n_batches = 0
        for batch in dataloader:
            losses = self.evaluate_batch(batch)
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    totals[key] = totals.get(key, 0.0) + float(value.item())
            n_batches += 1
        if n_batches == 0:
            return {}
        return {key: value / n_batches for key, value in totals.items()}

    def compute_losses(self, batch: dict) -> dict:
        source_inv, target_inv, extras = self._paired_invariants(batch)
        loss = compute_pair_iso_loss(
            source_inv,
            target_inv.detach(),
            batch,
            negative_margin=self.negative_margin,
            device=self.device,
            has_pairs=bool(extras.get("has_pairs", True)),
        )
        zero = loss.new_tensor(0.0)
        losses = {
            "loss": loss,
            "loss_total": loss,
            "loss_pair_iso": loss,
            "loss_iso": loss,
            "loss_recon": zero,
            "training_mode": "paired",
            "training_invariant": source_inv,
            "exported_invariant": extras.get("exported_invariant", source_inv),
            "routing_weights": extras.get("routing_weights", source_inv.new_zeros(source_inv.shape[0], 0)),
            "routing_entropy": zero,
            "expert_usage": source_inv.new_zeros(0),
            "topk_idx": torch.empty(source_inv.shape[0], 0, device=source_inv.device, dtype=torch.long),
            "topk_gate_weights": source_inv.new_zeros(source_inv.shape[0], 0),
        }
        return losses

    def _paired_invariants(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(self.model, HDIMCoreEngine):
            return self._core_engine_invariants(batch)
        return self._hdim_model_invariants(batch)

    def _hdim_model_invariants(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        source_domain_id = batch["domain_id"].to(self.device)
        has_pairs = "pair_domain_id" in batch
        target_domain_id = batch["pair_domain_id"].to(self.device) if has_pairs else source_domain_id

        if self._has_texts(batch):
            source_result, target_result, has_pairs = self._extract_text_pair_invariants(
                batch,
                source_domain_id,
                target_domain_id,
                has_pairs,
            )
        else:
            source_result, target_result = self._extract_tensor_pair_invariants(
                batch,
                source_domain_id,
                target_domain_id,
                has_pairs,
            )

        source_state = source_result.aux_state
        target_state = target_result.aux_state
        return (
            source_state.training_invariant,
            target_state.training_invariant,
            self._build_extras(source_result, has_pairs),
        )

    def _extract_text_pair_invariants(
        self,
        batch: dict,
        source_domain_id: torch.Tensor,
        target_domain_id: torch.Tensor,
        has_pairs: bool,
    ):
        """Build source/target invariant results from text batches.

        Args:
            batch: Training batch with text and optional pair_text fields.
        """
        model = cast("Any", self.model)
        source_texts = [str(text) for text in batch["text"]]
        if not self._has_pair_texts(batch):
            source_result = model.forward_texts(
                source_texts,
                source_domain_id,
                return_state=True,
                update_memory=False,
                memory_mode="retrieve",
            )
            return source_result, source_result, False

        target_texts = [str(text) for text in batch["pair_text"]]
        source_encoding = model.encode_texts(source_texts, device=self.device)
        source_result = model.transfer_pairs(
            source_encoding,
            source_domain_id,
            target_domain_id,
            update_memory=False,
            memory_mode="retrieve",
        )
        target_result = model.forward_texts(
            target_texts,
            target_domain_id,
            return_state=True,
            update_memory=False,
            memory_mode="retrieve",
        )
        return source_result, target_result, has_pairs

    def _extract_tensor_pair_invariants(
        self,
        batch: dict,
        source_domain_id: torch.Tensor,
        target_domain_id: torch.Tensor,
        has_pairs: bool,
    ):
        """Build source/target invariant results from tensor encodings.

        Args:
            batch: Training batch with encoding and optional pair_encoding fields.
        """
        model = cast("HDIMModel", self.model)
        source_encoding = batch["encoding"].to(self.device)
        if not has_pairs:
            source_result = model(
                source_encoding,
                source_domain_id,
                return_state=True,
                update_memory=False,
                memory_mode="retrieve",
            )
            return source_result, source_result

        pair_encoding = batch.get("pair_encoding", source_encoding).to(self.device)
        source_result = model.transfer_pairs(
            source_encoding,
            source_domain_id,
            target_domain_id,
            update_memory=False,
            memory_mode="retrieve",
        )
        target_result = model(
            pair_encoding,
            target_domain_id,
            return_state=True,
            update_memory=False,
            memory_mode="retrieve",
        )
        return source_result, target_result

    def _build_extras(self, state, has_pairs: bool) -> dict[str, Any]:
        """Collect auxiliary invariant outputs for loss reporting.

        Args:
            state: Forward result carrying aux_state and routing weights.
        """
        source_state = state.aux_state
        return {
            "exported_invariant": source_state.exported_invariant,
            "routing_weights": state.routing_weights,
            "has_pairs": has_pairs,
        }

    def _core_engine_invariants(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if "encoding" not in batch or "pair_encoding" not in batch:
            raise KeyError("HDIMCoreEngine batches must include encoding and pair_encoding")
        source_domain_id = batch["domain_id"].to(self.device)
        target_domain_id = batch["pair_domain_id"].to(self.device)
        source_inv = self._extract_core(batch["encoding"].to(self.device), source_domain_id)
        target_inv = self._extract_core(batch["pair_encoding"].to(self.device), target_domain_id)
        return source_inv, target_inv, {"exported_invariant": source_inv}

    def _extract_core(self, encoding: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        multivector = self.model.encode(encoding)
        invariant = torch.empty_like(multivector)
        domain_names = tuple(self.model.config.domain_names or ())
        for domain in torch.unique(domain_id).tolist():
            mask = domain_id == int(domain)
            invariant[mask] = self.model.extract(multivector[mask], domain_names[int(domain)])
        return invariant

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch

    def compute_iso_loss(self, u_inv_source: torch.Tensor, u_inv_target: torch.Tensor) -> torch.Tensor:
        from src.training.invariant_losses import compute_iso_loss

        return compute_iso_loss(u_inv_source, u_inv_target)

    def compute_routing_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
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
            from src.training.invariant_losses import compute_iso_loss

            return compute_iso_loss(exported_invariant, pair_exported_target)
        from src.training.invariant_losses import compute_infonce_loss

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
                new_key = "pipeline.moe." + key[len("pipeline.moe.kernel.") :]
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

    def _has_texts(self, batch: dict) -> bool:
        return "text" in batch and hasattr(self.model, "forward_texts")

    def _has_pair_texts(self, batch: dict) -> bool:
        return self._has_texts(batch) and "pair_text" in batch and hasattr(self.model, "encode_texts")
