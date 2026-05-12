"""Minimal invariant-only trainer for HDIM core learning."""

from __future__ import annotations

from typing import Union

import torch

from src.core.engine import HDIMCoreEngine
from src.models.hdim_model import HDIMModel
from src.training.invariant_losses import compute_pair_iso_loss


class InvariantTrainer:
    """Train HDIM invariants from paired source/target domain batches only."""

    def __init__(
        self,
        model: Union[HDIMModel, HDIMCoreEngine],
        optimizer,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)

    def training_step(self, batch: dict) -> dict:
        """Run one paired invariant optimization step and return loss metrics."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.compute_losses(batch)
        losses["loss"].backward()
        self.optimizer.step()
        return {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in losses.items()}

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
            losses = self.compute_losses(batch)
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
            negative_margin=1.0,
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
        return source_state.training_invariant, target_state.training_invariant, self._build_extras(source_result, has_pairs)

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
        source_texts = [str(text) for text in batch["text"]]
        if not self._has_pair_texts(batch):
            source_result = self.model.forward_texts(
                source_texts,
                source_domain_id,
                return_state=True,
                update_memory=False,
                memory_mode="retrieve",
            )
            return source_result, source_result, False

        target_texts = [str(text) for text in batch["pair_text"]]
        source_encoding = self.model.encode_texts(source_texts, device=self.device)
        source_result = self.model.transfer_pairs(
            source_encoding,
            source_domain_id,
            target_domain_id,
            update_memory=False,
            memory_mode="retrieve",
        )
        target_result = self.model.forward_texts(
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
        source_encoding = batch["encoding"].to(self.device)
        if not has_pairs:
            source_result = self.model(
                source_encoding,
                source_domain_id,
                return_state=True,
                update_memory=False,
                memory_mode="retrieve",
            )
            return source_result, source_result

        pair_encoding = batch.get("pair_encoding", source_encoding).to(self.device)
        source_result = self.model.transfer_pairs(
            source_encoding,
            source_domain_id,
            target_domain_id,
            update_memory=False,
            memory_mode="retrieve",
        )
        target_result = self.model(
            pair_encoding,
            target_domain_id,
            return_state=True,
            update_memory=False,
            memory_mode="retrieve",
        )
        return source_result, target_result

    def _build_extras(self, state, has_pairs: bool) -> dict[str, torch.Tensor]:
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
        domain_names = tuple(self.model.config.domain_names)
        for domain in torch.unique(domain_id).tolist():
            mask = domain_id == int(domain)
            invariant[mask] = self.model.extract(multivector[mask], domain_names[int(domain)])
        return invariant

    def _has_texts(self, batch: dict) -> bool:
        return "text" in batch and hasattr(self.model, "forward_texts")

    def _has_pair_texts(self, batch: dict) -> bool:
        return self._has_texts(batch) and "pair_text" in batch and hasattr(self.model, "encode_texts")
