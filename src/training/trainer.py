"""HDIMTrainer — training loop, loss computation, and checkpoint management."""

from __future__ import annotations
from dataclasses import dataclass
import logging
import math
import os
import pickle

logger = logging.getLogger(__name__)
from typing import Any, Dict, Literal, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from src.models.hdim_model import HDIMAuxState, HDIMModel
from src.models.results import ForwardResult
from src.training.augmentations import EmbeddingAugmenter
from src.training.losses import (
    compute_batch_losses as _compute_batch_losses_fn,
    compute_pair_ranking_loss as _pair_ranking_loss,
    compute_iso_loss as _iso_loss,
    compute_routing_loss as _routing_loss,
    compute_pair_iso_loss as _pair_iso_loss,
    compute_infonce_loss as _infonce_loss,
    LossConfig,
)
from src.training.temperature import effective_temperature, cluster_scaled_temperature


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
        lambda_routing: float = 0.01,  # MoE load balance (reduced: routing loss normalized by log(E))
        lambda_pair: float = 0.4,  # InfoNCE contrastive (optimal from Run 18)
        lambda_memory: float = 0.05,  # memory regularization (EMA stability)
        negative_margin: float = 1.0,
        ranking_margin: float = 0.2,
        use_infonce: bool = True,
        infonce_temperature: float = 0.10,
        lambda_z: float = 0.01,  # MoE z-loss (prevents router collapse)
        lambda_expert_ortho: float = 0.01,  # expert orthogonalization (Phase 26)
        lambda_online: float = 0.01,  # online self-evolution loss (Phase 31)
        learnable_temperature: bool = False,
        lambda_dcl: float = 0.05,
        lambda_uniformity: float = 0.02,
        use_sc_temperature: bool = False,
        lambda_matryoshka: float = 0.15,  # multi-scale embedding loss (optional)
        # Temperature scheduling parameters (exposed for configurability)
        temp_schedule: str = "none",
        tau_max: float = 0.1,
        tau_min: float = 0.01,
        temp_schedule_T_0: int = 20,
        focal_gamma: float = 1.0,
        # Embedding augmentation
        aug_noise_std: float = 0.0,
        aug_mixup_alpha: float = 0.0,
        # Gradient accumulation
        accum_steps: int = 1,
    ) -> None:
        self.model = model.to(device)
        self._sbert_encoder = self._get_sbert_encoder()
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.lambda_routing = lambda_routing
        self.lambda_pair = lambda_pair
        self.lambda_memory = lambda_memory
        self.negative_margin = negative_margin
        self.ranking_margin = ranking_margin
        self.use_infonce = use_infonce
        self.infonce_temperature = infonce_temperature
        self.lambda_z = lambda_z
        self.lambda_expert_ortho = lambda_expert_ortho
        self.lambda_online = lambda_online
        self.lambda_dcl = lambda_dcl
        self.lambda_uniformity = lambda_uniformity
        self.lambda_matryoshka = lambda_matryoshka
        self.use_sc_temperature = use_sc_temperature
        self.accum_steps = max(1, accum_steps)
        self._accum_counter = 0
        self._last_grad_norm: float = 0.0
        self._grad_norm_ema: float = 0.0
        self._last_batch_size: int = 1

        # Warn about conflicting loss combinations
        if focal_gamma < 1.0 and lambda_dcl > 0:
            logger.warning("Focal-InfoNCE + DCL simultaneously — DCL already removes positive from denominator, "
                           "focal modulation is redundant. Consider using DCL alone or InfoNCE+Focal alone.")
        if lambda_uniformity > lambda_pair * 2:
            logger.warning("lambda_uniformity >> lambda_pair — uniformity pushes points apart, "
                           "may prevent contrastive clustering. Recommend lambda_uniformity < lambda_pair.")
        if lambda_dcl > 0 and lambda_uniformity > 0:
            logger.info("DCL + Uniformity is a strong combo for embedding uniformity — ensure lambda_pair is tuned.")
        self.use_hard_negatives: bool = False
        self._last_cluster_temp: float | None = None
        self._step: int = 0
        self._focal_gamma: float = focal_gamma
        self._current_epoch: int = 0
        self._temp_schedule: str = temp_schedule
        self._tau_max: float = tau_max
        self._tau_min: float = tau_min
        self._temp_schedule_T_0: int = temp_schedule_T_0
        import math as _math
        if learnable_temperature:
            self._log_temp = nn.Parameter(
                torch.tensor(_math.log(infonce_temperature), device=torch.device(device))
            )
            self.model.register_parameter('_log_temp', self._log_temp)
        else:
            self._log_temp = None
        self._augmenter = EmbeddingAugmenter(
            noise_std=aug_noise_std,
            dropout_p=0.0,
            mixup_alpha=aug_mixup_alpha,
        ) if (aug_noise_std > 0 or aug_mixup_alpha > 0) else None

    def _get_sbert_encoder(self):
        """Get SBERT encoder stored outside nn.Module registry (via object.__setattr__)."""
        if hasattr(self.model, 'text_encoder'):
            try:
                return object.__getattribute__(self.model.text_encoder, '_sbert')
            except AttributeError:
                pass
        return None

    def _all_trainable_params(self):
        """All trainable params including SBERT (which bypasses nn.Module registration)."""
        params = list(self.model.parameters())
        if self._sbert_encoder is not None:
            params.extend(p for p in self._sbert_encoder.parameters() if p.requires_grad)
        return params

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
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor] | None]:
        if collect_matryoshka and hasattr(self.model, 'encode_texts_matryoshka'):
            full_enc, scales = self.model.encode_texts_matryoshka(
                texts, device=self.device,
            )
            return full_enc, scales
        enc = self.model.encode_texts(texts, device=self.device)
        return enc, None

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
        result = self._forward_batch(
            pair_encoding,
            pair_domain_id,
            TrainingRegime(
                mode="reconstruction",
                update_memory=False,
                memory_mode="retrieve",
            ),
        )
        return self._training_invariant(result.aux_state).detach()

    def _compute_pair_iso_targets_from_text(
        self, pair_texts: Sequence[str], pair_domain_id: torch.Tensor
    ) -> torch.Tensor:
        result = self._forward_text_batch(
            pair_texts,
            pair_domain_id,
            TrainingRegime(
                mode="reconstruction",
                update_memory=False,
                memory_mode="retrieve",
            ),
        )
        return self._training_invariant(result.aux_state).detach()

    def _extract_pair_target_state(
        self, batch: Dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._uses_text_model() and self._has_pair_texts(batch):
            pair_texts = self._extract_texts(batch, "pair_text")
            pair_domain_id = batch["pair_domain_id"].to(self.device)
            result = self._forward_text_batch(
                pair_texts,
                pair_domain_id,
                TrainingRegime(
                    mode="reconstruction",
                    update_memory=False,
                    memory_mode="retrieve",
                ),
            )
            aux_state = result.aux_state
            return (
                self._training_invariant(aux_state).detach(),
                aux_state.exported_invariant.detach(),
            )

        pair_encoding = batch.get("pair_encoding")
        pair_domain_id = batch.get("pair_domain_id")
        if pair_encoding is None or pair_domain_id is None:
            return None
        result = self._forward_batch(
            pair_encoding.to(self.device),
            pair_domain_id.to(self.device),
            TrainingRegime(
                mode="reconstruction",
                update_memory=False,
                memory_mode="retrieve",
            ),
        )
        aux_state = result.aux_state
        return (
            self._training_invariant(aux_state).detach(),
            aux_state.exported_invariant.detach(),
        )

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
        self._accum_counter = 0
        if epoch == 1 and hasattr(self.model, "reset_memory"):
            self.model.reset_memory(strategy="hard")

    # -- Loss delegation (public API preserved for test compatibility) ----------

    def compute_pair_ranking_loss(
        self,
        exported_invariant: torch.Tensor,
        pair_exported_target: torch.Tensor | None,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        config = self._build_loss_config()
        temp = effective_temperature(
            log_temp=self._log_temp,
            temp_schedule=self._temp_schedule,
            current_epoch=self._current_epoch,
            infonce_temperature=self.infonce_temperature,
            tau_max=self._tau_max,
            tau_min=self._tau_min,
            temp_schedule_T_0=self._temp_schedule_T_0,
        )
        if self.use_sc_temperature:
            temp = cluster_scaled_temperature(exported_invariant, temp)
            self._last_cluster_temp = temp
        return _pair_ranking_loss(
            exported_invariant, pair_exported_target, batch, config,
            effective_temp=temp, device=self.device,
            has_pairs=self._has_pairs(batch),
        )

    def compute_iso_loss(
        self, u_inv_source: torch.Tensor, u_inv_target: torch.Tensor
    ) -> torch.Tensor:
        return _iso_loss(u_inv_source, u_inv_target)

    def compute_routing_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        return _routing_loss(routing_weights)

    def _compute_pair_iso_loss(
        self,
        training_invariant: torch.Tensor,
        iso_target: torch.Tensor,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        return _pair_iso_loss(
            training_invariant, iso_target, batch,
            negative_margin=self.negative_margin,
            device=self.device,
            has_pairs=self._has_pairs(batch),
        )

    def _compute_infonce_loss(
        self,
        source_inv: torch.Tensor,
        target_inv: torch.Tensor,
        pair_relation_label: torch.Tensor,
        pair_weight: torch.Tensor,
        temperature: float = 0.07,
        pair_group_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _infonce_loss(
            source_inv, target_inv, pair_relation_label, pair_weight,
            temperature=temperature, pair_group_id=pair_group_id,
            device=self.device,
        )

    # -- Forward helpers ---------------------------------------------------------

    def _forward_text_batch(
        self,
        texts: Sequence[str],
        domain_id: torch.Tensor,
        regime: TrainingRegime,
        *,
        return_encoding: bool = False,
    ) -> ForwardResult:
        return self.model.forward_texts(
            texts,
            domain_id,
            return_state=True,
            update_memory=regime.update_memory,
            memory_mode=regime.memory_mode,
            return_encoding=return_encoding,
        )

    def _forward_batch(
        self,
        encoding: torch.Tensor,
        domain_id: torch.Tensor,
        regime: TrainingRegime,
    ) -> ForwardResult:
        return self.model(
            encoding,
            domain_id,
            return_state=True,
            update_memory=regime.update_memory,
            memory_mode=regime.memory_mode,
        )

    # -- Loss config builder ----------------------------------------------------

    def _build_loss_config(self) -> LossConfig:
        return LossConfig(
            lambda_routing=self.lambda_routing,
            lambda_pair=self.lambda_pair,
            lambda_memory=self.lambda_memory,
            negative_margin=self.negative_margin,
            ranking_margin=self.ranking_margin,
            use_infonce=self.use_infonce,
            infonce_temperature=self.infonce_temperature,
            lambda_z=self.lambda_z,
            lambda_expert_ortho=self.lambda_expert_ortho,
            lambda_online=self.lambda_online,
            learnable_temperature=self._log_temp is not None,
            lambda_dcl=self.lambda_dcl,
            lambda_uniformity=self.lambda_uniformity,
            use_sc_temperature=self.use_sc_temperature,
            lambda_matryoshka=self.lambda_matryoshka,
            temp_schedule=self._temp_schedule,
            tau_max=self._tau_max,
            tau_min=self._tau_min,
            temp_schedule_T_0=self._temp_schedule_T_0,
            focal_gamma=self._focal_gamma,
            use_hard_negatives=self.use_hard_negatives,
        )

    # -- Batch loss computation (delegates to losses.py) ------------------------

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
                tp_result = self.model.transfer_pairs(
                        _src_enc,
                        domain_id,
                        pair_domain_id,
                        update_memory=regime.update_memory,
                        memory_mode=regime.memory_mode,
                    )
                output = tp_result.output
                routing_weights = tp_result.routing_weights
                invariant = tp_result.invariant
                aux_state = tp_result.aux_state
                _prl = batch.get("pair_relation_label")
                _tgt_enc, _tgt_scales = self._encode_texts(pair_texts, collect_matryoshka=True)
                # Apply embedding augmentation to pair (non-anchor) side
                if self._augmenter is not None:
                    _tgt_enc = self._augmenter(_tgt_enc, pairs_only=False)
                if _src_scales is not None and _tgt_scales is not None:
                    # Compute matryoshka scales from exported_invariant (not SBERT space)
                    _exported = aux_state.exported_invariant
                    _exported_dim = _exported.shape[-1]
                    _matryoshka_dims = [d for d in sorted(_src_scales.keys()) if d < _exported_dim]
                    if _matryoshka_dims:
                        batch["matryoshka_source"] = {d: _exported[..., :d].contiguous() for d in _matryoshka_dims}
                        # For target, use pair_exported_target if available
                        _pair_tgt = batch.get("pair_exported_target", _exported.detach())
                        batch["matryoshka_target"] = {d: _pair_tgt[..., :d].contiguous() for d in _matryoshka_dims}
                    else:
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
                txt_result = self._forward_text_batch(
                    texts, domain_id, regime, return_encoding=True
                )
                output = txt_result.output
                routing_weights = txt_result.routing_weights
                invariant = txt_result.invariant
                aux_state = txt_result.aux_state
                recon_target = txt_result.encodings
        else:
            encoding = batch["encoding"].to(self.device)
            if regime.mode == "paired":
                pair_encoding = batch["pair_encoding"].to(self.device)
                # Apply embedding augmentation to pair (non-anchor) side
                if self._augmenter is not None:
                    pair_encoding = self._augmenter(pair_encoding, pairs_only=False)
                pair_domain_id = batch["pair_domain_id"].to(self.device)
                tp_result = self.model.transfer_pairs(
                    encoding,
                    domain_id,
                    pair_domain_id,
                    update_memory=regime.update_memory,
                    memory_mode=regime.memory_mode,
                )
                output = tp_result.output
                routing_weights = tp_result.routing_weights
                invariant = tp_result.invariant
                aux_state = tp_result.aux_state
                # For negative pairs: recon_target = source encoding (no transfer)
                # For positive pairs: recon_target = pair_encoding (cross-domain transfer)
                pair_relation_label = batch.get("pair_relation_label")
                if pair_relation_label is not None:
                    pos_mask = (pair_relation_label > 0.5).to(encoding.device).unsqueeze(-1).expand_as(encoding)
                    recon_target = torch.where(pos_mask, pair_encoding, encoding)
                else:
                    recon_target = pair_encoding
            else:
                fwd_result = self._forward_batch(
                    encoding, domain_id, regime
                )
                output = fwd_result.output
                routing_weights = fwd_result.routing_weights
                invariant = fwd_result.invariant
                aux_state = fwd_result.aux_state
                recon_target = encoding

        # Compute pair targets for loss delegation
        pair_target_state = self._extract_pair_target_state(batch)
        if pair_target_state is None:
            iso_target = self._extract_iso_targets(batch, aux_state)
            pair_exported_target = None
        else:
            iso_target, pair_exported_target = pair_target_state

        # Delegate loss computation to losses.py
        batch_losses = _compute_batch_losses_fn(
            batch=batch,
            output=output,
            recon_target=recon_target,
            aux_state=aux_state,
            routing_weights=routing_weights,
            invariant=invariant,
            iso_target=iso_target,
            pair_exported_target=pair_exported_target,
            config=self._build_loss_config(),
            device=self.device,
            current_epoch=self._current_epoch,
            log_temp=self._log_temp,
            model=self.model,
        )

        # Add training_mode (not in losses.py which is regime-agnostic)
        batch_losses["training_mode"] = regime.mode
        return batch_losses

    def evaluate_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            return self._compute_batch_losses(batch)

    def train_step(self, batch: Dict[str, Any], scaler=None) -> torch.Tensor:
        """Execute one training step with optional AMP scaler support.

        Phase 31: Includes OnlineLearner replay_step for experience replay.
        Supports gradient accumulation via self.accum_steps.
        """
        self.model.train()
        is_accum_boundary = (self._accum_counter == 0)
        if is_accum_boundary:
            self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            "cuda", enabled=scaler is not None and self.device.type == "cuda"
        ):
            losses = self._compute_batch_losses(batch)
        # Track batch size for effective_temperature instrumentation
        _domain_id = batch.get("domain_id")
        if _domain_id is not None and isinstance(_domain_id, torch.Tensor):
            self._last_batch_size = _domain_id.shape[0]
        if losses.get("_nan_skip"):
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_counter = 0
            return losses["loss_total"]
        loss_total = losses["loss_total"]
        unscaled_loss = loss_total.detach().clone()

        # Scale loss for gradient accumulation
        if self.accum_steps > 1:
            loss_total = loss_total / self.accum_steps

        # Phase 31: Experience replay from OnlineLearner buffer
        online_replay_loss = torch.zeros((), device=loss_total.device, dtype=loss_total.dtype)
        if hasattr(self.model, 'online_learner') and self.model.online_learner is not None:
            replay_loss = self.model.online_learner.replay_step(self.model)
            if replay_loss is not None:
                online_replay_loss = replay_loss
                loss_total = loss_total + (replay_loss / self.accum_steps if self.accum_steps > 1 else replay_loss)

        self._accum_counter += 1
        should_step = (self._accum_counter % self.accum_steps == 0)

        if scaler is not None:
            # NaN/Inf loss: skip step entirely and force scale reduction.
            # Without this, backward() produces NaN grads -> scaler.step()
            # considers the step "successful" (no exception) -> scaler.update()
            # keeps the high scale -> next normal batch overflows -> permanent
            # loss explosion (7 -> 2414 -> 18M over 3 epochs).
            if torch.isnan(loss_total) or torch.isinf(loss_total):
                # Force scale reduction for stability
                new_scale = max(scaler.get_scale() * 0.5, 1.0)
                scaler._scale.fill_(new_scale)
                scaler.update()
                self._last_grad_norm = float('inf')
                if should_step:
                    self._accum_counter = 0
                return loss_total.detach()

            scaler.scale(loss_total).backward()
            if should_step:
                scaler.unscale_(self.optimizer)
                # Zero NaN/Inf gradients from uninitialized memory/MoE state
                has_nan_grad = False
                for p in self._all_trainable_params():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        p.grad.zero_()
                        has_nan_grad = True
                grad_norm = nn.utils.clip_grad_norm_(self._all_trainable_params(), max_norm=1.0)
                self._last_grad_norm = grad_norm.item() if grad_norm.numel() == 1 else grad_norm.max().item()
                self._grad_norm_ema = 0.9 * self._grad_norm_ema + 0.1 * self._last_grad_norm
                if has_nan_grad:
                    # NaN grads found -- skip step and reduce scale to prevent
                    # permanent scaler corruption from non-finite gradients.
                    new_scale = max(scaler.get_scale() * 0.5, 1.0)
                    scaler._scale.fill_(new_scale)
                    scaler.update()
                else:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self._step += 1
                self._accum_counter = 0
        else:
            loss_total.backward()
            if should_step:
                has_nan_grad = False
                for p in self._all_trainable_params():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        p.grad.zero_()
                        has_nan_grad = True
                grad_norm = nn.utils.clip_grad_norm_(self._all_trainable_params(), max_norm=1.0)
                self._last_grad_norm = grad_norm.item() if grad_norm.numel() == 1 else grad_norm.max().item()
                self._grad_norm_ema = 0.9 * self._grad_norm_ema + 0.1 * self._last_grad_norm
                if not has_nan_grad:
                    self.optimizer.step()
                    self._step += 1
                self._accum_counter = 0
        self._last_loss_components = {
            k: v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v
            for k, v in losses.items()
            if not isinstance(v, torch.Tensor) or v.dim() == 0
        }
        self._last_loss_components["grad_norm"] = self._last_grad_norm
        self._last_loss_components["online_replay_loss"] = online_replay_loss.item()
        return unscaled_loss

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        totals: dict = {}
        n_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                losses = self._compute_batch_losses(batch)
                for key, val in losses.items():
                    if key.startswith("_"):
                        continue
                    v = val.item() if isinstance(val, torch.Tensor) and val.dim() == 0 else None
                    if v is not None and math.isfinite(v):
                        totals[key] = totals.get(key, 0.0) + v
                n_batches += 1
        if n_batches > 0:
            for key in totals:
                totals[key] /= n_batches
        # Instrumentation for empirical laws
        val_loss = totals.get("loss_total", 0.0)
        train_loss = self._last_loss_components.get("loss_total", 0.0) if hasattr(self, '_last_loss_components') else 0.0
        totals["grad_norm_ema"] = self._grad_norm_ema
        lr = self.optimizer.param_groups[0]["lr"]
        totals["effective_temperature"] = lr / (self._last_batch_size * self.accum_steps)
        totals["loss_gap"] = val_loss - train_loss
        return totals

    def save_checkpoint(self, path: str, scaler=None, scheduler=None) -> None:
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
        # Phase 31: Save OnlineLearner state for persistence between sessions
        if hasattr(self.model, 'online_learner') and self.model.online_learner is not None:
            checkpoint["online_learner_state"] = self.model.online_learner.save_state()

        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)

    def _load_checkpoint_safe(self, path: str) -> dict:
        """Safely load checkpoint with weights_only=True (OWASP A08 fix).

        Prevents arbitrary code execution via pickle deserialization.
        Only tensor data is loaded, no Python objects.
        """
        map_location = self.device
        try:
            ckpt = torch.load(path, map_location=map_location, weights_only=True)
        except (RuntimeError, pickle.UnpicklingError) as exc:
            raise RuntimeError(
                f"Failed to load checkpoint with weights_only=True: {exc}. "
                f"Checkpoint at '{path}' contains non-tensor objects and cannot be "
                f"loaded safely. Re-save the checkpoint with the current code version "
                f"or remove incompatible objects from it."
            ) from exc
        return ckpt

    @staticmethod
    def _migrate_checkpoint_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Old adapter checkpoints have keys like pipeline.moe.kernel.X
        # New MoEKernel checkpoints have keys like pipeline.moe.X
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
        new_state_dict = self._migrate_checkpoint_state_dict(state_dict)
        if new_state_dict != state_dict:
            checkpoint["model_state_dict"] = new_state_dict
            import logging as _logging
            _logging.getLogger(__name__).info(
                "Checkpoint migration: stripped 'kernel.' prefix from MoEKernelAdapter keys"
            )

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._step = checkpoint.get("step", 0)
        self._current_epoch = checkpoint.get("current_epoch", 0)
        # Restore temperature scheduling config
        self._temp_schedule = checkpoint.get("temp_schedule", "none")
        self._tau_max = checkpoint.get("tau_max", 0.1)
        self._tau_min = checkpoint.get("tau_min", 0.01)
        self._temp_schedule_T_0 = checkpoint.get("temp_schedule_T_0", 20)
        self._focal_gamma = checkpoint.get("focal_gamma", 1.0)
        # Phase 31: Restore OnlineLearner state from checkpoint
        if hasattr(self.model, 'online_learner') and self.model.online_learner is not None:
            if "online_learner_state" in checkpoint:
                self.model.online_learner.load_state(checkpoint["online_learner_state"])

        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
