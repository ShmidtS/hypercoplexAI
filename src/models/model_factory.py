"""Single authoritative factory for all HDIM model variants.

This is the *only* place where HDIMModel / TextHDIMModel instances are
assembled with optional components.  Call sites should import
from here rather than wiring models by hand.

Public API
----------
build_hdim_model(cfg)                    -> HDIMModel
build_text_hdim_model(cfg, ...)          -> TextHDIMModel
build_sbert_hdim_model(cfg, ...)         -> TextHDIMModel (with frozen SBERT encoder)
model_from_experiment_config(exp)        -> TextHDIMModel | HDIMModel
"""
from __future__ import annotations

from typing import Union

from src.models.hdim_model import HDIMConfig, HDIMModel
from src.models.text_hdim_model import TextHDIMModel
from src.training.experiment_config import ExperimentConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _patch_soft_router(core_model: HDIMModel, z_loss_weight: float = 0.0) -> None:
    """Replace pipeline.moe with SoftMoERouter in-place.

    The replacement uses the same input_dim / num_experts as the original
    R3MoERouter so the pipeline contract is preserved.
    """
    from src.core.soft_moe_router import SoftMoERouter

    pipeline = core_model.pipeline
    cfg = core_model.config
    # R3MoERouter operates on clifford-space vectors
    input_dim: int = pipeline.clifford_dim
    expert_dim: int = input_dim * 2  # sensible default

    new_moe = SoftMoERouter(
        input_dim=input_dim,
        num_experts=cfg.num_experts,
        expert_dim=expert_dim,
        top_k=cfg.top_k,
        z_loss_weight=z_loss_weight,
    )
    pipeline.moe = new_moe  # type: ignore[assignment]


def _patch_sbert_encoder(
    text_model: TextHDIMModel,
    *,
    model_name: str = "paraphrase-multilingual-mpnet-base-v2",
    freeze: bool = True,
    dropout: float = 0.1,
    unfreeze_layers: list | None = None,
    projection_hidden: int | None = None,
) -> None:
    """Replace text_model.text_encoder with SBERTEncoder in-place."""
    from src.models.sbert_encoder import SBERTEncoder

    core_cfg = text_model.core_model.config
    new_encoder = SBERTEncoder(
        output_dim=core_cfg.hidden_dim,
        model_name=model_name,
        freeze=freeze,
        dropout=dropout,
        unfreeze_layers=unfreeze_layers,
        projection_hidden=projection_hidden,
    )
    text_model.text_encoder = new_encoder  # type: ignore[assignment]




# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def build_hdim_model(cfg: HDIMConfig) -> HDIMModel:
    """Build a plain HDIMModel from an HDIMConfig."""
    return HDIMModel(cfg)


def build_text_hdim_model(
    cfg: HDIMConfig,
    *,
    soft_router: bool = False,
    z_loss_weight: float = 0.0,
) -> TextHDIMModel:
    """Build a TextHDIMModel with optional SoftMoERouter."""
    core_model = HDIMModel(cfg)

    if soft_router:
        _patch_soft_router(core_model, z_loss_weight=z_loss_weight)

    return TextHDIMModel(core_model)


def build_sbert_hdim_model(
    cfg: HDIMConfig,
    *,
    soft_router: bool = False,
    sbert_model_name: str = "paraphrase-multilingual-mpnet-base-v2",
    freeze_sbert: bool = True,
    sbert_dropout: float = 0.1,
    unfreeze_layers: list | None = None,
    projection_hidden: int | None = None,
    z_loss_weight: float = 0.0,
) -> TextHDIMModel:
    """Build a TextHDIMModel with frozen SBERT encoder and optional SoftMoERouter."""
    core_model = HDIMModel(cfg)

    if soft_router:
        _patch_soft_router(core_model, z_loss_weight=z_loss_weight)

    text_model = TextHDIMModel(core_model)

    _patch_sbert_encoder(
        text_model,
        model_name=sbert_model_name,
        freeze=freeze_sbert,
        dropout=sbert_dropout,
        unfreeze_layers=unfreeze_layers,
        projection_hidden=projection_hidden,
    )

    return text_model


def model_from_experiment_config(
    exp: ExperimentConfig,
) -> Union[TextHDIMModel, HDIMModel]:
    """Build the appropriate model from an ExperimentConfig.

    Returns TextHDIMModel when text_mode or any advanced component flag is
    set; otherwise returns a plain HDIMModel.
    """
    # Build HDIMConfig from the experiment settings.
    # We forward only the fields that HDIMConfig actually accepts.
    cfg = HDIMConfig(
        hidden_dim=exp.hidden_dim,
        num_experts=exp.num_experts,
        num_domains=exp.num_domains,
    )

    needs_text = (
        exp.text_mode
        or exp.soft_router
        or getattr(exp, "pretrained_encoder", False)
    )

    if needs_text:
        if getattr(exp, "pretrained_encoder", False):
            return build_sbert_hdim_model(
                cfg,
                soft_router=exp.soft_router,
            )
        return build_text_hdim_model(
            cfg,
            soft_router=exp.soft_router,
        )

    return HDIMModel(cfg)
