"""Single authoritative factory for all HDIM model variants.

This is the *only* place where HDIMModel / TextHDIMModel instances are
assembled with optional advanced components.  Call sites should import
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

def _patch_hierarchical_memory(core_model: HDIMModel) -> None:
    """Replace pipeline.memory with HierarchicalTitansMemory in-place.

    Dimensions are inferred from the already-built pipeline so that the
    replacement is dimensionally identical to the original TitansMemoryModule.
    """
    from src.core.hierarchical_memory import HierarchicalTitansMemory

    pipeline = core_model.pipeline
    # key_dim comes from memory_key_proj output dim; val_dim == clifford_dim
    key_dim: int = pipeline.memory_key_proj.out_features
    val_dim: int = pipeline.clifford_dim
    hidden_dim: int = max(key_dim, 32)  # gate MLP hidden size

    new_memory = HierarchicalTitansMemory(
        key_dim=key_dim,
        val_dim=val_dim,
        hidden_dim=hidden_dim,
    )
    pipeline.memory = new_memory  # type: ignore[assignment]


def _patch_soft_router(core_model: HDIMModel) -> None:
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
    )
    pipeline.moe = new_moe  # type: ignore[assignment]


def _patch_advanced_encoder(text_model: TextHDIMModel) -> None:
    """Replace text_model.text_encoder with AdvancedTextEncoder in-place."""
    from src.models.advanced_text_encoder import AdvancedTextEncoder

    core_cfg = text_model.core_model.config
    new_encoder = AdvancedTextEncoder.from_text_config(
        output_dim=core_cfg.hidden_dim,
        text_config=text_model.text_config,
        fallback_dropout=core_cfg.dropout,
    )
    text_model.text_encoder = new_encoder  # type: ignore[assignment]


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




def _patch_modular_moe(
    core_model: HDIMModel,
    *,
    routing_type: str = 'soft',
    expert_hidden_multiplier: int = 2,
) -> None:
    """Replace pipeline.moe with ModularMoERouter in-place.

    ModularMoERouter поддерживает динамическое add_expert/remove_expert.
    Совместим с SoftMoERouter/R3MoERouter по API router_state.
    """
    from src.core.modular_moe import build_modular_moe

    pipeline = core_model.pipeline
    cfg = core_model.config
    input_dim: int = pipeline.clifford_dim
    expert_hidden = input_dim * expert_hidden_multiplier

    new_moe = build_modular_moe(
        input_dim=input_dim,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        routing_type=routing_type,
        expert_hidden_dim=expert_hidden,
    )
    pipeline.moe = new_moe  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def build_hdim_model(cfg: HDIMConfig) -> HDIMModel:
    """Build a plain HDIMModel from an HDIMConfig."""
    return HDIMModel(cfg)


def build_text_hdim_model(
    cfg: HDIMConfig,
    *,
    advanced_encoder: bool = False,
    hierarchical_memory: bool = False,
    soft_router: bool = False,
    modular_moe: bool = False,
    modular_moe_routing_type: str = 'soft',
) -> TextHDIMModel:
    """Build a TextHDIMModel, optionally with Phase-2 advanced components.

    Assembly order
    --------------
    1. Build HDIMModel(cfg).
    2. If hierarchical_memory: swap pipeline.memory -> HierarchicalTitansMemory.
    3. If soft_router:         swap pipeline.moe    -> SoftMoERouter.
    4. If modular_moe:         swap pipeline.moe    -> ModularMoERouter.
    5. Wrap in TextHDIMModel.
    6. If advanced_encoder:    swap text_encoder    -> AdvancedTextEncoder.
    7. Return the assembled TextHDIMModel.
    """
    core_model = HDIMModel(cfg)

    if hierarchical_memory:
        _patch_hierarchical_memory(core_model)

    if soft_router:
        _patch_soft_router(core_model)

    if modular_moe:
        _patch_modular_moe(core_model, routing_type=modular_moe_routing_type)

    text_model = TextHDIMModel(core_model)

    if advanced_encoder:
        _patch_advanced_encoder(text_model)

    return text_model


def build_sbert_hdim_model(
    cfg: HDIMConfig,
    *,
    hierarchical_memory: bool = False,
    soft_router: bool = False,
    modular_moe: bool = False,
    modular_moe_routing_type: str = 'soft',
    sbert_model_name: str = "paraphrase-multilingual-mpnet-base-v2",
    freeze_sbert: bool = True,
    sbert_dropout: float = 0.1,
    unfreeze_layers: list | None = None,
    projection_hidden: int | None = None,
) -> TextHDIMModel:
    """Build a TextHDIMModel with frozen SBERT encoder (Phase 4 modernization).

    Assembly order
    --------------
    1. Build HDIMModel(cfg).
    2. If hierarchical_memory: swap pipeline.memory -> HierarchicalTitansMemory.
    3. If soft_router:         swap pipeline.moe    -> SoftMoERouter.
    4. Wrap in TextHDIMModel.
    5. Swap text_encoder -> frozen SBERTEncoder.
    6. Return assembled TextHDIMModel.

    Note: cfg.hidden_dim should match SBERT projection target (e.g. 256 or 512).
    The SBERT encoder projects from 768 → cfg.hidden_dim via trainable MLP.
    """
    core_model = HDIMModel(cfg)

    if hierarchical_memory:
        _patch_hierarchical_memory(core_model)

    if soft_router:
        _patch_soft_router(core_model)

    if modular_moe:
        _patch_modular_moe(core_model, routing_type=modular_moe_routing_type)

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
        or exp.advanced_encoder
        or exp.hierarchical_memory
        or exp.soft_router
        or getattr(exp, "pretrained_encoder", False)
    )

    if needs_text:
        if getattr(exp, "pretrained_encoder", False):
            return build_sbert_hdim_model(
                cfg,
                hierarchical_memory=exp.hierarchical_memory,
                soft_router=exp.soft_router,
            )
        return build_text_hdim_model(
            cfg,
            advanced_encoder=exp.advanced_encoder,
            hierarchical_memory=exp.hierarchical_memory,
            soft_router=exp.soft_router,
        )

    # Apply hierarchical_memory / soft_router even without text wrapper
    core_model = HDIMModel(cfg)
    if exp.hierarchical_memory:
        _patch_hierarchical_memory(core_model)
    if exp.soft_router:
        _patch_soft_router(core_model)
    return core_model
