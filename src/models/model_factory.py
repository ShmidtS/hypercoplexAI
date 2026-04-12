"""Single authoritative factory for all HDIM model variants.

This is the *only* place where HDIMModel / TextHDIMModel instances are
assembled with optional components. Call sites should import
from here rather than wiring models by hand.

Public API
----------
build_hdim_model(cfg) -> HDIMModel
build_text_hdim_model(cfg, ...) -> TextHDIMModel
build_sbert_hdim_model(cfg, ...) -> TextHDIMModel (with frozen SBERT encoder)
build_modernbert_hdim_model(cfg, ...) -> TextHDIMModel (with ModernBERT encoder)
model_from_experiment_config(exp) -> TextHDIMModel | HDIMModel
"""

from __future__ import annotations

from typing import Union

from src.models.hdim_model import HDIMConfig, HDIMModel
from src.models.text_hdim_model import TextHDIMModel
from src.training.experiment_config import ExperimentConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_sbert_encoder(
    text_model: TextHDIMModel,
    *,
    model_name: str = "paraphrase-multilingual-mpnet-base-v2",
    freeze: bool = True,
    dropout: float = 0.1,
    unfreeze_layers: list | None = None,
    freeze_bottom_frac: float | None = None,
    projection_hidden: int | None = None,
) -> None:
    """Set text_model.text_encoder to SBERTEncoder."""
    from src.models.sbert_encoder import SBERTEncoder

    core_cfg = text_model.core_model.config
    new_encoder = SBERTEncoder(
        output_dim=core_cfg.hidden_dim,
        model_name=model_name,
        freeze=freeze,
        dropout=dropout,
        unfreeze_layers=unfreeze_layers,
        freeze_bottom_frac=freeze_bottom_frac,
        projection_hidden=projection_hidden,
    )
    text_model.text_encoder = new_encoder


def _make_modernbert_encoder(
    text_model: TextHDIMModel,
    *,
    model_name: str = "answerdotai/ModernBERT-base",
    freeze: bool = True,
    use_cls_pooling: bool = True,
    max_length: int = 512,
    matryoshka_dims: list[int] | None = None,
) -> None:
    """Set text_model.text_encoder to ModernBertEncoder."""
    from src.models.modern_text_encoder import ModernBertEncoder

    core_cfg = text_model.core_model.config
    new_encoder = ModernBertEncoder(
        output_dim=core_cfg.hidden_dim,
        pretrained_model=model_name,
        freeze_pretrained=freeze,
        use_cls_pooling=use_cls_pooling,
        max_length=max_length,
        matryoshka_dims=matryoshka_dims,
    )
    text_model.text_encoder = new_encoder


def _make_moe_kernel(
    core_model: HDIMModel,
    *,
    expert_names: list | None = None,
    z_loss_weight: float = 0.01,
    ortho_loss_weight: float = 0.01,
    use_can_experts: bool = False,
) -> None:
    """Set pipeline.moe to MoEKernel (via MoEKernelAdapter).

    Uses the canonical MoEKernelAdapter from src.core.moe_kernel_adapter
    which wraps MoEKernel to the MoERouter interface.

    Supports both built-in experts (math, language, code, science) and
    custom experts registered via `register_expert()` from moe_kernel.

    num_experts is computed from expert_names if provided, otherwise from
    core_model.config.num_experts.

    Args:
        core_model: HDIMModel instance to patch
        expert_names: List of expert domain names. Built-in: math, language,
            code, science. Custom names must be registered via register_expert()
            before calling. Unknown names use generic DomainExpert.
        z_loss_weight: Weight for router Z-loss regularization
        ortho_loss_weight: Weight for expert orthogonalization loss

    Example:
        >>> from src.core.moe_kernel import register_expert, DomainExpert
        >>> class MedicalExpert(DomainExpert):
        ...     pass  # custom implementation
        >>> register_expert("medical", MedicalExpert)
        >>> _make_moe_kernel(model, expert_names=["math", "medical"])
    """
    from src.core.moe_kernel import MoEKernel, MoEKernelConfig
    from src.core.moe_kernel_adapter import MoEKernelAdapter

    pipeline = core_model.pipeline
    cfg = core_model.config
    input_dim: int = pipeline.clifford_dim

    # Compute num_experts from expert_names if provided
    if expert_names is not None:
        num_experts = len(expert_names)
    else:
        # Use config.num_experts (guaranteed to be set after __post_init__)
        num_experts = cfg.num_experts or 4
        # Default expert names for built-in experts
        expert_names = ["math", "language", "code", "science"][:num_experts]
        # Pad with generic names if needed
        if len(expert_names) < num_experts:
            expert_names += [
                f"expert_{i}" for i in range(len(expert_names), num_experts)
            ]

    kernel_cfg = MoEKernelConfig(
        input_dim=input_dim,
        expert_hidden_dim=input_dim * 2,
        num_experts=num_experts,
        slots_per_expert=1,
        temperature=1.0,
        z_loss_weight=z_loss_weight,
        ortho_loss_weight=ortho_loss_weight,
        use_shared_expert=True,
        use_aux_loss_free=True,
        use_expert_ortho=True,
        expert_names=expert_names,
        use_can_experts=use_can_experts,
    )

    pipeline.moe = MoEKernelAdapter(MoEKernel(kernel_cfg))


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
    if soft_router and z_loss_weight > 0:
        cfg.z_loss_weight = z_loss_weight

    return TextHDIMModel(HDIMModel(cfg))


def build_sbert_hdim_model(
    cfg: HDIMConfig,
    *,
    soft_router: bool = False,
    sbert_model_name: str = "paraphrase-multilingual-mpnet-base-v2",
    freeze_sbert: bool = True,
    sbert_dropout: float = 0.1,
    unfreeze_layers: list | None = None,
    freeze_bottom_frac: float | None = None,
    projection_hidden: int | None = None,
    z_loss_weight: float = 0.0,
) -> TextHDIMModel:
    """Build a TextHDIMModel with frozen SBERT encoder and optional SoftMoERouter."""
    text_model = build_text_hdim_model(cfg, soft_router=soft_router, z_loss_weight=z_loss_weight)

    _make_sbert_encoder(
        text_model,
        model_name=sbert_model_name,
        freeze=freeze_sbert,
        dropout=sbert_dropout,
        unfreeze_layers=unfreeze_layers,
        freeze_bottom_frac=freeze_bottom_frac,
        projection_hidden=projection_hidden,
    )

    return text_model


def build_modernbert_hdim_model(
    cfg: HDIMConfig,
    *,
    soft_router: bool = False,
    modernbert_model_name: str = "answerdotai/ModernBERT-base",
    freeze_modernbert: bool = True,
    use_cls_pooling: bool = True,
    max_length: int = 512,
    matryoshka_dims: list[int] | None = None,
    z_loss_weight: float = 0.0,
) -> TextHDIMModel:
    """Build a TextHDIMModel with ModernBERT encoder and optional SoftMoERouter."""
    text_model = build_text_hdim_model(cfg, soft_router=soft_router, z_loss_weight=z_loss_weight)

    _make_modernbert_encoder(
        text_model,
        model_name=modernbert_model_name,
        freeze=freeze_modernbert,
        use_cls_pooling=use_cls_pooling,
        max_length=max_length,
        matryoshka_dims=matryoshka_dims,
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
    cfg_kwargs = dict(
        hidden_dim=exp.hidden_dim,
        num_domains=exp.num_domains,
    )
    # Pass expert_names if provided, otherwise pass num_experts
    if exp.expert_names is not None:
        cfg_kwargs["expert_names"] = exp.expert_names
    elif exp.num_experts is not None:
        cfg_kwargs["num_experts"] = exp.num_experts

    cfg = HDIMConfig(**cfg_kwargs)

    needs_text = (
        exp.text_mode
        or exp.soft_router
        or getattr(exp, "pretrained_encoder", False)
        or getattr(exp, "modernbert_encoder", False)
    )

    z_loss_weight = getattr(exp, "z_loss_weight", 0.0)

    if needs_text:
        if getattr(exp, "modernbert_encoder", False):
            return build_modernbert_hdim_model(
                cfg,
                soft_router=exp.soft_router,
                modernbert_model_name=getattr(
                    exp, "modernbert_model_name", "answerdotai/ModernBERT-base"
                ),
                freeze_modernbert=getattr(exp, "freeze_modernbert", True),
                use_cls_pooling=getattr(exp, "modernbert_use_cls_pooling", True),
                max_length=getattr(exp, "modernbert_max_length", 512),
                matryoshka_dims=getattr(exp, "matryoshka_dims", None),
                z_loss_weight=z_loss_weight,
            )
        if getattr(exp, "pretrained_encoder", False):
            return build_sbert_hdim_model(
                cfg,
                soft_router=exp.soft_router,
                z_loss_weight=z_loss_weight,
            )
        return build_text_hdim_model(
            cfg,
            soft_router=exp.soft_router,
            z_loss_weight=z_loss_weight,
        )

    return HDIMModel(cfg)
