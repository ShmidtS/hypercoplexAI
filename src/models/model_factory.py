"""Single authoritative factory for HDIM core models and optional adapters.

Public API
----------
build_hdim_model(cfg) -> HDIMModel
build_text_adapter(cfg, engine) -> TextAdapter
model_from_experiment_config(exp) -> TextHDIMModel | HDIMModel
"""

from __future__ import annotations

import warnings
from typing import Any, Union

from src.adapters.text import SimpleTextEncoder, TextAdapter
from src.core.engine import HDIMCoreEngine
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


def attach_moe_plugin(
    core_model: HDIMModel,
    *,
    expert_names: list | None = None,
    z_loss_weight: float = 0.01,
    ortho_loss_weight: float = 0.01,
    use_can_experts: bool = False,
) -> None:
    """Set pipeline.moe to MoEKernel (implements MoERouter directly).

    Uses MoEKernel directly (implements MoERouter interface).

    Supports built-in experts (math, language, code, science) and expert
    behavior configured via EXPERT_CONFIGS from src.extensions.moe.

    num_experts is computed from expert_names if provided, otherwise from
    core_model.config.num_experts.

    Args:
        core_model: HDIMModel instance to patch
        expert_names: List of expert domain names. Built-in: math, language,
            code, science. Unknown names use generic expert config.
        z_loss_weight: Weight for router Z-loss regularization
        ortho_loss_weight: Weight for expert orthogonalization loss
    """
    from src.extensions.moe import MoEKernel, MoEKernelConfig

    pipeline = core_model.pipeline
    cfg = core_model.config
    input_dim: int = pipeline.clifford_dim

    # Compute num_experts from expert_names if provided
    if expert_names is not None:
        num_experts = len(expert_names)
    else:
        # Use config.moe.num_experts (guaranteed to be set after __post_init__)
        num_experts = cfg.moe.num_experts or 4
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

    pipeline.moe = MoEKernel(kernel_cfg)


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def build_hdim_model(cfg: HDIMConfig) -> HDIMModel:
    """Build a core-engine HDIMModel; optional systems attach separately."""
    return HDIMModel(cfg)


def build_text_adapter(cfg: HDIMConfig, engine: HDIMCoreEngine) -> TextAdapter:
    """Build the optional text adapter for a core engine."""
    text_encoder = getattr(cfg, "text_encoder", None)
    if text_encoder not in (None, "simple") and not isinstance(text_encoder, dict):
        warnings.warn(
            f"Unsupported text_encoder={text_encoder!r}; using SimpleTextEncoder",
            DeprecationWarning,
            stacklevel=2,
        )

    text_config = cfg.text
    if isinstance(text_encoder, dict):
        text_config = type(cfg.text)(**{**cfg.text.__dict__, **text_encoder})

    encoder = SimpleTextEncoder.from_text_config(
        output_dim=engine.config.input_dim,
        text_config=text_config,
        fallback_dropout=cfg.dropout,
    )
    return TextAdapter(encoder, engine)


def build_text_hdim_model(
    cfg: HDIMConfig,
    *,
    soft_router: bool = False,
    z_loss_weight: float = 0.0,
) -> TextHDIMModel:
    """Build a TextHDIMModel with optional SoftMoERouter."""
    if soft_router and z_loss_weight > 0:
        cfg.moe.z_loss_weight = z_loss_weight

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
    """Deprecated legacy SBERT text wrapper factory."""
    warnings.warn(
        "build_sbert_hdim_model is deprecated; use build_hdim_model + build_text_adapter",
        DeprecationWarning,
        stacklevel=2,
    )
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
    """Deprecated legacy ModernBERT text wrapper factory."""
    warnings.warn(
        "build_modernbert_hdim_model is deprecated; use build_hdim_model + build_text_adapter",
        DeprecationWarning,
        stacklevel=2,
    )
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


_patch_moe_kernel = attach_moe_plugin


def model_from_experiment_config(
    exp: ExperimentConfig,
) -> Union[TextHDIMModel, HDIMModel]:
    """Build the appropriate model from an ExperimentConfig.

    Returns TextHDIMModel when text_mode or any advanced component flag is
    set; otherwise returns a plain HDIMModel.
    """
    # Build HDIMConfig from the experiment settings.
    # We forward only the fields that HDIMConfig actually accepts.
    cfg_kwargs: dict[str, Any] = {
        "hidden_dim": exp.hidden_dim,
        "num_domains": exp.num_domains,
        "expert_names": exp.expert_names,
    }
    moe: dict[str, Any] = {}
    if exp.num_experts is not None:
        moe["num_experts"] = exp.num_experts
    if getattr(exp, "n_shared_experts", 0) > 0:
        moe["n_shared_experts"] = exp.n_shared_experts
    cfg_kwargs["moe"] = moe

    if getattr(exp, "use_domain_embedding", False):
        cfg_kwargs["use_domain_embedding"] = exp.use_domain_embedding

    if getattr(exp, "use_domain_lora", False):
        cfg_kwargs["use_domain_lora"] = exp.use_domain_lora
        cfg_kwargs["domain_lora_rank"] = getattr(exp, "domain_lora_rank", 4)

    cfg = HDIMConfig(**cfg_kwargs)

    needs_text = (
        exp.text_mode
        or exp.soft_router
        or getattr(exp, "pretrained_encoder", False)
        or getattr(exp, "modernbert_encoder", False)
    )

    z_loss_weight = getattr(exp, "z_loss_weight", 0.0)

    if needs_text:
        model = build_text_hdim_model(
            cfg,
            soft_router=exp.soft_router,
            z_loss_weight=z_loss_weight,
        )
        if getattr(exp, "pretrained_encoder", False) or getattr(exp, "modernbert_encoder", False):
            warnings.warn(
                "Pretrained text encoder factories are deprecated; using SimpleTextEncoder compatibility path",
                DeprecationWarning,
                stacklevel=2,
            )
    else:
        model = HDIMModel(cfg)

    if getattr(exp, "moe_kernel", False):
        core_model = model.core_model if isinstance(model, TextHDIMModel) else model
        attach_moe_plugin(
            core_model,
            expert_names=getattr(exp, "moe_kernel_expert_names", None),
            z_loss_weight=z_loss_weight,
            ortho_loss_weight=getattr(exp, "lambda_expert_ortho", 0.01),
            use_can_experts=getattr(exp, "use_can_experts", False),
        )

    return model
