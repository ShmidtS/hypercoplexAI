# train.py -- запуск: python -m src.training.train
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.models.hdim_model import HDIMConfig, HDIMModel
from src.models.metrics import compute_all_metrics
from src.models.model_factory import build_text_hdim_model
from src.models.text_hdim_model import TextHDIMModel
from src.training.dataset import (
    create_demo_dataset,
    create_group_aware_split,
    create_paired_demo_dataset,
)
from src.training.experiment_config import ExperimentConfig
from src.training.results_logger import append_ledger_row
from src.training.trainer import HDIMTrainer


def _coerce_override(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(marker in raw_value for marker in (".", "e", "E")):
            return float(raw_value)
        return int(raw_value)
    except ValueError:
        return raw_value


def _parse_overrides(pairs: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Override must use key=value format, got: {item}")
        key, value = item.split("=", 1)
        overrides[key] = _coerce_override(value)
    return overrides


def _load_experiment_config(config_path: Path | None) -> ExperimentConfig | None:
    if config_path is None:
        return None
    return ExperimentConfig.from_json(config_path)


def _apply_experiment_defaults(
    args: argparse.Namespace,
    experiment: ExperimentConfig | None,
) -> argparse.Namespace:
    if experiment is None:
        return args

    args.epochs = experiment.epochs
    args.batch_size = experiment.batch_size
    args.lr = experiment.lr
    args.device = experiment.device
    args.num_samples = experiment.num_samples
    args.use_pairs = experiment.use_pairs
    args.negative_ratio = experiment.negative_ratio
    args.train_fraction = experiment.train_fraction
    args.seed = experiment.seed
    args.text_mode = experiment.text_mode
    if experiment.results_json is not None:
        args.results_json = Path(experiment.results_json)
    if experiment.ledger_path is not None:
        args.ledger_path = Path(experiment.ledger_path)
    if experiment.description:
        args.description = experiment.description
    args.model_override.extend(
        f"{key}={value}" for key, value in experiment.model_overrides.items()
    )
    args.trainer_override.extend(
        f"{key}={value}" for key, value in experiment.trainer_overrides.items()
    )

    # Phase-2 advanced component flags from manifest
    if experiment.soft_router:
        args.soft_router = True

    # Phase-2 HDIMConfig overrides from manifest (only when not already
    # present as explicit CLI args -- we check for non-default values).
    if experiment.hidden_dim != 128:   # non-default -> forward as model_override
        args.model_override.append(f"hidden_dim={experiment.hidden_dim}")
    if experiment.num_experts is not None and experiment.num_experts != 4:
        args.model_override.append(f"num_experts={experiment.num_experts}")
    if experiment.expert_names is not None:
        args.model_override.append(f"expert_names={experiment.expert_names}")

    return args


def _build_config(args: argparse.Namespace) -> HDIMConfig:
    cfg = HDIMConfig()
    # Apply hidden_dim / num_experts from CLI flags first (take precedence
    # over model_override strings for these two well-known knobs).
    if getattr(args, "hidden_dim", None) is not None:
        cfg.hidden_dim = args.hidden_dim
    if getattr(args, "num_experts", None) is not None:
        cfg.num_experts = args.num_experts

    for key, value in _parse_overrides(args.model_override).items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown model override: {key}")
        setattr(cfg, key, value)
    return cfg


def _build_model(
    cfg: HDIMConfig,
    args: argparse.Namespace,
) -> HDIMModel | TextHDIMModel:
    """Build the model using model_factory when advanced flags are set."""
    soft_router: bool = getattr(args, "soft_router", False)

    needs_text = args.text_mode or soft_router

    if needs_text:
        return build_text_hdim_model(
            cfg,
            soft_router=soft_router,
        )

    # Plain baseline path (identical to pre-Phase-2 behaviour)
    core_model = HDIMModel(cfg)
    return TextHDIMModel(core_model) if args.text_mode else core_model


def _build_trainer(
    model: HDIMModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> HDIMTrainer:
    trainer_kwargs: dict[str, Any] = {
        "device": args.device,
    }
    for key, value in _parse_overrides(args.trainer_override).items():
        trainer_kwargs[key] = value
    return HDIMTrainer(model, optimizer, **trainer_kwargs)


def _build_run_summary(
    *,
    args: argparse.Namespace,
    cfg: HDIMConfig,
    val_metrics: dict,
    quality_metrics: dict,
    checkpoint_path: Path,
    config_hash: str | None,
    run_id: str | None,
    status: str,
) -> dict:
    return {
        "run_id": run_id,
        "config": {
            "hidden_dim": cfg.hidden_dim,
            "num_domains": cfg.num_domains,
            "num_experts": cfg.num_experts,
            "dropout": cfg.dropout,
            "clifford_p": cfg.clifford_p,
            "clifford_q": cfg.clifford_q,
            "clifford_r": cfg.clifford_r,
            "top_k": cfg.top_k,
            "memory_key_dim": cfg.memory_key_dim,
            "domain_names": cfg.get_domain_names(),
        },
        "run_args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": args.device,
            "num_samples": args.num_samples,
            "use_pairs": args.use_pairs,
            "negative_ratio": args.negative_ratio,
            "train_fraction": args.train_fraction,
            "seed": args.seed,
            "text_mode": args.text_mode,
            "description": args.description,
            "soft_router": getattr(args, "soft_router", False),
        },
        "validation": val_metrics,
        "quality": quality_metrics,
        "score": quality_metrics.get("pair_margin", 0.0),
        "nan_batches_total": 0,
        "checkpoint": checkpoint_path.as_posix(),
        "config_hash": config_hash,
        "status": status,
    }


def _resolve_run_id(experiment: ExperimentConfig | None) -> str | None:
    if experiment is None:
        return None
    rid = experiment.metadata.get("run_id")
    return str(rid) if rid is not None else None


def _resolve_checkpoint_path(
    args: argparse.Namespace,
    experiment: ExperimentConfig | None,
    run_id: str | None,
) -> Path:
    if args.results_json is not None:
        return args.results_json.parent / "checkpoints" / "hdim_final.pt"
    if experiment is not None and experiment.output_dir is not None:
        checkpoint_dir = Path(experiment.output_dir)
        if run_id is not None:
            checkpoint_dir = checkpoint_dir / run_id
        return checkpoint_dir / "checkpoints" / "hdim_final.pt"
    checkpoint_dir = Path(__file__).resolve().parents[2] / "checkpoints"
    return checkpoint_dir / "hdim_final.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HDIM model")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional experiment manifest JSON path.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument(
        "--use_pairs", action="store_true",
        help="Use paired cross-domain supervision dataset",
    )
    parser.add_argument("--negative_ratio", type=float, default=0.0)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--text_mode", action="store_true",
        help="Train through TextHDIMModel wrapper",
    )
    parser.add_argument("--description", default="baseline")
    parser.add_argument(
        "--model_override", action="append", default=[],
        help="Model override in key=value format",
    )
    parser.add_argument(
        "--trainer_override", action="append", default=[],
        help="Trainer override in key=value format",
    )
    parser.add_argument(
        "--results_json", type=Path, default=None,
        help="Optional path to write machine-readable run summary JSON.",
    )
    parser.add_argument(
        "--ledger_path", type=Path, default=None,
        help="Optional JSONL ledger path for keep/discard style logging.",
    )

    # ------------------------------------------------------------------ #
    # Phase-2 advanced component flags                                    #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--soft_router", action="store_true",
        help="Replace R3MoERouter with SoftMoERouter.",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=None,
        help="Override HDIMConfig.hidden_dim.",
    )
    parser.add_argument(
        "--num_experts", type=int, default=None,
        help="Override HDIMConfig.num_experts.",
    )

    args = parser.parse_args()

    experiment = _load_experiment_config(args.config)
    args = _apply_experiment_defaults(args, experiment)

    cfg = _build_config(args)
    model = _build_model(cfg, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = _build_trainer(model, optimizer, args)

    dataset_factory = create_paired_demo_dataset if args.use_pairs else create_demo_dataset
    dataset_kwargs: dict[str, Any] = {
        "n_samples": args.num_samples,
        "embed_dim": cfg.hidden_dim,
        "seed": args.seed,
    }
    if args.use_pairs:
        dataset_kwargs["negative_ratio"] = args.negative_ratio
    dataset = dataset_factory(**dataset_kwargs)
    train_ds, val_ds = create_group_aware_split(
        dataset, train_fraction=args.train_fraction, seed=args.seed
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    val_metrics: dict | None = None
    for epoch in range(args.epochs):
        trainer.set_epoch(epoch + 1)
        total_loss = 0.0
        for batch in train_loader:
            loss = trainer.train_step(batch)
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_metrics = trainer.validate(val_loader)
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"| train_loss={avg_loss:.4f} "
            f'| val_loss={val_metrics["loss_total"]:.4f}'
        )

    if val_metrics is None:
        val_metrics = trainer.validate(val_loader)

    quality_metrics = compute_all_metrics(model, val_loader)
    print(
        "Quality metrics | "
        f'STS_exported={quality_metrics["STS_exported"]:.4f} | '
        f'STS_training={quality_metrics["STS_training"]:.4f} | '
        f'DRS={quality_metrics["DRS"]:.4f} | '
        f'AFR={quality_metrics["AFR"]:.4f} | '
        f'pair_margin={quality_metrics["pair_margin"]:.4f}'
    )

    run_id = _resolve_run_id(experiment)
    checkpoint_path = _resolve_checkpoint_path(args, experiment, run_id)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(checkpoint_path))

    config_hash = experiment.config_hash() if experiment is not None else None
    status = "keep"
    if args.results_json is not None:
        run_summary = _build_run_summary(
            args=args,
            cfg=cfg,
            val_metrics=val_metrics,
            quality_metrics=quality_metrics,
            checkpoint_path=checkpoint_path,
            config_hash=config_hash,
            run_id=run_id,
            status=status,
        )
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(
            json.dumps(run_summary, indent=2), encoding="utf-8"
        )
        print(f"Wrote run summary to {args.results_json}")
        if args.ledger_path is not None:
            append_ledger_row(
                args.ledger_path,
                {
                    "run_id": run_id,
                    "status": status,
                    "config_hash": config_hash,
                    "results_json": args.results_json.as_posix(),
                    "checkpoint": checkpoint_path.as_posix(),
                    "quality": quality_metrics,
                    "validation": val_metrics,
                    "description": args.description,
                },
            )
            print(f"Appended run ledger row to {args.ledger_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
