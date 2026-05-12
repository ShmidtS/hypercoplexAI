# train.py -- запуск: python -m src.training.train
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.models.hdim_model import HDIMConfig
from src.models.hdim_model import HDIMModel
from src.models.metrics import compute_all_metrics
from src.training.dataset import create_group_aware_split
from src.training.dataset import create_paired_demo_dataset
from src.training.auto_config import TrainingDefaults
from src.training.invariant_trainer import InvariantTrainer
from src.training.real_dataset import load_real_pairs_dataset
from src.training.real_dataset import split_real_pairs

logger = logging.getLogger(__name__)

_TD = TrainingDefaults()
TRAINING_DEFAULTS = {
    "epochs": _TD.epochs,
    "batch_size": _TD.batch_size,
    "learning_rate": _TD.learning_rate,
    "device": _TD.device,
    "negative_ratio": 0.5,
    "train_fraction": _TD.train_fraction,
    "seed": _TD.seed,
}


def _build_config(args: argparse.Namespace) -> HDIMConfig:
    return HDIMConfig(
        hidden_dim=args.hidden_dim,
        num_domains=args.num_domains,
        clifford_p=args.clifford_p,
        clifford_q=args.clifford_q,
        clifford_r=args.clifford_r,
        dropout=args.dropout,
        text_encoder=args.encoder,
    )


def _build_run_summary(
    *,
    args: argparse.Namespace,
    cfg: HDIMConfig,
    val_metrics: dict,
    quality_metrics: dict,
    checkpoint_path: Path,
) -> dict:
    return {
        "config": {
            "encoder": args.encoder,
            "hidden_dim": cfg.hidden_dim,
            "num_domains": cfg.num_domains,
            "dropout": cfg.dropout,
            "clifford_p": cfg.clifford_p,
            "clifford_q": cfg.clifford_q,
            "clifford_r": cfg.clifford_r,
            "domain_names": cfg.get_domain_names(),
        },
        "run_args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": args.device,
            "data_path": args.data_path.as_posix() if args.data_path is not None else None,
            "num_samples": args.num_samples,
            "negative_ratio": args.negative_ratio,
            "train_fraction": args.train_fraction,
            "seed": args.seed,
        },
        "validation": val_metrics,
        "quality": quality_metrics,
        "checkpoint": checkpoint_path.as_posix(),
    }


def _build_datasets(args: argparse.Namespace, cfg: HDIMConfig):
    if args.data_path is not None:
        dataset = load_real_pairs_dataset(
            args.data_path,
            seed=args.seed,
            add_negatives=True,
            negative_ratio=args.negative_ratio,
        )
        return split_real_pairs(dataset, train_fraction=args.train_fraction, seed=args.seed)

    dataset = create_paired_demo_dataset(
        n_samples=args.num_samples,
        embed_dim=cfg.hidden_dim,
        seed=args.seed,
        negative_ratio=args.negative_ratio,
    )
    return create_group_aware_split(dataset, train_fraction=args.train_fraction, seed=args.seed)


def main() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    parser = argparse.ArgumentParser(description="Train HDIM invariant model")
    parser.add_argument("--encoder", default=None, help="Optional encoder name stored in HDIMConfig.text_encoder")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_domains", type=int, default=4)
    parser.add_argument("--clifford_p", type=int, default=3)
    parser.add_argument("--clifford_q", type=int, default=1)
    parser.add_argument("--clifford_r", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=TRAINING_DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int, default=TRAINING_DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAINING_DEFAULTS["learning_rate"])
    parser.add_argument("--device", default=TRAINING_DEFAULTS["device"])
    parser.add_argument("--data_path", type=Path, default=None, help="Optional real paired dataset JSON path")
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--negative_ratio", type=float, default=TRAINING_DEFAULTS["negative_ratio"])
    parser.add_argument("--train_fraction", type=float, default=TRAINING_DEFAULTS["train_fraction"])
    parser.add_argument("--seed", type=int, default=TRAINING_DEFAULTS["seed"])
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = _build_config(args)
    model = HDIMModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    trainer = InvariantTrainer(model, optimizer, device=device)
    train_ds, val_ds = _build_datasets(args, cfg)

    num_workers = 2 if device.type == "cuda" else 0
    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, **loader_kwargs)

    val_metrics: dict | None = None
    for epoch in range(args.epochs):
        trainer.set_epoch(epoch + 1)
        total_loss = 0.0
        for batch in train_loader:
            loss = trainer.train_step(batch)
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_metrics = trainer.validate(val_loader)
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} | train_loss={avg_loss:.4f} | val_loss={val_metrics['loss_total']:.4f}"
        )

    if val_metrics is None:
        val_metrics = trainer.validate(val_loader)

    quality_metrics = compute_all_metrics(model, val_loader)
    logger.info(
        "Quality metrics | "
        f"STS_exported={quality_metrics['STS_exported']:.4f} | "
        f"STS_training={quality_metrics['STS_training']:.4f} | "
        f"DRS={quality_metrics['DRS']:.4f} | "
        f"AFR={quality_metrics['AFR']:.4f} | "
        f"pair_margin={quality_metrics['pair_margin']:.4f}"
    )

    checkpoint_path = args.output_dir / "checkpoints" / "hdim_final.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(checkpoint_path))

    run_summary = _build_run_summary(
        args=args,
        cfg=cfg,
        val_metrics=val_metrics,
        quality_metrics=quality_metrics,
        checkpoint_path=checkpoint_path,
    )
    results_path = args.output_dir / "run_summary.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    logger.info(f"Wrote run summary to {results_path}")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
