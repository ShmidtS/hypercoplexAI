# train.py — запуск: python -m src.training.train
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.hdim_model import HDIMConfig, HDIMModel
from src.models.metrics import compute_all_metrics
from src.training.dataset import (
    create_demo_dataset,
    create_group_aware_split,
    create_paired_demo_dataset,
)
from src.training.trainer import HDIMTrainer


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
        },
        "validation": val_metrics,
        "quality": quality_metrics,
        "checkpoint": checkpoint_path.as_posix(),
    }


def main():
    parser = argparse.ArgumentParser(description='Train HDIM model')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--use_pairs', action='store_true', help='Use paired cross-domain supervision dataset')
    parser.add_argument(
        '--results_json',
        type=Path,
        default=None,
        help='Optional path to write machine-readable run summary JSON for autoresearch-style orchestration.',
    )
    args = parser.parse_args()

    cfg = HDIMConfig()
    model = HDIMModel(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = HDIMTrainer(model, optimizer, device=args.device)

    dataset_factory = create_paired_demo_dataset if args.use_pairs else create_demo_dataset
    dataset = dataset_factory(n_samples=args.num_samples, embed_dim=cfg.hidden_dim)
    train_ds, val_ds = create_group_aware_split(dataset, train_fraction=0.8, seed=42)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    val_metrics = None
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in train_loader:
            loss = trainer.train_step(batch)
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_metrics = trainer.validate(val_loader)
        print(f'Epoch {epoch+1}/{args.epochs} | train_loss={avg_loss:.4f} | val_loss={val_metrics["loss_total"]:.4f}')

    if val_metrics is None:
        val_metrics = trainer.validate(val_loader)

    quality_metrics = compute_all_metrics(model, val_loader)
    print(
        'Quality metrics | '
        f'STS_exported={quality_metrics["STS_exported"]:.4f} | '
        f'STS_training={quality_metrics["STS_training"]:.4f} | '
        f'DRS={quality_metrics["DRS"]:.4f} | '
        f'AFR={quality_metrics["AFR"]:.4f} | '
        f'pair_margin={quality_metrics["pair_margin"]:.4f}'
    )

    checkpoint_dir = Path(__file__).resolve().parents[2] / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / 'hdim_final.pt'
    trainer.save_checkpoint(str(checkpoint_path))

    if args.results_json is not None:
        run_summary = _build_run_summary(
            args=args,
            cfg=cfg,
            val_metrics=val_metrics,
            quality_metrics=quality_metrics,
            checkpoint_path=checkpoint_path,
        )
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(run_summary, indent=2), encoding='utf-8')
        print(f'Wrote run summary to {args.results_json}')

    print('Training complete.')


if __name__ == '__main__':
    main()
