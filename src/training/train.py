# train.py — запуск: python -m src.training.train
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.hdim_model import HDIMConfig, HDIMModel
from src.models.metrics import compute_all_metrics
from src.training.dataset import create_demo_dataset, create_paired_demo_dataset
from src.training.trainer import HDIMTrainer


def main():
    parser = argparse.ArgumentParser(description='Train HDIM model')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--use_pairs', action='store_true', help='Use paired cross-domain supervision dataset')
    args = parser.parse_args()

    cfg = HDIMConfig()
    model = HDIMModel(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = HDIMTrainer(model, optimizer, device=args.device)

    dataset_factory = create_paired_demo_dataset if args.use_pairs else create_demo_dataset
    dataset = dataset_factory()
    # 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in train_loader:
            loss = trainer.train_step(batch)
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_metrics = trainer.validate(val_loader)
        print(f'Epoch {epoch+1}/{args.epochs} | train_loss={avg_loss:.4f} | val_loss={val_metrics["loss_total"]:.4f}')

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
    trainer.save_checkpoint(str(checkpoint_dir / 'hdim_final.pt'))
    print('Training complete.')


if __name__ == '__main__':
    main()
