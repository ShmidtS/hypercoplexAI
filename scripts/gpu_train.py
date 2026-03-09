#!/usr/bin/env python
"""
HDIM GPU Training Script с AMP, gradient checkpointing и live monitoring.

Функции:
  - Автоматическое определение GPU/CPU
  - Mixed Precision Training (AMP) для GPU
  - Gradient checkpointing для экономии памяти
  - Живой мониторинг через progress bar (tqdm) и tensorboard/json
  - Checkpoint сохранение каждые N эпох и по лучшей валидации
  - Поддержка AdvancedTextEncoder и HierarchicalTitansMemory
  - Совместим с ExperimentConfig/AutoResearchRunner

Запуск:
  python scripts/gpu_train.py --epochs 50 --device cuda --use_pairs --text_mode --advanced_encoder
  python scripts/gpu_train.py --config path/to/config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.hdim_model import HDIMConfig, HDIMModel
from src.models.metrics import compute_all_metrics
from src.models.text_hdim_model import TextHDIMModel
from src.training.dataset import (
    create_demo_dataset,
    create_group_aware_split,
    create_paired_demo_dataset,
)
from src.training.experiment_config import ExperimentConfig
from src.training.results_logger import append_ledger_row
from src.training.trainer import HDIMTrainer


def detect_device(requested: str) -> torch.device:
    """Определяет лучшее доступное устройство."""
    if requested == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            props = torch.cuda.get_device_properties(device)
            print(f"GPU: {props.name} | Memory: {props.total_memory / 1e9:.1f} GB")
            return device
        else:
            print("WARNING: CUDA not available, falling back to CPU")
            return torch.device("cpu")
    elif requested == "auto":
        if torch.cuda.is_available():
            return detect_device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def build_model_with_advanced_components(
    cfg: HDIMConfig,
    use_advanced_encoder: bool = False,
    use_hierarchical_memory: bool = False,
    use_soft_router: bool = False,
    text_mode: bool = False,
) -> nn.Module:
    """
    Собирает модель с опциональными продвинутыми компонентами.
    """
    model = HDIMModel(cfg)

    if use_hierarchical_memory:
        from src.core.hierarchical_memory import HierarchicalTitansMemory
        old_memory = model.pipeline.memory
        new_memory = HierarchicalTitansMemory(
            key_dim=old_memory.key_dim,
            val_dim=old_memory.val_dim,
            hidden_dim=64,
        )
        model.pipeline.memory = new_memory
        print("Using HierarchicalTitansMemory")

    if use_soft_router:
        from src.core.soft_moe_router import SoftMoERouter
        old_router = model.pipeline.moe
        new_router = SoftMoERouter(
            input_dim=model.pipeline.clifford_dim,
            num_experts=cfg.num_experts,
            expert_dim=256,
            top_k=cfg.top_k,
        )
        model.pipeline.moe = new_router
        print("Using SoftMoERouter")

    if text_mode:
        text_model = TextHDIMModel(model)

        if use_advanced_encoder:
            from src.models.advanced_text_encoder import AdvancedTextEncoder
            old_encoder = text_model.text_encoder
            new_encoder = AdvancedTextEncoder(
                output_dim=cfg.hidden_dim,
                vocab_size=old_encoder.vocab_size,
                max_length=old_encoder.max_length,
                num_layers=2,
                num_heads=max(1, cfg.hidden_dim // 32),
                dropout=cfg.dropout,
            )
            text_model.text_encoder = new_encoder
            print(f"Using AdvancedTextEncoder (layers=2, heads={max(1, cfg.hidden_dim // 32)})")

        return text_model

    return model


class GPUTrainingMonitor:
    """Мониторинг GPU памяти и throughput."""

    def __init__(self, device: torch.device, log_path: Optional[Path] = None):
        self.device = device
        self.log_path = log_path
        self.history: list[dict] = []
        self.start_time = time.time()
        self._use_tqdm = self._check_tqdm()

    def _check_tqdm(self) -> bool:
        try:
            import tqdm  # noqa
            return True
        except ImportError:
            return False

    def gpu_stats(self) -> dict:
        if self.device.type != "cuda":
            return {}
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        return {"gpu_allocated_gb": round(allocated, 3), "gpu_reserved_gb": round(reserved, 3)}

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_metrics: dict,
        quality_metrics: dict,
        lr: float,
    ) -> None:
        elapsed = time.time() - self.start_time
        gpu_stats = self.gpu_stats()
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics.get("loss_total", 0.0), 6),
            "pair_margin": round(quality_metrics.get("pair_margin", 0.0), 6),
            "STS_exported": round(quality_metrics.get("STS_exported", 0.0), 6),
            "lr": lr,
            "elapsed_s": round(elapsed, 1),
            **gpu_stats,
        }
        self.history.append(row)

        # Print
        msg = (
            f"Epoch {epoch:3d}/{total_epochs} "
            f"| train={train_loss:.4f} "
            f"| val={val_metrics.get('loss_total', 0):.4f} "
            f"| margin={quality_metrics.get('pair_margin', 0):.4f} "
            f"| STS={quality_metrics.get('STS_exported', 0):.4f}"
        )
        if gpu_stats:
            msg += f" | GPU={gpu_stats.get('gpu_allocated_gb', 0):.2f}GB"
        print(msg)

        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    def summary(self) -> dict:
        if not self.history:
            return {}
        best = max(self.history, key=lambda x: x.get("pair_margin", -float("inf")))
        return {
            "best_epoch": best["epoch"],
            "best_pair_margin": best["pair_margin"],
            "best_STS_exported": best["STS_exported"],
            "total_epochs": len(self.history),
            "total_time_s": round(time.time() - self.start_time, 1),
        }


def run_gpu_training(
    cfg: HDIMConfig,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Основной цикл GPU обучения."""

    # Создаём модель
    model = build_model_with_advanced_components(
        cfg,
        use_advanced_encoder=args.advanced_encoder,
        use_hierarchical_memory=args.hierarchical_memory,
        use_soft_router=args.soft_router,
        text_mode=args.text_mode,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )

    # LR Scheduler — cosine with warmup
    # LR Scheduler: cosine annealing with linear warmup
    # Use CosineAnnealingLR to avoid ZeroDivisionError when epochs <= warmup_epochs
    warmup_steps = max(1, min(args.warmup_epochs, args.epochs - 1))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=warmup_steps,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - warmup_steps),
                eta_min=args.lr * 1e-2,
            ),
        ],
        milestones=[warmup_steps],
    )
    # AMP Scaler для GPU
    use_amp = (device.type == "cuda") and args.amp
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using Automatic Mixed Precision (AMP)")

    # Dataset
    trainer = HDIMTrainer(
        model, optimizer,
        device=device,
        lambda_iso=args.lambda_iso,
        lambda_pair=args.lambda_pair,
        lambda_routing=args.lambda_routing,
        lambda_memory=args.lambda_memory,
    )

    dataset_factory = create_paired_demo_dataset if args.use_pairs else create_demo_dataset
    dataset_kwargs: dict = {
        "n_samples": args.num_samples,
        "embed_dim": cfg.hidden_dim,
        "seed": args.seed,
    }
    if args.use_pairs:
        dataset_kwargs["negative_ratio"] = args.negative_ratio

    dataset = dataset_factory(**dataset_kwargs)
    train_ds, val_ds = create_group_aware_split(dataset, train_fraction=args.train_fraction, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=(device.type=="cuda"))

    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val samples")

    # Monitor
    monitor = GPUTrainingMonitor(
        device,
        log_path=output_dir / "training_log.jsonl",
    )

    best_margin = float("-inf")
    best_checkpoint = output_dir / "checkpoints" / "best.pt"
    final_checkpoint = output_dir / "checkpoints" / "hdim_final.pt"
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    val_metrics: dict = {}
    quality_metrics: dict = {"STS_exported": 0.0, "STS_training": 0.0, "DRS": 0.0, "AFR": 0.0, "pair_margin": 0.0}

    nan_batches_total = 0

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        nan_batches_epoch = 0
        for batch in train_loader:
            if use_amp:
                # AMP-aware loop: не используем trainer.train_step (не поддерживает scaler)
                optimizer.zero_grad()
                try:
                    with autocast():
                        losses = trainer._compute_batch_losses(batch)
                        loss = losses["loss_total"]
                    if torch.isnan(loss) or torch.isinf(loss):
                        nan_batches_epoch += 1
                        optimizer.zero_grad()
                        continue
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                except RuntimeError as e:
                    if "nan" in str(e).lower() or "inf" in str(e).lower():
                        nan_batches_epoch += 1
                        optimizer.zero_grad()
                        if scaler is not None:
                            scaler.update()
                        continue
                    raise
            else:
                # trainer.train_step уже делает zero_grad, backward, clip_grad, step
                try:
                    loss = trainer.train_step(batch)
                    if torch.isnan(loss) or torch.isinf(loss):
                        nan_batches_epoch += 1
                        continue
                except RuntimeError as e:
                    if "nan" in str(e).lower() or "inf" in str(e).lower():
                        nan_batches_epoch += 1
                        continue
                    raise
            epoch_loss += loss.item()
            n_batches += 1

        if nan_batches_epoch > 0:
            nan_batches_total += nan_batches_epoch
            print(f"  [WARN] Epoch {epoch}: {nan_batches_epoch} NaN batches skipped")

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation каждые eval_every эпох
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = trainer.validate(val_loader)
            quality_metrics = compute_all_metrics(model, val_loader)

            # Checkpoint если лучше
            current_margin = quality_metrics.get("pair_margin", 0.0)
            if current_margin > best_margin:
                best_margin = current_margin
                trainer.save_checkpoint(str(best_checkpoint))

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        monitor.log_epoch(
            epoch=epoch,
            total_epochs=args.epochs,
            train_loss=avg_loss,
            val_metrics=val_metrics,
            quality_metrics=quality_metrics,
            lr=current_lr,
        )

        # Checkpoint каждые save_every эпох
        if epoch % args.save_every == 0:
            trainer.save_checkpoint(str(output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"))

    # Final save
    trainer.save_checkpoint(str(final_checkpoint))

    training_summary = monitor.summary()
    print(f"\nTraining complete. Best pair_margin={best_margin:.4f} at epoch {training_summary.get('best_epoch', '?')}")

    # Results JSON
    # Serialize args and device: convert non-JSON types to strings
    run_args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    results = {
        "run_args": run_args_dict,
        "config": {
            "hidden_dim": cfg.hidden_dim,
            "num_domains": cfg.num_domains,
            "num_experts": cfg.num_experts,
            "num_params": total_params,
        },
        "device": str(device),
        "validation": val_metrics,
        "quality": quality_metrics,
        "training_summary": training_summary,
        "checkpoint": str(final_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "status": "keep",
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="HDIM GPU Training")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--use_pairs", action="store_true")
    parser.add_argument("--negative_ratio", type=float, default=0.2)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text_mode", action="store_true")
    parser.add_argument("--advanced_encoder", action="store_true", help="Use AdvancedTextEncoder")
    parser.add_argument("--hierarchical_memory", action="store_true", help="Use HierarchicalTitansMemory")
    parser.add_argument("--soft_router", action="store_true", help="Use SoftMoERouter")
    parser.add_argument("--amp", action="store_true", default=True, help="Use AMP on GPU")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--lambda_iso", type=float, default=0.1)
    parser.add_argument("--lambda_pair", type=float, default=0.1)
    parser.add_argument("--lambda_routing", type=float, default=0.05)
    parser.add_argument("--lambda_memory", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_domains", type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    # Load experiment config if provided
    if args.config:
        exp = ExperimentConfig.from_json(args.config)
        args.epochs = exp.epochs
        args.batch_size = exp.batch_size
        args.lr = exp.lr
        args.device = exp.device
        args.num_samples = exp.num_samples
        args.use_pairs = exp.use_pairs
        args.negative_ratio = exp.negative_ratio
        args.train_fraction = exp.train_fraction
        args.seed = exp.seed
        args.text_mode = exp.text_mode

    device = detect_device(args.device)

    output_dir = args.output_dir or Path("artifacts") / "gpu_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = HDIMConfig(
        hidden_dim=args.hidden_dim,
        num_domains=args.num_domains,
        num_experts=args.num_experts,
    )

    print(f"\n{'='*60}")
    print(f"HDIM GPU Training")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"Advanced Encoder: {args.advanced_encoder} | Hierarchical Memory: {args.hierarchical_memory}")
    print(f"Soft Router: {args.soft_router} | Text Mode: {args.text_mode}")
    print(f"{'='*60}\n")

    results = run_gpu_training(cfg, args, device, output_dir)

    quality = results.get("quality", {})
    print(f"\nFinal Quality Metrics:")
    for k, v in quality.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()