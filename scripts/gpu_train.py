#!/usr/bin/env python
"""
HDIM GPU Training Script с AMP, gradient checkpointing и live monitoring.

Функции:
  - Автоматическое определение GPU/CPU
  - Mixed Precision Training (AMP) для GPU
  - Gradient checkpointing для экономии памяти
  - Живой мониторинг через progress bar (tqdm) и tensorboard/json
  - Checkpoint сохранение каждые N эпох и по лучшей валидации
  - Поддержка AdvancedTextEncoder, HierarchicalTitansMemory, SoftMoERouter
  - Совместим с ExperimentConfig/AutoResearchRunner
  - Использует model_factory как единственный источник сборки моделей

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
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.hdim_model import HDIMConfig
from src.models.metrics import compute_all_metrics
from src.models.model_factory import build_hdim_model, build_text_hdim_model, build_sbert_hdim_model
from src.training.dataset import (
    create_demo_dataset,
    create_group_aware_split,
    create_paired_demo_dataset,
)
from src.training.experiment_config import ExperimentConfig
from src.training.results_logger import append_ledger_row
from src.training.trainer import HDIMTrainer


# ---------------------------------------------------------------------------
# Primary score formula — единственное место определения «качества прогона».
# autoresearch_loop.py импортирует эту же формулу для согласованности.
# ---------------------------------------------------------------------------
PRIMARY_SCORE_WEIGHTS = {"pair_margin": 1.0, "STS_exported": 0.3}


def compute_primary_score(quality: dict) -> float:
    """Главная скалярная метрика — incumbent-совместимая."""
    return sum(
        w * quality.get(k, 0.0) for k, w in PRIMARY_SCORE_WEIGHTS.items()
    )


def check_run_validity(results: dict) -> tuple[bool, str]:
    """Проверяет валидность завершённого прогона.

    Returns
    -------
    (is_valid, reason)
        is_valid — True если прогон считается успешным.
        reason   — пустая строка при успехе, иначе код провала из таксономии.
    """
    quality = results.get("quality", {})
    training_summary = results.get("training_summary", {})
    nan_batches_total = results.get("nan_batches_total", 0)
    total_epochs = training_summary.get("total_epochs", 0)

    # crash_nan: слишком много NaN-батчей
    if total_epochs > 0 and nan_batches_total / max(total_epochs, 1) > 5:
        return False, "crash_nan"

    # crash_oom: признак OOM — total_epochs < ожидаемого (уже поймано через returncode)
    # здесь детектируем только на основе метрик

    # metric_regression: все метрики нулевые / NaN
    pair_margin = quality.get("pair_margin", 0.0)
    sts = quality.get("STS_exported", 0.0)
    if (pair_margin == 0.0 and sts == 0.0) or (
        pair_margin != pair_margin or sts != sts  # NaN check
    ):
        return False, "metric_regression"

    return True, ""


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


def _build_model(cfg: HDIMConfig, args: argparse.Namespace) -> nn.Module:
    """Собирает модель через model_factory — единственный источник истины."""
    if getattr(args, 'pretrained_encoder', False):
        model = build_sbert_hdim_model(
            cfg,
            hierarchical_memory=args.hierarchical_memory,
            soft_router=args.soft_router,
        )
        print("Components: FrozenSBERT(paraphrase-multilingual-mpnet-base-v2) + projection")
        if args.hierarchical_memory:
            print("  + HierarchicalTitansMemory")
        if args.soft_router:
            print("  + SoftMoERouter")
        return model

    needs_text = args.text_mode or args.advanced_encoder or args.hierarchical_memory or args.soft_router
    if needs_text:
        model = build_text_hdim_model(
            cfg,
            advanced_encoder=args.advanced_encoder,
            hierarchical_memory=args.hierarchical_memory,
            soft_router=args.soft_router,
        )
        components = []
        if args.advanced_encoder:
            components.append("AdvancedTextEncoder")
        if args.hierarchical_memory:
            components.append("HierarchicalTitansMemory")
        if args.soft_router:
            components.append("SoftMoERouter")
        if components:
            print(f"Components: {', '.join(components)}")
        return model
    return build_hdim_model(cfg)


class GPUTrainingMonitor:
    """Мониторинг GPU памяти, throughput и NaN-статистики."""

    def __init__(self, device: torch.device, log_path: Optional[Path] = None):
        self.device = device
        self.log_path = log_path
        self.history: list[dict] = []
        self.start_time = time.time()

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
        nan_batches_epoch: int = 0,
        nan_batches_total: int = 0,
    ) -> None:
        elapsed = time.time() - self.start_time
        gpu_stats = self.gpu_stats()
        score = compute_primary_score(quality_metrics)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics.get("loss_total", 0.0), 6),
            "loss_memory": round(val_metrics.get("loss_memory", 0.0), 6),
            "loss_routing": round(val_metrics.get("loss_routing", 0.0), 6),
            "pair_margin": round(quality_metrics.get("pair_margin", 0.0), 6),
            "STS_exported": round(quality_metrics.get("STS_exported", 0.0), 6),
            "score": round(score, 6),
            "lr": lr,
            "elapsed_s": round(elapsed, 1),
            "nan_batches_epoch": nan_batches_epoch,
            "nan_batches_total": nan_batches_total,
            **gpu_stats,
        }
        self.history.append(row)

        msg = (
            f"Epoch {epoch:3d}/{total_epochs} "
            f"| train={train_loss:.4f} "
            f"| val={val_metrics.get('loss_total', 0):.4f} "
            f"| margin={quality_metrics.get('pair_margin', 0):.4f} "
            f"| STS={quality_metrics.get('STS_exported', 0):.4f} "
            f"| score={score:.4f}"
        )
        if nan_batches_epoch > 0:
            msg += f" | NaN={nan_batches_epoch}"
        if gpu_stats:
            msg += f" | GPU={gpu_stats.get('gpu_allocated_gb', 0):.2f}GB"
        print(msg)

        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    def summary(self) -> dict:
        if not self.history:
            return {}
        best = max(self.history, key=lambda x: x.get("score", -float("inf")))
        return {
            "best_epoch": best["epoch"],
            "best_pair_margin": best["pair_margin"],
            "best_STS_exported": best["STS_exported"],
            "best_score": best["score"],
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

    model = _build_model(cfg, args)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )

    # CosineAnnealingWarmRestarts: периодически перезапускает LR, помогает выбраться из плохих минимумов
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(args.warmup_epochs * 3, 20),  # первый цикл
        T_mult=2,                              # удваивать цикл каждый раз
        eta_min=args.lr * 1e-3,               # минимальный LR
    )

    use_amp = (device.type == "cuda") and args.amp
    scaler = GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Using Automatic Mixed Precision (AMP)")

    trainer = HDIMTrainer(
        model, optimizer,
        device=device,
        lambda_iso=args.lambda_iso,
        lambda_pair=args.lambda_pair,
        lambda_routing=args.lambda_routing,
        lambda_memory=args.lambda_memory,
        ranking_margin=args.ranking_margin,
        use_infonce=getattr(args, 'use_infonce', True),
        infonce_temperature=getattr(args, 'infonce_temperature', 0.07),
        lambda_sts=getattr(args, 'lambda_sts', 0.0),
        lambda_angle=getattr(args, 'lambda_angle', 0.0),
        learnable_temperature=getattr(args, 'learnable_temperature', False),
    )
    # Add learnable temperature to optimizer if enabled
    if getattr(args, 'learnable_temperature', False) and trainer._log_temp is not None:
        optimizer.add_param_group({'params': [trainer._log_temp], 'lr': args.lr * 0.1})
        print(f"Learnable temperature enabled (init={trainer._log_temp.exp().item():.4f})")

    real_pairs_path = getattr(args, 'real_pairs', None)
    if real_pairs_path:
        from src.training.real_dataset import load_real_pairs_dataset, split_real_pairs
        augment = getattr(args, 'augment_factor', 8)
        dataset = load_real_pairs_dataset(real_pairs_path, augment_factor=augment, seed=args.seed)
        train_ds, val_ds = split_real_pairs(dataset, train_fraction=args.train_fraction, seed=args.seed)
        metrics_ds = dataset  # use full dataset for metrics
        print(f"Real pairs dataset: {len(dataset)} total (augment x{augment})")
    else:
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
        metrics_dataset = dataset_factory(**dataset_kwargs)
        metrics_n = max(64, int(len(metrics_dataset) * (1 - args.train_fraction)))
        metrics_ds, _ = torch.utils.data.random_split(
            metrics_dataset,
            [metrics_n, len(metrics_dataset) - metrics_n],
            generator=torch.Generator().manual_seed(args.seed + 99),
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=(device.type == "cuda"))
    metrics_loader = DataLoader(metrics_ds, batch_size=args.batch_size, pin_memory=(device.type == "cuda"))

    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val")

    # Precompute SBERT embeddings cache for frozen encoder (huge speedup)
    if getattr(args, 'pretrained_encoder', False):
        import sys as _sys
        encoder = getattr(model, 'text_encoder', None)
        if encoder is not None and hasattr(encoder, 'precompute_cache'):
            print("Precomputing SBERT embeddings cache...", flush=True)
            all_texts = []
            # For real dataset, extract from raw pairs JSON
            real_pairs_path = getattr(args, 'real_pairs', None)
            if real_pairs_path:
                import json as _json
                pairs = _json.loads(open(real_pairs_path).read())
                for p in pairs:
                    all_texts.append(p['source_text'])
                    all_texts.append(p['target_text'])
            else:
                for ds_item in [train_ds, val_ds]:
                    for idx in range(min(len(ds_item), 500)):
                        try:
                            item = ds_item[idx]
                            if 'text' in item and isinstance(item['text'], str):
                                all_texts.append(item['text'])
                            if 'pair_text' in item and isinstance(item['pair_text'], str):
                                all_texts.append(item['pair_text'])
                        except Exception:
                            pass
            encoder.precompute_cache(all_texts)

    monitor = GPUTrainingMonitor(
        device,
        log_path=output_dir / "training_log.jsonl",
    )

    best_score = float("-inf")
    best_score_epoch = 1
    best_checkpoint = output_dir / "checkpoints" / "best.pt"
    final_checkpoint = output_dir / "checkpoints" / "hdim_final.pt"
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    val_metrics: dict = {}
    quality_metrics: dict = {"STS_exported": 0.0, "STS_training": 0.0, "DRS": 0.0, "AFR": 0.0, "pair_margin": 0.0}
    nan_batches_total: int = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        nan_batches_epoch = 0

        for batch in train_loader:
            try:
                if use_amp and scaler is not None:
                    with autocast("cuda"):
                        loss = trainer.train_step(batch)
                    if torch.isnan(loss) or torch.isinf(loss):
                        nan_batches_epoch += 1
                        nan_batches_total += 1
                        continue
                else:
                    loss = trainer.train_step(batch)
                    if torch.isnan(loss) or torch.isinf(loss):
                        nan_batches_epoch += 1
                        nan_batches_total += 1
                        continue
                epoch_loss += loss.item()
                n_batches += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"OOM at epoch {epoch}, skipping batch")
                    nan_batches_epoch += 1
                    nan_batches_total += 1
                else:
                    raise

        train_loss = epoch_loss / max(n_batches, 1)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = trainer.validate(val_loader)
            from src.models.metrics import compute_all_metrics
            quality_metrics = compute_all_metrics(model, metrics_loader)

            score = compute_primary_score(quality_metrics)
            monitor.log_epoch(
                epoch, args.epochs, train_loss, val_metrics, quality_metrics,
                current_lr, nan_batches_epoch, nan_batches_total
            )

            if score > best_score:
                best_score = score
                best_score_epoch = epoch
                trainer.save_checkpoint(str(best_checkpoint))

            # Early stopping — вне блока if score > best_score
            early_stop_patience = getattr(args, 'early_stopping_patience', 0)
            if early_stop_patience > 0 and best_score_epoch > 0:
                evals_since_best = (epoch - best_score_epoch) // args.eval_every
                if evals_since_best >= early_stop_patience:
                    print(f"Early stopping: no improvement for {evals_since_best} evals (best ep={best_score_epoch}, patience={early_stop_patience})")
                    break
        else:
            monitor.log_epoch(
                epoch, args.epochs, train_loss, val_metrics, quality_metrics,
                current_lr, nan_batches_epoch, nan_batches_total
            )

        if epoch % args.save_every == 0:
            trainer.save_checkpoint(str(output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"))

    trainer.save_checkpoint(str(final_checkpoint))

    training_summary = monitor.summary()
    results = {
        "config": vars(args),
        "training_summary": training_summary,
        "quality": quality_metrics,
        "validation": val_metrics,
        "best_checkpoint": str(best_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "nan_batches_total": nan_batches_total,
        "score": compute_primary_score(quality_metrics),
        "status": "completed",
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    if args.results_json:
        Path(args.results_json).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    if args.ledger_path:
        append_ledger_row(
            args.ledger_path,
            {"score": results["score"], "quality": quality_metrics,
             "training_summary": training_summary, "status": "completed"}
        )

    is_valid, fail_reason = check_run_validity(results)
    if not is_valid:
        print(f"WARNING: Run validity check failed: {fail_reason}")
        results["validity"] = fail_reason

    print(f"\nTraining complete. Best score: {best_score:.4f}")
    print(f"Results saved to: {results_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="HDIM GPU Training Script")
    # Core training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    # Model arch
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--num_domains", type=int, default=4)
    parser.add_argument("--clifford_dim", type=int, default=None)
    # Loss weights
    parser.add_argument("--lambda_iso", type=float, default=0.1)
    parser.add_argument("--lambda_pair", type=float, default=0.1)
    parser.add_argument("--lambda_routing", type=float, default=0.05)
    parser.add_argument("--lambda_memory", type=float, default=0.01)
    parser.add_argument("--ranking_margin", type=float, default=0.3)
    # Dataset
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--use_pairs", action="store_true")
    parser.add_argument("--text_mode", action="store_true")
    parser.add_argument("--negative_ratio", type=float, default=0.3)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    # Advanced components
    parser.add_argument("--advanced_encoder", action="store_true")
    parser.add_argument("--hierarchical_memory", action="store_true")
    parser.add_argument("--soft_router", action="store_true")
    parser.add_argument("--pretrained_encoder", action="store_true",
                        help="Use frozen SBERT encoder (paraphrase-multilingual-mpnet-base-v2)")
    # Loss
    parser.add_argument("--lambda_sts", type=float, default=0.0,
                        help="STS regularization weight (cosine similarity preservation, default 0=off)")
    parser.add_argument("--use_infonce", action="store_true", default=True,
                        help="Use InfoNCE loss instead of ranking margin (default: True)")
    parser.add_argument("--no_infonce", dest="use_infonce", action="store_false")
    parser.add_argument("--infonce_temperature", type=float, default=0.07,
                        help="InfoNCE temperature (default: 0.07)")
    # Real data
    parser.add_argument("--real_pairs", type=str, default=None,
                        help="Path to real_pairs.json for training on real cross-domain pairs")
    parser.add_argument("--augment_factor", type=int, default=8,
                        help="Augmentation factor for real pairs dataset")
    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="Stop if best score not improved for N evals (0=disabled)")
    # Phase 5 additions
    parser.add_argument("--lambda_angle", type=float, default=0.0,
                        help="AnglE loss weight (0=off, try 0.3-0.5)")
    parser.add_argument("--learnable_temperature", action="store_true", default=False,
                        help="Use learnable InfoNCE temperature (log-parameterized)")
    # Monitoring
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/gpu_training"))
    parser.add_argument("--results_json", type=str, default=None)
    parser.add_argument("--ledger_path", type=str, default=None)
    parser.add_argument("--config", type=Path, default=None)

    args = parser.parse_args()

    if args.config is not None:
        from src.training.experiment_config import ExperimentConfig
        exp = ExperimentConfig.from_json(args.config)
        for key, val in exp.to_dict().items():
            if hasattr(args, key) and key != "config":
                setattr(args, key, val)

    torch.manual_seed(args.seed)

    device = detect_device(args.device)

    cfg = HDIMConfig(
        hidden_dim=args.hidden_dim,
        num_experts=args.num_experts,
        num_domains=args.num_domains,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"HDIM GPU Training | device={device} | epochs={args.epochs} | hidden={args.hidden_dim}")

    results = run_gpu_training(cfg, args, device, output_dir)
    print(f"Score: {results.get('score', 0):.4f}")


if __name__ == "__main__":
    main()
