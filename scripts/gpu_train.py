#!/usr/bin/env python
"""
HDIM GPU Training Script с AMP, gradient checkpointing и live monitoring.

Функции:
  - Автоматическое определение GPU/CPU
  - Mixed Precision Training (AMP) для GPU
  - Gradient checkpointing для экономии памяти
  - Живой мониторинг через progress bar (tqdm) и json
  - Checkpoint сохранение каждые N эпох и по лучшей валидации
  - Поддержка SoftMoERouter и frozen SBERT encoder
  - Совместим с ExperimentConfig
  - Использует model_factory как единственный источник сборки моделей

Запуск:
  python scripts/gpu_train.py --pretrained_encoder --soft_router --use_pairs --amp --real_pairs data/real_pairs_v10.json --epochs 60 --device cuda
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
from src.models.model_factory import build_hdim_model, build_text_hdim_model, build_sbert_hdim_model, build_modernbert_hdim_model
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
    if getattr(args, 'modernbert_encoder', False):
        matryoshka_dims = None
        _mkr_str = getattr(args, 'modernbert_matryoshka_dims', None)
        if _mkr_str:
            matryoshka_dims = [int(d.strip()) for d in _mkr_str.split(',') if d.strip()]
        model = build_modernbert_hdim_model(
            cfg,
            soft_router=args.soft_router,
            modernbert_model_name=getattr(args, 'modernbert_model_name', 'answerdotai/ModernBERT-base'),
            freeze_modernbert=getattr(args, 'freeze_modernbert', True),
            use_cls_pooling=getattr(args, 'modernbert_use_cls_pooling', True),
            max_length=getattr(args, 'modernbert_max_length', 512),
            matryoshka_dims=matryoshka_dims,
        )
        print("Components: ModernBERT(answerdotai/ModernBERT-base) + Linear Projection")
        if matryoshka_dims:
            print(f"  + Matryoshka dims: {matryoshka_dims}")
        if args.soft_router:
            print("  + SoftMoERouter")
        return model

    if getattr(args, 'pretrained_encoder', False):
        unfreeze_layers = None
        _unfreeze_str = getattr(args, 'unfreeze_sbert_layers', None)
        if _unfreeze_str:
            unfreeze_layers = [s.strip() for s in _unfreeze_str.split(',') if s.strip()]
        _freeze_frac = getattr(args, 'freeze_sbert_bottom_frac', None)
        model = build_sbert_hdim_model(
            cfg,
            soft_router=args.soft_router,
            unfreeze_layers=unfreeze_layers,
            freeze_bottom_frac=_freeze_frac,
            projection_hidden=getattr(args, 'sbert_projection_hidden', None),
        )
        print("Components: SBERT(paraphrase-multilingual-mpnet-base-v2) + SimpleMLP")
        if unfreeze_layers:
            print(f"  + Partial SBERT unfreeze: {unfreeze_layers}")
        if _freeze_frac is not None:
            print(f"  + Freeze bottom {_freeze_frac*100:.0f}% SBERT layers")
        if args.soft_router:
            print("  + SoftMoERouter")
        return model

    if args.soft_router:
        model = build_text_hdim_model(
            cfg,
            soft_router=True,
        )
        print("Components: SimpleTextEncoder + SoftMoERouter")
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
        val_metrics: Optional[dict],
        quality_metrics: Optional[dict],
        lr: float,
        nan_batches_epoch: int = 0,
        nan_batches_total: int = 0,
    ) -> None:
        # A6 FIX: accept None for val_metrics/quality_metrics on non-eval epochs
        val_metrics = val_metrics or {}
        quality_metrics = quality_metrics or {}
        elapsed = time.time() - self.start_time
        gpu_stats = self.gpu_stats()
        score = compute_primary_score(quality_metrics)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics.get("loss_total", float("nan")), 6) if val_metrics else None,
            "loss_memory": round(val_metrics.get("loss_memory", float("nan")), 6) if val_metrics else None,
            "loss_routing": round(val_metrics.get("loss_routing", float("nan")), 6) if val_metrics else None,
            "pair_margin": round(quality_metrics.get("pair_margin", float("nan")), 6) if quality_metrics else None,
            "STS_exported": round(quality_metrics.get("STS_exported", float("nan")), 6) if quality_metrics else None,
            "score": round(score, 6) if quality_metrics else None,
            "lr": lr,
            "elapsed_s": round(elapsed, 1),
            "nan_batches_epoch": nan_batches_epoch,
            "nan_batches_total": nan_batches_total,
            **gpu_stats,
        }
        self.history.append(row)

        if val_metrics and quality_metrics:
            msg = (
                f"Epoch {epoch:3d}/{total_epochs} "
                f"| train={train_loss:.4f} "
                f"| val={val_metrics.get('loss_total', 0):.4f} "
                f"| margin={quality_metrics.get('pair_margin', 0):.4f} "
                f"| STS={quality_metrics.get('STS_exported', 0):.4f} "
                f"| score={score:.4f}"
            )
        else:
            msg = (
                f"Epoch {epoch:3d}/{total_epochs} "
                f"| train={train_loss:.4f} "
                f"| lr={lr:.6f}"
            )
        if nan_batches_epoch > 0:
            msg += f" | NaN={nan_batches_epoch}"
        if gpu_stats:
            msg += f" | GPU={gpu_stats.get('gpu_allocated_gb', 0):.2f}GB"
        print(msg)

        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")

    def summary(self) -> dict:
        if not self.history:
            return {}
        scored = [x for x in self.history if x.get("score") is not None]
        if not scored:
            return {}
        best = max(scored, key=lambda x: x["score"])
        return {
            "best_epoch": best["epoch"],
            "best_pair_margin": best["pair_margin"],
            "best_STS_exported": best["STS_exported"],
            "best_score": best["score"],
            "total_epochs": len(self.history),
            "total_time_s": round(time.time() - self.start_time, 1),
        }



def _build_scheduler(optimizer, args, total_steps: int):
    """Выбирает LR scheduler по args.scheduler_type.

    - cosine_restarts: CosineAnnealingWarmRestarts — периодические рестарты LR
      (хорошо для многомодальных loss landscapes, риск дёрганья метрик)
    - cosine_decay: CosineAnnealingLR — монотонное снижение до eta_min
      (стабильнее, лучше если есть early stopping)
    - plateau: ReduceLROnPlateau — снижение при стагнации val_score
      (адаптивный, вызывать scheduler.step(score) вместо scheduler.step())
    - onecycle: OneCycleLR — warmup + decay за один цикл
      (быстрая сходимость, хорошо для малого числа эпох)
    """
    stype = getattr(args, "scheduler_type", "cosine_restarts")
    lr = args.lr
    epochs = args.epochs

    if stype == "cosine_restarts":
        # T_0 in STEPS — Phase 8e record: T_0=20 epochs * steps_per_epoch
        # Cycles: 20ep -> 40ep -> 80ep. Best score on cycle 3 (ep45-60).
        steps_per_epoch = max(1, total_steps // max(1, epochs))
        T_0_epochs = max(getattr(args, "warmup_epochs", 20), 15)
        T_0 = T_0_epochs * steps_per_epoch
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=getattr(args, "t_mult", 2),
            eta_min=lr * 1e-3,
        ), False  # (scheduler, needs_score)
    elif stype == "cosine_decay":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr * 1e-3,
        ), False
    elif stype == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=10,
            min_lr=lr * 1e-3,
        ), True  # needs_score=True: вызывать step(score)
    elif stype == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr * 3,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        ), False
    else:
        raise ValueError(f"Unknown scheduler_type: {stype}")


def freeze_sbert_bottom_half(model):
    """Freeze bottom 50% of SBERT encoder layers."""
    sbert = None
    for name, module in model.named_modules():
        if hasattr(module, 'encoder') and hasattr(module.encoder, 'layer'):
            sbert = module
            break
        elif 'sbert' in name.lower() or 'text_encoder' in name.lower():
            sbert = module
            break
    
    if sbert is None:
        print("WARNING: Could not find SBERT encoder for freezing")
        return
    
    # Freeze embeddings
    if hasattr(sbert, 'embeddings'):
        for param in sbert.embeddings.parameters():
            param.requires_grad = False
    
    # Freeze bottom 50% of transformer layers
    if hasattr(sbert.encoder, 'layer'):
        total_layers = len(sbert.encoder.layer)
        freeze_layers = total_layers // 2
        print(f"Freezing bottom {freeze_layers}/{total_layers} SBERT layers")
        
        for i in range(freeze_layers):
            for param in sbert.encoder.layer[i].parameters():
                param.requires_grad = False
    
    # Count trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} total, {trainable_params:,} trainable ({trainable_params/total_params*100:.1f}%)")


def run_gpu_training(
    cfg: HDIMConfig,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Основной цикл GPU обучения."""

    model = _build_model(cfg, args)
    model = model.to(device)
    
    # Freeze bottom 50% SBERT encoder if requested
    if getattr(args, 'freeze_sbert', False):
        freeze_sbert_bottom_half(model)

    # Enable gradient checkpointing if requested
    if getattr(args, 'gradient_checkpointing', False):
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
            print("Gradient checkpointing: ENABLED (memory + MoE paths)")
        else:
            print("WARNING: gradient_checkpointing flag set but model doesn't support it")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Build param groups: separate LR for unfrozen SBERT layers
    _unfreeze_str = getattr(args, 'unfreeze_sbert_layers', None)
    _freeze_frac = getattr(args, 'freeze_sbert_bottom_frac', None)
    _sbert_lr = getattr(args, 'sbert_lr', 1e-5)
    _has_sbert = hasattr(model, 'text_encoder') and hasattr(model.text_encoder, '_sbert')
    if _has_sbert and (_unfreeze_str or _freeze_frac is not None):
        sbert_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'text_encoder._sbert' in name:
                sbert_params.append(param)
            else:
                other_params.append(param)
        wd = getattr(args, 'weight_decay', 1e-4)
        param_groups = [
            {'params': other_params, 'lr': args.lr, 'weight_decay': wd},
            {'params': sbert_params, 'lr': _sbert_lr, 'weight_decay': wd * 100},
        ]
        print(f"Optimizer: {len(other_params)} HDIM params (lr={args.lr}), {len(sbert_params)} SBERT params (lr={_sbert_lr})")
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=getattr(args, 'weight_decay', 1e-4),
        betas=(0.9, 0.999),
    )

    use_amp = (device.type == "cuda") and args.amp
    scaler = GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Using Automatic Mixed Precision (AMP)")

    total_steps = args.epochs * max(1, getattr(args, 'num_samples', 500) // args.batch_size)
    scheduler, scheduler_needs_score = _build_scheduler(optimizer, args, total_steps)

    trainer = HDIMTrainer(
        model, optimizer,
        device=device,
        lambda_iso=args.lambda_iso,
        lambda_pair=args.lambda_pair,
        lambda_routing=args.lambda_routing,
        lambda_memory=args.lambda_memory,
        ranking_margin=args.ranking_margin,
        use_infonce=getattr(args, 'use_infonce', True),
        infonce_temperature=getattr(args, 'infonce_temperature', 0.15),
        lambda_sts=getattr(args, 'lambda_sts', 0.0),
        lambda_angle=getattr(args, 'lambda_angle', 0.0),
        lambda_supcon=getattr(args, 'lambda_supcon', 0.0),
        lambda_z=getattr(args, 'lambda_z', 0.0),
        lambda_expert_ortho=getattr(args, 'lambda_expert_ortho', 0.0),
        learnable_temperature=getattr(args, 'learnable_temperature', False),
        lambda_dcl=getattr(args, 'lambda_dcl', 0.0),
        lambda_uniformity=getattr(args, 'lambda_uniformity', 0.0),
        lambda_diversity_var=getattr(args, 'lambda_diversity_var', 0.01),
        lambda_diversity_ortho=getattr(args, 'lambda_diversity_ortho', 0.005),
        lambda_matryoshka=getattr(args, 'lambda_matryoshka', 0.1),
    )
    # Focal-InfoNCE
    _focal_gamma = getattr(args, 'focal_gamma', 1.0)
    trainer._focal_gamma = _focal_gamma
    if _focal_gamma < 1.0:
        print(f"Focal-InfoNCE: gamma={_focal_gamma}")
    # Temperature scheduling
    _temp_schedule = getattr(args, 'temp_schedule', 'none')
    if _temp_schedule != 'none':
        trainer._temp_schedule = _temp_schedule
        trainer._tau_max = getattr(args, 'tau_max', 0.1)
        trainer._tau_min = getattr(args, 'tau_min', 0.01)
        print(f"Temperature schedule: {_temp_schedule} (tau_max={trainer._tau_max}, tau_min={trainer._tau_min})")
    # Attach hard negative mining flag
    trainer.use_hard_negatives = getattr(args, 'use_hard_negatives', False)
    if trainer.use_hard_negatives:
        print("Hard Negative Mining: ENABLED")
    # Phase 22: SC-InfoNCE cluster temperature
    if getattr(args, 'sc_temperature', False):
        trainer.use_sc_temperature = True
        print("SC-InfoNCE cluster temperature: ENABLED")
    # Phase 22: Enable model-level flags
    _p22_flags = []
    if getattr(args, 'gradient_surprise', False) and hasattr(model, 'enable_gradient_surprise'):
        model.enable_gradient_surprise()
        _p22_flags.append("gradient_surprise")
    if getattr(args, 'adaptive_forgetting', False) and hasattr(model, 'enable_adaptive_forgetting'):
        model.enable_adaptive_forgetting()
        _p22_flags.append("adaptive_forgetting")
    if getattr(args, 'learnable_metric', False) and hasattr(model, 'enable_learnable_metric'):
        model.enable_learnable_metric()
        _p22_flags.append("learnable_metric")
    if _p22_flags:
        print(f"Phase 22 features: {', '.join(_p22_flags)}")
    # Phase 26: MoE Expert features
    _p26_flags = []
    if getattr(args, 'shared_expert', False) and hasattr(model, 'enable_shared_expert'):
        model.enable_shared_expert()
        _p26_flags.append("shared_expert")
    if getattr(args, 'aux_loss_free', False) and hasattr(model, 'enable_aux_loss_free'):
        model.enable_aux_loss_free(aux_lr=getattr(args, 'aux_lr', 0.001))
        _p26_flags.append("aux_loss_free")
    if getattr(args, 'expert_ortho', False) and hasattr(model, 'enable_expert_ortho'):
        model.enable_expert_ortho()
        _p26_flags.append("expert_ortho")
    if _p26_flags:
        print(f"Phase 26 features: {', '.join(_p26_flags)}")
    # Add learnable temperature to optimizer (must be before scheduler)
    if getattr(args, 'learnable_temperature', False) and trainer._log_temp is not None:
        temp_lr_mult = getattr(args, 'temperature_lr_mult', 0.1)
        optimizer.add_param_group({'params': [trainer._log_temp], 'lr': args.lr * temp_lr_mult})
        print(f"Learnable temperature enabled (init={trainer._log_temp.exp().item():.4f})")
    real_pairs_path = getattr(args, 'real_pairs', None)
    if real_pairs_path:
        from src.training.real_dataset import load_real_pairs_dataset, split_real_pairs
        augment = getattr(args, 'augment_factor', 8)
        dataset = load_real_pairs_dataset(
            real_pairs_path,
            augment_factor=augment,
            seed=args.seed,
            add_negatives=True,
            negative_ratio=1.0,
        )
        train_ds, val_ds = split_real_pairs(dataset, train_fraction=args.train_fraction, seed=args.seed)
        metrics_ds = val_ds  # use held-out val split for honest metrics
        n_pos = sum(1 for it in dataset._items if it['relation'] == 'positive')
        n_neg = sum(1 for it in dataset._items if it['relation'] == 'negative')
        print(f"Real pairs dataset: {len(dataset)} total (augment x{augment}, pos={n_pos}, neg={n_neg})")
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

    # HDIM_NUM_WORKERS=0 отключает workers (используется auto_tune для экономии RAM)
    import os as _os
    _nw_env = _os.environ.get("HDIM_NUM_WORKERS")
    if _nw_env is not None:
        _num_workers = int(_nw_env)
    else:
        _num_workers = min(4, _os.cpu_count() or 1) if device.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == "cuda"), num_workers=_num_workers, persistent_workers=(_num_workers > 0), prefetch_factor=(2 if _num_workers > 0 else None))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=(device.type == "cuda"), num_workers=_num_workers, persistent_workers=(_num_workers > 0), prefetch_factor=(2 if _num_workers > 0 else None))
    metrics_loader = DataLoader(metrics_ds, batch_size=args.batch_size, pin_memory=(device.type == "cuda"), num_workers=_num_workers, persistent_workers=(_num_workers > 0), prefetch_factor=(2 if _num_workers > 0 else None))
    print(f"DataLoader: num_workers={_num_workers}, pin_memory={device.type == chr(99)+chr(117)+chr(100)+chr(97)}")

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
                pairs = _json.loads(open(real_pairs_path, encoding="utf-8").read())
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
            # Phase 25: log cache size after precomputation
            _cache = getattr(encoder, '_embedding_cache', None)
            if _cache is not None:
                print(f"SBERT cache active: {len(_cache)} embeddings preloaded", flush=True)
            else:
                print(f"SBERT cache precomputed for {len(all_texts)} texts", flush=True)

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
    consecutive_nan: int = 0

    for epoch in range(1, args.epochs + 1):
        trainer.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        nan_batches_epoch = 0

        for batch in train_loader:
            try:
                if use_amp and scaler is not None:
                    with autocast("cuda"):
                        loss = trainer.train_step(batch, scaler=scaler)
                else:
                    loss = trainer.train_step(batch)
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_batches_epoch += 1
                    nan_batches_total += 1
                    consecutive_nan += 1
                    if consecutive_nan >= 3:
                        old_lr = optimizer.param_groups[0]["lr"]
                        new_lr = old_lr * 0.5
                        for pg in optimizer.param_groups:
                            pg["lr"] = new_lr
                        consecutive_nan = 0
                        print(f"  NaN recovery: LR reduced from {old_lr:.2e} to {new_lr:.2e}")
                    continue
                consecutive_nan = 0
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

        # Log loss breakdown if available from trainer
        loss_breakdown = getattr(trainer, '_last_loss_components', None)
        if loss_breakdown and epoch % args.eval_every == 0:
            breakdown_str = " | ".join(
                f"{k}={v:.4f}" for k, v in loss_breakdown.items()
                if isinstance(v, (int, float)) and v != 0
            )
            if breakdown_str:
                print(f"  Loss breakdown: {breakdown_str}")

        if scheduler_needs_score:
            # ReduceLROnPlateau: нужен score (лучше вызывать после eval)
            _sched_score = compute_primary_score(quality_metrics) if quality_metrics.get("pair_margin", 0) > 0 else 0.0
            scheduler.step(_sched_score)
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            try:
                current_lr = scheduler.get_last_lr()[0]
            except Exception:
                current_lr = optimizer.param_groups[0]["lr"]
            # A2 FIX (upgraded): detect LR restart (CosineWarmRestarts)
            # Use 'stabilize' instead of hard reset — preserves memory patterns,
            # only normalizes momentum to prevent exploding gradients after LR spike.
            # Hard reset would cause score drop at each restart cycle.
            if current_lr > prev_lr * 1.5 and hasattr(model, 'reset_memory'):
                model.reset_memory(strategy='stabilize')
                print(f"[LR restart ep{epoch}] Memory stabilized. LR: {prev_lr:.6f} -> {current_lr:.6f}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = trainer.validate(val_loader)
            quality_metrics = compute_all_metrics(model, metrics_loader)

            score = compute_primary_score(quality_metrics)
            monitor.log_epoch(
                epoch, args.epochs, train_loss, val_metrics, quality_metrics,
                current_lr, nan_batches_epoch, nan_batches_total
            )

            if score > best_score:
                best_score = score
                best_score_epoch = epoch
                trainer.save_checkpoint(str(best_checkpoint), scaler=scaler)

            # Early stopping
            early_stop_patience = getattr(args, 'early_stopping_patience', 0)
            if early_stop_patience > 0 and best_score_epoch > 0:
                evals_since_best = (epoch - best_score_epoch) // args.eval_every
                if evals_since_best >= early_stop_patience:
                    print(f"Early stopping: no improvement for {evals_since_best} evals (best ep={best_score_epoch}, patience={early_stop_patience})")
                    break
        else:
            # A6 FIX: log only real train-time metrics — no fake zeros for score/margin/STS
            monitor.log_epoch(
                epoch, args.epochs, train_loss,
                None,   # val_metrics not available on non-eval epochs
                None,   # quality_metrics not available on non-eval epochs
                current_lr, nan_batches_epoch, nan_batches_total
            )

        if epoch % args.save_every == 0:
            trainer.save_checkpoint(str(output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"), scaler=scaler)

    trainer.save_checkpoint(str(final_checkpoint), scaler=scaler)

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
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for AdamW optimizer (default: 1e-4)")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override model dropout probability (default: use HDIMConfig default)")
    parser.add_argument("--warmup_epochs", type=int, default=20,
                        help="T_0 for cosine_restarts in epochs (default: 20, Phase8e record)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce activation memory")
    # Model arch
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--num_domains", type=int, default=4)
    parser.add_argument("--clifford_p", type=int, default=4,
                        help="Clifford algebra positive bases (default=4, Cl(4,1,0) dim=32)")
    parser.add_argument("--clifford_q", type=int, default=1,
                        help="Clifford algebra negative bases (default=1)")
    parser.add_argument("--clifford_r", type=int, default=0,
                        help="Clifford algebra nilpotent bases (default=0)")
    # Loss weights
    parser.add_argument("--lambda_iso", type=float, default=0.1)
    parser.add_argument("--lambda_pair", type=float, default=0.1)
    parser.add_argument("--lambda_routing", type=float, default=0.05)
    parser.add_argument("--lambda_z", type=float, default=0.0,
                        help="Router z-loss weight (ST-MoE stability, default=0=disabled)")
    parser.add_argument("--lambda_dcl", type=float, default=0.0,
                        help="DCL loss weight (Decoupled Contrastive, Yeh et al. 2022, try 0.3-0.5)")
    parser.add_argument("--lambda_uniformity", type=float, default=0.0,
                        help="Uniformity+Alignment loss weight (Wang & Isola 2020, try 0.1-0.3)")
    parser.add_argument("--lambda_memory", type=float, default=0.01)
    parser.add_argument("--lambda_diversity_var", type=float, default=0.01,
                        help="Diversity variance weight (anti-collapse)")
    parser.add_argument("--lambda_diversity_ortho", type=float, default=0.005,
                        help="Diversity orthogonality weight (anti-collapse)")
    parser.add_argument("--lambda_matryoshka", type=float, default=0.1,
                        help="Matryoshka multi-scale loss weight")
    parser.add_argument("--ranking_margin", type=float, default=0.3)
    # Dataset
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--use_pairs", action="store_true")
    parser.add_argument("--negative_ratio", type=float, default=0.3)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    # Advanced components
    parser.add_argument("--soft_router", action="store_true")
    parser.add_argument("--pretrained_encoder", action="store_true",
                        help="Use frozen SBERT encoder (paraphrase-multilingual-mpnet-base-v2)")
    parser.add_argument("--modernbert_encoder", action="store_true",
                        help="Use ModernBERT encoder (answerdotai/ModernBERT-base)")
    parser.add_argument("--modernbert_model_name", type=str, default="answerdotai/ModernBERT-base",
                        help="ModernBERT model name (default: answerdotai/ModernBERT-base)")
    parser.add_argument("--freeze_modernbert", action="store_true", default=True,
                        help="Freeze ModernBERT weights (default: True)")
    parser.add_argument("--no_freeze_modernbert", dest="freeze_modernbert", action="store_false",
                        help="Allow ModernBERT fine-tuning")
    parser.add_argument("--modernbert_use_cls_pooling", action="store_true", default=True,
                        help="Use CLS token pooling for ModernBERT (default: True)")
    parser.add_argument("--modernbert_max_length", type=int, default=512,
                        help="Max sequence length for ModernBERT tokenizer (default: 512)")
    parser.add_argument("--modernbert_matryoshka_dims", type=str, default=None,
                        help="Comma-separated Matryoshka output dims (e.g. '64,128,256,768')")
    parser.add_argument("--unfreeze_sbert_layers", type=str, default=None,
                        help="Comma-separated SBERT layer names to unfreeze (e.g. '10,11,pooling')")
    parser.add_argument("--sbert_projection_hidden", type=int, default=None,
                        help="Hidden dim for SBERT projection MLP (default: max(hidden_dim, 384))")
    parser.add_argument("--use_hard_negatives", action="store_true", default=False,
                        help="Enable online hard negative mining in InfoNCE loss")
    parser.add_argument("--sbert_lr", type=float, default=1e-5,
                        help="LR for unfrozen SBERT layers (default: 1e-5)")
    parser.add_argument("--freeze_sbert_bottom_frac", type=float, default=None,
                        help="Freeze bottom N%% of SBERT transformer layers (e.g. 0.5 = freeze bottom 50%%)")
    parser.add_argument("--freeze_sbert", action="store_true",
                        help="Freeze bottom 50%% of SBERT encoder (standalone freezing without build_sbert)")
    # Loss
    parser.add_argument("--lambda_sts", type=float, default=0.0,
                        help="STS regularization weight (cosine similarity preservation, default 0=off)")
    parser.add_argument("--use_infonce", action="store_true", default=True,
                        help="Use InfoNCE loss instead of ranking margin (default: True)")
    parser.add_argument("--no_infonce", dest="use_infonce", action="store_false")
    parser.add_argument("--infonce_temperature", type=float, default=0.15,
                        help="InfoNCE temperature (default: 0.15)")
    # Real data
    parser.add_argument("--real_pairs", type=str, default=None,
                        help="Path to real_pairs.json for training on real cross-domain pairs")
    parser.add_argument("--augment_factor", type=int, default=8,
                        help="Augmentation factor for real pairs dataset")
    # Embedding augmentations
    parser.add_argument("--aug_noise_std", type=float, default=0.0,
                        help="Gaussian noise std for embedding augmentation (0.0=disabled)")
    parser.add_argument("--aug_mixup_alpha", type=float, default=0.0,
                        help="Mixup Beta alpha for embedding augmentation (0.0=disabled)")
    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="Stop if best score not improved for N evals (0=disabled)")
    # Phase 5 additions
    parser.add_argument("--lambda_angle", type=float, default=0.0,
                        help="AnglE loss weight (0=off, try 0.3-0.5)")
    parser.add_argument("--lambda_supcon", type=float, default=0.0,
                        help="SupCon loss weight (Supervised Contrastive, нужен pair_group_id, try 0.1-0.5)")
    parser.add_argument("--scheduler_type", default="cosine_restarts",
                        choices=["cosine_restarts", "cosine_decay", "plateau", "onecycle"],
                        help="LR scheduler type (default: cosine_restarts)")
    parser.add_argument("--t_mult", type=int, default=2,
                        help="T_mult for cosine_restarts scheduler (default: 2, use 1 for stable LR)")
    parser.add_argument("--learnable_temperature", action="store_true", default=False,
                        help="Use learnable InfoNCE temperature (log-parameterized)")
    parser.add_argument("--temperature_lr_mult", type=float, default=0.1,
                        help="LR multiplier for learnable temperature param (default: 0.1)")
    # Focal-InfoNCE (Hou & Li, EMNLP 2023)
    parser.add_argument("--focal_gamma", type=float, default=1.0,
                        help="Focal-InfoNCE gamma (1.0=standard, 0.5=moderate, <1=focus on hard negatives)")
    # Phase 22: Gradient-based surprise memory (Titans, NeurIPS 2025)
    parser.add_argument("--gradient_surprise", action="store_true", default=False,
                        help="Use gradient-based surprise metric in Titans memory (Titans 2025)")
    parser.add_argument("--adaptive_forgetting", action="store_true", default=False,
                        help="Adaptive forgetting based on surprise (high surprise = less forgetting)")
    # Phase 22: Learnable Clifford metric (CliffordNet, 2026)
    parser.add_argument("--learnable_metric", action="store_true", default=False,
                        help="Learnable per-blade metric scaling in Clifford algebra")
    # Phase 22: SC-InfoNCE cluster temperature (Cheng et al., Nov 2025)
    parser.add_argument("--sc_temperature", action="store_true", default=False,
                        help="SC-InfoNCE cluster-aware temperature scaling")
    # Phase 26: MoE Expert features
    parser.add_argument("--shared_expert", action="store_true", default=False,
                        help="Enable DeepSeek-V3 always-on shared expert in SoftMoERouter")
    parser.add_argument("--aux_loss_free", action="store_true", default=False,
                        help="Enable Auxiliary-Loss-Free load balancing (DeepSeek-V3)")
    parser.add_argument("--aux_lr", type=float, default=0.001,
                        help="Bias adjustment rate for auxiliary-loss-free balancing")
    parser.add_argument("--expert_ortho", action="store_true", default=False,
                        help="Enable expert orthogonalization loss (arXiv:2505.22323)")
    parser.add_argument("--lambda_expert_ortho", type=float, default=0.0,
                        help="Weight for expert orthogonalization loss (try 0.01-0.05)")
    parser.add_argument("--memory_type", type=str, default="titans",
                        choices=["titans", "hippocampus", "neocortex", "cls", "hbma"],
                        help="Memory module type (default: titans)")
    # Temperature scheduling
    parser.add_argument("--temp_schedule", type=str, default="none",
                        choices=["none", "warm_restart"],
                        help="Temperature scheduling strategy (default: none)")
    parser.add_argument("--tau_max", type=float, default=0.1,
                        help="Max temperature for warm_restart schedule")
    parser.add_argument("--tau_min", type=float, default=0.01,
                        help="Min temperature for warm_restart schedule")
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

    _dropout_override = getattr(args, 'dropout', None)
    cfg = HDIMConfig(
        hidden_dim=args.hidden_dim,
        num_experts=args.num_experts,
        num_domains=args.num_domains,
        dropout=_dropout_override if _dropout_override is not None else 0.1,
        clifford_p=args.clifford_p,
        clifford_q=args.clifford_q,
        clifford_r=args.clifford_r,
        memory_type=getattr(args, 'memory_type', 'titans'),
    )
    clifford_dim = 2 ** (args.clifford_p + args.clifford_q + args.clifford_r)
    print(f"Clifford algebra: Cl({args.clifford_p},{args.clifford_q},{args.clifford_r}) dim={clifford_dim}")
    if _dropout_override is not None:
        print(f"Override dropout: {_dropout_override}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"HDIM GPU Training | device={device} | epochs={args.epochs} | hidden={args.hidden_dim}")

    results = run_gpu_training(cfg, args, device, output_dir)
    print(f"Score: {results.get('score', 0):.4f}")


if __name__ == "__main__":
    main()
