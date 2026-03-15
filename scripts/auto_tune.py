#!/usr/bin/env python
"""
HDIM Auto-Tuner v22 — Optuna hyperparameter search for Phase 22 ModernBERT config.

Запуск:
  python scripts/auto_tune.py                        # 20 trials, 30 эпох каждый (SBERT)
  python scripts/auto_tune.py --modernbert_encoder    # 20 trials с ModernBERT
  python scripts/auto_tune.py --modernbert_encoder --n_trials 50 --epochs 50
  python scripts/auto_tune.py --resume               # продолжить прерванное исследование
  python scripts/auto_tune.py --data v8 --n_trials 30 # использовать v8 данные (330 пар)
  python scripts/auto_tune.py --phase quick           # быстрый 15-эпох прогон
  python scripts/auto_tune.py --phase deep            # глубокий 60-эпох прогон

Phase 22 ModernBERT improvements (all active):
  - ModernBERT encoder (替代 SBERT)
  - Gated Memory Fusion (learned gate before memory addition)
  - Expert Dropout (0.1) in MoE experts
  - Similarity-Preserving Router (ICLR 2026)
  - Gradient Isolation for Memory (.detach() on retrieved)
  - Precomputed Clifford signs (performance fix)
  - DCL + Uniformity + Alignment losses
  - Focal-InfoNCE (gamma)
  - Router Z-Loss (lambda_z >= 0.005)
  - Learnable temperature with checkpoint persistence
  - Matryoshka Representation Learning (MRL)

Optuna подбирает:
  - lr, batch_size, lambda_pair, lambda_angle, lambda_sts
  - lambda_dcl, lambda_uniformity, lambda_z
  - lambda_iso, lambda_routing, lambda_memory
  - infonce_temperature, focal_gamma
  - augment_factor, warmup_epochs
  - data_version (v5 vs v8)
  - modernbert_model_name, modernbert_pooling
  - modernbert_max_length, modernbert_matryoshka_dims
  - hidden_dim (64 vs 128 vs 256)

Зафиксировано:
  - num_experts=4 (Phase8e рекорд)
  - soft_router, t_mult=2
  - simple MLP projection (рекордная архитектура)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TRAIN_SCRIPT = str(REPO_ROOT / "scripts" / "gpu_train.py")
STUDY_DB = str(REPO_ROOT / "artifacts" / "optuna_study.db")
STUDY_NAME = "hdim_autotune"

# Phase presets
PHASE_PRESETS = {
    "micro": {"epochs": 5, "eval_every": 2, "early_stop": 3},
    "quick": {"epochs": 10, "eval_every": 3, "early_stop": 5},
    "standard": {"epochs": 30, "eval_every": 5, "early_stop": 15},
    "deep": {"epochs": 60, "eval_every": 5, "early_stop": 25},
}

DATA_VERSIONS = {
    "v5": "data/real_pairs_v5.json",  # 175 пар — Phase8e рекорд
    "v7": "data/real_pairs_v7.json",  # 232 пары
    "v8": "data/real_pairs_v8.json",  # 330 пар — 35.8% neg, все домены
}


def objective(
    trial: optuna.Trial,
    epochs: int = 30,
    data_version: str = "v5",
    modernbert: bool = False,
) -> float:
    """Один trial: запустить обучение с предложенными параметрами, вернуть score."""

    # === Данные ===
    data_file = DATA_VERSIONS.get(data_version, DATA_VERSIONS["v5"])

    # === Параметры для подбора ===

    # Learning rate — лог-шкала
    lr = trial.suggest_float("lr", 5e-5, 2e-3, log=True)

    # Batch size — 32 рекорд, но пробуем больше
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 48, 64])

    # Augment factor — v5 рекорд=30, но зависит от данных
    max_augment = 50 if data_version == "v5" else 30
    augment_factor = trial.suggest_int("augment_factor", 5, max_augment, step=5)

    # === Loss weights (Phase 22 ModernBERT / Phase 21 SOTA) ===

    # Pair loss — основной contrastive signal
    lambda_pair = trial.suggest_float("lambda_pair", 0.2, 0.6, step=0.05)

    # AnglE loss — угол между векторами
    lambda_angle = trial.suggest_float("lambda_angle", 0.1, 0.5, step=0.05)

    # STS export loss — STS benchmark alignment
    lambda_sts = trial.suggest_float("lambda_sts", 0.05, 0.35, step=0.05)

    # Isomorphism loss — domain invariant extraction
    lambda_iso = trial.suggest_float("lambda_iso", 0.05, 0.2, step=0.025)

    # Routing loss — MoE load balancing
    lambda_routing = trial.suggest_float("lambda_routing", 0.02, 0.1, step=0.01)

    # Memory loss — Titans memory regularization
    lambda_memory = trial.suggest_float("lambda_memory", 0.005, 0.03, step=0.005)

    # DCL — Decoupled Contrastive Loss (NeurIPS 2022)
    lambda_dcl = trial.suggest_float("lambda_dcl", 0.0, 0.5, step=0.05)

    # Uniformity + Alignment (Wang & Isola 2020)
    lambda_uniformity = trial.suggest_float("lambda_uniformity", 0.0, 0.3, step=0.025)

    # Router Z-Loss — ОБЯЗАТЕЛЬНО >= 0.005 для num_experts=4
    lambda_z = trial.suggest_float("lambda_z", 0.005, 0.03, step=0.005)

    # === Temperature & Focal ===
    temperature = trial.suggest_float("infonce_temperature", 0.05, 0.25, log=True)
    focal_gamma = trial.suggest_float("focal_gamma", 0.2, 1.0, step=0.1)

    # === Scheduler ===
    # t_mult=1 исключён — Phase 12 anti-pattern (деградация на каждом рестарте)
    t_mult = trial.suggest_categorical("t_mult", [2])
    warmup_epochs = trial.suggest_int("warmup_epochs", 2, 5)

    # === ModernBERT-specific parameters ===
    if modernbert:
        # Model name — выбор между размерами ModernBERT
        modernbert_model_name = trial.suggest_categorical(
            "modernbert_model_name",
            ["answerdotai/ModernBERT-base", "answerdotai/ModernBERT-large"],
        )
        # Pooling strategy
        modernbert_pooling = trial.suggest_categorical(
            "modernbert_pooling", ["cls", "mean"]
        )
        # Max sequence length
        modernbert_max_length = trial.suggest_categorical(
            "modernbert_max_length", [128, 256, 512, 1024]
        )
        # Matryoshka dimensions (comma-separated)
        use_matryoshka = trial.suggest_categorical("use_matryoshka", [True, False])
        if use_matryoshka:
            matryoshka_dims = trial.suggest_categorical(
                "modernbert_matryoshka_dims", ["64,128,256", "128,256,512"]
            )
        else:
            matryoshka_dims = None
        # Freeze ModernBERT encoder
        freeze_modernbert = True  # Замораживаем для скорости
        # Hidden dim — ModernBERT может работать с меньшими hidden_dim
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    else:
        # SBERT defaults
        modernbert_model_name = None
        modernbert_pooling = None
        modernbert_max_length = None
        matryoshka_dims = None
        freeze_modernbert = False
        hidden_dim = 256  # Phase8e рекорд

    # === Output ===
    out_dir = REPO_ROOT / "artifacts" / f"optuna_{trial.number:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_every = max(1, epochs // 4)
    early_stop = max(5, epochs // 2)

    cmd = [
        PYTHON,
        TRAIN_SCRIPT,
        "--epochs",
        str(epochs),
        "--hidden_dim",
        str(hidden_dim),
        "--num_experts",
        "4",
        "--num_domains",
        "4",
        "--soft_router",
        "--real_pairs",
        str(REPO_ROOT / data_file),
        "--augment_factor",
        str(augment_factor),
        "--lambda_pair",
        str(lambda_pair),
        "--lambda_angle",
        str(lambda_angle),
        "--lambda_sts",
        str(lambda_sts),
        "--lambda_iso",
        str(lambda_iso),
        "--lambda_routing",
        str(lambda_routing),
        "--lambda_memory",
        str(lambda_memory),
        "--lambda_z",
        str(lambda_z),
        "--lambda_dcl",
        str(lambda_dcl),
        "--lambda_uniformity",
        str(lambda_uniformity),
        "--use_infonce",
        "--infonce_temperature",
        str(temperature),
        "--learnable_temperature",
        "--focal_gamma",
        str(focal_gamma),
        "--lr",
        str(lr),
        "--seed",
        "42",
        "--batch_size",
        str(batch_size),
        "--scheduler_type",
        "cosine_restarts",
        "--t_mult",
        str(t_mult),
        "--warmup_epochs",
        str(warmup_epochs),
        "--eval_every",
        str(eval_every),
        "--early_stopping_patience",
        str(early_stop),
        "--save_every",
        "100",
        "--output_dir",
        str(out_dir),
        "--results_json",
        str(out_dir / "results.json"),
        "--device",
        "auto",
        "--amp",
    ]

    # Добавляем ModernBERT флаги если нужно
    if modernbert:
        cmd.append("--modernbert_encoder")
        if modernbert_model_name:
            cmd.extend(["--modernbert_model_name", modernbert_model_name])
        if modernbert_pooling:
            cmd.extend(["--modernbert_pooling", modernbert_pooling])
        if modernbert_max_length:
            cmd.extend(["--modernbert_max_length", str(modernbert_max_length)])
        if matryoshka_dims:
            cmd.extend(["--modernbert_matryoshka_dims", matryoshka_dims])
        if freeze_modernbert:
            cmd.append("--freeze_modernbert")
    else:
        # SBERT encoder
        cmd.append("--pretrained_encoder")

    start = time.time()
    log_file = out_dir / "trial.log"
    try:
        env = os.environ.copy()
        # num_workers=0 для autotune чтобы не плодить процессы
        env["HDIM_NUM_WORKERS"] = "0"
        with open(log_file, "w", encoding="utf-8") as logf:
            result = subprocess.run(
                cmd, stdout=logf, stderr=logf, timeout=epochs * 300, env=env
            )
    except subprocess.TimeoutExpired:
        print(f"  Trial {trial.number}: TIMEOUT")
        return 0.0

    elapsed = time.time() - start

    # Читаем score из results.json
    results_file = out_dir / "results.json"
    if not results_file.exists():
        print(f"  Trial {trial.number}: NO RESULTS (returncode={result.returncode})")
        if log_file.exists():
            tail = log_file.read_text(encoding="utf-8", errors="ignore")[-500:]
            print("  LOG tail:", tail[-300:])
        return 0.0

    try:
        data = json.loads(results_file.read_text(encoding="utf-8"))
        score = float(data.get("score", 0.0))
    except Exception as e:
        print(f"  Trial {trial.number}: PARSE ERROR {e}")
        return 0.0

    quality = data.get("quality", {})
    pair_margin = quality.get("pair_margin", 0)
    sts = quality.get("STS_exported", 0)
    best_epoch = data.get("training_summary", {}).get("best_epoch", 0)

    # Report to Optuna for pruning
    trial.set_user_attr("pair_margin", pair_margin)
    trial.set_user_attr("STS_exported", sts)
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("elapsed_s", elapsed)
    trial.set_user_attr("hidden_dim", hidden_dim)
    if modernbert:
        trial.set_user_attr("modernbert_model_name", modernbert_model_name)
        trial.set_user_attr("modernbert_pooling", modernbert_pooling)
        trial.set_user_attr("modernbert_max_length", modernbert_max_length)
        trial.set_user_attr("freeze_modernbert", freeze_modernbert)
        trial.set_user_attr("use_matryoshka", use_matryoshka)
        trial.set_user_attr("freeze_modernbert", freeze_modernbert)

    print(
        f"  Trial {trial.number:3d}: score={score:.4f} "
        f"margin={pair_margin:.4f} STS={sts:.4f} "
        f"ep={best_epoch} t={elapsed:.0f}s "
        f"lr={lr:.5f} bs={batch_size} dcl={lambda_dcl:.2f} "
        f"z={lambda_z:.3f} aug={augment_factor} "
        f"hidden={hidden_dim} modernbert={modernbert}"
    )
    return score


def main():
    parser = argparse.ArgumentParser(
        description="HDIM Auto-Tuner v22 (Optuna + ModernBERT)"
    )
    parser.add_argument(
        "--n_trials", type=int, default=20, help="Количество trials (default: 20)"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Эпох на один trial (default: 30)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Продолжить существующее исследование"
    )
    parser.add_argument("--study_name", type=str, default=STUDY_NAME)
    parser.add_argument("--db", type=str, default=STUDY_DB)
    parser.add_argument(
        "--data",
        type=str,
        default="v5",
        choices=["v5", "v7", "v8"],
        help="Версия данных (default: v5 — рекордная)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        choices=["micro", "quick", "standard", "deep"],
        help="Preset фазы: quick=15ep, standard=30ep, deep=60ep",
    )
    parser.add_argument(
        "--modernbert_encoder",
        action="store_true",
        help="Использовать ModernBERT encoder вместо SBERT",
    )
    args = parser.parse_args()

    # Phase preset override
    if args.phase:
        preset = PHASE_PRESETS[args.phase]
        args.epochs = preset["epochs"]
        print(f"Phase preset '{args.phase}': {preset['epochs']} epochs")

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{args.db}"

    if args.resume:
        study = optuna.load_study(study_name=args.study_name, storage=storage)
        existing = len(study.trials)
        print(f"Resuming study '{args.study_name}' with {existing} existing trials")
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
            load_if_exists=True,
        )

    data_info = {
        "v5": "175 pairs (Phase8e record)",
        "v7": "232 pairs",
        "v8": "330 pairs, 35.8% neg, all domains",
    }[args.data]

    encoder_info = "ModernBERT" if args.modernbert_encoder else "SBERT"

    print(
        f"HDIM Auto-Tuner v22 | n_trials={args.n_trials} | epochs={args.epochs}/trial"
    )
    print(f"Study: {args.study_name} | DB: {args.db}")
    print(f"Data: {args.data} — {data_info}")
    print(f"Encoder: {encoder_info}")
    if args.modernbert_encoder:
        print(f"Fixed: experts=4, t_mult=2, simple MLP, ModernBERT encoder")
    else:
        print(f"Fixed: hidden=256, experts=4, t_mult=2, simple MLP, frozen SBERT")
    print(
        f"SOTA: gated_memory, expert_dropout(0.1), sim_preserving_router, grad_isolation"
    )
    print(f"Tuning: lr, bs, augment, 9 loss lambdas, temp, focal_gamma, warmup")
    if args.modernbert_encoder:
        print(
            f"ModernBERT tuning: model_name, pooling, max_length, matryoshka, freeze, hidden_dim"
        )
    print()

    study.optimize(
        lambda trial: objective(
            trial,
            epochs=args.epochs,
            data_version=args.data,
            modernbert=args.modernbert_encoder,
        ),
        n_trials=args.n_trials,
        show_progress_bar=False,
    )

    print("\n" + "=" * 70)
    print("BEST TRIAL:")
    best = study.best_trial
    print(f"  Score: {best.value:.4f}")
    print(f"  Data: {args.data}")
    print(f"  Encoder: {encoder_info}")
    print(f"  Params:")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.6f}")
        else:
            print(f"    {k}: {v}")

    # Сохраняем лучшую конфигурацию как .bat
    best_bat = REPO_ROOT / "scripts" / "best_autotune.bat"
    p = best.params
    data_path = DATA_VERSIONS[args.data].replace("/", "\\")

    # Формируем ModernBERT параметры
    modernbert_block = ""
    if args.modernbert_encoder:
        modernbert_params = []
        if p.get("modernbert_model_name"):
            modernbert_params.append(
                f"    --modernbert_model_name {p['modernbert_model_name']}"
            )
        if p.get("modernbert_pooling"):
            modernbert_params.append(
                f"    --modernbert_pooling {p['modernbert_pooling']}"
            )
        if p.get("modernbert_max_length"):
            modernbert_params.append(
                f"    --modernbert_max_length {p['modernbert_max_length']}"
            )
        if p.get("use_matryoshka") and p.get("modernbert_matryoshka_dims"):
            modernbert_params.append(
                f"    --modernbert_matryoshka_dims {p['modernbert_matryoshka_dims']}"
            )
        if p.get("freeze_modernbert"):
            modernbert_params.append(f"    --freeze_modernbert")

        if modernbert_params:
            modernbert_block = "\n".join(modernbert_params) + " ^"

        encoder_flag = "    --modernbert_encoder ^"
        hidden_dim = p.get("hidden_dim", 256)
    else:
        encoder_flag = "    --pretrained_encoder ^"
        hidden_dim = 256

    bat_content = f"""@echo off
REM Auto-tuned config v22 (Optuna, score={best.value:.4f}, data={args.data}, encoder={encoder_info})
REM Generated by scripts/auto_tune.py
REM Phase 22 ModernBERT: {encoder_info} + Gated Memory + Expert Dropout + Sim-Preserving Router

cd /d E:\\hypercoplexAI
call .venv\\Scripts\\activate.bat

python scripts\\gpu_train.py ^
    --epochs 200 ^
    --hidden_dim {hidden_dim} ^
    --num_experts 4 ^
    --num_domains 4 ^
{encoder_flag}
{modernbert_block}
    --soft_router ^
    --real_pairs {data_path} ^
    --augment_factor {p['augment_factor']} ^
    --lambda_pair {p['lambda_pair']} ^
    --lambda_angle {p['lambda_angle']} ^
    --lambda_sts {p['lambda_sts']} ^
    --lambda_iso {p['lambda_iso']} ^
    --lambda_routing {p['lambda_routing']} ^
    --lambda_memory {p['lambda_memory']} ^
    --lambda_z {p['lambda_z']} ^
    --lambda_dcl {p['lambda_dcl']} ^
    --lambda_uniformity {p['lambda_uniformity']} ^
    --use_infonce ^
    --infonce_temperature {p['infonce_temperature']:.5f} ^
    --learnable_temperature ^
    --focal_gamma {p['focal_gamma']} ^
    --early_stopping_patience 40 ^
    --lr {p['lr']:.6f} ^
    --seed 42 ^
    --batch_size {p['batch_size']} ^
    --scheduler_type cosine_restarts ^
    --t_mult {p['t_mult']} ^
    --warmup_epochs {p['warmup_epochs']} ^
    --eval_every 5 ^
    --save_every 25 ^
    --output_dir artifacts\\best_autotune ^
    --results_json artifacts\\best_autotune\\results.json ^
    --device auto ^
    --amp

pause
"""
    best_bat.write_text(bat_content, encoding="utf-8")
    print(f"\nBest config saved to: {best_bat}")

    # Сохраняем все результаты в JSON
    all_results = []
    for t in sorted(study.trials, key=lambda x: x.value or 0, reverse=True):
        if t.value is not None:
            entry = {"trial": t.number, "score": t.value, **t.params}
            entry.update(t.user_attrs)
            all_results.append(entry)

    results_file = REPO_ROOT / "artifacts" / "optuna_results.json"
    results_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"All results saved to: {results_file}")

    print(f"\nTop 5 trials:")
    for r in all_results[:5]:
        margin = r.get("pair_margin", 0)
        sts = r.get("STS_exported", 0)
        print(
            f"  Trial {r['trial']:3d}: score={r['score']:.4f} "
            f"margin={margin:.4f} STS={sts:.4f} "
            f"lr={r['lr']:.5f} bs={r['batch_size']}"
        )

    # Генерация сравнительного отчёта
    print(f"\n=== Parameter Importance (top 5) ===")
    try:
        importances = optuna.importance.get_param_importances(study)
        for i, (param, imp) in enumerate(list(importances.items())[:5]):
            print(f"  {i+1}. {param}: {imp:.4f}")
    except Exception:
        print("  (not available)")


if __name__ == "__main__":
    main()
