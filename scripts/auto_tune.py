#!/usr/bin/env python
"""
HDIM Auto-Tuner v27 — Optuna hyperparameter search for Phase 26 config.

Запуск:
  python scripts/auto_tune.py                        # 20 trials, 30 эпох каждый
  python scripts/auto_tune.py --n_trials 50 --epochs 50
  python scripts/auto_tune.py --resume               # продолжить прерванное исследование
  python scripts/auto_tune.py --phase quick           # быстрый 10-эпох прогон
  python scripts/auto_tune.py --phase deep            # глубокий 60-эпох прогон

Phase 26 зафиксировано (всегда включено):
  - --pretrained_encoder (frozen SBERT)
  - --shared_expert (DeepSeek-V3 always-on FFN)
  - --aux_loss_free (bias-based load balancing)
  - --expert_ortho (expert orthogonalization loss)
  - --learnable_temperature
  - --soft_router
  - data=v10 (1036 пар: 636 pos / 400 neg)

Optuna подбирает:
  - lr, batch_size, augment_factor
  - lambda_pair, lambda_sts, lambda_dcl, lambda_uniformity
  - lambda_iso, lambda_routing, lambda_memory, lambda_z
  - lambda_expert_ortho
  - focal_gamma
  - warmup_epochs, t_mult

Зафиксировано:
  - hidden_dim=256, num_experts=4 (Phase8e рекорд)
  - soft_router, pretrained_encoder
  - learnable_temperature=True
  - shared_expert + aux_loss_free + expert_ortho (Phase 26)

История рекордов:
  Phase 26a: score=1.1063 @ ep45 (augment=3, no sts/dcl)
  Phase 26b: score=1.1513 @ ep15 (augment=5, sts=0.15, dcl=0.2, learnable_temp)
  Phase 26c: score=1.1542 @ ep15 (augment=5, sts=0.3, uniformity=0.1)
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
STUDY_DB = str(REPO_ROOT / "artifacts" / "optuna_study_v27.db")
STUDY_NAME = "hdim_autotune_v27"
DATA_PATH = str(REPO_ROOT / "data" / "real_pairs_v10.json")

# Phase presets
PHASE_PRESETS = {
    "micro":    {"epochs": 5,  "eval_every": 2,  "early_stop": 3},
    "quick":    {"epochs": 10, "eval_every": 3,  "early_stop": 5},
    "standard": {"epochs": 30, "eval_every": 5,  "early_stop": 15},
    "deep":     {"epochs": 60, "eval_every": 5,  "early_stop": 25},
}

# Лучшие конфиги из истории для warm-start Optuna
SOTA_SEEDS = [
    # Phase 26c рекорд
    {
        "lr": 3e-4, "batch_size": 48, "augment_factor": 5,
        "lambda_pair": 0.5, "lambda_sts": 0.3, "lambda_dcl": 0.2,
        "lambda_uniformity": 0.1, "lambda_iso": 0.1, "lambda_routing": 0.02,
        "lambda_memory": 0.01, "lambda_z": 0.01, "lambda_expert_ortho": 0.02,
        "focal_gamma": 1.0, "warmup_epochs": 20, "t_mult": 2,
    },
    # Phase 26b рекорд
    {
        "lr": 3e-4, "batch_size": 48, "augment_factor": 5,
        "lambda_pair": 0.5, "lambda_sts": 0.15, "lambda_dcl": 0.2,
        "lambda_uniformity": 0.0, "lambda_iso": 0.1, "lambda_routing": 0.02,
        "lambda_memory": 0.01, "lambda_z": 0.01, "lambda_expert_ortho": 0.02,
        "focal_gamma": 1.0, "warmup_epochs": 20, "t_mult": 2,
    },
]


def objective(trial: optuna.Trial, epochs: int = 30) -> float:
    """Один trial: запустить обучение с предложенными параметрами, вернуть score."""

    # === Learning rate ===
    lr = trial.suggest_float("lr", 5e-5, 8e-4, log=True)

    # === Batch size ===
    batch_size = trial.suggest_categorical("batch_size", [24, 32, 48, 64])

    # === Augment factor (v10 1036 пар — не нужно много) ===
    augment_factor = trial.suggest_int("augment_factor", 2, 10)

    # === Loss weights ===
    lambda_pair = trial.suggest_float("lambda_pair", 0.3, 0.7, step=0.05)
    lambda_sts  = trial.suggest_float("lambda_sts",  0.05, 0.5, step=0.05)
    lambda_dcl  = trial.suggest_float("lambda_dcl",  0.0,  0.4, step=0.05)
    lambda_uniformity = trial.suggest_float("lambda_uniformity", 0.0, 0.2, step=0.025)
    lambda_iso      = trial.suggest_float("lambda_iso",      0.05, 0.2,  step=0.025)
    lambda_routing  = trial.suggest_float("lambda_routing",  0.01, 0.05, step=0.005)
    lambda_memory   = trial.suggest_float("lambda_memory",   0.005, 0.02, step=0.005)
    lambda_z        = trial.suggest_float("lambda_z",        0.005, 0.02, step=0.005)
    lambda_expert_ortho = trial.suggest_float("lambda_expert_ortho", 0.005, 0.05, step=0.005)

    # === Focal gamma (1.0 = standard InfoNCE) ===
    focal_gamma = trial.suggest_float("focal_gamma", 0.5, 2.0, step=0.1)

    # === Scheduler ===
    t_mult = trial.suggest_categorical("t_mult", [1, 2, 3])
    warmup_epochs = trial.suggest_int("warmup_epochs", 5, 25, step=5)

    # === Output ===
    out_dir = REPO_ROOT / "artifacts" / f"optuna_{trial.number:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_every = max(1, epochs // 6)
    early_stop = max(5, epochs // 2)

    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--epochs",           str(epochs),
        "--hidden_dim",       "256",
        "--num_experts",      "4",
        "--num_domains",      "4",
        "--soft_router",
        "--pretrained_encoder",
        "--shared_expert",
        "--aux_loss_free",
        "--expert_ortho",
        "--learnable_temperature",
        "--real_pairs",       DATA_PATH,
        "--augment_factor",   str(augment_factor),
        "--lambda_pair",      str(lambda_pair),
        "--lambda_sts",       str(lambda_sts),
        "--lambda_iso",       str(lambda_iso),
        "--lambda_routing",   str(lambda_routing),
        "--lambda_memory",    str(lambda_memory),
        "--lambda_z",         str(lambda_z),
        "--lambda_dcl",       str(lambda_dcl),
        "--lambda_uniformity",str(lambda_uniformity),
        "--lambda_expert_ortho", str(lambda_expert_ortho),
        "--use_infonce",
        "--focal_gamma",      str(focal_gamma),
        "--lr",               str(lr),
        "--seed",             "42",
        "--batch_size",       str(batch_size),
        "--scheduler_type",   "cosine_restarts",
        "--t_mult",           str(t_mult),
        "--warmup_epochs",    str(warmup_epochs),
        "--eval_every",       str(eval_every),
        "--early_stopping_patience", str(early_stop),
        "--save_every",       "100",
        "--output_dir",       str(out_dir),
        "--results_json",     str(out_dir / "results.json"),
        "--device",           "auto",
        "--amp",
    ]

    start = time.time()
    log_file = out_dir / "trial.log"
    try:
        env = os.environ.copy()
        env["HDIM_NUM_WORKERS"] = "0"
        with open(log_file, "w", encoding="utf-8") as logf:
            result = subprocess.run(
                cmd, stdout=logf, stderr=logf, timeout=epochs * 300, env=env
            )
    except subprocess.TimeoutExpired:
        print(f"  Trial {trial.number}: TIMEOUT")
        return 0.0

    elapsed = time.time() - start

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
    best_score = data.get("training_summary", {}).get("best_score", score)

    trial.set_user_attr("pair_margin", pair_margin)
    trial.set_user_attr("STS_exported", sts)
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("best_score", best_score)
    trial.set_user_attr("elapsed_s", elapsed)

    print(
        f"  Trial {trial.number:3d}: score={best_score:.4f} "
        f"margin={pair_margin:.4f} STS={sts:.4f} "
        f"ep={best_epoch} t={elapsed:.0f}s "
        f"lr={lr:.5f} bs={batch_size} aug={augment_factor} "
        f"sts={lambda_sts:.2f} dcl={lambda_dcl:.2f} uni={lambda_uniformity:.3f} "
        f"ortho={lambda_expert_ortho:.3f}"
    )
    # Возвращаем best_score (пик за все эпохи), а не финальный
    return best_score


def enqueue_sota_seeds(study: optuna.Study, max_seeds: int = 2) -> None:
    """Добавляем исторически лучшие конфиги как первые trials для warm-start."""
    existing_params = {str(t.params) for t in study.trials if t.params}
    for seed in SOTA_SEEDS[:max_seeds]:
        if str(seed) not in existing_params:
            study.enqueue_trial(seed)


def main():
    parser = argparse.ArgumentParser(description="HDIM Auto-Tuner v27 (Optuna, Phase 26)")
    parser.add_argument("--n_trials",  type=int, default=20, help="Количество trials")
    parser.add_argument("--epochs",    type=int, default=30, help="Эпох на один trial")
    parser.add_argument("--resume",    action="store_true",  help="Продолжить существующее исследование")
    parser.add_argument("--no_seeds",  action="store_true",  help="Не добавлять SOTA seeds")
    parser.add_argument("--study_name", type=str, default=STUDY_NAME)
    parser.add_argument("--db",         type=str, default=STUDY_DB)
    parser.add_argument(
        "--phase", type=str, default=None,
        choices=["micro", "quick", "standard", "deep"],
        help="Preset: micro=5ep, quick=10ep, standard=30ep, deep=60ep",
    )
    args = parser.parse_args()

    if args.phase:
        preset = PHASE_PRESETS[args.phase]
        args.epochs = preset["epochs"]
        print(f"Phase preset '{args.phase}': {preset['epochs']} epochs")

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{args.db}"

    if args.resume:
        study = optuna.load_study(study_name=args.study_name, storage=storage)
        print(f"Resuming study '{args.study_name}' with {len(study.trials)} existing trials")
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
            load_if_exists=True,
        )

    # Warm-start с историческими рекордами
    if not args.no_seeds:
        enqueue_sota_seeds(study)

    print(f"HDIM Auto-Tuner v27 | n_trials={args.n_trials} | epochs={args.epochs}/trial")
    print(f"Study: {args.study_name} | DB: {args.db}")

    print(f"Data: {DATA_PATH} (v10, 1036 pairs: 636 pos / 400 neg)")
    print(f"Fixed: hidden=256, experts=4, soft_router, pretrained_encoder,")
    print(f"       shared_expert, aux_loss_free, expert_ortho, learnable_temperature")
    print(f"SOTA baseline: Phase 26c score=1.1542 @ ep15")
    print()

    study.optimize(
        lambda trial: objective(trial, epochs=args.epochs),
        n_trials=args.n_trials,
        show_progress_bar=False,
    )

    print("\n" + "=" * 70)
    print("BEST TRIAL:")
    best = study.best_trial
    print(f"  Score: {best.value:.4f}")
    print(f"  Params:")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.6f}")
        else:
            print(f"    {k}: {v}")


    # Save best config as .bat
    best_bat = REPO_ROOT / "scripts" / "best_autotune.bat"
    p = best.params
    data_bat = str(REPO_ROOT / "data" / "real_pairs_v10.json").replace("/", "\\")

    bat_lines = [
        "@echo off",
        f"REM Auto-tuned config v27 (Optuna, score={best.value:.4f})",
        "REM Generated by scripts/auto_tune.py",
        "REM Phase 26: shared_expert + aux_loss_free + expert_ortho + learnable_temp",
        "",
        "cd /d E:\\hypercoplexAI",
        "call .venv\\Scripts\\activate.bat",
        "",
        "python scripts\\gpu_train.py ^",
        "    --epochs 200 ^",
        "    --hidden_dim 256 ^",
        "    --num_experts 4 ^",
        "    --num_domains 4 ^",
        "    --soft_router ^",
        "    --pretrained_encoder ^",
        "    --shared_expert ^",
        "    --aux_loss_free ^",
        "    --expert_ortho ^",
        "    --learnable_temperature ^",
        f"    --real_pairs {data_bat} ^",
        f"    --augment_factor {p['augment_factor']} ^",
        f"    --lambda_pair {p['lambda_pair']} ^",
        f"    --lambda_sts {p['lambda_sts']} ^",
        f"    --lambda_iso {p['lambda_iso']} ^",
        f"    --lambda_routing {p['lambda_routing']} ^",
        f"    --lambda_memory {p['lambda_memory']} ^",
        f"    --lambda_z {p['lambda_z']} ^",
        f"    --lambda_dcl {p['lambda_dcl']} ^",
        f"    --lambda_uniformity {p['lambda_uniformity']} ^",
        f"    --lambda_expert_ortho {p['lambda_expert_ortho']} ^",
        "    --use_infonce ^",
        f"    --focal_gamma {p['focal_gamma']} ^",
        "    --early_stopping_patience 40 ^",
        f"    --lr {p['lr']:.6f} ^",
        "    --seed 42 ^",
        f"    --batch_size {p['batch_size']} ^",
        "    --scheduler_type cosine_restarts ^",
        f"    --t_mult {p['t_mult']} ^",
        f"    --warmup_epochs {p['warmup_epochs']} ^",
        "    --eval_every 5 ^",
        "    --save_every 25 ^",
        "    --output_dir artifacts\\best_autotune ^",
        "    --results_json artifacts\\best_autotune\\results.json ^",
        "    --device auto ^",
        "    --amp",
        "",
        "pause",
    ]
    best_bat.write_text("\n".join(bat_lines), encoding="utf-8")
    print(f"\nBest config saved to: {best_bat}")

    # Save all results
    all_results = []
    for t in sorted(study.trials, key=lambda x: x.value or 0, reverse=True):
        if t.value is not None:
            entry = {"trial": t.number, "score": t.value, **t.params}
            entry.update(t.user_attrs)
            all_results.append(entry)

    results_file = REPO_ROOT / "artifacts" / "optuna_results_v27.json"
    results_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"All results saved to: {results_file}")

    print(f"\nTop 5 trials:")
    for r in all_results[:5]:
        margin = r.get("pair_margin", 0)
        sts_val = r.get("STS_exported", 0)
        best_ep = r.get("best_epoch", 0)
        print(
            f"  Trial {r['trial']:3d}: score={r['score']:.4f} "
            f"margin={margin:.4f} STS={sts_val:.4f} ep={best_ep} "
            f"lr={r['lr']:.5f} bs={r['batch_size']} aug={r['augment_factor']} "
            f"sts={r['lambda_sts']:.2f} dcl={r['lambda_dcl']:.2f}"
        )

    print(f"\n=== Parameter Importance (top 5) ===")
    try:
        importances = optuna.importance.get_param_importances(study)
        for i, (param, imp) in enumerate(list(importances.items())[:5]):
            print(f"  {i+1}. {param}: {imp:.4f}")
    except Exception:
        print("  (not available - need >= 5 completed trials)")


if __name__ == "__main__":
    main()
