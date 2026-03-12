#!/usr/bin/env python
"""
HDIM Auto-Tuner — автоматический подбор гиперпараметров через Optuna.

Запуск:
  python scripts/auto_tune.py                        # 20 trials, 30 эпох каждый
  python scripts/auto_tune.py --n_trials 50          # 50 trials
  python scripts/auto_tune.py --epochs 50 --n_trials 30
  python scripts/auto_tune.py --resume               # продолжить прерванное исследование

Optuna подбирает:
  - lr, batch_size, lambda_pair, lambda_angle, lambda_sts
  - lambda_dcl, lambda_uniformity, lambda_z
  - infonce_temperature, focal_gamma
  - augment_factor

Зафиксировано (из экспериментов):
  - hidden_dim=256, num_experts=4 (Phase8e рекорд)
  - pretrained_encoder, soft_router
  - simple MLP (не менять архитектуру)
  - data=v5 (рекордные данные)
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
STUDY_NAME = "hdim_autotune_v1"


def objective(trial: optuna.Trial, epochs: int = 30) -> float:
    """Один trial: запустить обучение с предложенными параметрами, вернуть score."""

    # === Параметры для подбора ===
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 96])
    augment_factor = trial.suggest_int("augment_factor", 10, 50, step=10)

    # Loss weights
    lambda_pair = trial.suggest_float("lambda_pair", 0.2, 0.6, step=0.1)
    lambda_angle = trial.suggest_float("lambda_angle", 0.1, 0.5, step=0.1)
    lambda_sts = trial.suggest_float("lambda_sts", 0.0, 0.3, step=0.1)
    lambda_dcl = trial.suggest_float("lambda_dcl", 0.0, 0.5, step=0.1)
    lambda_uniformity = trial.suggest_float("lambda_uniformity", 0.0, 0.3, step=0.05)
    lambda_z = trial.suggest_float("lambda_z", 0.0, 0.02, step=0.005)

    # Temperature
    temperature = trial.suggest_float("infonce_temperature", 0.05, 0.2, log=True)
    focal_gamma = trial.suggest_float("focal_gamma", 0.3, 1.0, step=0.1)

    # Scheduler
    t_mult = trial.suggest_categorical("t_mult", [1, 2])

    out_dir = REPO_ROOT / "artifacts" / f"optuna_trial_{trial.number:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--epochs", str(epochs),
        "--hidden_dim", "256",
        "--num_experts", "4",
        "--num_domains", "4",
        "--pretrained_encoder",
        "--soft_router",
        "--real_pairs", str(REPO_ROOT / "data" / "real_pairs_v5.json"),
        "--augment_factor", str(augment_factor),
        "--lambda_pair", str(lambda_pair),
        "--lambda_angle", str(lambda_angle),
        "--lambda_sts", str(lambda_sts),
        "--lambda_iso", "0.1",
        "--lambda_routing", "0.05",
        "--lambda_memory", "0.01",
        "--lambda_z", str(lambda_z),
        "--lambda_dcl", str(lambda_dcl),
        "--lambda_uniformity", str(lambda_uniformity),
        "--use_infonce",
        "--infonce_temperature", str(temperature),
        "--learnable_temperature",
        "--focal_gamma", str(focal_gamma),
        # early_stopping_patience set above dynamically
        "--lr", str(lr),
        "--seed", "42",
        "--batch_size", str(batch_size),
        "--scheduler_type", "cosine_restarts",
        "--t_mult", str(t_mult),
        "--warmup_epochs", "3",
        "--eval_every", str(max(1, epochs // 3)),
        "--early_stopping_patience", str(max(5, epochs // 2)),
        "--save_every", "100",
        "--output_dir", str(out_dir),
        "--results_json", str(out_dir / "results.json"),
        "--device", "auto",
        "--amp",
    ]

    start = time.time()
    log_file = out_dir / "trial.log"
    try:
        env = os.environ.copy()
        env["HDIM_NUM_WORKERS"] = "0"  # отключаем DataLoader workers (RAM)
        with open(log_file, "w", encoding="utf-8") as logf:
            result = subprocess.run(
                cmd, stdout=logf, stderr=logf,
                timeout=epochs * 200, env=env
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
            tail = log_file.read_text(encoding="utf-8", errors="ignore")[-400:]
            print("  LOG tail:", tail)
        return 0.0

    try:
        data = json.loads(results_file.read_text(encoding="utf-8"))
        score = float(data.get("score", 0.0))
    except Exception as e:
        print(f"  Trial {trial.number}: PARSE ERROR {e}")
        return 0.0

    pair_margin = data.get("quality", {}).get("pair_margin", 0)
    sts = data.get("quality", {}).get("STS_exported", 0)
    best_epoch = data.get("training_summary", {}).get("best_epoch", 0)

    print(
        f"  Trial {trial.number:3d}: score={score:.4f} "
        f"margin={pair_margin:.4f} STS={sts:.4f} "
        f"ep={best_epoch} t={elapsed:.0f}s "
        f"lr={lr:.5f} bs={batch_size} dcl={lambda_dcl:.2f} uni={lambda_uniformity:.2f}"
    )
    return score


def main():
    parser = argparse.ArgumentParser(description="HDIM Auto-Tuner (Optuna)")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Количество trials (default: 20)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Эпох на один trial (default: 30)")
    parser.add_argument("--resume", action="store_true",
                        help="Продолжить существующее исследование")
    parser.add_argument("--study_name", type=str, default=STUDY_NAME)
    parser.add_argument("--db", type=str, default=STUDY_DB)
    args = parser.parse_args()

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

    print(f"HDIM Auto-Tuner | n_trials={args.n_trials} | epochs={args.epochs}/trial")
    print(f"Study: {args.study_name} | DB: {args.db}")
    print(f"Fixed: hidden=256, experts=4, v5 data, simple MLP, frozen SBERT")
    print(f"Tuning: lr, batch_size, augment, lambdas, temperature, focal_gamma")
    print()

    study.optimize(
        lambda trial: objective(trial, epochs=args.epochs),
        n_trials=args.n_trials,
        show_progress_bar=False,
    )

    print("\n" + "=" * 60)
    print("BEST TRIAL:")
    best = study.best_trial
    print(f"  Score: {best.value:.4f}")
    print(f"  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Сохраняем лучшую конфигурацию как .bat
    best_bat = REPO_ROOT / "scripts" / "best_autotune.bat"
    params = best.params
    bat_content = f"""@echo off
REM Auto-tuned config (Optuna, score={best.value:.4f})
REM Generated by scripts/auto_tune.py

cd /d E:\\hypercoplexAI
call .venv\\Scripts\\activate.bat

python scripts\\gpu_train.py ^
    --epochs 200 ^
    --hidden_dim 256 ^
    --num_experts 4 ^
    --num_domains 4 ^
    --pretrained_encoder ^
    --soft_router ^
    --real_pairs data\\real_pairs_v5.json ^
    --augment_factor {params['augment_factor']} ^
    --lambda_pair {params['lambda_pair']} ^
    --lambda_angle {params['lambda_angle']} ^
    --lambda_sts {params['lambda_sts']} ^
    --lambda_iso 0.1 ^
    --lambda_routing 0.05 ^
    --lambda_memory 0.01 ^
    --lambda_z {params['lambda_z']} ^
    --lambda_dcl {params['lambda_dcl']} ^
    --lambda_uniformity {params['lambda_uniformity']} ^
    --use_infonce ^
    --infonce_temperature {params['infonce_temperature']:.5f} ^
    --learnable_temperature ^
    --focal_gamma {params['focal_gamma']} ^
    --early_stopping_patience 40 ^
    --lr {params['lr']:.6f} ^
    --seed 42 ^
    --batch_size {params['batch_size']} ^
    --scheduler_type cosine_restarts ^
    --t_mult {params['t_mult']} ^
    --warmup_epochs 3 ^
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
            all_results.append({"trial": t.number, "score": t.value, **t.params})

    results_file = REPO_ROOT / "artifacts" / "optuna_results.json"
    results_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"All results saved to: {results_file}")
    print(f"\nTop 5 trials:")
    for r in all_results[:5]:
        print(f"  Trial {r['trial']:3d}: score={r['score']:.4f} lr={r['lr']:.5f} bs={r['batch_size']}")


if __name__ == "__main__":
    main()
