@echo off
REM ============================================================
REM  Phase 17 Training Script — HypercomplexAI HDIM
REM  Fixes applied: C1-C7, A1-A2, A5-A7
REM  Key changes vs phase16_baseline_v8:
REM    - infonce_temperature 0.07 -> 0.15
REM    - focal_gamma 0.5 (hard negative focus)
REM    - t_mult=1 (stable cosine decay, no LR restarts)
REM    - memory reset per epoch (via trainer.set_epoch)
REM    - lambda_memory 0.05 (reduced from default)
REM ============================================================

set PYTHONPATH=%~dp0..
set PYTHONIOENCODING=utf-8

python scripts\gpu_train.py ^
    --epochs 60 ^
    --batch_size 32 ^
    --lr 3e-4 ^
    --hidden_dim 128 ^
    --num_experts 4 ^
    --num_domains 4 ^
    --lambda_iso 0.1 ^
    --lambda_pair 0.2 ^
    --lambda_routing 0.05 ^
    --lambda_memory 0.05 ^
    --lambda_z 0.01 ^
    --use_infonce ^
    --infonce_temperature 0.15 ^
    --focal_gamma 0.5 ^
    --scheduler_type cosine_restarts ^
    --t_mult 1 ^
    --warmup_epochs 5 ^
    --use_pairs ^
    --num_samples 1000 ^
    --train_fraction 0.8 ^
    --eval_every 5 ^
    --save_every 10 ^
    --output_dir artifacts\phase17 ^
    --results_json artifacts\phase17\results.json ^
    --device auto ^
    --amp ^
    --seed 42

echo.
echo Phase 17 training complete.
pause
