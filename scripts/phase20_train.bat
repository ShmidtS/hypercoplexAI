@echo off
REM Phase 20 Training — HDIM with DCL + Uniformity + v5 + num_workers
REM Target: score > 1.20 (рекорд Phase8e = 1.1370)
REM Ключевые изменения:
REM   + DCL loss (lambda_dcl=0.3) — Decoupled Contrastive (Yeh et al. 2022)
REM   + Uniformity+Alignment (lambda_uniformity=0.1) — Wang & Isola 2020
REM   + batch_size=64 — больше in-batch negatives
REM   + lambda_z=0.01 — Router Z-Loss
REM   + v5 данные (как Phase8e рекорд)
REM   + num_workers=4 (DataLoader оптимизация)

cd /d E:\hypercoplexAI

call .venv\Scripts\activate.bat

python scripts\gpu_train.py ^
    --epochs 200 ^
    --hidden_dim 256 ^
    --num_experts 4 ^
    --num_domains 4 ^
    --pretrained_encoder ^
    --soft_router ^
    --real_pairs data\real_pairs_v5.json ^
    --augment_factor 30 ^
    --lambda_pair 0.4 ^
    --lambda_sts 0.2 ^
    --lambda_angle 0.3 ^
    --lambda_iso 0.1 ^
    --lambda_routing 0.05 ^
    --lambda_memory 0.01 ^
    --lambda_z 0.01 ^
    --lambda_dcl 0.3 ^
    --lambda_uniformity 0.1 ^
    --use_infonce ^
    --infonce_temperature 0.1 ^
    --learnable_temperature ^
    --focal_gamma 0.5 ^
    --early_stopping_patience 40 ^
    --lr 0.0005 ^
    --seed 42 ^
    --batch_size 64 ^
    --scheduler_type cosine_restarts ^
    --t_mult 2 ^
    --warmup_epochs 3 ^
    --eval_every 5 ^
    --save_every 25 ^
    --output_dir artifacts\phase20_dcl_uniform ^
    --results_json artifacts\phase20_dcl_uniform\results.json ^
    --device auto ^
    --amp

pause
