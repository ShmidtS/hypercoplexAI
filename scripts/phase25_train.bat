@echo off
REM Phase 25b Training — HDIM Optimal Configuration (Calibrated)
REM Key changes vs Phase 25:
REM   - --freeze_sbert_bottom_frac 0.5: freeze bottom 50%% SBERT layers
REM   - --weight_decay 0.05: stronger regularization (was default 1e-4)
REM   - --early_stopping_patience 3: stop fast if no improvement (was 50)
REM   - --dropout 0.3: increased dropout for regularization (was default 0.1)
REM   - --lambda_sts 0.5: increased STS weight in loss (was 0.25)
REM   - Output dir: phase25b (not overwriting phase25)
REM   - Added --freeze_sbert flag (via freeze_sbert_bottom_frac)
REM Hardware: RTX 3070 Laptop 8.6GB, CUDA 12.4
REM Target: score > 1.15 (break Phase 8e record)

cd /d E:\hypercoplexAI

python scripts\gpu_train.py ^
    --epochs 200 ^
    --hidden_dim 256 ^
    --num_experts 4 ^
    --num_domains 4 ^
    --batch_size 16 ^
    --lr 0.000288 ^
    --weight_decay 0.05 ^
    --dropout 0.3 ^
    --pretrained_encoder ^
    --freeze_sbert_bottom_frac 0.5 ^
    --soft_router ^
    --real_pairs data\real_pairs_v8.json ^
    --augment_factor 25 ^
    --lambda_pair 0.45 ^
    --lambda_angle 0.4 ^
    --lambda_sts 0.5 ^
    --lambda_iso 0.175 ^
    --lambda_routing 0.08 ^
    --lambda_memory 0.015 ^
    --lambda_z 0.015 ^
    --lambda_dcl 0.1 ^
    --lambda_uniformity 0.3 ^
    --use_infonce ^
    --infonce_temperature 0.131 ^
    --learnable_temperature ^
    --focal_gamma 0.5 ^
    --scheduler_type cosine_restarts ^
    --t_mult 2 ^
    --warmup_epochs 3 ^
    --gradient_checkpointing ^
    --similarity_preserving_router ^
    --early_stopping_patience 3 ^
    --eval_every 5 ^
    --save_every 25 ^
    --output_dir artifacts\phase25b_cl41 ^
    --results_json artifacts\phase25b_cl41\results.json ^
    --device auto ^
    --amp ^
    --seed 42

pause
