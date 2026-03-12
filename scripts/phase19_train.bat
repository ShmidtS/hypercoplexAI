@echo off
REM Phase 19 Training — HDIM with anti-collapse fixes
REM Fixes: double trainer bug (P0), real focal_gamma (P0), memory drift (P1), diversity loss (P0)
REM Config: Based on Phase8e record (1.1370) but with v7 data + fixes

python scripts\gpu_train.py ^
    --epochs 200 ^
    --hidden_dim 256 ^
    --num_experts 4 ^
    --num_domains 4 ^
    --pretrained_encoder ^
    --soft_router ^
    --real_pairs data\real_pairs_v7.json ^
    --augment_factor 30 ^
    --lambda_pair 0.4 ^
    --lambda_sts 0.2 ^
    --lambda_angle 0.3 ^
    --lambda_iso 0.1 ^
    --lambda_routing 0.05 ^
    --lambda_memory 0.01 ^
    --use_infonce ^
    --infonce_temperature 0.1 ^
    --learnable_temperature ^
    --focal_gamma 0.5 ^
    --early_stopping_patience 40 ^
    --lr 0.0005 ^
    --seed 42 ^
    --batch_size 32 ^
    --scheduler_type cosine_restarts ^
    --t_mult 2 ^
    --warmup_epochs 3 ^
    --eval_every 5 ^
    --save_every 25 ^
    --output_dir artifacts\phase19_fixed ^
    --results_json artifacts\phase19_fixed\results.json ^
    --device auto ^
    --amp
