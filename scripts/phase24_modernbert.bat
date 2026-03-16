@echo off
REM Phase 24 Training — ModernBERT + Matryoshka Multi-Scale
REM Key changes vs Phase 23/25:
REM   - --modernbert_encoder: ModernBERT-base (answerdotai/ModernBERT-base)
REM   - --modernbert_matryoshka_dims 64,128,256,768: multi-scale Matryoshka
REM   - --modernbert_use_cls_pooling: CLS token pooling
REM   - --freeze_modernbert: freeze ModernBERT weights (frozen backbone)
REM   - --lambda_matryoshka 0.1: Matryoshka multi-scale loss weight
REM   - --lambda_diversity_var 0.015: slightly stronger anti-collapse
REM   - Hidden dim 256 (compatible with Matryoshka 256 scale)
REM   - Uses real_pairs_v8.json (330 pairs, v10 also available)
REM Hardware: RTX 3070 Laptop 8.6GB, CUDA 12.4
REM Target: score > 1.1542 (beat Phase 8e record)

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
    --modernbert_encoder ^
    --modernbert_matryoshka_dims 64,128,256,768 ^
    --freeze_modernbert ^
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
    --lambda_matryoshka 0.1 ^
    --lambda_diversity_var 0.015 ^
    --lambda_diversity_ortho 0.005 ^
    --use_infonce ^
    --infonce_temperature 0.131 ^
    --learnable_temperature ^
    --focal_gamma 0.5 ^
    --scheduler_type cosine_restarts ^
    --t_mult 2 ^
    --warmup_epochs 3 ^
    --gradient_checkpointing ^
    --early_stopping_patience 5 ^
    --eval_every 5 ^
    --save_every 25 ^
    --output_dir artifacts\phase24_modernbert ^
    --results_json artifacts\phase24_modernbert\results.json ^
    --device auto ^
    --amp ^
    --seed 42

pause
