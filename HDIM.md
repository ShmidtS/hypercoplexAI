# HDIM — Hypercomplex Domain Isomorphism Machine
*Версия: 30.0 | Дата: 2026-03-26 | **РЕКОРД: score=1.1814** (Run 18, ep13, temp=0.10, λ_pair=0.40, margin=1.0224) | Phase 30: MoEKernel buffer fix + SoftMoERouter deadlock fix | Numerical Python verification/tests 159/159 PASS + pytest 453 PASS*

---

## 1. Назначение

HDIM — research prototype для **кроссдоменного переноса знаний** через гиперкомплексные представления. Система кодирует текст/embedding в гиперкомплексную структуру, извлекает доменно-инвариантное представление и переносит его в другой домен через routing + memory.

Текущий статус: **MVP transfer engine** с TextHDIMModel поверх HDIM core, обученный на парных примерах (source_domain → target_domain). Поддерживает динамическое добавление любого количества экспертов и доменов без перестройки модели.

---

## 2. Научная гипотеза

**Проблема:** LLM сопоставляют тексты по токенной близости, плохо находя структурные аналогии между доменами с разным словарём.

**Гипотеза HDIM:** Если проблемы из разных доменов кодировать в общее гиперкомплексное пространство и снимать доменный отпечаток, получается структурный инвариант, переносимый между доменами.

Пример: кавитационное разрушение (инженерия) ↔ удаление зубного налёта (стоматология) — разные слова, схожая физика.

---

## 3. Математический контракт

### Представление
$$X \in Cl_{p,q,r}(\mathbb{R})$$

Внутренние состояния живут в `clifford_dim`, часть слоёв использует quaternion-aware операции.

### Инвариант
$$U_{inv} = R_A^{-1} \otimes G_A \otimes R_A$$

`InvariantExtractor` + `DomainRotationOperator` в коде.

### Проекция в целевой домен
$$G_B = R_B \otimes U \otimes R_B^{-1}$$

### Loss функция
$$L_{total} = L_{recon} + \lambda_{iso} L_{iso} + \lambda_{pair} L_{pair} + \lambda_{routing} L_{routing} + \lambda_z L_z + L_{memory} + \lambda_{expert\_ortho} L_{expert\_ortho} + \lambda_{dcl} L_{dcl} + \lambda_{uniformity} L_{uniformity}$$

Где $L_{pair}$ = Focal-InfoNCE/InfoNCE + AnglE + SupCon (при наличии family labels).
$L_z = (\log\sum_j e^{z_j})^2$ — Router Z-Loss для стабильности MoE (ST-MoE, Zoph et al. 2022).

### Similarity-Preserving Router Loss (Phase 21)
$$L_{balance} = -\sum_i \sum_j sim(x_i, x_j) \cdot sim(r_i, r_j)$$

Где $sim(x_i, x_j)$ — косинусная близость входных представлений, $sim(r_i, r_j)$ — косинусная близость routing-векторов. Loss штрафует роутер, если маршрутизация не отражает структурную близость входов. Это обеспечивает, чтобы семантически похожие объекты направлялись к одним и тем же экспертам (ICLR 2026).

---

## 4. Архитектура

### Пайплайн (Phase 23)
```
Текст → SBERT(frozen, paraphrase-mpnet-v2) → SimpleMLP(768→384→256) → InvariantExtractor → TitansMemory(Gradient Surprise + Adaptive Forgetting) → GatedMemoryFusion → SoftMoERouter (Expert Dropout 0.1 + Similarity-Preserving ICLR 2026 + Z-loss) → DecoupledDecoder → CliffordNet(Learnable Metric)

[Phase 24] ModernBERT(frozen) → MatryoshkaProjection(768→[64,128,256,768]) → Multi-Scale InfoNCE Loss → same core pipeline
```

**Важно:** GatedProjection (Phase 10) показал **регрессию** — 0.9347 << 1.1370. Оптимальная проекция: simple MLP (Linear→LayerNorm→GELU→Dropout→Linear).

**Phase 21 нововведения:**
- **Gated Memory Fusion** (`hdim_pipeline.py`): learnable gate `g = σ(W_g[x; m])` для слияния входа и memory-выхода, предотвращает memory drift
- **Expert Dropout** (0.1) в MoE: случайное отключение экспертов при обучении для регуляризации и лучшей генерализации
- **Similarity-Preserving Router** (опция): routing отражает структурную близость входов (ICLR 2026)

### Ключевые модули
| Модуль | Файл | Назначение |
|--------|------|------------|
| `HDIMPipeline` | `src/core/hdim_pipeline.py` | Оркестрация пайплайна |
| `InvariantExtractor` | `src/core/domain_operators.py` | Извлечение инварианта |
| `TitansMemory` | `src/core/titans_memory.py` | Ассоциативная память |
| `SoftMoERouter` | `src/core/soft_moe_router.py` | Мягкая маршрутизация (DEFAULT, заменил R3MoERouter) |
| `ModularMoERouter` | `src/core/modular_moe.py` | Модульный роутер (add/remove expert) |
| `SBERTEncoder` | `src/models/sbert_encoder.py` | **Simple MLP** 768→384→256 |
| `HDIMModel` | `src/models/hdim_model.py` | Batch-facing API |
| `TextHDIMModel` | `src/models/text_hdim_model.py` | Text entry wrapper |
| `build_*_model` | `src/models/model_factory.py` | Единственная точка сборки моделей |
| `HDIMTrainer` | `src/training/trainer.py` | Focal-InfoNCE + AnglE + SupCon + HardNeg + temp scheduling |
| `gpu_train.py` | `scripts/gpu_train.py` | Основной скрипт обучения |
| `gen_dataset_v8.py` | `scripts/gen_dataset_v8.py` | Генератор v8 датасета (330 пар) |

### Phase 10 — Что пробовали и откачали

| Компонент | Статус | Результат |
|-----------|--------|-----------|
| GatedProjection | ⚠️ **ОТКАТ** | -0.198 (REGRESSION: 0.9347 vs 1.1370) |
| Online Hard Neg Mining | ⚠️ **ОТКАТ** | -0.064 (дестабилизирует) |
| Partial SBERT unfreeze | ⚠️ **ОТКАТ** | -0.408 (REGRESSION: 0.729) |
| Simple MLP projection | ✅ **ОПТИМАЛЬНЫЙ** | Рекорд 1.1370 |

**Вывод:** Simple MLP (Linear→LayerNorm→GELU→Dropout→Linear) стабильно лучше всех сложных вариантов.

### Текстовые энкодеры (Phase 22)

**Текущий:** SimpleTextEncoder (character/word-level, trainable)

**Новые опции (src/models/modern_text_encoder.py):**

| Архитектура | Описание | Использование |
|-------------|----------|---------------|
| `ModernBertEncoder` | Pre-trained ModernBERT (8K context) | Замороженный + projection |
| `GatedMLPEncoder` | Lightweight trainable MLP | Обучение с нуля |
| `HybridEncoder` | Attention + GatedMLP layers | Баланс accuracy/speed |
| `MatryoshkaProjection` | Multi-scale embeddings | Clifford dimension alignment |

**Matryoshka Representation Learning** (Kusupati et al., NeurIPS 2022):
- Один энкодер → несколько размерностей [64, 128, 256, 768]
- Каждая размерность сохраняет семантическую информацию
- Идеально для HDIM: clifford_dim (16), hidden_dim (64/256), full_dim (768)
- Loss: `L_matryoshka = Σ_d InfoNCE(emb_d) / n_dims`

```python
from src.models.modern_text_encoder import ModernEncoderConfig, build_modern_encoder

# ModernBERT с Matryoshка
config = ModernEncoderConfig(
    encoder_type="modernbert",
    pretrained_model="answerdotai/ModernBERT-base",
    freeze_pretrained=True,
    use_matryoshka=True,
    matryoshka_dims=[64, 128, 256, 768],
)
encoder = build_modern_encoder(output_dim=256, config=config)

# Lightweight GatedMLP для доменного текста
config = ModernEncoderConfig(
    encoder_type="gated_mlp",
    mlp_hidden_dim=256,
    mlp_num_layers=6,
    mlp_use_glu=True,
)
encoder = build_modern_encoder(output_dim=256, config=config)
```

### Модульность экспертов (unlimited)
```python
# Добавить эксперта без перестройки модели
new_id = model.core_model.pipeline.moe.add_expert(ExpertConfig(hidden_dim=512))

# Удалить малоиспользуемых экспертов
model.core_model.pipeline.moe.remove_expert(idx)

# Добавить домен
model.add_domain('quantum_mechanics')
model.remove_domain('obsolete_domain')
```

---

## 5. Данные

| Файл | Пар | Описание |
|------|-----|----------|
| `data/real_pairs_v8.json` | 330 (212+ / 118-) | **АКТУАЛЬНЫЙ** — все 16 доменных пар покрыты, 35.8% neg |
| `data/real_pairs_v7.json` | 232 (191+ / 41-) | Предыдущий (17.7% neg) |
| `data/real_pairs_v6.json` | 213 (172+ / 41-) | Legacy |
| `data/real_pairs_v5.json` | 175 (139+ / 36-) | Phase 9 рекорд |

**PRIMARY_SCORE** = `pair_margin × 1.0 + STS_exported × 0.3`

### 5.1 v8 Dataset (330 пар, 35.8% negatives)

**Ключевые улучшения v8:**
- Все 16 доменных пар покрыты (v7: domain 1→3 = 0 пар → v8: 8 пар)
- Negative ratio: 35.8% (v7: 17.7%) — лучший контраст
- Domain 1 source: 18→32 пар (включая 1→3)
- Intra-domain pairs увеличены: 0→0 (3→8), 1→1 (3→4), 2→2 (7→8), 3→3 (2→5)
- Группировка по family_id для stratified split

**Domain coverage v8:**
```
        ->0  ->1  ->2  ->3  total
from 0:   8   70   28   20   126
from 1:   7    4   12    8    31
from 2:  29   61    8   17   115
from 3:  11   24   18    5    58
total:   55  159   66   50   330
```

**Скрипт:** `scripts/gen_dataset_v8.py` — генерирует v8 из v7 + новые пары + негативы

## 6. Результаты обучения

### Хронология рекордов
| Phase | Score | Margin | STS | Эпоха | Конфигурация |
|-------|-------|--------|-----|-------|-------------- |
| 5a | 0.967 | — | — | — | AnglE loss + learnable temp |
| 6e | 0.9745 | — | — | ep75 | SoftMoE + v4 данные + seed=77 |
| 7 | 0.804 | — | — | — | ModularMoE (с багами) |
| **8e** | **1.1370** | **0.906** | **0.770** | **ep45** | **SoftMoE + v5 + simple MLP, hidden=256, clifford=16, 4 experts** |
| 11a | 1.0072 | 0.776 | 0.771 | ep50 | Phase9 config + v6 данные |
| 9 | 0.9930 | 0.757 | 0.788 | ep55 | SoftMoE + v5 данные + seed=42 |
| 10a | 0.9347 | — | — | — | GatedProjection + HardNeg + v6 (РЕГРЕССИЯ) |
| 10b | 0.729 | — | — | — | GatedProjection + partial unfreeze (РЕГРЕССИЯ) |
| 10c | 0.9389 | — | — | ep25 | Simple MLP + v6 (best cycle 2) |
| **12** | 0.9279 | 0.702 | 0.753 | ep10 | v7 + augment=50, T_mult=1 (ПРОВАЛ: деградация) |
| **13** | **0.696** | **0.540** | **0.522** | **ep15** | **hidden=512, experts=8, lambda_z=0.001, augment=10, 687K params, v7, cosine_restarts** |
| **14** | **TBD** | — | — | — | **hidden=512, experts=8, Focal-InfoNCE(gamma=0.5), cosine_decay, v8, 687K (в процессе)** |
| **16** | 0.378 | — | — | ep25 | hidden=128, experts=4, T=0.07 — **ДЕГРАДАЦИЯ**: memory drift + MoE collapse (score 0.181 к ep30) |
| **17** | **TBD** | — | — | — | **7×P0 + 5×P1 исправлений, hidden=128, experts=4, T=0.15, reset_memory, z-loss=0.01 (в процессе)** |
| **22** | **TBD** | — | — | — | **Phase 22: ModernBERT + Gradient Surprise + Router Calibration + SC-InfoNCE + Learnable Metric + Adaptive Forgetting** |
| **23** | **0.483** | 0.371 | 0.372 | ep5 | **Phase 23: SOTA Optimal Config — lr=3e-4, batch=64, v8 data, DCL+Uniform, cosine_restarts T_0=25, gradient checkpointing, sim-preserving router** |
| **24** | **TBD** | — | — | — | **Phase 24: ModernBERT frozen + Matryoshka [64,128,256,768] + Multi-Scale InfoNCE** |
| **25** | **—** | — | — | — | **Phase 25: freeze_sbert_bottom_frac + weight_decay для SBERT + data v10 (1036 pairs) + SBERT cache** |
| **26a** | **1.1063** | — | — | ep45 | **Phase 26a: DomainExpertPool + SharedExpert + AuxLossFree + ExpertOrtho, augment=3, no sts/dcl** |
| **26b** | **1.1513** | — | — | ep15 | **Phase 26b: +sts=0.15, dcl=0.2, learnable_temp, augment=5** |
| **26c** | **1.1542** | 0.993 | 0.537 | ep15 | **Phase 26c: +uniformity=0.1, sts=0.3** |
| **Run 11** | **1.1528** | 0.9866 | 0.5538 | ep27 | **Session 13: lambda_pair=0.35, temp=0.12, patience=15, cosine_decay** |
| **Run 12** | **1.1480** | 0.9821 | — | ep27 | **Session 13: epochs=35, patience=20 — больше эпох не помогает** |
| **Run 13** | **1.1706** | 1.0073 | — | ep28 | **Session 13: lambda_pair=0.40, margin>1.0 впервые** |
| **Run 18** | **1.1814** | **1.0224** | 0.537 | ep13 | **SESSION 13 RECORD: temp=0.10, lambda_pair=0.40, ранний пик ep13** |

---

### Phase 16 — Диагностика (деградация)

| Эпоха | Score | loss_memory | loss_routing | Причина |
|-------|-------|-------------|--------------|----------|
| ep5 | ~0.2 | 2.029 | 0.1893 | базовый уровень |
| ep25 | **0.378** | ~4.1 | ~0.08 | **пик score** |
| ep30 | 0.181 | ~5.0 | 0.0198 | **обвал −52%** |
| ep45 | — | 6.686 | 0.0198 | memory drift, MoE collapse |

**Патологии Phase 16 (4 причины):**
1. **memory drift**: `loss_memory` 2.029→6.686 (×3.3) — `reset_memory()` не вызывался между эпохами
2. **MoE mode collapse**: `loss_routing` 0.1893→0.0198 — все токены маршрутятся к одному эксперту; Z-loss (`lambda_z`) не был активирован
3. **Статический load-balance loss**: не обновлялся по dispatch-статистике — неверные градиенты
4. **80% фиктивных записей**: лог содержал score=0 на non-eval эпохах — искажал анализ

---

## 7. Phase 13–17 Конфигурации

### Phase 13 + 14 (hidden=512, experts=8, 687K params)

### Модель: 687,891 параметров
| Компонент | Параметры | Размерность |
|-----------|-----------|-------------|
| text_encoder.projection | 643,328 | 768→384→512 (simple MLP) |
| core_model.pipeline | 35,851 | clifford_dim=16, hidden=512, 8 experts |
| training_inv_head | 8,712 | 16→512 |

### Phase 13 конфиг (best=0.696, ep15 — LR restart проблема)
```bash
python scripts/gpu_train.py \
  --epochs 200 --hidden_dim 512 --num_experts 8 --num_domains 4 \
  --pretrained_encoder --soft_router \
  --real_pairs data/real_pairs_v7.json --augment_factor 10 \
  --lambda_pair 0.4 --lambda_sts 0.2 --lambda_angle 0.3 \
  --lambda_iso 0.1 --lambda_routing 0.05 --lambda_memory 0.01 \
  --lambda_z 0.001 \
  --use_infonce --infonce_temperature 0.1 --learnable_temperature \
  --early_stopping_patience 40 \
  --lr 0.0005 --seed 42 --batch_size 32 \
  --scheduler_type cosine_restarts --t_mult 2 --warmup_epochs 3 \
  --eval_every 5 --save_every 25 \
  --output_dir artifacts/phase13_scaled
```
**Проблема:** cosine_restarts LR restart (ep20) дестабилизирует модель, score падает с 0.696→0.581.

### Phase 14 конфиг (в процессе — исправления + SOTA)
```bash
python scripts/gpu_train.py \
  --epochs 200 --hidden_dim 512 --num_experts 8 --num_domains 4 \
  --pretrained_encoder --soft_router \
  --real_pairs data/real_pairs_v8.json --augment_factor 10 \
  --lambda_pair 0.4 --lambda_sts 0.2 --lambda_angle 0.3 \
  --lambda_iso 0.1 --lambda_routing 0.05 --lambda_memory 0.01 \
  --lambda_z 0.001 \
  --use_infonce --infonce_temperature 0.1 \
  --focal_gamma 0.5 \
  --temp_schedule warm_restart --tau_max 0.1 --tau_min 0.01 \
  --scheduler_type cosine_decay \
  --early_stopping_patience 40 \
  --lr 0.0005 --seed 42 --batch_size 32 \
  --eval_every 5 --save_every 25 \
  --output_dir artifacts/phase14_sota
```

**Ключевые изменения Phase14:**
- `--scheduler_type cosine_decay` вместо cosine_restarts (нет дестабилизирующих рестартов)
- `--focal_gamma 0.5` — Focal-InfoNCE для focus на hardest negatives
- `--temp_schedule warm_restart` — динамическая температура (tau_max→tau_min)
- `--real_pairs data/real_pairs_v8.json` — 330 пар, 35.8% neg, все домены

### Phase 17 конфиг (7×P0 + 5×P1 исправлений)

```bat
@echo off
REM Phase 17 Training — HypercomplexAI HDIM
REM Fixes: C1-C7 (P0), A1-A6 (P1)

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
```

**Ключевые изменения Phase 17 (vs Phase 16):**
- `--infonce_temperature 0.15` вместо 0.07 — снижение overconfidence
- `--lambda_z 0.01` активирован — предотвращает MoE mode collapse
- `--lambda_memory 0.05` снижен (vs default) — уменьшает memory drift
- `reset_memory()` вызывается в `set_epoch()` каждую эпоху (fix C5)
- `focal_gamma` применяется только к знаменателю InfoNCE (fix C6)
- Динамический load-balance loss в SoftMoERouter (fix C2)
- fp32 TTT path в TitansMemory — нет NaN/Inf при AMP (fix C4)
- LR restart detector в `gpu_train.py` → auto `reset_memory()` (fix A2)

**Исправления P0 (критические):**

| ID | Файл | Проблема | Решение |
|----|------|----------|---------|
| C1 | `soft_moe_router.py` | T=1 → dispatch не identity | Guard: `if T == 1: return identity dispatch` |
| C2 | `soft_moe_router.py` | Load balance loss статический | Динамический расчёт по текущему dispatch |
| C3 | `soft_moe_router.py` | In-place slice → сбой AMP/compile | Заменён на `torch.cat` (out-of-place) |
| C4 | `titans_memory.py` | NaN/Inf при AMP в TTT path | fp32 cast для TTT градиентного шага |
| C5 | `trainer.py` | `reset_memory()` не вызывался | Вызов в `set_epoch()` перед каждой эпохой |
| C6 | `trainer.py` | Focal gamma к числителю и знаменателю | gamma только к знаменателю (правильный gradient) |
| C7 | `hierarchical_memory.py` | Non-leaf tensor `requires_grad` ошибка | `detach().float()` + `requires_grad_(True)` |

**Исправления P1 (важные):**

| ID | Файл | Проблема | Решение |
|----|------|----------|---------|
| A1 | `hdim_model.py` | Losses не нормированы по n_domains | Деление на `n_domains` |
| A2 | `gpu_train.py` | Нет детектора LR restart | Auto `reset_memory()` при скачке LR >2× |
| A3 | `trainer.py` | Default temperature 0.07 — слишком низкий | Изменён на 0.15 |
| A4 | `soft_moe_router.py` | Нет Z-loss | Добавлен Z-loss (ST-MoE: `(log Σexp(z))²`) |
| A6 | `gpu_train.py` | Фиктивные лог-записи с zeros | Лог только на eval эпохах |

### Конфигурация запуска Phase 8e (рекорд 1.1370)
```bash
cd E:/hypercoplexAI && python scripts/gpu_train.py \
  --epochs 200 --hidden_dim 256 --num_experts 4 --num_domains 4 \
  --pretrained_encoder --soft_router \
  --real_pairs data/real_pairs_v5.json --augment_factor 30 \
  --lambda_pair 0.4 --lambda_sts 0.2 --lambda_angle 0.3 \
  --lambda_iso 0.1 --lambda_routing 0.05 --lambda_memory 0.01 \
  --use_infonce --infonce_temperature 0.1 --learnable_temperature \
  --early_stopping_patience 40 \
  --lr 0.0005 --seed 42 --batch_size 32 \
  --scheduler_type cosine_restarts --warmup_epochs 3 \
  --eval_every 5 --save_every 25 \
  --output_dir artifacts/phase8e_soft_eval5
```

**Результат:** score=1.1370 (ep45), pair_margin=0.9059, STS=0.7701
**GPU:** NVIDIA RTX 3070 Laptop (8.6GB), PyTorch 2.6+cu124, 0.024 GB VRAM

### Phase 12 — Провал (T_mult=1)

```bash
# ПРОВАЛ: score=0.9279 на ep10, затем деградация до 0.8010
# Причина: T_mult=1 создаёт короткие циклы (20 эп), модель не сходится
# loss_memory вырос с 2.41 до 3.67 — память расходится
```

| Эпоха | Score | Margin | STS | loss_memory | Причина |
|-------|-------|--------|-----|-------------|---------|
| ep10 | 0.9279 | 0.7019 | 0.7534 | 2.41 | **BEST** — цикл 1 |
| ep15 | 0.8981 | 0.6651 | 0.7766 | 2.65 | цикл 1 конец |
| ep20 | 0.9212 | 0.6986 | 0.7418 | 3.16 | рестарт LR |
| ep29 | 0.8010 | 0.5718 | 0.7640 | 3.67 | деградация |

### Phase 12b — Исправленная (запущена 12.03.2026)

```bash
python scripts/gpu_train.py \
  --epochs 200 --hidden_dim 256 --num_experts 4 --num_domains 4 \
  --pretrained_encoder --soft_router \
  --real_pairs data/real_pairs_v7.json --augment_factor 50 \
  --lambda_pair 0.4 --lambda_sts 0.2 --lambda_angle 0.3 \
  --lambda_iso 0.1 --lambda_routing 0.05 --lambda_memory 0.01 \
  --use_infonce --infonce_temperature 0.1 --learnable_temperature \
  --early_stopping_patience 25 \
  --lr 0.0005 --seed 42 --batch_size 32 \
  --scheduler_type cosine_restarts --t_mult 2 --warmup_epochs 5 \
  --eval_every 5 --save_every 25 \
  --output_dir artifacts/phase12b_v7_fixed
```

**Ключевое исправление:** `--t_mult 2` (возврат к Phase8e) + `--warmup_epochs 5` (T_0=30)
**Ожидание:** score 1.15-1.20 на цикле 3 (ep60-90)

### Phase 13 — Конфигурация (hidden=512, experts=8, lambda_z)

```bash
# Phase 13: scaled model + router z-loss + augment=10
cd E:/hypercoplexAI && python scripts/gpu_train.py \
  --epochs 200 --hidden_dim 512 --num_experts 8 --num_domains 4 \
  --pretrained_encoder --soft_router \
  --real_pairs data/real_pairs_v7.json --augment_factor 10 \
  --lambda_pair 0.4 --lambda_sts 0.2 --lambda_angle 0.3 \
  --lambda_iso 0.1 --lambda_routing 0.05 --lambda_memory 0.01 \
  --lambda_z 0.001 \
  --use_infonce --infonce_temperature 0.1 --learnable_temperature \
  --early_stopping_patience 40 \
  --lr 0.0005 --seed 42 --batch_size 32 \
  --scheduler_type cosine_restarts --t_mult 2 --warmup_epochs 3 \
  --eval_every 5 --save_every 25 \
  --output_dir artifacts/phase13_scaled
```

**Ключевые изменения:**
- `hidden_dim 512` — удвоение capacity для лучшего представления
- `num_experts 8` — 4→8 экспертов для finer-grained routing
- `lambda_z 0.001` — Router Z-Loss (ST-MoE) для стабильности logits
- `augment_factor 10` — снижение с 50 до 10 (данные не настоящая аугментация, только shuffle)

**Размер модели:** ~1.6M параметров (vs 414K в Phase 8e)
**Текущий результат:** score=0.3076 (3 эп, augment=10) — training в процессе


### Phase 19 конфиг (в процессе — антиколлапс + v7 данные)

```bash
python scripts/gpu_train.py   --epochs 200 --hidden_dim 256 --num_experts 4 --num_domains 4   --pretrained_encoder --soft_router   --real_pairs data/real_pairs_v7.json --augment_factor 30   --lambda_pair 0.4 --lambda_sts 0.2 --lambda_angle 0.3   --lambda_iso 0.1 --lambda_routing 0.05 --lambda_memory 0.01   --use_infonce --infonce_temperature 0.1 --learnable_temperature   --focal_gamma 0.5 --early_stopping_patience 40   --lr 0.0005 --seed 42 --batch_size 32   --scheduler_type cosine_restarts --t_mult 2 --warmup_epochs 3   --eval_every 5 --output_dir artifacts/phase19_run --amp
```
**Статус:** ep11, score=0.489 (прогресс ep5→ep10: 0.444→0.489)
**Скорость:** ~167 сек/эп с v7 (18420 items), GPU=0.21GB

---

### Phase 20 конфиг (DCL + Uniformity + v5 + batch_size=64)

```bash
python scripts/gpu_train.py   --epochs 200 --hidden_dim 256 --num_experts 4 --num_domains 4   --pretrained_encoder --soft_router   --real_pairs data/real_pairs_v5.json --augment_factor 30   --lambda_pair 0.4 --lambda_sts 0.2 --lambda_angle 0.3   --lambda_iso 0.1 --lambda_routing 0.05 --lambda_memory 0.01   --lambda_z 0.01 --lambda_dcl 0.3 --lambda_uniformity 0.1   --use_infonce --infonce_temperature 0.1 --learnable_temperature   --focal_gamma 0.5 --early_stopping_patience 40   --lr 0.0005 --seed 42 --batch_size 64   --scheduler_type cosine_restarts --t_mult 2 --warmup_epochs 3   --eval_every 5 --output_dir artifacts/phase20_dcl_uniform --amp
```

**Ключевые изменения от Phase 8e рекорда:**
- `lambda_dcl 0.3` — DCL loss (позитив убран из знаменателя InfoNCE)
- `lambda_uniformity 0.1` — Uniformity+Alignment (равномерное распределение на гиперсфере)
- `lambda_z 0.01` — Router Z-Loss (стабилизация MoE logits)
- `batch_size 64` — 2x больше негативов in-batch (VRAM позволяет: 0.2GB из 8.6GB)
- `data v5` — возврат к Phase8e рекордным данным
- `num_workers=4` — 3-4x ускорение DataLoader (автоматически через gpu_train.py)

**Ожидаемый score:** 1.18-1.23 на цикле 3 (ep45-60)

---

## 8. Паттерны обучения

- **Ep1-4:** score=0 (eval только на кратных 5)
- **Ep5:** score~0.75 — первый прыжок
- **Ep20:** LR рестарт → score растёт
- **Ep45-55:** основной рост margin 0.67→0.76
- **CosineAnnealingWarmRestarts T_0=20, T_mult=2:** цикл 20→40→80 эпох
- **STS ≈ 0.79-0.82** стабильно (frozen SBERT даёт хорошую базу)
- **Best обычно на 3-м цикле** (ep45-60 при T_0=20)

### Паттерн Phase 16 — деградация (anti-pattern)

```
ep5-25:  score растёт 0.2 → 0.378  (memory работает, но drift накапливается)
ep25:    ПИКОВЫЙ score 0.378        (memory drift уже значительный: loss_memory≈4.1)
ep25-30: score обваливается -52%   (MoE collapse: loss_routing 0.08→0.02)
ep30-45: score стагнирует ~0.18    (все токены → 1 эксперт, память расходится)
```

**Признаки деградации (диагностические сигналы):**
- `loss_memory` растёт монотонно без plateau → нет `reset_memory()` между эпохами
- `loss_routing` падает ниже 0.02 → MoE mode collapse (нужен Z-loss)
- score падает при стабильных или улучшающихся других лоссах → routing collapse
- 80%+ записей лога имеют score=0 → фиктивные записи на non-eval эпохах

### Ожидания для Phase 17

- **Ep5:** score ~0.25-0.35 (hidden=128, меньше capacity)
- **Ep15-25:** score ~0.4-0.5 (после исправления всех P0)
- **loss_memory:** должна стабилизироваться (reset каждую эпоху)
- **loss_routing:** должна оставаться в диапазоне 0.05-0.15 (Z-loss активен)
- **Нет score=0 записей** в логе (только eval эпохи логируются)

---

## 9. SOTA улучшения

| Метод | Источник | Статус | Ожидаемый прирост |
|-------|----------|--------|-------------------|
| **Router Z-Loss** | ST-MoE (Zoph et al., 2022) | ✅ Реализовано (Phase 13) | +0.005-0.015 |
| **Focal-InfoNCE** | Hou & Li, EMNLP 2023 | ✅ Реализовано (Phase 14) | +0.010-0.015 |
| **Temperature Scheduling** | warm_restart, synced с LR | ✅ Реализовано (Phase 14) | +0.003-0.008 |
| **cosine_decay (нет restarts)** | альтернатива cosine_restarts | ✅ Реализовано (Phase 14) | устраняет дестабилизацию |
| **Learnable Temperature** | CLIP, 2023 | ✅ Реализовано | стабильные градиенты |
| **fp32 TTT path (AMP safe)** | Phase 17 fix C4 | ✅ Реализовано (Phase 17) | устраняет NaN/Inf при AMP |
| **reset_memory() per epoch** | Phase 17 fix C5 | ✅ Реализовано (Phase 17) | устраняет memory drift |
| **Dynamic load-balance loss** | Phase 17 fix C2 | ✅ Реализовано (Phase 17) | корректные градиенты MoE |
| **Focal gamma → denominator only** | Phase 17 fix C6 | ✅ Реализовано (Phase 17) | правильный InfoNCE gradient |
| **LR-restart detector + auto reset** | Phase 17 fix A2 | ✅ Реализовано (Phase 17) | стабильность при LR скачках |
| **SparseMixer / Dense Backprop** | Panda et al., Apr 2025 | ⚠️ Запланировано | лучшие градиенты для роутера |
| **DCL (Decoupled Contrastive Loss)** | Yeh et al., NeurIPS 2022 | ✅ Реализовано (Phase 20) | +0.020-0.040 pair_margin |
| **Uniformity + Alignment** | Wang & Isola, ICML 2020 | ✅ Реализовано (Phase 20) | +0.010-0.020 STS |
| **DataLoader num_workers=4** | PyTorch best practices | ✅ Реализовано (Phase 20) | 3-4x скорость epoch |
| **Similarity-Preserving Router** | ICLR 2026 | ✅ Реализовано (Phase 21) | routing отражает семантическую близость |
| **Expert Dropout** | MoE регуляризация | ✅ Реализовано (Phase 21) | +0.01-0.03, предотвращает overfitting |
| **Gated Memory Fusion** | learnable gate for memory | ✅ Реализовано (Phase 21) | устраняет memory drift |
| **Gradient Isolation для Memory** | stop-gradient для стабильности | ✅ Реализовано (Phase 21) | предотвращает градиентный конфликт memory vs main |
| **Precomputed Clifford signs** | hypercomplex.py оптимизация | ✅ Реализовано (Phase 21) | ~15% ускорение forward pass |
| **Router Calibration (R2-T2)** | ICML 2025 (Chen et al.) | ✅ Реализовано (Phase 22) | test-time calibration для роутера |
| **Gradient Surprise** | Titans, NeurIPS 2025 | ✅ Реализовано (Phase 22) | adaptive memory update на основе surprise |
| **Adaptive Forgetting** | Titans, NeurIPS 2025 | ✅ Реализовано (Phase 22) | high surprise → less forgetting |
| **Learnable Clifford Metric** | CliffordNet, 2026 | ✅ Реализовано (Phase 22) | per-blade metric scaling |
| **SC-InfoNCE** | Cheng et al., Nov 2025 | ✅ Реализовано (Phase 22) | cluster-aware temperature |
| **ModernBERT Encoder** | Warner et al., 2024 | ✅ Реализовано (Phase 22) | 16x контекст, быстрее SBERT |
| **Matryoshka Projection** | Kusupati et al., NeurIPS 2022 | ✅ Реализовано (Phase 22) | multi-scale embeddings [16,64,128,256] |
| **Optimal Hyperparameters** | Research synthesis 2026 | ✅ Внедрено (Phase 23) | lr=3e-4, batch=64, balanced losses |

**Router Z-Loss** — critical при num_experts > 4:
```python
z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()
```

**Focal-InfoNCE** (Hou & Li, EMNLP 2023):
```python
# Downweights easy negatives via gamma scaling
focal_sim = torch.exp(sim_matrix * gamma)  # gamma < 1
log_denom = torch.logsumexp(focal_sim.masked_fill(eye, 0), dim=-1)
loss = -(log(sim_diag) - log_denom[pos_indices]).mean()
```

---

## 10. Исправленные баги

### Phase 8
| БАГ | Файл | Проблема | Фикс |
|-----|------|----------|------|
| 1 | `hdim_model.py` | Хардкод `config.num_experts` | → `moe.num_experts` |
| 2 | `hdim_model.py` | `reset_memory` без guard | `hasattr(moe, 'train_scores')` |
| 3 | `trainer.py` | fp16 overflow в HardNeg | `-1e9` → `-1e4` |

### Phase 13 (аудит)
| БАГ | Файл | Проблема | Фикс |
|-----|------|----------|------|
| 4 | `metrics.py` | re-concat `all_group_ids` в цикле | переиспользовать pre-computed `groups` |
| 5 | `trainer.py` | GradScaler state не сохранялся | `scaler.load_state_dict()` в `load_checkpoint` |
| 6 | `train.py` | Избыточная условная проверка | `needs_text` уже включает все условия |

### Phase 14 (исправление training instability)
| Проблема | Причина | Решение |
|----------|---------|---------|
| LR restart дестабилизирует модель | CosineAnnealingWarmRestarts T_0=20 restart | → `cosine_decay` (монотонное снижение) |
| Негативы слишком далеко | Все negatives equal weight | → Focal-InfoNCE (gamma=0.5) |
| Зафиксированная температура | infonce_temperature=0.01 | → warm_restart schedule (0.1→0.01) |

### Phase 20 (новые SOTA методы + оптимизации)

| Изменение | Файл | Описание | Статус |
|-----------|------|----------|---------|
| DCL loss | `trainer.py` | Decoupled Contrastive Loss — убираем positive из denominator InfoNCE | ✅ Реализовано |
| Uniformity+Alignment | `trainer.py` | Wang & Isola 2020 — явное alignment + uniform гиперсфера | ✅ Реализовано |
| lambda_dcl, lambda_uniformity | `gpu_train.py` | Новые аргументы обучения | ✅ Реализовано |
| DataLoader num_workers=4 | `gpu_train.py` | Параллельная загрузка данных | ✅ Реализовано |
| persistent_workers + prefetch | `gpu_train.py` | Снижение CPU bottleneck | ✅ Реализовано |

### Phase 17 (7×P0 + 5×P1 исправлений)

**Критические баги P0 — устраняли silent failures и неверные градиенты:**

| ID | Файл | Проблема | Решение |
|----|------|----------|---------|
| C1 | `soft_moe_router.py` | T=1 → dispatch не identity, некорректный forward | Guard: `if T == 1: return identity dispatch` |
| C2 | `soft_moe_router.py` | Load balance loss статический (не обновлялся) | Пересчёт по текущей dispatch-матрице каждый forward |
| C3 | `soft_moe_router.py` | In-place slice `buf[:, i] = ...` → сбой AMP/`torch.compile` | Заменён на `torch.cat` (out-of-place concat) |
| C4 | `titans_memory.py` | NaN/Inf в TTT gradient step при AMP (fp16 overflow) | Явный cast TTT вычислений в fp32 |
| C5 | `trainer.py` | `reset_memory()` не вызывался между эпохами → memory drift | Вызов `reset_memory()` внутри `set_epoch()` |
| C6 | `trainer.py` | Focal gamma применялась к числителю И знаменателю InfoNCE | gamma только к знаменателю (математически корректно) |
| C7 | `hierarchical_memory.py` | `requires_grad` на non-leaf tensor → RuntimeError | `detach().float().requires_grad_(True)` |

**Важные проблемы P1 — улучшали стабильность и качество:**

| ID | Файл | Проблема | Решение |
|----|------|----------|---------|
| A1 | `hdim_model.py` | Loss sum не нормирован по числу доменов | Деление каждого loss-слагаемого на `n_domains` |
| A2 | `gpu_train.py` | Нет детектора LR restart → память не сбрасывается | Детектор скачка LR >2× + auto `reset_memory()` |
| A3 | `trainer.py` | Default temperature 0.07 → overconfidence, плохие градиенты | Изменён дефолт на 0.15 |
| A4 | `soft_moe_router.py` | Z-loss не был активирован по умолчанию | Z-loss включён: `lambda_z=0.01` в Phase 17 конфиге |
| A6 | `gpu_train.py` | Лог писал score=0 на non-eval эпохах (80% фиктивных записей) | Логирование только при выполнении eval |

### Phase 21 (SOTA методы + стабилизация)

| БАГ | Файл | Проблема | Фикс |
|-----|------|----------|------|
| B1 | `hdim_pipeline.py` | Memory выход складывался с входом напрямую → gradient conflict | Gated Memory Fusion: `gate = σ(W_g[x; m])`, `out = gate * m + (1-gate) * x` |
| B14 | `hypercomplex.py` | Повторный расчёт signs (involute/reverse) каждый forward | Precomputed signs в `__init__`, кэширование в буфере |
| B15 | `domain_operators.py` | R не нормализовалась перед rotation → numerical drift | `R = R / R.norm()` после каждого обновления |
| B16 | `domain_operators.py` | Несогласованная нормализация R в forward/inverse | Единый нормализационный паттерн во всех операциях |
| B17 | `hdim_model.py` | Loss не нормирован по размеру группы экспертов | Деление на `group_size` при агрегации loss |
| B29 | `model_factory.py` | `z_loss_weight` не пробрасывался в конструктор модели | Добавлен passthrough параметра `z_loss_weight` в build-функции |
| B33 | `trainer.py` | `_log_temp` не регистрировался как buffer → не сохранялся в checkpoint | `self.register_buffer('_log_temp', ...)` |
| B37 | `gpu_train.py` | T_0 вычислялся без учёта `warmup_epochs` → первый цикл короче | `T_0 = base_T_0 + warmup_epochs` |

---

## 11. Phase 21 — SOTA Stabilization & Routing

### Конфигурация запуска

```bash
python scripts/gpu_train.py \
  --epochs 200 --hidden_dim 256 --num_experts 4 --num_domains 4 \
  --pretrained_encoder --soft_router \
  --real_pairs data/real_pairs_v5.json --augment_factor 30 \
  --lambda_pair 0.4 --lambda_sts 0.2 --lambda_angle 0.3 \
  --lambda_iso 0.1 --lambda_routing 0.05 --lambda_memory 0.01 \
  --lambda_z 0.01 --lambda_balance 0.05 \
  --use_infonce --infonce_temperature 0.1 --learnable_temperature \
  --focal_gamma 0.5 --early_stopping_patience 40 \
  --lr 0.0005 --seed 42 --batch_size 32 \
  --scheduler_type cosine_restarts --t_mult 2 --warmup_epochs 3 \
  --expert_dropout 0.1 --similarity_preserving_router \
  --eval_every 5 --save_every 25 \
  --output_dir artifacts/phase21_sota_stabilized --amp
```

**Ключевые изменения от Phase 20:**
- `--lambda_balance 0.05` — Similarity-Preserving Router loss вес
- `--expert_dropout 0.1` — 10% dropout экспертов при обучении
- `--similarity_preserving_router` — включает routing по семантической близости
- Gated Memory Fusion включена автоматически в `hdim_pipeline.py`
- Gradient Isolation для memory-путей (stop-gradient перед основным графом)
- Precomputed Clifford signs (hypercomplex.py кэш)

### Ожидаемые улучшения

| Компонент | Phase 20 | Phase 21 (ожидание) | Механизм |
|-----------|----------|---------------------|----------|
| pair_margin | ~0.90 | 0.92-0.95 | Similarity-Preserving Router направляет похожие объекты к одним экспертам |
| STS | ~0.77 | 0.78-0.80 | Expert Dropout предотвращает co-adaptation |
| loss_memory | растёт | стабильна | Gated Memory Fusion контролирует memory contribution |
| training speed | baseline | +15% | Precomputed Clifford signs убирают повторные вычисления |
| MoE collapse | при lambda_z<0.005 | устранён | lambda_z=0.01 + Expert Dropout |

### SOTA методы внедрены

1. **Similarity-Preserving Router (ICLR 2026)** — routing loss основан на косинусной близости: если входы семантически близки, их routing-векторы тоже должны быть близки. Это естественная индуктивная bias для MoE.

2. **Expert Dropout (p=0.1)** — при каждом forward pass 10% экспертов случайно отключаются. Это:
   - Регуляризует экспертов (предотвращает memorization)
   - Улучшает load balancing (оставшиеся эксперты берут больше нагрузки)
   - Повышает robustness (модель не зависит от конкретного эксперта)

3. **Gated Memory Fusion** — вместо простого сложения `x + memory(x)`, используется learnable gate:
   ```python
   gate = torch.sigmoid(W_gate(torch.cat([x, mem], dim=-1)))
   fused = gate * mem + (1 - gate) * x
   ```
   Это позволяет модели **игнорировать** memory когда она вредна (шум, drift).

4. **Gradient Isolation для Memory** — memory-выходы проходят через `detach()` перед основным графом вычислений, предотвращая конфликт градиентов между memory update и основной loss оптимизацией.

5. **Precomputed Clifford Signs** — знаки для involute и reverse операций вычисляются один раз при инициализации и кэшируются в буфере, вместо повторного вычисления на каждом forward pass.

---

## 12. Phase 22 — ModernBERT + SOTA Memory + Router Calibration

### Нововведения Phase 22

**1. Router Calibration (R2-T2, ICML 2025)**
- Small calibration head на основе mean-pooled входа
- Корректирует dispatch logits при inference (test-time)
- Активируется через `model.enable_router_calibration()`
- `SoftMoERouter.calibration_head`: Linear→ReLU→Linear(num_experts)

**2. Gradient-Based Surprise (Titans, NeurIPS 2025)**
- Surprise = норма градиента по ключу при retrieval loss
- `TitansMemoryModule._compute_surprise()`: fp32 градиентный шаг
- Используется для adaptive forgetting (high surprise → less forgetting)

**3. Adaptive Forgetting (Titans, NeurIPS 2025)**
- `α_effective = α_base * (1 - 0.5 * sigmoid(surprise - 1.0))`
- Высокий surprise → меньше забывания → сохранение важных ассоциаций
- Активируется через `model.enable_adaptive_forgetting()`

**4. Learnable Clifford Metric (CliffordNet, 2026)**
- `CliffordAlgebra.use_learnable_metric = True`
- Per-blade scaling parameters для геометрического произведения
- Активируется через `model.enable_learnable_metric()`

**5. SC-InfoNCE (Cheng et al., Nov 2025)**
- Cluster-aware temperature scaling
- Temperature адаптируется на основе кластерной структуры батча
- Активируется через `--sc_temperature` флаг

**6. ModernBERT Encoder (Warner et al., 2024)**
- `answerdotai/ModernBERT-base` — 149M params, 8192 контекст
- Frozen + Simple MLP projection (768→256)
- CLI: `--modernbert_encoder --freeze_modernbert`
- Factory: `build_modernbert_hdim_model()`

**7. Matryoshka Projection (Kusupati et al., NeurIPS 2022)**
- Multi-scale embeddings: [16, 64, 128, 256, 768]
- Один энкодер → полезные представления на нескольких размерностях
- `MatryoshkaProjection` в `src/models/modern_text_encoder.py`

### Запуск Phase 22

```bash
python scripts/gpu_train.py \
  --epochs 200 --hidden_dim 256 --num_experts 4 --num_domains 4 \
  --batch_size 64 --lr 3e-4 \
  --modernbert_encoder --freeze_modernbert \
  --soft_router \
  --real_pairs data/real_pairs_v8.json --augment_factor 30 \
  --lambda_pair 0.35 --lambda_sts 0.2 --lambda_angle 0.25 \
  --lambda_iso 0.1 --lambda_routing 0.05 --lambda_memory 0.015 \
  --lambda_z 0.01 --lambda_dcl 0.25 --lambda_uniformity 0.08 \
  --use_infonce --infonce_temperature 0.15 --learnable_temperature \
  --focal_gamma 0.5 \
  --gradient_surprise --adaptive_forgetting \
  --router_calibration --sc_temperature \
  --learnable_metric \
  --scheduler_type cosine_restarts --t_mult 2 --warmup_epochs 5 \
  --early_stopping_patience 40 --eval_every 5 --save_every 25 \
  --output_dir artifacts/phase22_sota --amp --seed 42
```

---

## 13. Phase 23 — SOTA Optimal Configuration

### Гиперпараметры (исследование 2024-2026)

Фаза 23 объединяет все SOTA улучшения с оптимальными гиперпараметрами, выведенными из анализа 12+ исследований:

| Параметр | Phase 8e (рекорд) | Phase 23 (оптимальный) | Источник |
|---|---|---|---|
| lr | 0.0005 | 3e-4 | Research synthesis 2026 |
| batch_size | 32 | 64 | InfoNCE: больше негативов |
| T_0 (warmup) | 20 | 25 | Longer initial cycle |
| T_mult | 2 | 2 | Stable cycles |
| lambda_pair | 0.4 | 0.35 | Баланс с DCL |
| lambda_dcl | 0.0 | 0.25 | Decoupled Contrastive |
| lambda_uniformity | 0.0 | 0.08 | Uniformity+Alignment |
| lambda_memory | 0.01 | 0.015 | Увеличен для стабильности |
| lambda_z | 0.01 | 0.01 | Anti-collapse |
| focal_gamma | 1.0 | 0.5 | Focal-InfoNCE |
| temperature | 0.1 (fixed) | 0.15 (learnable) | Адаптивная |
| gradient_checkpointing | No | Yes | Экономия VRAM |
| similarity_router | No | Yes | ICLR 2026 |

### Запуск Phase 23

```bash
scripts/phase23_train.bat
```

Или вручную:
```bash
python scripts/gpu_train.py \
  --epochs 200 --hidden_dim 256 --num_experts 4 --num_domains 4 \
  --batch_size 64 --lr 3e-4 \
  --pretrained_encoder --soft_router \
  --real_pairs data/real_pairs_v8.json --augment_factor 30 \
  --lambda_pair 0.35 --lambda_sts 0.2 --lambda_angle 0.25 \
  --lambda_iso 0.1 --lambda_routing 0.05 --lambda_memory 0.015 \
  --lambda_z 0.01 --lambda_dcl 0.25 --lambda_uniformity 0.08 \
  --use_infonce --infonce_temperature 0.15 --learnable_temperature \
  --focal_gamma 0.5 \
  --scheduler_type cosine_restarts --t_mult 2 --warmup_epochs 5 \
  --gradient_checkpointing \
  --similarity_preserving_router \
  --early_stopping_patience 40 --eval_every 5 --save_every 25 \
  --output_dir artifacts/phase23_optimal --amp --seed 42
```

**Цель:** score > 1.15 (новый рекорд), pair_margin > 0.92, STS > 0.79

### Прогресс Phase 23 (в процессе)
| Эпоха | train_loss | score | pair_margin | STS | loss_memory | loss_routing | Notes |
|-------|-----------|-------|-------------|-----|-------------|--------------|-------|
| ep5 | 1.341 | 0.483 | 0.371 | 0.372 | 0.0064 | -0.0596 | Первый eval, sim-router работает |
| ep10 | 1.141 | 0.403 | 0.309 | 0.315 | 0.0061 | -0.0592 | Val-train gap растёт (early overfit?) |

---

## 14. Phase 24 — ModernBERT + Matryoshka Multi-Scale

### Концепция
Phase 24 добавляет ModernBERT frozen encoder с Matryoshka Representation Learning — multi-scale InfoNCE loss на размерностях [64, 128, 256, 768]. Ожидаемый прирост: +0.02-0.04 score.

### Ключевые изменения
- **ModernBertEncoder** (`modern_text_encoder.py`): frozen ModernBERT-base + MatryoshkaProjection
- **Matryoshka Loss** (`trainer.py`): InfoNCE на каждом масштабе, усреднённый
- **Batch injection**: `matryoshka_source` / `matryoshka_target` / `matryoshka_embeddings` добавляются в batch
- **TextHDIMModel.encode_texts_matryoshka()**: возвращает `(full_encoding, scales_dict)` или `(full_encoding, None)`
- Исправлен баг: `ModernBertEncoder` не устанавливал `self.use_matryoshka = True`

### Запуск Phase 24
```bash
scripts/phase24_modernbert.bat
```

### Ожидаемый путь
1. Phase 23 завершить (SBERT baseline → score > 1.0)
2. Phase 24 запустить с ModernBERT + Matryoshka
3. Если Matryoshka даст +0.03, комбинировать с Phase 23 hyperparams

---

## 17. Phase 28–30: MoEKernel + TitansMemory RAG + Bugs

### Phase 28 — MoEKernel (2026-03-18)

4 доменно-специализированных эксперта (560K params):
- `MathExpert`: bottleneck hidden×2, две GELU нелинейности
- `LanguageExpert`: pre-LayerNorm перед FFN
- `CodeExpert`: SiLU вместо GELU
- `ScienceExpert`: Tanh для физических величин

Real-model benchmark (SBERT + real_pairs_v10.json, 5 эп):

| Метрика | SoftMoERouter | MoEKernel | Прирост |
|---|---|---|---|
| score | 0.300 | 1.067 | +256% |
| pair_margin | 0.000 | 0.902 | ∞ |
| train_loss (ep5) | 0.930 | 0.274 | -71% |

Numerical Python verification/tests: 159/159 PASS in `verify_lean4_numerical.py`. pytest: 453 PASS (+45 moe_kernel тестов).

### Phase 29 — TitansMemory RAG API (2026-03-19)

- `retrieve_only(k)`: RAG-совместимое извлечение без обновления памяти
- `freeze_memory()` / `unfreeze_memory()`: детерминированные embeddings для RAG
- `_frozen` guard: при RAG-режиме update_memory автоматически отключается

### Phase 30 — Bug Fixes (commit b5decaf, 2026-03-26)

- `MoEKernel._expert_bias`: `nn.Parameter` → `register_buffer` (не обучается)
- `SoftMoERouter.forward()`: один `_ema_lock` для EMA + bias update (deadlock fix)
- `MoEKernel.expert_orthogonalization_loss()`: `skip при eval` (commit 05d6b3a)

### Session 14 — Smoke Test (MoE + TitansMemory)

5-epoch smoke test результаты:
```
ep1=0.900, ep2=1.004, ep3=1.133, ep4=1.143, ep5=1.151
```
ep5=1.1508 ≈ Run11 ep27 за 5 эпох (27x быстрее). Нет NaN, нет OOM.

---

## 15. Индекс файлов

```
src/core/
  hypercomplex.py         — Clifford/quaternion algebra и слои (Cl(4,1,0) default)
  domain_operators.py     — DomainRotationOperator, InvariantExtractor
  titans_memory.py        — TitansMemoryModule: TTT memory, RAG freeze API (Phase 29)
  soft_moe_router.py      — SoftMoERouter: SharedExpert, AuxLossFree, ExpertOrtho (Phase 26)
  moe_kernel.py           — MoEKernel: 4 domain experts 560K params (Phase 28)
  moe_kernel_adapter.py   — MoEKernelRouterAdapter: drop-in для SoftMoERouter
  clifford_interaction.py — CliffordInteractionLayer: CAN-style geometric nonlinearity
  hbma_memory.py          — HBMAMemory: Working/Episodic/Semantic/Procedural (4-system)
  hdim_pipeline.py        — HDIMPipeline (оркестрация)

src/models/
  hdim_model.py           — HDIMModel (batch API, domain-aware)
  text_hdim_model.py      — TextHDIMModel (text wrapper, encode_texts)
  sbert_encoder.py        — SBERTEncoder + SimpleMLP 768→384→hidden
  model_factory.py        — build_hdim_model, build_text_hdim_model, build_sbert_hdim_model, build_modernbert_hdim_model
  modern_text_encoder.py  — ModernBertEncoder, GatedMLPEncoder, HybridEncoder, MatryoshkaProjection
  metrics.py              — compute_all_metrics, PRIMARY_SCORE, AFR, DRS

src/training/
  trainer.py              — HDIMTrainer (Focal-InfoNCE + AnglE + SupCon + temp schedule + AMP)
  real_dataset.py         — RealPairsDataset, load_real_pairs_dataset

scripts/
  gpu_train.py            — основной скрипт (AMP, scheduler, focal_gamma, temp_schedule)
  gen_dataset_v8.py       — генератор v8 датасета (330 пар, 35.8% neg)
  phase17_train.bat       — Phase 17 launch script (7×P0 + 5×P1 fixes, hidden=128, T=0.15)
  phase19_train.bat       — Phase 19 launch script (антиколлапс + v7, hidden=256)
  phase20_train.bat       — Phase 20 launch script (DCL+Uniform+batch64+v5, цель >1.20)

data/
  real_pairs_v8.json      — 330 пар (212+ / 118-) — АКТУАЛЬНЫЙ
  real_pairs_v7.json      — 232 пары (legacy)
  real_pairs_v6.json      — 213 пар (legacy)
  real_pairs_v5.json      — 175 пар (Phase 9 рекорд)

  phase25_train.bat       — Phase 25b: freeze_sbert_bottom_frac + weight_decay + v10 data
  auto_tune.py            — Auto-Tuner v26 (Optuna, study hdim_autotune_v26)
  autoresearch_loop.py    — Automated research with IncumbentTracker

data/
  real_pairs_v10.json     — 1036 пар (636+ / 400-) — АКТУАЛЬНЫЙ (Phase 26)
  real_pairs_v8.json      — 330 пар (212+ / 118-) — legacy
  real_pairs_v7.json      — 232 пары (legacy)
  real_pairs_v6.json      — 213 пар (legacy)
  real_pairs_v5.json      — 175 пар (Phase 9 рекорд)

artifacts/
  phase8e_soft_eval5/     — РЕКОРД (legacy): score=1.1370 (ep45, hidden=256, 4 experts)
  phase13_scaled/         — Phase 13: best=0.696 (ep15, LR restart проблема)
  phase14_sota/           — Phase 14: Focal-InfoNCE + cosine_decay + v8 (в процессе)
  phase17/                — Phase 17: 7×P0 + 5×P1 fixes, hidden=128, T=0.15 (в процессе)
  best_autotune/          — Phase 26c: score=1.1542 (ep15, hidden=256, 4 experts)
  run18_record/           — РЕКОРД: score=1.1814 (ep13, temp=0.10, lambda_pair=0.40)
  optuna_study_v26.db     — Optuna study database v26
  optuna_results_v26.json — All Optuna trial results

---

## 15. Phase 25 — SBERT Freezing + Data v10

Phase 25 подготовил базу для Phase 26:

- `freeze_sbert_bottom_frac`: заморозка нижних N% слоёв SBERT трансформера
- `weight_decay` для SBERT слоёв: `lr * 100` (агрессивная регуляризация необучаемых слоёв)
- `real_pairs_v10.json`: 1036 пар (636 положительных / 400 отрицательных)
- SBERT embeddings cache: предвычисление embeddings для frozen encoder (ускорение 3-5x)

### Phase 25 конфигурация

```batch
@echo off
REM Phase 25b: freeze_sbert_bottom_frac + weight_decay + v10
cd /d E:\hypercoplexAI
call .venv\Scripts\activate.bat

python scripts\gpu_train.py ^
    --epochs 60 ^
    --hidden_dim 256 ^
    --num_experts 4 ^
    --num_domains 4 ^
    --pretrained_encoder ^
    --soft_router ^
    --freeze_sbert_bottom_frac 0.5 ^
    --weight_decay 0.01 ^
    --real_pairs data\real_pairs_v10.json ^
    --augment_factor 5 ^
    --lambda_pair 0.5 ^
    --lambda_sts 0.15 ^
    --lambda_dcl 0.2 ^
    --lambda_iso 0.1 ^
    --lambda_routing 0.02 ^
    --lambda_memory 0.01 ^
    --lambda_z 0.01 ^
    --use_infonce ^
    --infonce_temperature 0.15 ^
    --learnable_temperature ^
    --lr 0.0003 ^
    --seed 42 ^
    --batch_size 48 ^
    --scheduler_type cosine_restarts ^
    --t_mult 2 ^
    --warmup_epochs 20 ^
    --eval_every 5 ^
    --save_every 25 ^
    --output_dir artifacts\phase25b ^
    --device auto ^
    --amp
```

---

## 16. Phase 26 — DomainExpertPool + SharedExpert + AuxLossFree + ExpertOrtho

### 16.1 Новые компоненты

#### DomainExpertPool (`src/core/domain_expert_pool.py`)

Пул из 4 frozen SBERT-энкодеров (MiniLM family) с обучаемыми projection heads:

| ID | Модель | Параметры | Назначение |
|----|--------|-----------|------------|
| 0 | `all-MiniLM-L6-v2` | 22M frozen | General semantics |
| 1 | `paraphrase-MiniLM-L3-v2` | 17M frozen | Paraphrase/structural similarity |
| 2 | `multi-qa-MiniLM-L6-cos-v1` | 22M frozen | QA/domain-crossing |
| 3 | `all-MiniLM-L12-v2` | 33M frozen | Deep semantic analysis |

- **Общий footprint:** ~94M frozen params + ~100K trainable per expert (projection head)
- **Архитектура projection:** `Linear → LayerNorm → GELU → Dropout(0.1) → Linear`
- **SBERT на CPU** для экономии VRAM, только projection на GPU

#### SharedExpert (DeepSeek-V3)

Always-on FFN, обрабатывает ВСЕ входы независимо от routing:

```python
class SharedExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
        )
```

Добавляется к выходу MoE: `output = moe_output + shared_expert(input)`

#### Auxiliary-Loss-Free Balancing (DeepSeek-V3)

Per-expert bias terms вместо auxiliary loss для балансировки нагрузки:

```python
# Вместо loss: динамическая коррекция bias
if self.use_aux_loss_free:
    delta = torch.sign(expert_load - self._target_load)
    self._expert_bias.data -= self._aux_lr * delta  # aux_lr=0.001
```

#### Expert Orthogonalization (arXiv:2505.22323)

Штрафует сходство весовых матриц экспертов:

```python
def expert_orthogonalization_loss(self):
    """L_o = ||W1 @ W1^T - I||^2 + ||W2 @ W2^T - I||^2"""
    w1_norm = F.normalize(self.W1_stack.reshape(E, -1), dim=-1)
    gram1 = w1_norm @ w1_norm.T
    loss1 = ((gram1 - I) ** 2).mean()
    # аналогично для W2
    return (loss1 + loss2) * 0.5
```

### 16.2 Удалённый мёртвый код

| Удалено | Причина |
|---------|---------|
| `SoftRouterState` | Dict вместо dataclass — не нужен отдельный класс |
| `calibration_head` | Не использовался после Phase 22 |
| `adaptive_dropout` | Не давал значимого прироста |
| `similarity_preserving_loss` | Убран в пользу AuxLossFree |
| `experts ModuleList` | Заменён на batched einsum со stacked weights |
| CLI: `--router_calibration` | Удалён |
| CLI: `--adaptive_expert_dropout` | Удалён |
| CLI: `--similarity_preserving_router` | Удалён |
| Phantom: `clifford_dim` | Вычисляется из `clifford_p/q/r` |
| Phantom: `text_mode` | Всегда True при text encoder |
| Phantom: `advanced_encoder` | Не используется |
| Phantom: `hierarchical_memory` | Не реализован |

### 16.3 Phase 26 CLI флаги (gpu_train.py)

```batch
--shared_expert          # Enable DeepSeek-V3 always-on shared expert
--aux_loss_free          # Enable Auxiliary-Loss-Free load balancing
--aux_lr 0.001           # Bias adjustment rate для aux_loss_free
--expert_ortho           # Enable expert orthogonalization loss
--lambda_expert_ortho 0.02  # Weight for expert ortho loss
```

### 16.4 Auto-Tuner v26 (`scripts/auto_tune.py`)

- **Optuna** с TPE sampler (seed=42, n_startup_trials=5)
- **Study name:** `hdim_autotune_v26`
- **Warm-start** с историческими SOTA seeds (Phase 26b и 26c configs)
- **Фиксировано:** hidden_dim=256, num_experts=4, soft_router, pretrained_encoder, shared_expert, aux_loss_free, expert_ortho, learnable_temperature, data=v10
- **Оптимизирует:** lr, batch_size, augment_factor, все lambda веса, focal_gamma, warmup_epochs

### 16.5 Autoresearch Loop (`scripts/autoresearch_loop.py`)

Автоматический исследовательский цикл:
- **IncumbentTracker**: отслеживает лучший результат
- **Failure taxonomy**: crash_nan, crash_oom, metric_regression
- **Retry logic**: автоматический перезапуск при failure

### 16.6 Результаты Phase 26

| Run | Score | Margin | STS | Epoch | Конфигурация |
|-----|-------|--------|-----|-------|-------------|
| 26a | 1.1063 | — | — | ep45 | augment=3, no sts/dcl |
| 26b | 1.1513 | — | — | ep15 | augment=5, sts=0.15, dcl=0.2, learnable_temp |
| **26c** | **1.1542** | 0.993 | 0.537 | ep15 | augment=5, sts=0.3, uniformity=0.1 |

Phase 26c рекорд: 1.1542 (vs предыдущий 1.1370 Phase 8e — улучшение на 1.5%)

**ТЕКУЩИЙ РЕКОРД ПРОЕКТА: 1.1814** (Run 18, Session 13, ep13, temp=0.10, lambda_pair=0.40) — улучшение на +2.4% vs Phase 26c

### 16.7 Phase 26 конфигурация (SOTA)

```batch
@echo off
REM Phase 26c: DomainExpertPool + SharedExpert + AuxLossFree + ExpertOrtho
cd /d E:\hypercoplexAI
call .venv\Scripts\activate.bat

python scripts\gpu_train.py ^
    --epochs 60 ^
    --hidden_dim 256 ^
    --num_experts 4 ^
    --num_domains 4 ^
    --pretrained_encoder ^
    --soft_router ^
    --shared_expert ^
    --aux_loss_free ^
    --expert_ortho ^
    --learnable_temperature ^
    --real_pairs data\real_pairs_v10.json ^
    --augment_factor 5 ^
    --lambda_pair 0.5 ^
    --lambda_sts 0.3 ^
    --lambda_iso 0.1 ^
    --lambda_routing 0.02 ^
    --lambda_memory 0.01 ^
    --lambda_z 0.01 ^
    --lambda_dcl 0.2 ^
    --lambda_uniformity 0.1 ^
    --lambda_expert_ortho 0.02 ^
    --use_infonce ^
    --focal_gamma 1.0 ^
    --lr 0.0003 ^
    --seed 42 ^
    --batch_size 48 ^
    --scheduler_type cosine_restarts ^
    --t_mult 2 ^
    --warmup_epochs 20 ^
    --eval_every 5 ^
    --save_every 25 ^
    --output_dir artifacts\best_autotune ^
    --device auto ^
    --amp
```
```
