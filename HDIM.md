# HDIM — Hypercomplex Domain Isomorphism Machine
*Версия: 12.0 | Дата: 2026-03-12 | Рекорд: score=1.1370 (Phase 8e, ep45) | Phase 13: 0.3076 (3 эп, augment=10, quick run)*

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
$$L_{total} = L_{recon} + \lambda_{iso} L_{iso} + \lambda_{pair} L_{pair} + \lambda_{routing} L_{routing} + L_{memory}$$

Где $L_{pair}$ = InfoNCE + AnglE + SupCon (при наличии family labels).

---

## 4. Архитектура

### Пайплайн
```
Текст → SBERT(frozen) → SimpleMLP(768→384→256) → InvariantExtractor → TitansMemory → SoftMoERouter → DecoupledDecoder
```

**Важно:** GatedProjection (Phase 10) показал **регрессию** — 0.9347 << 1.1370. Оптимальная проекция: simple MLP (Linear→LayerNorm→GELU→Dropout→Linear).

### Ключевые модули
| Модуль | Файл | Назначение |
|--------|------|------------|
| `HDIMPipeline` | `src/core/hdim_pipeline.py` | Оркестрация пайплайна |
| `InvariantExtractor` | `src/core/domain_operators.py` | Извлечение инварианта |
| `TitansMemory` | `src/core/titans_memory.py` | Ассоциативная память |
| `SoftMoERouter` | `src/core/soft_moe_router.py` | Мягкая маршрутизация (Phase 9 рекорд) |
| `ModularMoERouter` | `src/core/modular_moe.py` | Модульный роутер (add/remove expert) |
| `SBERTEncoder` | `src/models/sbert_encoder.py` | **Simple MLP** 768→384→256 |
| `HDIMModel` | `src/models/hdim_model.py` | Batch-facing API |
| `TextHDIMModel` | `src/models/text_hdim_model.py` | Text entry wrapper |
| `build_*_model` | `src/models/model_factory.py` | Единственная точка сборки моделей |
| `HDIMTrainer` | `src/training/trainer.py` | InfoNCE + AnglE + SupCon + HardNeg |
| `gpu_train.py` | `scripts/gpu_train.py` | Основной скрипт обучения |

### Phase 10 — Новые компоненты

#### GatedProjection (SBERT 768 → hidden_dim)
```python
h = GELU(LayerNorm(W_down · sbert_emb))     # [B, phidden]
out = LayerNorm(W_up(h) * σ(W_gate(h)))     # gated output [B, hidden_dim]
```
Мотивация: gated MLP стабильно превосходит простые MLP, особенно при переносе из 768-мерного SBERT пространства.

#### Partial SBERT Unfreezing
```python
# Размораживаем только слои 10, 11 (14.6M из 278M параметров)
--unfreeze_sbert_layers "10,11" --sbert_lr 1e-5
```

#### Online Hard Negative Mining
```python
# Находит hardest negatives в batche по cosine similarity
--use_hard_negatives
```
Алгоритм: строим sim_matrix (B×B), маскируем диагональ и истинные позитивы, берём argmax по строке.

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
| `data/real_pairs_v7.json` | 232 (191+ / 41-) | **Актуальный** — сбалансированные домены |
| `data/real_pairs_v6.json` | 213 (172+ / 41-) | Phase 10-11 |
| `data/real_pairs_v5.json` | 175 (139+ / 36-) | Phase 9 рекорд |
| `data/real_pairs_v4.json` | 140 | Phase 6e рекорд |

**Новые домены в v6:**
- Квантовая механика ↔ Термодинамика (туннелирование, декогеренция)
- Нейронные сети ↔ Статистическая физика (backprop ↔ Monte Carlo)
- Экономика ↔ Физика (броуновское движение, арбитраж)
- Биология ↔ Информационные технологии (ДНК репликация, кэш)
- Химия ↔ Астрофизика (катализ ↔ звёздообразование)
- Теория информации ↔ Термодинамика (Шеннон ↔ Больцман)

Структура элемента:
```json
{"source_text": "...", "source_domain": 0, "target_text": "...", "target_domain": 1,
 "relation": "positive", "group_id": 42, "family": "thermodynamics"}```

**PRIMARY_SCORE** = `pair_margin × 1.0 + STS_exported × 0.3`

### 5.1 Рекомендации для v8 dataset

**Критические недостатки v7:**
- `augment_factor` — это НЕ аугментация, только shuffle/repeat одних и тех же 423 текстов
- Негативные пары: 17.7% (41/232) — недостаточно для контрастного обучения
- Domain 2->3 отсутствует полностью (0 пар)
- Domain 3 как источник: 0 пар
- Каждый pair имеет уникальный group_id → split становится случайным per-pair

**План v8 (цель: 300+ пар, 35-40% негативных):**

| Приоритет | Действие | Ожидаемый эффект |
|-----------|----------|------------------|
| HIGH | Добавить Domain 2->3 пары (15-20) | Заполнить критический пробел |
| HIGH | Добавить Domain 3 как source (10-15) | Полная coverage домена Physics |
| HIGH | Увеличить негативные пары до 35-40% | Улучшить контрастное обучение |
| MEDIUM | Добавить backward pairs для всех негативов | 2x negative data без нового контента |
| MEDIUM | T5-based парфразинг для позитивных пар | Настоящая аугментация текста |
| LOW | Back-translation (en-de, en-fr, en-ja) | Лексическое разнообразие |
| LOW | Standardize family naming convention | Улучшить split quality |

**Не использовать:** Embedding mixup (экспериментально, может генерировать бессмысленный текст).

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
| **12b** | TBD | — | — | — | v7 + T_mult=2, warmup=5 (запущена 12.03) |
| **13** | **0.3076** | **0.027** | **0.934** | **ep3** | **hidden=512, experts=8, lambda_z=0.001, augment=10 (в процессе)** |

### Phase 9 — детальный прогресс
| Эпоха | Score | Margin | STS | Событие |
|-------|-------|--------|-----|---------|
| 5 | 0.754 | 0.507 | 0.822 | Первый eval |
| 20 | 0.780 | 0.539 | 0.801 | LR рестарт (T_0=20) |
| 45 | 0.906 | 0.670 | 0.786 | Резкий рост |
| 50 | 0.961 | 0.725 | 0.788 | Прорыв |
| **55** | **0.993** | **0.757** | **0.788** | **Рекорд** |

---

## 7. Рекордная конфигурация (Phase 8e) + Оптимизированная Phase 12

### Модель: 414,103 параметров (Phase 8e)
| Компонент | Параметры | Размерность |
|-----------|-----------|-------------|
| text_encoder.projection | 394,624 | 768→384→256 (simple MLP) |
| core_model.pipeline | 15,127 | clifford_dim=16, hidden=256 |
| training_inv_head | 4,352 | 16→256 |

### Конфигурация запуска
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

---

## 8. Phase 10 — Команда запуска

```bash
# Phase 10: GatedProjection + HardNeg + v6 датасет (213 пар)
cd E:/hypercoplexAI && python scripts/gpu_train.py \
  --epochs 200 --hidden_dim 256 --num_experts 4 --num_domains 4 \
  --pretrained_encoder --soft_router \
  --real_pairs data/real_pairs_v6.json --augment_factor 30 \
  --lambda_pair 0.4 --lambda_sts 0.2 --lambda_angle 0.3 \
  --lambda_iso 0.1 --lambda_routing 0.05 --lambda_memory 0.01 \
  --use_infonce --infonce_temperature 0.1 --learnable_temperature \
  --use_hard_negatives \
  --early_stopping_patience 40 \
  --lr 0.0005 --seed 42 --batch_size 32 \
  --scheduler_type cosine_restarts --warmup_epochs 3 \
  --eval_every 5 --save_every 25 \
  --output_dir artifacts/phase10_v6_gated_hardneg
```

```bash
# Phase 10b: + Partial SBERT unfreeze (слои 10, 11)
cd E:/hypercoplexAI && python scripts/gpu_train.py \
  --epochs 200 --hidden_dim 256 --num_experts 4 --num_domains 4 \
  --pretrained_encoder --soft_router \
  --real_pairs data/real_pairs_v6.json --augment_factor 30 \
  --lambda_pair 0.4 --lambda_sts 0.2 --lambda_angle 0.3 \
  --lambda_iso 0.1 --lambda_routing 0.05 --lambda_memory 0.01 \
  --use_infonce --infonce_temperature 0.1 --learnable_temperature \
  --use_hard_negatives \
  --unfreeze_sbert_layers "10,11" --sbert_lr 1e-5 \
  --early_stopping_patience 40 \
  --lr 0.0005 --seed 42 --batch_size 32 \
  --scheduler_type cosine_restarts --warmup_epochs 5 \
  --eval_every 5 --save_every 25 \
  --output_dir artifacts/phase10b_v6_partial_unfreeze
```

---

## 9. Паттерны обучения

- **Ep1-4:** score=0 (eval только на кратных 5)
- **Ep5:** score~0.75 — первый прыжок
- **Ep20:** LR рестарт → score растёт
- **Ep45-55:** основной рост margin 0.67→0.76
- **CosineAnnealingWarmRestarts T_0=20, T_mult=2:** цикл 20→40→80 эпох
- **STS ≈ 0.79-0.82** стабильно (frozen SBERT даёт хорошую базу)
- **Best обычно на 3-м цикле** (ep45-60 при T_0=20)

---

## 10. Phase 10 — Улучшения архитектуры

### Что добавлено (и что откатили)

| Компонент | Статус | Результат |
|-----------|--------|-----------|
| GatedProjection 768→hidden | ⚠️ Реализовано, **ОТКАТ** | -0.198 (REGRESSION: 0.9347 vs 1.1370) |
| Online Hard Negative Mining | ⚠️ Реализовано, **ОТКАТ** | -0.064 (дестабилизирует) |
| Датасет v6 (213 пар) | ✅ Создано | v7 (232 пары) лучше |
| Partial SBERT unfreeze | ⚠️ Реализовано, **ОТКАТ** | -0.408 (REGRESSION: 0.729) |
| Simple MLP projection | ✅ **ОПТИМАЛЬНЫЙ** | Рекорд 1.1370 |

**Ключевой вывод:** Simple MLP (Linear→LayerNorm→GELU→Dropout→Linear) стабильно превосходит GatedProjection и сложные варианты.

### 10.1 SOTA улучшения (Focal-InfoNCE, temperature scheduling)

| Метод | Источник | Статус | Ожидаемый прирост |
|-------|----------|--------|-------------------|
| **Router Z-Loss** | ST-MoE (Zoph et al., 2022) | ✅ Реализовано | +0.005-0.015 (стабильность) |
| **Focal-InfoNCE** | Hou & Li, EMNLP 2023 | ⚠️ Запланировано | +0.010-0.015 |
| **Temperature-Free InfoNCE** | Kim et al., Jan 2025 | ⚠️ Запланировано | убирает гиперпараметр |
| **Learnable Temperature** | CLIP, 2023 | ✅ Реализовано | стабильные градиенты |
| **SparseMixer / Dense Backprop** | Panda et al., Apr 2025 | ⚠️ Запланировано | лучшие градиенты для роутера |

**Router Z-Loss (lambda_z=0.001):**
```python
# Формула: L_z = (1/B) * Σ_b (log Σ_e exp(router_logit_b,e))^2
# Штрафует большие magnitudes router logits → стабильные dispatch weights
z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()
```
**Для HDIM:** critical при num_experts > 4 (8 экспертов в Phase 13).

**Focal-InfoNCE (следующий приоритет):**
```python
# Downweights easy negatives, focuses on hard negatives
# +1.64 Spearman improvement на SimCSE-BERTbase
# Рекомендация: использовать если Hard Neg Mining нестабилен
```

### GatedProjection
```python
# Вместо простого MLP:
h = GELU(LayerNorm(Linear(768 → phidden)))
out = LayerNorm(Linear_up(h) * sigmoid(Linear_gate(h)))
```
Mотивация: GLU-варианты (gated linear units) стабильно превосходят plain MLP в задачах трансформации embedding пространств.

### Ожидаемый потолок Phase 10
| Сценарий | Оценка |
|----------|--------|
| GatedProj + HardNeg + v6 | 0.993-0.997 |
| + Partial SBERT unfreeze | 0.997-0.999 |
| + hidden_dim=512, experts=8 | >0.999 |

---

## 11. Исправленные баги (Phase 8)

### БАГ 1: Хардкод `config.num_experts` в `_allocate_state_tensors`
**Файл:** `src/models/hdim_model.py`
**Фикс:** заменить `self.config.num_experts` → `self.pipeline.moe.num_experts`

### БАГ 2: `reset_memory` без guard на `train_scores`
**Файл:** `src/models/hdim_model.py`
**Фикс:** `if hasattr(self.pipeline.moe, 'train_scores'):` перед обращением

### БАГ 3: Hard Negative Mining fp16 overflow
**Файл:** `src/training/trainer.py`
**Фикс:** использовать `-1e4` вместо `-1e9` при masked_fill для fp16 совместимости

### 11.4 Исправленные баги (Phase 13)

#### БАГ 4: val_metrics не содержит loss_z
**Файл:** `src/training/trainer.py:774-794`
**Проблема:** `validate()` возвращает только 6 loss ключей (recon, iso, pair, routing, memory, total), но не включает `loss_z`.
**Фикс:** добавить `loss_z` в totals dict:
```python
def validate(self, dataloader):
    totals = {
        "loss_recon": 0.0, "loss_iso": 0.0, "loss_pair": 0.0,
        "loss_routing": 0.0, "loss_memory": 0.0, "loss_z": 0.0,  # NEW
        "loss_total": 0.0,
    }
```
**Причина:** val_loss не учитывает z-loss, что даёт неверное представление о стабильности роутера.

#### БАГ 5: GradScaler checkpoint не восстанавливается в load_checkpoint
**Файл:** `src/training/trainer.py:806-810`
**Проблема:** `load_checkpoint()` восстанавливает model/optimizer/step, но не восстанавливает GradScaler state.
**Фикс:** добавить `scaler.load_state_dict()`:
```python
def load_checkpoint(self, path, scaler=None):
    checkpoint = torch.load(path, map_location=self.device)
    self.model.load_state_dict(checkpoint["model_state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self._step = checkpoint.get("step", 0)
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
```
**Важно:** без этого восстановление AMP training после checkpoint даёт потерю scale factor → gradient explosion.

---

## 12. Индекс файлов

```
src/core/
  hypercomplex.py         — Clifford/quaternion algebra и слои
  domain_operators.py     — DomainRotationOperator, InvariantExtractor
  titans_memory.py        — TitansMemory, HierarchicalTitansMemory
  moe_router.py           — R3MoERouter (baseline)
  soft_moe_router.py      — SoftMoERouter (Phase 9 рекорд)
  modular_moe.py          — ModularMoERouter (add/remove expert)
  hdim_pipeline.py        — HDIMPipeline (оркестрация)

src/models/
  hdim_model.py           — HDIMModel (batch API)
  text_hdim_model.py      — TextHDIMModel (text wrapper)
  sbert_encoder.py        — SBERTEncoder + GatedProjection + partial unfreeze
  model_factory.py        — build_hdim_model, build_text_hdim_model, build_sbert_hdim_model
  metrics.py              — compute_all_metrics, PRIMARY_SCORE

src/training/
  trainer.py              — HDIMTrainer + Hard Negative Mining
  dataset.py              — demo datasets
  real_dataset.py         — load_real_pairs_dataset, split_real_pairs

scripts/
  gpu_train.py            — основной скрипт обучения (AMP, scheduler, early stopping)
  gen_dataset_v6.py       — генератор датасета v6 (213 пар)

data/
  real_pairs_v6.json      — 213 пар (актуальный датасет, Phase 10)
  real_pairs_v5.json      — 175 пар (Phase 9 рекорд)
  real_pairs_v4.json      — 140 пар (Phase 6e рекорд)

artifacts/
  phase9_v5_baseline/     — результаты Phase 9 (score=0.9930)
  phase10_v6_gated_hardneg/ — Phase 10 результаты (TBD)
```
