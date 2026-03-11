# HDIM — Hypercomplex Domain Isomorphism Machine
*Версия: 9.0 | Дата: 2026-03-11 | Рекорд: score=0.9930 (Phase 9, ep55)*

---

## 1. Назначение

HDIM — research prototype для **кроссдоменного переноса знаний** через гиперкомплексные представления. Система кодирует текст/embedding в гиперкомплексную структуру, извлекает доменно-инвариантное представление и переносит его в другой домен через routing + memory.

Текущий статус: **MVP transfer engine** с TextHDIMModel поверх HDIM core, обученный на парных примерах (source_domain → target_domain).

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

---

## 4. Архитектура MVP

### Пайплайн
```
Текст → FrozenSBERT → проекция → InvariantExtractor → TitansMemory → SoftMoERouter → DecoupledDecoder
```

### Ключевые модули
| Модуль | Файл | Назначение |
|--------|------|------------|
| `HDIMPipeline` | `src/core/hdim_pipeline.py` | Оркестрация пайплайна |
| `InvariantExtractor` | `src/core/domain_operators.py` | Извлечение инварианта |
| `TitansMemory` | `src/core/titans_memory.py` | Ассоциативная память |
| `SoftMoERouter` | `src/core/moe_router.py` | Мягкая маршрутизация |
| `ModularMoERouter` | `src/core/modular_moe.py` | Модульный роутер (add/remove expert) |
| `HDIMModel` | `src/models/hdim_model.py` | Batch-facing API |
| `TextHDIMModel` | `src/models/text_hdim_model.py` | Text entry wrapper |
| `build_*_model` | `src/models/model_factory.py` | Единственная точка сборки моделей |
| `gpu_train.py` | `scripts/gpu_train.py` | Основной скрипт обучения |

### Модульность экспертов
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
| `data/real_pairs_v5.json` | 175 (139+ / 36-) | **Актуальный** |
| `data/real_pairs_v4.json` | 140 | Phase 6e рекорд |

Структура элемента:
```json
{"source_text": "...", "source_domain": 0, "target_text": "...", "target_domain": 1,
 "relation": "positive", "group_id": 42, "family": "thermodynamics"}
```

**PRIMARY_SCORE** = `pair_margin × 1.0 + STS_exported × 0.3`

---

## 6. Результаты обучения

### Хронология рекордов
| Phase | Score | Эпоха | Конфигурация |
|-------|-------|-------|--------------|
| 5a | 0.967 | — | AnglE loss + learnable temp |
| 6e | 0.9745 | ep75 | SoftMoE + v4 данные + seed=77 |
| 7 | 0.804 | — | ModularMoE (с багами) |
| **9** | **0.9930** | **ep55** | **SoftMoE + v5 данные + seed=42** |

### Phase 9 — детальный прогресс
| Эпоха | Score | Margin | STS | Событие |
|-------|-------|--------|-----|---------|
| 5 | 0.754 | 0.507 | 0.822 | Первый eval |
| 20 | 0.780 | 0.539 | 0.801 | LR рестарт (T_0=20) |
| 25 | 0.811 | 0.568 | 0.809 | Начало 2-го цикла |
| 45 | 0.906 | 0.670 | 0.786 | Резкий рост |
| 50 | 0.961 | 0.725 | 0.788 | Прорыв |
| **55** | **0.993** | **0.757** | **0.788** | **Рекорд** |
| 59 | 0.990 | 0.757 | 0.788 | Конец 3-го цикла |

---

## 7. Рекордная конфигурация (Phase 9)

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
  --output_dir artifacts/phase9_v5_baseline
```

**GPU:** NVIDIA RTX 3070 Laptop (8.6GB), PyTorch 2.6+cu124  
**Время:** ~3.94h до ep59

---

## 8. Паттерны обучения

- **Ep1-4:** score=0 (eval только на кратных 5)
- **Ep5:** score~0.75 — первый прыжок
- **Ep20:** LR рестарт → score растёт
- **Ep45-55:** основной рост margin 0.67→0.76
- **CosineAnnealingWarmRestarts T_0=20, T_mult=2:** цикл 20→40→80 эпох
- **STS ≈ 0.79-0.82** стабильно (frozen SBERT даёт хорошую базу)
- **Best обычно на 3-м цикле** (ep45-60 при T_0=20)

---

## 9. Выводы и рекомендации для Phase 10

### 9.1 Почему Phase 9 побила рекорд
1. **v5 данные (175 пар)** vs v4 (140 пар) — больше уникальных паттернов
2. **seed=42** даёт лучшую начальную инициализацию
3. **eval_every=5** — правильный баланс скорость/мониторинг
4. **T_0=20** с T_mult=2 — более короткие начальные циклы → быстрее находит оптимум

### 9.2 Направления оптимизации

**Быстрые победы:**
- `eval_every=1` + `early_stopping_patience=5` — остановиться точно на пике
- `T_0=40` — удлинить первый цикл, пик должен быть выше
- Grid по seed: [7, 13, 21, 42, 77] — найти лучший

**Архитектурные улучшения:**
- Частичное размораживание SBERT (слои 10-11, LR=1e-5): STS↑ → score↑
- `hidden_dim=512, num_experts=8`: VRAM=0.02GB из 8.6GB — огромный запас
- SupCon loss с `family_id` из датасета
- Семантические domain ID (SBERT-embedding домена вместо integer)

**Данные:**
- Цель: 300+ пар (сейчас 175)
- Добавить: квантмех↔термодинамика, экономика↔физика, нейросети↔статфизика
- Качественные hard negatives

### 9.3 Ожидаемый потолок
| Сценарий | Оценка |
|----------|--------|
| Только seed tuning | 0.993-0.995 |
| Partial SBERT unfreeze | 0.995-0.998 |
| 300+ пар + архитектура | >0.999 |

---

## 10. Исправленные баги (Phase 8)

### БАГ 1: Хардкод `config.num_experts` в `_allocate_state_tensors`
**Файл:** `src/models/hdim_model.py`  
**Фикс:** заменить `self.config.num_experts` → `self.pipeline.moe.num_experts`

### БАГ 2: `reset_memory` без guard на `train_scores`
**Файл:** `src/models/hdim_model.py`  
**Фикс:** `if hasattr(self.pipeline.moe, 'train_scores'):` перед обращением

### БАГ 3: `build_text_hdim_model` не поддерживал `modular_moe`
**Файл:** `src/models/model_factory.py`  
**Фикс:** добавить параметры `modular_moe`, `modular_moe_routing_type`

### БАГ 4: `_build_model` в `gpu_train.py` не передавал `modular_moe` в text path
**Файл:** `scripts/gpu_train.py`  
**Фикс:** добавить `getattr(args, 'modular_moe', False)` в needs_text и build_text_hdim_model

---

## 11. Индекс файлов

```
src/core/
  hypercomplex.py         — Clifford/quaternion algebra и слои
  domain_operators.py     — DomainRotationOperator, InvariantExtractor
  titans_memory.py        — TitansMemory, HierarchicalTitansMemory
  moe_router.py           — SoftMoERouter, R3MoERouter
  modular_moe.py          — ModularMoERouter (add/remove expert)
  hdim_pipeline.py        — HDIMPipeline (оркестрация)

src/models/
  hdim_model.py           — HDIMModel (batch API)
  text_hdim_model.py      — TextHDIMModel (text wrapper)
  model_factory.py        — build_hdim_model, build_text_hdim_model, build_sbert_hdim_model
  metrics.py              — compute_all_metrics, PRIMARY_SCORE

src/training/
  trainer.py              — HDIMTrainer (train_step, validate)
  dataset.py              — demo datasets
  real_dataset.py         — load_real_pairs_dataset, split_real_pairs
  experiment_config.py    — ExperimentConfig
  results_logger.py       — append_ledger_row

scripts/
  gpu_train.py            — основной скрипт обучения (AMP, scheduler, early stopping)

data/
  real_pairs_v5.json      — 175 пар (актуальный датасет)
  real_pairs_v4.json      — 140 пар (Phase 6e рекорд)

artifacts/
  phase9_v5_baseline/     — результаты Phase 9 (score=0.9930)
  phase6e_run/            — результаты Phase 6e (score=0.9745)
```
