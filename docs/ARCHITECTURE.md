# HDIM Architecture Documentation

> **Дата:** 2026-03-13  
> **Версия:** Research Prototype  
> **Источники:** Исследовательские отчёты в `[.omc/research/](../.omc/research/)`

---

## Оглавление

1. [Обзор системы](#1-обзор-системы)
2. [Слойная архитектура](#2-слойная-архитектура)
3. [Core Layer](#3-core-layer)
4. [Model Layer](#4-model-layer)
5. [Training Layer](#5-training-layer)
6. [Scripts Layer](#6-scripts-layer)
7. [Потоки данных](#7-потоки-данных)
8. [Публичный API](#8-публичный-api)
9. [Конфигурации](#9-конфигурации)
10. [Stable vs Experimental](#10-stable-vs-experimental)

---

## 1. Обзор системы

**HDIM** (Hypercomplex Domain Isomorphism Machine) — research-прототип для **кросс-доменного переноса структурных аналогий** через гиперкомплексные инварианты.

### Ключевая идея

В отличие от LLM, которые сравнивают тексты по токеновой близости, HDIM находит **структурные изоморфизмы** между задачами из разных доменов с разной лексикой, но одинаковой глубинной структурой.

**Канонический пример:** Кавитационная эрозия (инженерия) vs удаление зубного налёта (стоматология). Разные слова, одинаковая физика. HDIM находит эту связь через доменно-инвариантное представление.

### Математический контракт


| Операция          | Формула                                  | Описание                          |
| ----------------- | ---------------------------------------- | --------------------------------- |
| Encode A          | `G_A = MLP(SBERT(text_A))`               | Мультивектор домена A             |
| Extract invariant | `U = R⁻¹ ⊗ G_A ⊗ R`                      | Снятие доменной сигнатуры         |
| Transfer to B     | `G_B = R_B ⊗ U ⊗ R_B⁻¹`                  | Реконструкция в домене B          |
| Isomorphism loss  | `L_iso = MSE(G_B, G_B_target)`           | Штраф за структурное несовпадение |
| PRIMARY score     | `pair_margin × 1.0 + STS_exported × 0.3` | Целевая метрика                   |


**Лучший результат:** 1.1370 (Phase 8e, ep45): `pair_margin=0.906`, `STS=0.770`

---

## 2. Слойная архитектура

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HYPERCOREPLEX AI (HDIM)                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Scripts Layer                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │  gpu_train.py   │  │    train.py     │  │   hdim_demo.py  │          │
│  │  (AMP, GPU)     │  │  (CLI entry)    │  │  (demos)        │          │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘          │
│           │                    │                    │                   │
├───────────┼────────────────────┼────────────────────┼───────────────────┤
│  Training Layer                                                         │
│  ┌────────┴───────┐    ┌───────┴────────┐   ┌───────┴────────┐          │
│  │  HDIMTrainer   │    │ ExperimentRun  │   │   Datasets     │          │
│  │  (losses, AMP) │    │  (orchestr.)   │   │(DomainProblem) │          │
│  └────────┬───────┘    └────────────────┘   └────────────────┘          │
│           │                                                             │
├───────────┼─────────────────────────────────────────────────────────────┤
│  Model Layer                                                            │
│  ┌────────┴────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │  HDIMModel      │  │ TextHDIMModel   │  │ SBERTEncoder    │          │
│  │  (core wrapper) │◄─┤ (text wrapper)  │◄─┤ (frozen encoder)│          │
│  └────────┬────────┘  └─────────────────┘  └─────────────────┘          │
│           │              ┌─────────────────┐                            │
│           └──────────────┤ model_factory   │                            │
│                          │ (build_*)       │                            │
│                          └─────────────────┘                            │
├───────────┼─────────────────────────────────────────────────────────────┤
│  Core Layer                                                             │
│  ┌────────┴────────┐                                                    │
│  │  HDIMPipeline   │                                                    │
│  │  (orchestrator) │                                                    │
│  └────────┬────────┘                                                    │
│           │                                                             │
│  ┌────────┴────────┬─────────────────┬─────────────────┐                │
│  │ hypercomplex.py │domain_operators │  titans_memory  │                │
│  │ (CliffordAlgebra│(DomainRotor,    │  (TTT memory)   │                │
│  │  Quaternion)    │ InvariantExtr.) │                 │                │
│  └─────────────────┴─────────────────┴─────────────────┘                │
│           │                                                             │
│  ┌────────┴────────┬─────────────────┐                                  │
│  │ soft_moe_router │  hdim_pipeline  │                                  │
│  │ (SoftMoE)       │  (HDIMEncoder,  │                                  │
│  │                 │   HDIMDecoder)  │                                  │
│  └─────────────────┴─────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Таблица слоёв


| Слой         | Файлы                                                                                                                           | Ответственность                        | Публичный API                             |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ----------------------------------------- |
| **Scripts**  | `[scripts/gpu_train.py](../scripts/gpu_train.py)`, `[scripts/train.py](../scripts/train.py)`, `[hdim_demo.py](../hdim_demo.py)` | Entrypoints, CLI, GPU training         | `main()`, `demo_*()`                      |
| **Training** | `[src/training/*.py](../src/training/)`                                                                                         | Losses, regimes, datasets, checkpoints | `HDIMTrainer`, `ExperimentRunner`         |
| **Model**    | `[src/models/*.py](../src/models/)`                                                                                             | Wrappers, encoders, factories, metrics | `HDIMModel`, `TextHDIMModel`, `build_*()` |
| **Core**     | `[src/core/*.py](../src/core/)`                                                                                                 | Алгебра, память, роутинг, pipeline     | `HDIMPipeline`, `CliffordAlgebra`         |


---

## 3. Core Layer

### 3.1 Обзор модулей


| Модуль                                                         | Класс                      | Назначение                   | Статус       |
| -------------------------------------------------------------- | -------------------------- | ---------------------------- | ------------ |
| `[hypercomplex.py](../src/core/hypercomplex.py)`               | `CliffordAlgebra`          | Алгебра Клиффорда Cl_{p,q,r} | **Stable**   |
|                                                                | `QuaternionLinear`         | Кватернионный слой           | **Stable**   |
|                                                                | `PHMLinear`                | Parameterized Hypercomplex   | *Удалён* |
| `[domain_operators.py](../src/core/domain_operators.py)`       | `DomainRotationOperator`   | Обучаемый ротор домена       | **Stable**   |
|                                                                | `InvariantExtractor`       | Извлечение U_inv = R⁻¹GR     | **Stable**   |
|                                                                | `DomainRegistry`           | Реестр доменов               | **Stable**   |
| `[titans_memory.py](../src/core/titans_memory.py)`             | `TitansMemoryModule`       | Test-Time Training память    | **Stable**   |
| `[soft_moe_router.py](../src/core/soft_moe_router.py)`         | `SoftMoERouter`            | Soft Mixture-of-Experts      | **Stable**   |
| `[hdim_pipeline.py](../src/core/hdim_pipeline.py)`             | `HDIMPipeline`             | Главный orchestrator         | **Stable**   |
|                                                                | `HDIMEncoder`              | Кодирование → мультивектор   | **Stable**   |
|                                                                | `HDIMDecoder`              | Мультивектор → выход         | **Stable**   |
|                                                                | `TransferState`            | State-контейнер              | **Stable**   |


### 3.2 CliffordAlgebra

**Файл:** `[src/core/hypercomplex.py:20](../src/core/hypercomplex.py:20)`

**Назначение:** Вырожденная алгебра Клиффорда Cl_{p,q,r}(R). По умолчанию Cl_{3,1,0} с размерностью мультивектора 16.

**Ключевые методы:**


| Метод                                                        | Строка | Формула   | Описание                                                     |
| ------------------------------------------------------------ | ------ | --------- | ------------------------------------------------------------ |
| `[geometric_product(a, b)](../src/core/hypercomplex.py:105)` | L105   | `a ⊗ b`   | Геометрическое произведение через scattering по Cayley table |
| `[sandwich(R, x)](../src/core/hypercomplex.py:171)`          | L171   | `R x R⁻¹` | Сэндвич-произведение — основа доменных вращений              |
| `[reverse(x)](../src/core/hypercomplex.py:149)`              | L149   | `x̃`      | Реверсия мультивектора (инволюция)                           |
| `[norm(x)](../src/core/hypercomplex.py:161)`                 | L161   | `√⟨xx̃⟩₀` | Норма через скалярную часть                                  |


**Особенности реализации:**

- Таблица Кэли предвычисляется при инициализации — zero runtime overhead
- `geometric_product` использует `scatter_add`_ для эффективного суммирования
- Встроенный `nan_to_num` + `clamp` для предотвращения NaN и gradient explosion

### 3.3 DomainRotationOperator

**Файл:** `[src/core/domain_operators.py:19](../src/core/domain_operators.py:19)`

**Назначение:** Обучаемый ротор домена с именем и нормализацией.

**Методы:**


| Метод                                                    | Строка | Формула             | Описание                               |
| -------------------------------------------------------- | ------ | ------------------- | -------------------------------------- |
| `[_normalized_R()](../src/core/domain_operators.py:35)`  | L35    | `R / ‖R‖`           | Нормализация для стабильности сэндвича |
| `[get_inverse()](../src/core/domain_operators.py:40)`    | L40    | `reverse(R) / ‖R‖²` | Вычисление обратного ротора            |
| `[forward(x)](../src/core/domain_operators.py:47)`       | L47    | `R x R⁻¹`           | Сэндвич-преобразование                 |
| `[apply_inverse(x)](../src/core/domain_operators.py:50)` | L50    | `R⁻¹ x R`           | Обратное преобразование                |


### 3.4 InvariantExtractor

**Файл:** `[src/core/domain_operators.py:54](../src/core/domain_operators.py:54)`

**Назначение:** Извлечение структурного инварианта через сэндвич-произведение.

**Математика:**

```
U_inv = R_source⁻¹ ⊗_Cl G_source ⊗_Cl R_source
```

Это "снимает" доменную сигнатуру, оставляя чистый структурный инвариант.

### 3.5 TitansMemoryModule

**Файл:** `[src/core/titans_memory.py:30](../src/core/titans_memory.py:30)`

**Назначение:** Test-Time Training ассоциативная память.

**Атрибуты:**


| Атрибут      | Размер                     | Описание             |
| ------------ | -------------------------- | -------------------- |
| `memory`     | `Linear(key_dim, val_dim)` | Матрица памяти M     |
| `momentum_S` | `Buffer(val_dim, key_dim)` | Momentum state S     |
| `gate_proj`  | `Linear(key_dim, 3)`       | Проекция для α, η, θ |


**Методы:**


| Метод                                                           | Строка | Описание                          |
| --------------------------------------------------------------- | ------ | --------------------------------- |
| `[retrieve(k, v)](../src/core/titans_memory.py:60)`             | L60    | Только извлечение, без обновления |
| `[update(k, v)](../src/core/titans_memory.py:74)`               | L74    | TTT-обновление в fp32             |
| `[retrieve_and_update(k, v)](../src/core/titans_memory.py:115)` | L115   | Атомарный retrieve + update       |
| `[reset_memory(strategy)](../src/core/titans_memory.py:150)`    | L150   | Умный сброс памяти                |


**Алгоритм TTT:**

```python
gates = sigmoid(gate_proj(k_agg))  # [α, η, θ]
M_fp32 = memory.weight.detach().float().requires_grad_(True)
pred = k @ M_fp32.T
loss = MSE(pred, v)
grad = ∂loss/∂M_fp32
momentum_S = η * momentum_S - θ * grad_clamped
memory.weight = (1-α) * M + momentum_S
```

**Критично:** Весь TTT путь выполняется в `float32` для стабильности при AMP.

### 3.6 SoftMoERouter

**Файл:** `[src/core/soft_moe_router.py:43](../src/core/soft_moe_router.py:43)`

**Назначение:** Soft Mixture-of-Experts без token dropping (Puigcerver et al. ICLR 2024).

**Алгоритм Soft MoE:**

```python
logits = dispatch_proj(x) / temperature
dispatch = softmax(logits, dim=0)   # нормализация по токенам
combine = softmax(logits, dim=-1)   # нормализация по слотам

slot_inputs = dispatch.T @ x        # агрегация токенов в слоты
slot_outputs = [expert(slot_input) for expert in experts]
output = combine @ slot_outputs     # агрегация выходов
```

**Ключевое отличие от Hard MoE:**

- Все токены получают взвешенную смесь ВСЕХ экспертов
- Нет token dropping при перегрузке
- Полностью дифференцируемый routing

### 3.7 HDIMPipeline

**Файл:** `[src/core/hdim_pipeline.py:128](../src/core/hdim_pipeline.py:128)`

**Назначение:** Полный пайплайн кроссдоменного переноса.

**Атрибуты:**


| Атрибут               | Описание                                     |
| --------------------- | -------------------------------------------- |
| `algebra`             | `CliffordAlgebra(p, q, r)`                   |
| `encoder`             | `HDIMEncoder(input_dim, clifford_dim)`       |
| `decoder`             | `HDIMDecoder(clifford_dim, output_dim)`      |
| `domain_rotors`       | `ModuleDict[str, DomainRotationOperator]`    |
| `invariant_extractor` | `InvariantExtractor(algebra)`                |
| `invariant_norm`      | `LayerNorm(clifford_dim)`                    |
| `memory`              | `TitansMemoryModule(key_dim, val_dim)`       |
| `memory_key_proj`     | `Linear(clifford_dim, memory_key_dim)`       |
| `moe`                 | `SoftMoERouter(input_dim, num_experts, ...)` |


**Ключевые методы:**


| Метод                                                                           | Строка | Описание                            |
| ------------------------------------------------------------------------------- | ------ | ----------------------------------- |
| `[encode_domain(x, domain_name)](../src/core/hdim_pipeline.py:176)`             | L176   | Кодирование + извлечение инварианта |
| `[transfer(x, source_domain, target_domain)](../src/core/hdim_pipeline.py:215)` | L215   | Полный кроссдоменный перенос        |
| `[add_domain(domain_name)](../src/core/hdim_pipeline.py:285)`                   | L285   | Добавление домена в runtime         |
| `[remove_domain(domain_name)](../src/core/hdim_pipeline.py:300)`                | L300   | Удаление домена                     |
| `[reset_memory(strategy)](../src/core/hdim_pipeline.py:313)`                    | L313   | Сброс памяти                        |


---

## 4. Model Layer

### 4.1 Иерархия моделей

```
HDIMConfig (dataclass)
    └── HDIMModel(nn.Module)
            └── TextHDIMModel(nn.Module)
                    ├── SimpleTextEncoder (по умолчанию)
                    └── SBERTEncoder (опционально, frozen)
```

### 4.2 HDIMConfig

**Файл:** `[src/models/hdim_model.py:83](../src/models/hdim_model.py:83)`

```python
@dataclass
class HDIMConfig:
    hidden_dim: int = 64           # Входная/выходная размерность
    num_domains: int = 4           # Число доменных роторов
    num_experts: int = 4           # Число MoE экспертов
    dropout: float = 0.1           # Dropout после encoder
    clifford_p: int = 3            # Положительные базисы Cl_{p,q,r}
    clifford_q: int = 1            # Отрицательные базисы
    clifford_r: int = 0            # Nilpotent базисы
    top_k: int = 2                 # Активных экспертов на токен
    memory_key_dim: int = 32       # Размерность ключей Titans памяти
    domain_names: Optional[List[str]] = None  # Явные имена доменов
```

### 4.3 HDIMModel

**Файл:** `[src/models/hdim_model.py:117](../src/models/hdim_model.py:117)`

**Назначение:** Основная модель с batch API для кроссдоменного переноса.

**Публичный API:**


| Метод                                                 | Строка | Сигнатура                                                             | Описание                                     |
| ----------------------------------------------------- | ------ | --------------------------------------------------------------------- | -------------------------------------------- |
| `[forward()](../src/models/hdim_model.py:289)`        | L289   | `forward(x, domain_id, *, return_state=False)`                        | Batch forward для same-domain reconstruction |
| `[transfer()](../src/models/hdim_model.py:399)`       | L399   | `transfer(encoding, source_domain, target_domain)`                    | Кроссдоменный перенос                        |
| `[transfer_pairs()](../src/models/hdim_model.py:481)` | L481   | `transfer_pairs(source_encoding, source_domain_id, target_domain_id)` | Batch transfer для mixed-domain pairs        |
| `[add_domain()](../src/models/hdim_model.py:433)`     | L433   | `add_domain(domain_name: str)`                                        | Добавить домен в runtime                     |
| `[remove_domain()](../src/models/hdim_model.py:446)`  | L446   | `remove_domain(domain_name: str)`                                     | Удалить домен                                |
| `[reset_memory()](../src/models/hdim_model.py:458)`   | L458   | `reset_memory(strategy: str = 'geometric')`                           | Сброс памяти между эпохами                   |


### 4.4 TextHDIMModel

**Файл:** `[src/models/text_hdim_model.py:191](../src/models/text_hdim_model.py:191)`

**Назначение:** Text-entry wrapper вокруг HDIMModel.

**Публичный API:**


| Метод                                                                   | Строка | Описание                      |
| ----------------------------------------------------------------------- | ------ | ----------------------------- |
| `[encode_texts()](../src/models/text_hdim_model.py:255)`                | L255   | Кодирование текстов в векторы |
| `[forward_texts()](../src/models/text_hdim_model.py:268)`               | L268   | Forward с текстовым входом    |
| `[transfer_texts()](../src/models/text_hdim_model.py:287)`              | L287   | Кроссдоменный перенос текста  |
| `[score_text_pairs_with_state()](../src/models/text_hdim_model.py:340)` | L340   | Сходство + полные состояния   |


### 4.5 SBERTEncoder

**Файл:** `[src/models/sbert_encoder.py:20](../src/models/sbert_encoder.py:20)`

**Архитектура:**

```
Text → SBERT (frozen, 768-dim) → Linear→LayerNorm→GELU→Dropout→Linear → output_dim
```

**Константы:**

- `MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"`
- `SBERT_DIM = 768`

**Особенности:**

- SBERT остаётся на CPU при `freeze=True` для экономии VRAM (278M params)
- Projection MLP обучается (trainable)
- Встроенный кэш embeddings для frozen режима

### 4.6 Model Factory

**Файл:** `[src/models/model_factory.py](../src/models/model_factory.py)`


| Функция                                                                   | Строка | Возвращает              |
| ------------------------------------------------------------------------- | ------ | ----------------------- |
| `[build_hdim_model(cfg)](../src/models/model_factory.py:143)`             | L143   | `HDIMModel`             |
| `[build_text_hdim_model(cfg, ...)](../src/models/model_factory.py:148)`   | L148   | `TextHDIMModel`         |
| `[build_sbert_hdim_model(cfg, ...)](../src/models/model_factory.py:189)`  | L189   | `TextHDIMModel + SBERT` |
| `[model_from_experiment_config(exp)](../src/models/model_factory.py:242)` | L242   | `HDIMModel              |


---

## 5. Training Layer

### 5.1 HDIMTrainer

**Файл:** `[src/training/trainer.py:19](../src/training/trainer.py:19)`

**Конструктор:**

```python
class HDIMTrainer:
    def __init__(
        self,
        model: HDIMModel,
        optimizer: Optimizer,
        device: str | torch.device = "cpu",
        # Loss weights
        lambda_iso: float = 0.1,
        lambda_pair: float = 0.1,
        lambda_routing: float = 0.05,
        lambda_memory: float = 0.05,
        lambda_z: float = 0.0,      # MoE anti-collapse
        # InfoNCE
        infonce_temperature: float = 0.15,
        focal_gamma: float = 1.0,
        ...
    )
```

### 5.2 Losses — Полный каталог


| Loss              | Вес            | Фаза     | Формула                         | Описание               |
| ----------------- | -------------- | -------- | ------------------------------- | ---------------------- |
| `loss_recon`      | 1.0            | Phase 1  | `MSE(output, target)`           | Реконструкция          |
| `loss_iso`        | `λ_iso`        | Phase 1  | `MSE(training_inv, iso_target)` | Изоморфизм             |
| `loss_pair`       | `λ_pair`       | Phase 3  | InfoNCE / Focal-InfoNCE         | Pair ranking           |
| `loss_routing`    | `λ_routing`    | Phase 7  | `-entropy(routing_weights)`     | Routing entropy        |
| `router_z_loss`   | `λ_z`          | Phase 9  | `(logsumexp(logits))²`          | MoE anti-collapse      |
| `loss_memory`     | `λ_memory`     | Phase 6  | `MSE(retrieved, target)`        | Titans memory          |
| `loss_sts`        | `λ_sts`        | Phase 8  | `1 - cos_sim(inv, iso_target)`  | STS regularization     |
| `loss_angle`      | `λ_angle`      | Phase 11 | AnglE loss                      | Angular similarity     |
| `loss_dcl`        | `λ_dcl`        | Phase 20 | DCL loss                        | Decoupled Contrastive  |
| `loss_uniformity` | `λ_uniformity` | Phase 20 | Uniformity+Alignment            | Representation quality |


**Total Loss:**

```
L_total = L_recon + λ_iso L_iso + λ_pair L_pair + λ_routing L_routing + 
          λ_memory L_memory + λ_z L_z + λ_sts L_sts + λ_angle L_angle + 
          λ_dcl L_dcl + λ_uniformity L_uniformity
```

### 5.3 Training Regimes


| Режим              | Условие                                 | Поведение                                      |
| ------------------ | --------------------------------------- | ---------------------------------------------- |
| **Reconstruction** | Нет пар в батче                         | `forward()` с `source_domain == target_domain` |
| **Paired**         | Есть `pair_encoding` + `pair_domain_id` | `transfer_pairs()` для кроссдоменного переноса |


### 5.4 Memory Modes


| Режим        | Обновление | Использование           |
| ------------ | ---------- | ----------------------- |
| `"none"`     | Нет        | Memory не используется  |
| `"retrieve"` | Нет        | Только чтение из памяти |
| `"update"`   | Да         | Чтение + запись (TTT)   |


### 5.5 Checkpoints

**Сохраняется:**

- `step` — номер шага
- `model_state_dict` — веса модели
- `optimizer_state_dict` — состояние оптимизатора
- `scaler_state_dict` — состояние AMP GradScaler (если есть)

**Checkpoint policy:**

1. **Best checkpoint:** `checkpoints/best.pt` — сохраняется когда `score > best_score`
2. **Periodic checkpoints:** `checkpoints/epoch_NNNN.pt` — каждые `save_every` эпох
3. **Final checkpoint:** `checkpoints/hdim_final.pt` — в конце обучения

---

## 6. Scripts Layer

### 6.1 Entrypoints


| Entrypoint                                | Команда                                             | Назначение               |
| ----------------------------------------- | --------------------------------------------------- | ------------------------ |
| `[gpu_train.py](../scripts/gpu_train.py)` | `python scripts/gpu_train.py --use_pairs --amp`     | GPU training с AMP       |
| `[train.py](../src/training/train.py)`    | `python -m src.training.train --config config.json` | CLI training             |
| `[hdim_demo.py](../hdim_demo.py)`         | `python hdim_demo.py`                               | Демонстрации компонентов |


### 6.2 Ключевые опции gpu_train.py


| Опция                   | Default | Описание                                    |
| ----------------------- | ------- | ------------------------------------------- |
| `--epochs`              | 30      | Число эпох                                  |
| `--batch_size`          | 32      | Размер батча                                |
| `--hidden_dim`          | 128     | HDIM hidden dimension                       |
| `--num_experts`         | 4       | Число MoE экспертов                         |
| `--lambda_iso`          | 0.1     | Iso loss weight                             |
| `--lambda_pair`         | 0.1     | Pair loss weight                            |
| `--lambda_z`            | 0.0     | Router Z-loss weight (>= 0.01 рекомендуемо) |
| `--infonce_temperature` | 0.15    | Temperature для InfoNCE                     |
| `--focal_gamma`         | 1.0     | Gamma для Focal-InfoNCE                     |
| `--amp`                 | True    | Mixed precision                             |
| `--soft_router`         | False   | Использовать SoftMoERouter                  |
| `--real_pairs`          | None    | Path to JSON pairs file                     |


---

## 7. Потоки данных

### 7.1 Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Text Input                                                              │
│  ┌─────────┐                                                             │
│  │ "text"  │                                                             │
│  └────┬────┘                                                             │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────┐     ┌───────────────────┐     ┌─────────────────┐   │
│  │SimpleTextEncoder│ OR  │AdvancedTextEncoder│ OR  │  SBERTEncoder   │   │
│  │(trainable)      │     │(Transformer+RoPE) │     │ (frozen+proj)   │   │
│  └────────┬────────┘     └────────┬──────────┘     └────────┬────────┘   │
│           │                       │                         │            │
│           └───────────────────────┼─────────────────────────┘            │
│                                   │                                      │
│                                   ▼                                      │
│  Embedding (B, hidden_dim)                                               │
│  ┌─────────────────┐                                                     │
│  │ encoding tensor │                                                     │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                     │
│  │   HDIMEncoder   │  ←── Linear/Quaternion → LayerNorm                  │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  Multivector (B, clifford_dim)                                           │
│  ┌─────────────────┐                                                     │
│  │   g_source      │  ←── Clifford multivector                           │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌───────────────────┐                                                   │
│  │InvariantExtractor │  ←── R_source⁻¹ · g_source · R_source             │
│  └────────┬──────────┘                                                   │
│           │                                                              │
│           ▼                                                              │
│  Invariant (B, clifford_dim)                                             │
│  ┌─────────────────┐                                                     │
│  │     u_inv       │  ←── Raw structural invariant                       │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                     │
│  │  LayerNorm      │  ←── Stabilization                                  │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                     │
│  │ memory_key_proj │  ←── Linear(clifford_dim → key_dim)                 │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌───────────────────┐                                                   │
│  │TitansMemoryModule │ ←── retrieve_and_update(key, u_inv)               │
│  └────────┬──────────┘                                                   │
│           │                                                              │
│           ▼                                                              │
│  Memory-Augmented Invariant                                              │
│  ┌─────────────────┐                                                     │
│  │     u_mem       │  ←── u_inv + retrieved                              │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                     │
│  │ SoftMoERouter   │  ←── slot_inputs → experts → combine                │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  Routed Invariant                                                        │
│  ┌─────────────────┐                                                     │
│  │    u_route      │  ←── MoE-processed invariant                        │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                     │
│  │sandwich_transfer│ ←── R_target · u_route · R_target⁻¹                 │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  Target Multivector                                                      │
│  ┌─────────────────┐                                                     │
│  │    g_target     │  ←── Domain-specific representation                 │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                     │
│  │   HDIMDecoder   │  ←── Linear → LayerNorm                             │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  Output Tensor (B, output_dim)                                           │
│  ┌─────────────────┐                                                     │
│  │     output      │                                                     │
│  └─────────────────┘                                                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Публичный API

### 8.1 Быстрый старт

```python
from src.models.model_factory import build_sbert_hdim_model
from src.models.hdim_model import HDIMConfig

# Создание модели
config = HDIMConfig(
    hidden_dim=256,
    num_domains=4,
    num_experts=4,
    top_k=2,
    memory_key_dim=32,
    clifford_p=3,
    clifford_q=1,
)

model = build_sbert_hdim_model(
    config,
    soft_router=True,      # SoftMoERouter (рекомендуется)
    freeze_sbert=True,     # Frozen SBERT (рекомендуется)
    z_loss_weight=0.01,     # MoE anti-collapse
)

# Кодирование текста
texts = ["example problem description", "another problem"]
encodings = model.encode_texts(texts, device="cuda")

# Same-domain forward
import torch
domain_id = torch.tensor([0, 0], device="cuda")
output, routing, invariant = model.forward_texts(texts, domain_id)

# Cross-domain transfer
target_domain = torch.tensor([1, 1], device="cuda")
output, routing, invariant, state = model.transfer_text_pairs(
    texts, domain_id, target_domain
)
```

### 8.2 Кастомная конфигурация

```python
from src.models.hdim_model import HDIMConfig, HDIMTextConfig

config = HDIMConfig(
    hidden_dim=256,
    num_domains=10,  # Расширенный набор доменов
    num_experts=8,
    top_k=3,
    memory_key_dim=64,
    clifford_p=3,
    clifford_q=1,
    domain_names=["physics", "chemistry", "biology", "engineering", ...],
    text=HDIMTextConfig(
        max_length=256,
        tokenizer_name="char",
    ),
)
```

---

## 9. Конфигурации

### 9.1 HDIMConfig


| Параметр         | Default | Влияние                          |
| ---------------- | ------- | -------------------------------- |
| `hidden_dim`     | 64      | Размерность входа/выхода модели  |
| `num_domains`    | 4       | Число доменных роторов           |
| `num_experts`    | 4       | Число MoE экспертов              |
| `top_k`          | 2       | Активных экспертов на токен      |
| `dropout`        | 0.1     | Dropout после encoder            |
| `memory_key_dim` | 32      | Размерность ключей Titans памяти |
| `clifford_p`     | 3       | Положительные базисы Cl_{p,q,r}  |
| `clifford_q`     | 1       | Отрицательные базисы             |
| `clifford_r`     | 0       | Nilpotent базисы                 |
| `domain_names`   | None    | Явные имена доменов              |


### 9.2 ExperimentConfig


| Параметр              | Default | Влияние                               |
| --------------------- | ------- | ------------------------------------- |
| `epochs`              | 3       | Число эпох обучения                   |
| `batch_size`          | 16      | Размер батча (>= 32 для InfoNCE)      |
| `lr`                  | 1e-3    | Learning rate                         |
| `lambda_iso`          | 0.1     | Вес isomorphism loss                  |
| `lambda_pair`         | 0.1     | Вес pair ranking loss                 |
| `lambda_z`            | 0.0     | Вес MoE Z-loss (>= 0.01 рекомендуемо) |
| `infonce_temperature` | 0.15    | Temperature для InfoNCE               |
| `focal_gamma`         | 1.0     | Gamma для Focal-InfoNCE               |
| `soft_router`         | False   | Использовать SoftMoERouter            |


### 9.3 Factory Flags


| Флаг                       | Эффект                                            |
| -------------------------- | ------------------------------------------------- |
| `soft_router=True`         | Заменяет R3MoERouter на SoftMoERouter             |
| `freeze_sbert=True`        | Frozen SBERT + trainable projection               |


---

## 10. Stable vs Experimental

### 10.1 Production-Ready


| Компонент                | Файл                                                             | Статус     |
| ------------------------ | ---------------------------------------------------------------- | ---------- |
| `CliffordAlgebra`        | `[hypercomplex.py:20](../src/core/hypercomplex.py:20)`           | **Stable** |
| `DomainRotationOperator` | `[domain_operators.py:19](../src/core/domain_operators.py:19)`   | **Stable** |
| `InvariantExtractor`     | `[domain_operators.py:54](../src/core/domain_operators.py:54)`   | **Stable** |
| `TitansMemoryModule`     | `[titans_memory.py:30](../src/core/titans_memory.py:30)`         | **Stable** |
| `SoftMoERouter`          | `[soft_moe_router.py:43](../src/core/soft_moe_router.py:43)`     | **Stable** |
| `HDIMEncoder/Decoder`    | `[hdim_pipeline.py:90](../src/core/hdim_pipeline.py:90)`         | **Stable** |
| `HDIMPipeline`           | `[hdim_pipeline.py:128](../src/core/hdim_pipeline.py:128)`       | **Stable** |
| `HDIMModel`              | `[hdim_model.py:117](../src/models/hdim_model.py:117)`           | **Stable** |
| `TextHDIMModel`          | `[text_hdim_model.py:191](../src/models/text_hdim_model.py:191)` | **Stable** |
| `SBERTEncoder`           | `[sbert_encoder.py:20](../src/models/sbert_encoder.py:20)`       | **Stable** |
| `HDIMTrainer`            | `[trainer.py:19](../src/training/trainer.py:19)`                 | **Stable** |


### 10.2 Удалено (Occam's razor)

Компоненты удалены как неиспользуемый мёртвый код:


---

## Расхождения README vs Код


| Аспект                     | README        | Код                            | Примечание                                |
| -------------------------- | ------------- | ------------------------------ | ----------------------------------------- |
| Default `hidden_dim`       | 256           | 64                             | README описывает типичную конфигурацию    |
| Default MoE router         | SoftMoERouter | R3MoERouter                    | `HDIMPipeline` создаёт R3, factory — Soft |
| `invariant_norm`           | Не указан     | LayerNorm после извлечения     | Добавлен для стабильности                 |
| `memory_key_proj`          | Не указан     | Linear(clifford_dim → key_dim) | Для проекции ключей                       |
| `input_dim` в HDIMPipeline | 256           | 64                             | Отличается от HDIMConfig default          |


---

## Антипаттерны


| Паттерн                | Проблема          | Решение                     |
| ---------------------- | ----------------- | --------------------------- |
| `ModularMoERouter`     | Нестабилен        | `SoftMoERouter`             |
| `reset_memory('zero')` | Уничтожает знания | `reset_memory('geometric')` |
| `batch_size < 32`      | Мало негативов    | `batch_size >= 32`          |
| `temperature < 0.15`   | Overconfidence    | `temperature >= 0.15`       |
| `lambda_z = 0`         | MoE collapse      | `lambda_z >= 0.01`          |
| Нет `reset_memory()`   | Memory drift      | Вызывать каждую эпоху       |


---

## Ключевые ссылки

### Файлы

- `[src/core/hypercomplex.py](../src/core/hypercomplex.py)` — Алгебраическая база
- `[src/core/domain_operators.py](../src/core/domain_operators.py)` — Доменные операторы
- `[src/core/titans_memory.py](../src/core/titans_memory.py)` — Titans memory
- `[src/core/soft_moe_router.py](../src/core/soft_moe_router.py)` — Soft MoE router
- `[src/core/hdim_pipeline.py](../src/core/hdim_pipeline.py)` — Главный pipeline
- `[src/models/hdim_model.py](../src/models/hdim_model.py)` — HDIMModel, конфигурации
- `[src/models/text_hdim_model.py](../src/models/text_hdim_model.py)` — TextHDIMModel
- `[src/models/model_factory.py](../src/models/model_factory.py)` — Factory functions
- `[src/training/trainer.py](../src/training/trainer.py)` — HDIMTrainer, losses

### Исследовательские отчёты

- `[.omc/research/core-architecture.md](../.omc/research/core-architecture.md)` — Ядро
- `[.omc/research/model-stack.md](../.omc/research/model-stack.md)` — Модели
- `[.omc/research/training-and-ops.md](../.omc/research/training-and-ops.md)` — Обучение
- `[.omc/research/architecture-synthesis.md](../.omc/research/architecture-synthesis.md)` — Синтез архитектуры

---

*Документация сгенерирована на основе анализа исходного кода.*