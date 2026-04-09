# HDIM Architecture Documentation

> **Дата:** 2026-04-09
> **Версия:** Research Prototype (Phase 30+)
> **Источники:** Исследовательские отчёты в `[.omc/research/](../.omc/research/)`
> **Verification:** pytest 754 PASS (10 skipped) | Lean4 161/163 PASS

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
11. [Hallucination & Safety Subsystem](#11-hallucination--safety-subsystem)
12. [Online Learning Subsystem](#12-online-learning-subsystem)

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


**Лучший результат:** 1.1814 (Run 18, ep13, temp=0.10, lambda_pair=0.40): `pair_margin=1.0224`, `STS=0.537`

---

## 2. Слойная архитектура

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            HYPERCOREPLEX AI (HDIM)                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Scripts Layer (20 entrypoints)                                                 │
│  ┌───────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐     │
│  │ gpu_train.py  │ │moe_chat.py   │ │auto_tune.py  │ │benchmark_comparison│     │
│  │ (AMP, GPU)    │ │(interactive) │ │(Optuna)      │ │(MoE vs FFN vs SBERT│     │
│  └───────┬───────┘ └──────┬───────┘ └──────┬───────┘ └────────┬───────────┘     │
│          └────────────────┼────────────────┼──────────────────┘                 │
├───────────────────────────┼────────────────┼────────────────────────────────────┤
│  Training Layer                                                                 │
│  ┌────────┴───────┐    ┌───────┴────────┐   ┌────────────────┐                  │
│  │  HDIMTrainer   │    │ ExperimentRun  │   │   Datasets     │                  │
│  │  (losses, AMP) │    │  (orchestr.)   │   │(DomainProblem) │                  │
│  └────────┬───────┘    └────────────────┘   └────────────────┘                  │
│           │                                                                     │
├───────────┼─────────────────────────────────────────────────────────────────────┤
│  Model Layer                                                                    │
│  ┌────────┴────────┐  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  HDIMModel      │  │ TextHDIMModel   │  │ SBERTEncoder    │                  │
│  │  (core wrapper) │◄─┤ (text wrapper)  │◄─┤ (frozen encoder)│                  │
│  └────────┬────────┘  └─────────────────┘  └─────────────────┘                  │
│           │              ┌─────────────────┐                                    │
│           └──────────────┤ model_factory   │                                    │
│                          │ (build_*)       │                                  │
│                          └─────────────────┘                                  │
├───────────┼───────────────────────────────────────────────────────────────────┤
│  Core Layer                                                                   │
│  ┌────────┴────────┐                                                           │
│  │  HDIMPipeline   │                                                           │
│  │  (orchestrator) │                                                           │
│  └────────┬────────┘                                                           │
│           │                                                                    │
│  ┌────────┴────────┬──────────────────┬──────────────────┐                    │
│  │ hypercomplex.py │domain_operators  │  hdim_pipeline   │                    │
│  │ (CliffordAlgebra│(DomainRotor,     │ (HDIMEncoder,    │                    │
│  │  Quaternion)     │ InvariantExtr.)  │  HDIMDecoder)    │                    │
│  └─────────────────┴──────────────────┴──────────────────┘                    │
│           │                                                                    │
│  ┌────────┴────────┬──────────────────┬──────────────────┐                    │
│  │ soft_moe_router │   moe_kernel     │  moe_interface   │                    │
│  │ (SoftMoE,       │ (4 named experts │  (MoERouter ABC) │                    │
│  │  SharedExpert,  │  560K params)    │                  │                    │
│  │  AuxLossFree,   │                  │                  │                    │
│  │  ExpertOrtho)   │                  │                  │                    │
│  └─────────────────┴──────────────────┴──────────────────┘                    │
│  ┌─────────────────┬──────────────────┬──────────────────┐                    │
│  │maxscore_router  │moe_kernel_adapter│  hbma_memory     │                    │
│  │ (Wang ACL 2025, │ (MoEKernel→      │ (4-system brain: │                    │
│  │  min-cost flow)  │  MoERouter)      │  Working/Episodic│                    │
│  │                 │                  │  /Semantic/Proc.)│                    │
│  └─────────────────┴──────────────────┴──────────────────┘                    │
│  ┌─────────────────┬──────────────────┬──────────────────┐                    │
│  │ msa_attention   │memory_interface  │memory_persistence│                    │
│  │ (MSA Sparse     │ (MemoryInterface │ (save/load/      │                    │
│  │  Index, top-k,  │  ABC, TitansAdpt │  checkpoint,     │                    │
│  │  chunk compress)│  HBMAMemoryAdpt) │  atomic backup)  │                    │
│  └─────────────────┴──────────────────┴──────────────────┘                    │
│  ┌─────────────────┬──────────────────┬──────────────────┐                    │
│  │transfer_engine  │ domain_encoder   │invariant_processor│                   │
│  │ (MoE+sandwich+  │ (encoder+rotors+ │ (memory-based    │                    │
│  │  decode)         │  inv. extraction)│  processing)     │                    │
│  └─────────────────┴──────────────────┴──────────────────┘                    │
│  ┌─────────────────┬──────────────────┬──────────────────┐                    │
│  │transfer_state   │ online_learner   │  online_lora     │                    │
│  │ (cross-domain   │ (continual learn,│ (task-free LoRA, │                    │
│  │  state dataclass│  3 grad modes)   │  Linear+Conv2d)  │                    │
│  └─────────────────┴──────────────────┴──────────────────┘                    │
│  ┌─────────────────┬──────────────────┬──────────────────┐                    │
│  │ continual_norm  │hallucination_    │hallucination_    │                    │
│  │ (streaming norm │ detector         │ feedback         │                    │
│  │  stability)     │ (5-signal risk)  │ (risk rerouting) │                    │
│  └─────────────────┴──────────────────┴──────────────────┘                    │
│  ┌─────────────────┐                                                        │
│  │semantic_entropy │                                                        │
│  │ _probe          │                                                        │
│  │ (uncertainty    │                                                        │
│  │  quantification)│                                                        │
│  └─────────────────┘                                                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Таблица слоёв


| Слой         | Файлы                                          | Ответственность                                    | Публичный API                             |
| ------------ | ---------------------------------------------- | -------------------------------------------------- | ----------------------------------------- |
| **Scripts**  | `[scripts/*.py](../scripts/)` (20 entrypoints) | Entrypoints, CLI, GPU training, benchmarks, demos  | `main()`, `demo_*()`                      |
| **Training** | `[src/training/*.py](../src/training/)`        | Losses, regimes, datasets, checkpoints             | `HDIMTrainer`, `ExperimentRunner`         |
| **Model**    | `[src/models/*.py](../src/models/)`            | Wrappers, encoders, factories, metrics             | `HDIMModel`, `TextHDIMModel`, `build_*()` |
| **Core**     | `[src/core/*.py](../src/core/)` (25 модулей)   | Алгебра, память, роутинг, pipeline, safety, online | `HDIMPipeline`, `CliffordAlgebra`         |


---

## 3. Core Layer

### 3.1 Обзор модулей


| Модуль                                                               | Класс                        | Назначение                                                                     | Статус     |
| -------------------------------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------ | ---------- |
| `[hypercomplex.py](../src/core/hypercomplex.py)`                     | `CliffordAlgebra`            | Алгебра Клиффорда Cl_{p,q,r}                                                   | **Stable** |
|                                                                      | `QuaternionLinear`           | Кватернионный слой                                                             | **Stable** |
| `[domain_operators.py](../src/core/domain_operators.py)`             | `DomainRotationOperator`     | Обучаемый ротор домена                                                         | **Stable** |
|                                                                      | `InvariantExtractor`         | Извлечение U_inv = R⁻¹GR                                                       | **Stable** |
|                                                                      | `DomainRegistry`             | Реестр доменов                                                                 | **Stable** |
| `[titans_memory.py](../src/core/titans_memory.py)`                   | `TitansMemoryModule`         | Test-Time Training память                                                      | **Stable** |
| `[soft_moe_router.py](../src/core/soft_moe_router.py)`               | `SoftMoERouter`              | Soft Mixture-of-Experts (SharedExpert, AuxLossFree, ExpertOrtho)               | **Stable** |
| `[moe_kernel.py](../src/core/moe_kernel.py)`                         | `MoEKernel`                  | 4 доменных эксперта (math/language/code/science), 560K params                  | **Stable** |
|                                                                      | `MoEKernelConfig`            | Dataclass конфигурации MoE-ядра                                                | **Stable** |
|                                                                      | `MoEKernelState`             | State-контейнер forward-прохода                                                | **Stable** |
| `[moe_interface.py](../src/core/moe_interface.py)`                   | `MoERouter` ABC              | Абстрактный интерфейс роутера (forward, get_expert_load, ortho_loss)           | **Stable** |
| `[moe_kernel_adapter.py](../src/core/moe_kernel_adapter.py)`         | `MoEKernelAdapter`           | Адаптер MoEKernel → MoERouter interface                                        | **Stable** |
| `[maxscore_router.py](../src/core/maxscore_router.py)`               | `MaxScoreRouter`             | Maximum Score Routing (Wang et al. ACL 2025), min-cost max-flow, SoftTopk      | **Stable** |
| `[msa_attention.py](../src/core/msa_attention.py)`                   | `MSASparseIndex`             | Memory Sparse Attention: top-k + chunk compression, O(log N) retrieval         | **Stable** |
|                                                                      | `MSAOverflowBuffer`          | MSA-backed overflow для EpisodicMemory, multi-hop retrieval                    | **Stable** |
|                                                                      | `MSAMemory`                  | Prototype-based sparse retrieval с MSA index, ring buffer overflow, batched store | **Stable** |
| `[hbma_memory.py](../src/core/hbma_memory.py)`                       | `HBMAMemory`                 | 4-system brain-inspired: Working/Episodic/Semantic/Procedural                  | **Stable** |
|                                                                      | `WorkingMemory`              | Круговой буфер, salience-filtered attention                                    | **Stable** |
|                                                                      | `EpisodicMemory`             | Surprise-gated binding, temporal ordering, overflow                            | **Stable** |
|                                                                      | `SemanticMemory`             | EMA prototypes + confidence + MSA sparse retrieval                             | **Stable** |
|                                                                      | `ProceduralMemory`           | Learnable pattern store, trigger detection                                     | **Stable** |
|                                                                      | `ConsolidationEngine`        | Working→Episodic→Semantic pipeline                                             | **Stable** |
|                                                                      | `CLSMemory`                  | Обратная совместимость (alias HBMAMemory)                                      | **Stable** |
| `[memory_interface.py](../src/core/memory_interface.py)`             | `MemoryInterface` ABC        | Unified memory contract: forward, reset, memory_loss                           | **Stable** |
|                                                                      | `TitansAdapter`              | TitansMemoryModule → MemoryInterface                                           | **Stable** |
|                                                                      | `HBMAMemoryAdapter`          | HBMAMemory → MemoryInterface                                                   | **Stable** |
| `[memory_persistence.py](../src/core/memory_persistence.py)`         | `MemoryPersistence`          | Save/load/checkpoint с atomicity и backup rotation                             | **Stable** |
| `[online_learner.py](../src/core/online_learner.py)`                 | `OnlineLearner`              | Continual learning: TTT-style, replay buffer, 3 gradient modes                 | **Stable** |
| `[online_lora.py](../src/core/online_lora.py)`                       | `OnlineLoRA`                 | Task-free low-rank adaptation для Linear и Conv2d                              | **Stable** |
|                                                                      | `OnlineLoRALinear`           | Convenience wrapper для nn.Linear                                              | **Stable** |
|                                                                      | `OnlineLoRAConv`             | Convenience wrapper для nn.Conv2d                                              | **Stable** |
|                                                                      | `OnlineLoRAManager`          | Координация: batch EMA, consolidation, stats                                   | **Stable** |
| `[continual_norm.py](../src/core/continual_norm.py)`                 | `ContinualNorm`              | Streaming normalization без task reset (IL-ETransformer)                       | **Stable** |
|                                                                      | `ContinualNormLayer`         | LayerNorm alternative с continual EMA monitoring                               | **Stable** |
| `[hallucination_detector.py](../src/core/hallucination_detector.py)` | `HallucinationDetector`      | 5-signal weighted risk: entropy+confidence+mismatch+eigen+semantic             | **Stable** |
| `[hallucination_feedback.py](../src/core/hallucination_feedback.py)` | `HallucinationFeedbackLoop`  | Risk-based rerouting, confidence adjustment, memory consolidation trigger      | **Stable** |
| `[semantic_entropy_probe.py](../src/core/semantic_entropy_probe.py)` | `SemanticEntropyProbe`       | Linear probe для uncertainty quantification (Kossen ICLR 2024)                 | **Stable** |
| `[transfer_engine.py](../src/core/transfer_engine.py)`               | `TransferEngine`             | MoE routing + sandwich transfer + decode, инкапсуляция кроссдоменного переноса | **Stable** |
| `[domain_encoder.py](../src/core/domain_encoder.py)`                 | `DomainEncoder`              | Encoder + rotors + invariant extraction + normalization                        | **Stable** |
| `[invariant_processor.py](../src/core/invariant_processor.py)`       | `InvariantProcessor`         | Memory-based обработка инвариантов через MemoryInterface                       | **Stable** |
| `[transfer_state.py](../src/core/transfer_state.py)`                 | `TransferState`              | Dataclass состояния кроссдоменного переноса                                    | **Stable** |
| `[hdim_pipeline.py](../src/core/hdim_pipeline.py)`                   | `HDIMPipeline`               | Главный orchestrator                                                           | **Stable** |
|                                                                      | `HDIMEncoder`                | Кодирование → мультивектор                                                     | **Stable** |
|                                                                      | `HDIMDecoder`                | Мультивектор → выход                                                           | **Stable** |


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

# Phase 26: AuxLossFree — добавить per-expert bias
if self.use_aux_loss_free:
    logits = logits + self._expert_bias.unsqueeze(0)

dispatch = softmax(logits, dim=0)   # нормализация по токенам
combine = softmax(logits, dim=-1)   # нормализация по слотам

slot_inputs = dispatch.T @ x        # агрегация токенов в слоты
# Batched expert execution via stacked weights + einsum
h = einsum('esd,ehd->esh', slot_inputs, W1_stack) + b1
h = GELU(h)
slot_outputs = einsum('esh,edh->esd', h, W2_stack) + b2
output = combine @ slot_outputs     # агрегация выходов

# Phase 26: Shared Expert — всегда-включённый FFN
if self.use_shared_expert:
    output = output + self._shared_expert(x)
```

**Ключевое отличие от Hard MoE:**

- Все токены получают взвешенную смесь ВСЕХ экспертов
- Нет token dropping при перегрузке
- Полностью дифференцируемый routing

**Phase 26 нововведения:**


| Фича                                 | Описание                                      | Включение                        |
| ------------------------------------ | --------------------------------------------- | -------------------------------- |
| Shared Expert (DeepSeek-V3)          | Always-on FFN обрабатывает ВСЕ входы          | `enable_shared_expert()`         |
| AuxLoss-Free Balancing (DeepSeek-V3) | Per-expert bias для динамической балансировки | `enable_aux_loss_free(lr=0.001)` |
| Expert Orthogonalization             | `L_o =                                        |                                  |


**Phase 26 удаления:**

- `SoftRouterState` dataclass — заменён на plain dict
- `calibration_head`, `adaptive_dropout` — не использовались
- `similarity_preserving_loss` — заменён на AuxLossFree
- `experts` ModuleList — заменён на batched einsum со stacked weights (`W1_stack`, `W2_stack`, `b1_stack`, `b2_stack`)

### 3.7 MoEKernel

**Файл:** `[src/core/moe_kernel.py](../src/core/moe_kernel.py)`

**Назначение:** 4 именованных доменных эксперта (math/language/code/science) с 560K параметров. Заменяет удалённый `DomainExpertPool`.

**Ключевые классы:**


| Класс             | Назначение                                           |
| ----------------- | ---------------------------------------------------- |
| `MoEKernel`       | MoE ядро с named domain experts                      |
| `MoEKernelConfig` | Dataclass конфигурации (input_dim, num_experts, ...) |
| `MoEKernelState`  | State-контейнер forward-прохода                      |


### 3.8 MoERouter ABC

**Файл:** `[src/core/moe_interface.py](../src/core/moe_interface.py)`

**Назначение:** Абстрактный интерфейс для всех MoE-роутеров. Обеспечивает полиморфное использование SoftMoERouter, MoEKernel (через адаптер) и MaxScoreRouter.

**Абстрактные методы:**


| Метод                             | Сигнатура                  | Описание                   |
| --------------------------------- | -------------------------- | -------------------------- |
| `forward(x)`                      | `(Tensor, Dict[str, Any])` | Route input через эксперты |
| `get_expert_load()`               | `Tensor[num_experts]`      | EMA load статистика        |
| `expert_orthogonalization_loss()` | `Tensor`                   | Loss за диверсификацию     |


### 3.9 MoEKernelAdapter

**Файл:** `[src/core/moe_kernel_adapter.py](../src/core/moe_kernel_adapter.py)`

**Назначение:** Адаптер, оборачивающий `MoEKernel` в интерфейс `MoERouter`. Переводит `MoEKernelState` → `Dict[str, Any]`, позволяя MoEKernel использоваться везде, где ожидается MoERouter.

### 3.10 MaxScoreRouter

**Файл:** `[src/core/maxscore_router.py](../src/core/maxscore_router.py)`

**Назначение:** Maximum Score Routing (Wang et al., ACL 2025). Моделирует routing как min-cost max-flow задачу. SoftTopk оператор обеспечивает дифференцируемый top-k selection без token dropping.

**Отличие от SoftMoERouter:**

- SoftMoE: каждый токен использует ВСЕ эксперты через dispatch/combine
- MaxScore: каждый токен использует top-k экспертов, но selection дифференцируем

**Ключевые фичи:**

- 0% token dropping
- Дифференцируемый top-k через SoftTopk
- Load balancing через entropy regularization
- Checkpoint/rollback для fault tolerance

### 3.11 MSAAttention (Memory Sparse Attention)

**Файл:** `[src/core/msa_attention.py](../src/core/msa_attention.py)`

**Назначение:** Sparse Index для SemanticMemory. Иерархическая компрессия и sparse retrieval.

**Ключевые классы:**


| Класс                        | Назначение                                                           |
| ---------------------------- | -------------------------------------------------------------------- |
| `MSAConfig`                  | Конфигурация: dim, top_k, chunk_size, num_heads, temperature         |
| `MSASparseIndex`             | Router projectors (W_KR, W_QR) + top-k selection + chunk compression |
| `MSAOverflowBuffer`          | MSA-backed overflow для EpisodicMemory, multi-hop retrieval          |
| `MSAMemory`                  | Prototype-based sparse retrieval, ring buffer overflow, batched store |


**Алгоритм MSA:**

```
KR = H @ W_KR    # Router K Projector
QR = H @ W_QR    # Router Q Projector
si = max_token(mean_head(cos(QR, KR)))  # routing scores
I = Top-k({si})  # sparse selection
compressed = mean_pool(P=64)  # chunk compression
```

**Результат:** O(log N) retrieval вместо O(N) dense cosine similarity.

### 3.12 HBMAMemory

**Файл:** `[src/core/hbma_memory.py](../src/core/hbma_memory.py)`

**Назначение:** 4-system brain-inspired memory (McClelland et al. 1995). Drop-in замена для TitansMemory.

**Подсистемы:**


| Подсистема        | Класс              | Аналог              | Назначение                                |
| ----------------- | ------------------ | ------------------- | ----------------------------------------- |
| Working Memory    | `WorkingMemory`    | Prefrontal cortex   | Круговой буфер, salience-filtered context |
| Episodic Memory   | `EpisodicMemory`   | Hippocampus CA3/CA1 | Surprise-gated binding, temporal ordering |
| Semantic Memory   | `SemanticMemory`   | Neocortex           | EMA prototypes + confidence + MSA sparse  |
| Procedural Memory | `ProceduralMemory` | Basal ganglia       | Learnable patterns, trigger detection     |


**Дополнительные компоненты:**


| Компонент               | Назначение                                                 |
| ----------------------- | ---------------------------------------------------------- |
| `ConsolidationEngine`   | Working→Episodic→Semantic pipeline                         |
| `SalienceScorer`        | Multi-factor: similarity+recency+frequency+importance+type |
| `MemorySubsystemPlugin` | ABC для расширения 5-й подсистемой                         |
| `CLSMemory`             | Обратная совместимость (alias HBMAMemory)                  |


### 3.13 MemoryInterface

**Файл:** `[src/core/memory_interface.py](../src/core/memory_interface.py)`

**Назначение:** Unified memory contract. Bridges TitansMemoryModule (k, v API) и HBMAMemory (single-input API) через общий ABC.

**Ключевые классы:**


| Класс                 | Назначение                                                 |
| --------------------- | ---------------------------------------------------------- |
| `MemoryResult`        | Dataclass: output, loss, updated, alpha/eta/theta/surprise |
| `MemoryInterface` ABC | forward(x, update_memory) → MemoryResult                   |
| `TitansAdapter`       | TitansMemoryModule → MemoryInterface                       |
| `HBMAMemoryAdapter`   | HBMAMemory → MemoryInterface                               |


### 3.14 MemoryPersistence

**Файл:** `[src/core/memory_persistence.py](../src/core/memory_persistence.py)`

**Назначение:** Save/load/checkpoint для HDIM memory систем.

**Поддерживаемые типы:** TitansMemoryModule, HBMAMemory/CLSMemory, TitansAdapter, HBMAMemoryAdapter.

**Ключевые методы:**


| Метод           | Описание                                   |
| --------------- | ------------------------------------------ |
| `save()`        | Versioned torch.save с metadata и checksum |
| `load()`        | Load с version validation и type matching  |
| `checkpoint()`  | Atomic checkpoint с backup rotation        |
| `export_json()` | Human-readable JSON экспорт с tensor stats |


### 3.15 OnlineLearner

**Файл:** `[src/core/online_learner.py](../src/core/online_learner.py)`

**Назначение:** Continual learning модуль с TTT-style gradient updates, experience replay, surprise detection, EMA моделью.

**3 режима градиентов:**


| Режим       | Градиенты            | Описание                                            |
| ----------- | -------------------- | --------------------------------------------------- |
| `DETACHED`  | Нет                  | Безопасный режим (default), нет градиентного потока |
| `SELECTIVE` | Только replay buffer | Градиенты для replay consolidation                  |
| `FULL`      | Полные               | Experimental, может дестабилизировать               |


**Ключевые компоненты:** `ReplayBuffer` (prioritized, surprise-based), `OnlineLearnerConfig`, `compute_surprise()` (1 - cosine similarity to EMA).

### 3.16 OnlineLoRA

**Файл:** `[src/core/online_lora.py](../src/core/online_lora.py)`

**Назначение:** Task-free low-rank adaptation для continual learning без catastrophic forgetting (Wei et al., WACV 2025).

**Ключевые классы:**


| Класс               | Назначение                                   |
| ------------------- | -------------------------------------------- |
| `OnlineLoRA`        | Base: LoRA A+B + importance weighting + EMA  |
| `OnlineLoRALinear`  | Convenience wrapper для nn.Linear            |
| `OnlineLoRAConv`    | Convenience wrapper для nn.Conv2d            |
| `OnlineLoRAManager` | Batch EMA updates, coordinated consolidation |
| `OnlineLoRAConfig`  | rank, alpha, importance_decay, ema_decay     |


**Алгоритм:** `output = base(x) + (x * importance) @ lora_A @ lora_B * scaling`, с gradient-based importance EMA и периодической консолидацией к EMA весам.

### 3.17 ContinualNorm

**Файл:** `[src/core/continual_norm.py](../src/core/continual_norm.py)`

**Назначение:** Streaming normalization без task reset (IL-ETransformer-style). В отличие от BatchNorm, ContinualNorm поддерживает EMA running statistics через task boundaries.

**Ключевые классы:**


| Класс                | Назначение                                    |
| -------------------- | --------------------------------------------- |
| `ContinualNorm`      | BatchNorm-style continual norm, EMA без reset |
| `ContinualNormLayer` | LayerNorm-style с continual monitoring        |


### 3.18 HallucinationDetector

**Файл:** `[src/core/hallucination_detector.py](../src/core/hallucination_detector.py)`

**Назначение:** 5-signal weighted risk detection. Комбинирует сигналы из MoE routing, memory, и hidden states в hallucination_risk ∈ [0, 1].

**5 сигналов:**


| Сигнал           | Вес | Источник                   | Высокий риск при    |
| ---------------- | --- | -------------------------- | ------------------- |
| Routing Entropy  | 25% | MoE gate distribution      | Высокая энтропия    |
| MoE Confidence   | 20% | Max gate weight            | Низкий confidence   |
| Memory Mismatch  | 20% | Titans surprise gradient   | Большой mismatch    |
| Semantic Entropy | 20% | SemanticEntropyProbe       | Высокая диверсность |
| EigenScore       | 15% | SVD routing representation | Высокая дисперсия   |


### 3.19 HallucinationFeedbackLoop

**Файл:** `[src/core/hallucination_feedback.py](../src/core/hallucination_feedback.py)`

**Назначение:** Risk-based self-correction. При высоком hallucination risk: reroute к safer expert, скорректировать confidence, trigger memory consolidation.

**5 уровней ответа (FeedbackAction):**


| Уровень               | Условие risk | Действие                            |
| --------------------- | ------------ | ----------------------------------- |
| NONE                  | < 0.3        | Нет коррекции                       |
| ADJUST_CONFIDENCE     | 0.3 - 0.5    | Снижение output confidence          |
| REROUTE               | 0.5 - 0.7    | Перенаправление к safer expert      |
| TRIGGER_CONSOLIDATION | 0.7 - 0.85   | Принудительная memory consolidation |
| FULL_CORRECTION       | >= 0.85      | Reroute + consolidate               |


### 3.20 SemanticEntropyProbe

**Файл:** `[src/core/semantic_entropy_probe.py](../src/core/semantic_entropy_probe.py)`

**Назначение:** Linear probe для предсказания semantic entropy из hidden states (Kossen et al., ICLR 2024). 45x-450x быстрее полной multi-sample генерации.

**Архитектура:** `Linear(hidden_dim, 1)` с zero initialization → `sigmoid` → entropy ∈ [0, 1]. Mean pooling по sequence dimension.

### 3.21 TransferEngine

**Файл:** `[src/core/transfer_engine.py](../src/core/transfer_engine.py)`

**Назначение:** Инкапсуляция кроссдоменного переноса: MoE routing → sandwich transfer → decode.

**Pipeline:**

```
u_mem → MoE → u_route → sandwich_transfer → g_target → decoder → output
```

Поддерживает gradient checkpointing через `enable_gradient_checkpointing()`.

### 3.22 DomainEncoder

**Файл:** `[src/core/domain_encoder.py](../src/core/domain_encoder.py)`

**Назначение:** Инкапсуляция кодирования входа в доменный инвариант.

**Pipeline:**

```
x → encoder → g_source → domain_rotor → invariant_extractor → u_inv → norm → u_inv_normalized
```

Поддерживает `add_domain()` для runtime-расширения.

### 3.23 InvariantProcessor

**Файл:** `[src/core/invariant_processor.py](../src/core/invariant_processor.py)`

**Назначение:** Обработка инвариантов через MemoryInterface с поддержкой режимов none/retrieve/update.

**Unified path:** все memory types проходят через `MemoryInterface.forward(x, update_memory)`.

### 3.24 TransferState

**Файл:** `[src/core/transfer_state.py](../src/core/transfer_state.py)`

**Назначение:** Dataclass состояния кроссдоменного переноса. Вынесен в отдельный модуль для избежания циклических импортов.

**Ключевые поля:** `g_source`, `u_inv`, `u_mem`, `u_route`, `g_target`, `output`, `memory_loss`, `router_state`, `memory_mode`.

### 3.25 HDIMPipeline

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
        lambda_pair: float = 0.40,     # Optimal: Run 18
        lambda_routing: float = 0.05,
        lambda_memory: float = 0.05,
        lambda_z: float = 0.0,         # MoE anti-collapse
        # InfoNCE
        infonce_temperature: float = 0.10,  # Optimum Run 18
        focal_gamma: float = 1.0,
        ...
    )
```

### 5.2 Losses — Полный каталог


| Loss              | Вес            | Оптимум        | Фаза     | Формула                         | Описание               |
| ----------------- | -------------- | -------------- | -------- | ------------------------------- | ---------------------- |
| `loss_recon`      | 1.0            | 1.0            | Phase 1  | `MSE(output, target)`           | Реконструкция          |
| `loss_iso`        | `λ_iso`        | 0.1            | Phase 1  | `MSE(training_inv, iso_target)` | Изоморфизм             |
| `loss_pair`       | `λ_pair`       | **0.40**       | Phase 3  | InfoNCE / Focal-InfoNCE         | Pair ranking           |
| `loss_routing`    | `λ_routing`    | 0.05           | Phase 7  | `-entropy(routing_weights)`     | Routing entropy        |
| `router_z_loss`   | `λ_z`          | >=0.01         | Phase 9  | `(logsumexp(logits))²`          | MoE anti-collapse      |
| `loss_memory`     | `λ_memory`     | 0.05           | Phase 6  | `MSE(retrieved, target)`        | Titans memory          |
| `loss_sts`        | `λ_sts`        | **0.0 (MAND)** | Phase 8  | `1 - cos_sim(inv, iso_target)`  | STS regularization     |
| `loss_angle`      | `λ_angle`      | --             | Phase 11 | AnglE loss                      | Angular similarity     |
| `loss_dcl`        | `λ_dcl`        | --             | Phase 20 | DCL loss                        | Decoupled Contrastive  |
| `loss_uniformity` | `λ_uniformity` | --             | Phase 20 | Uniformity+Alignment            | Representation quality |


**Total Loss:**

```
L_total = L_recon + λ_iso L_iso + λ_pair L_pair + λ_routing L_routing + 
          λ_memory L_memory + λ_z L_z + λ_sts L_sts + λ_angle L_angle + 
          λ_dcl L_dcl + λ_uniformity L_uniformity
```

**Критичные оптимальные значения:**

- `lambda_pair = 0.40` — лучший результат Run 18; выше не тестировалось
- `lambda_sts = 0.0` — ОБЯЗАТЕЛЬНО; любое значение > 0 подавляет pair_margin
- `infonce_temperature = 0.10` — оптимум Run 18; 0.12 даёт -0.0108

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

### 6.1 Entrypoints (20 скриптов)


| Скрипт                                                                    | Назначение                                                       |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `[gpu_train.py](../scripts/gpu_train.py)`                                 | Main GPU training: AMP, gradient checkpointing                   |
| `[benchmark_comparison.py](../scripts/benchmark_comparison.py)`           | Сравнение архитектур: MoEKernel vs SoftMoERouter vs FFN vs SBERT |
| `[moe_chat.py](../scripts/moe_chat.py)`                                   | Интерактивный чат с MoE domain experts                           |
| `[autoresearch_loop.py](../scripts/autoresearch_loop.py)`                 | Итеративный architecture search                                  |
| `[test_checkpoint_variants.py](../scripts/test_checkpoint_variants.py)`   | Тестирование checkpoint'ов через конфигурации                    |
| `[auto_tune.py](../scripts/auto_tune.py)`                                 | Optuna hyperparameter search                                     |
| `[interactive_kernel_chat.py](../scripts/interactive_kernel_chat.py)`     | Нейрогенеративный чат                                            |
| `[benchmark_can_integration.py](../scripts/benchmark_can_integration.py)` | FFN vs CAN expert benchmarks                                     |
| `[hdim_decoder.py](../scripts/hdim_decoder.py)`                           | HDIM-integrated decoder                                          |
| `[benchmark_sota.py](../scripts/benchmark_sota.py)`                       | SOTA comparison на STS/MTEB                                      |
| `[compare_memory_train.py](../scripts/compare_memory_train.py)`           | Сравнительное обучение: Titans vs HBMA vs CLS vs Hippocampus     |
| `[run_moe_demo.py](../scripts/run_moe_demo.py)`                           | MoEKernel demo                                                   |
| `[verify_moe_kernel_real.py](../scripts/verify_moe_kernel_real.py)`       | Полная цепочка верификации                                       |
| `[perf_profile.py](../scripts/perf_profile.py)`                           | torch.profiler профилирование                                    |
| `[gpu_memory_profile.py](../scripts/gpu_memory_profile.py)`               | GPU memory профилирование                                        |
| `[test_all_modules.py](../scripts/test_all_modules.py)`                   | Комплексный тест post-Clifford-fix                               |
| `[neural_decoder.py](../scripts/neural_decoder.py)`                       | NeuralDecoder: текст из embeddings                               |
| `[run_with_llm_experts.py](../scripts/run_with_llm_experts.py)`           | HDIM router + HuggingFace LLMs                                   |
| `[test_router.py](../scripts/test_router.py)`                             | Быстрый тест роутера                                             |
| `[test_kernel_chat.py](../scripts/test_kernel_chat.py)`                   | Программный тест kernel chat                                     |


### 6.2 Ключевые опции gpu_train.py


| Опция                   | Default | Описание                                    |
| ----------------------- | ------- | ------------------------------------------- |
| `--epochs`              | 30      | Число эпох                                  |
| `--batch_size`          | 32      | Размер батча                                |
| `--hidden_dim`          | 128     | HDIM hidden dimension                       |
| `--num_experts`         | 4       | Число MoE экспертов                         |
| `--lambda_iso`          | 0.1     | Iso loss weight                             |
| `--lambda_pair`         | 0.40    | Pair loss weight (оптимум Run 18)           |
| `--lambda_z`            | 0.0     | Router Z-loss weight (>= 0.01 рекомендуемо) |
| `--infonce_temperature` | 0.10    | Temperature для InfoNCE (оптимум Run 18)    |
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
│  │  OR HBMAMemory    │                                                   │
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
│  │ MoE Router      │  ←── slot_inputs → experts → combine                │
│  │ (SoftMoE /      │      (SoftMoERouter, MoEKernel, or MaxScoreRouter)  │
│  │  MoEKernel /    │                                                     │
│  │  MaxScore)      │                                                     │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  Routed Invariant                                                        │
│  ┌─────────────────┐                                                     │
│  │    u_route      │  ←── MoE-processed invariant                        │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────┐                                                    │
│  │ Hallucination    │  ←── optional: 5-signal risk detection             │
│  │ Detector +       │      + feedback loop rerouting                      │
│  │ FeedbackLoop     │                                                    │
│  └────────┬──────────┘                                                   │
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


| Параметр              | Default | Влияние                                  |
| --------------------- | ------- | ---------------------------------------- |
| `epochs`              | 3       | Число эпох обучения                      |
| `batch_size`          | 16      | Размер батча (>= 32 для InfoNCE)         |
| `lr`                  | 1e-3    | Learning rate                            |
| `lambda_iso`          | 0.1     | Вес isomorphism loss                     |
| `lambda_pair`         | 0.40    | Вес pair ranking loss (оптимум Run 18)   |
| `lambda_z`            | 0.0     | Вес MoE Z-loss (>= 0.01 рекомендуемо)    |
| `infonce_temperature` | 0.10    | Temperature для InfoNCE (оптимум Run 18) |
| `focal_gamma`         | 1.0     | Gamma для Focal-InfoNCE                  |
| `soft_router`         | False   | Использовать SoftMoERouter               |


### 9.3 Factory Flags


| Флаг                | Эффект                                |
| ------------------- | ------------------------------------- |
| `soft_router=True`  | Заменяет R3MoERouter на SoftMoERouter |
| `freeze_sbert=True` | Frozen SBERT + trainable projection   |


---

## 10. Stable vs Experimental

### 10.1 Production-Ready


| Компонент                   | Файл                                                                 | Статус     |
| --------------------------- | -------------------------------------------------------------------- | ---------- |
| `CliffordAlgebra`           | `[hypercomplex.py:20](../src/core/hypercomplex.py:20)`               | **Stable** |
| `DomainRotationOperator`    | `[domain_operators.py:19](../src/core/domain_operators.py:19)`       | **Stable** |
| `InvariantExtractor`        | `[domain_operators.py:54](../src/core/domain_operators.py:54)`       | **Stable** |
| `TitansMemoryModule`        | `[titans_memory.py:30](../src/core/titans_memory.py:30)`             | **Stable** |
| `SoftMoERouter`             | `[soft_moe_router.py:43](../src/core/soft_moe_router.py:43)`         | **Stable** |
| `MoEKernel`                 | `[moe_kernel.py](../src/core/moe_kernel.py)`                         | **Stable** |
| `MoEKernelConfig`           | `[moe_kernel.py](../src/core/moe_kernel.py)`                         | **Stable** |
| `MoEKernelState`            | `[moe_kernel.py](../src/core/moe_kernel.py)`                         | **Stable** |
| `MoERouter` ABC             | `[moe_interface.py](../src/core/moe_interface.py)`                   | **Stable** |
| `MoEKernelAdapter`          | `[moe_kernel_adapter.py](../src/core/moe_kernel_adapter.py)`         | **Stable** |
| `MaxScoreRouter`            | `[maxscore_router.py](../src/core/maxscore_router.py)`               | **Stable** |
| `HBMAMemory`                | `[hbma_memory.py](../src/core/hbma_memory.py)`                       | **Stable** |
| `MemoryInterface` ABC       | `[memory_interface.py](../src/core/memory_interface.py)`             | **Stable** |
| `MemoryPersistence`         | `[memory_persistence.py](../src/core/memory_persistence.py)`         | **Stable** |
| `MSASparseIndex`            | `[msa_attention.py](../src/core/msa_attention.py)`                   | **Stable** |
| `HallucinationDetector`     | `[hallucination_detector.py](../src/core/hallucination_detector.py)` | **Stable** |
| `HallucinationFeedbackLoop` | `[hallucination_feedback.py](../src/core/hallucination_feedback.py)` | **Stable** |
| `SemanticEntropyProbe`      | `[semantic_entropy_probe.py](../src/core/semantic_entropy_probe.py)` | **Stable** |
| `OnlineLearner`             | `[online_learner.py](../src/core/online_learner.py)`                 | **Stable** |
| `OnlineLoRA`                | `[online_lora.py](../src/core/online_lora.py)`                       | **Stable** |
| `ContinualNorm`             | `[continual_norm.py](../src/core/continual_norm.py)`                 | **Stable** |
| `MSAAttention`              | `[msa_attention.py](../src/core/msa_attention.py)`                   | **Stable** |
| `TransferEngine`            | `[transfer_engine.py](../src/core/transfer_engine.py)`               | **Stable** |
| `TransferState`             | `[transfer_state.py](../src/core/transfer_state.py)`                 | **Stable** |
| `DomainEncoder`             | `[domain_encoder.py](../src/core/domain_encoder.py)`                 | **Stable** |
| `InvariantProcessor`        | `[invariant_processor.py](../src/core/invariant_processor.py)`       | **Stable** |
| `HDIMEncoder/Decoder`       | `[hdim_pipeline.py:90](../src/core/hdim_pipeline.py:90)`             | **Stable** |
| `HDIMPipeline`              | `[hdim_pipeline.py:128](../src/core/hdim_pipeline.py:128)`           | **Stable** |
| `HDIMModel`                 | `[hdim_model.py:117](../src/models/hdim_model.py:117)`               | **Stable** |
| `TextHDIMModel`             | `[text_hdim_model.py:191](../src/models/text_hdim_model.py:191)`     | **Stable** |
| `SBERTEncoder`              | `[sbert_encoder.py:20](../src/models/sbert_encoder.py:20)`           | **Stable** |
| `HDIMTrainer`               | `[trainer.py:19](../src/training/trainer.py:19)`                     | **Stable** |


### 10.2 Удалено (Occam's razor)


| Компонент          | Файл (удалён)           | Причина удаления                  | Замена            |
| ------------------ | ----------------------- | --------------------------------- | ----------------- |
| `DomainExpertPool` | `domain_expert_pool.py` | Дублирование с MoEKernel          | `MoEKernel`       |
| `SharedExpert`     | `domain_expert_pool.py` | Встроен в SoftMoERouter/MoEKernel | `SoftMoERouter`   |
| `ExpertProjection` | `domain_expert_pool.py` | Не нужен с MoEKernel              | `MoEKernel`       |
| `PHMLinear`        | `hypercomplex.py`       | Не использовался                  | Удалён без замены |


---

## 11. Hallucination & Safety Subsystem

3-компонентный пайплайн для детекции и самокоррекции галлюцинаций:

```
┌──────────────────────┐     ┌──────────────────────────┐     ┌──────────────────────┐
│ HallucinationDetector│     │ HallucinationFeedbackLoop│     │ SemanticEntropyProbe │
│ (5-signal risk)      │────>│ (risk-based rerouting)   │────>│ (uncertainty quant.) │
│                      │     │                          │     │                      │
│ routing_entropy  25% │     │ FeedbackAction:          │     │ Linear probe         │
│ moe_confidence   20% │     │ NONE / ADJUST_CONF /     │     │ hidden_dim → 1       │
│ memory_mismatch  20% │     │ REROUTE / CONSOLIDATE /  │     │ sigmoid → [0, 1]     │
│ semantic_entropy 20% │     │ FULL_CORRECTION          │     │                      │
│ eigen_score      15% │     │                          │     │ Kossen ICLR 2024     │
│                      │     │ Risk thresholds:         │     │ 45-450x faster than  │
│ risk = weighted_sum  │     │ low=0.3 / med=0.5 /     │     │ full semantic entropy │
│ risk ∈ [0, 1]        │     │ high=0.7 / crit=0.85    │     │                      │
└──────────────────────┘     └──────────────────────────┘     └──────────────────────┘
```

**Интеграция в pipeline:**

`SemanticEntropyProbe` является подмодулем `HallucinationDetector`. Результат `HallucinationDetector.compute_hallucination_risk()` подаётся в `HallucinationFeedbackLoop.check_and_respond()`, который возвращает `FeedbackResult` с действием, скорректированным confidence и выбранным экспертом.

---

## 12. Online Learning Subsystem

3-компонентная подсистема для continual learning без catastrophic forgetting:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   OnlineLearner   │     │    OnlineLoRA    │     │  ContinualNorm   │
│ (global coord.)   │     │ (per-layer adapt)│     │ (streaming norm)  │
│                   │     │                  │     │                  │
│ ReplayBuffer      │     │ lora_A + lora_B  │     │ EMA running_mean │
│ (prioritized,     │     │ + importance EMA │     │ EMA running_var  │
│  surprise-based)  │     │ + EMA weights    │     │ No task reset    │
│                   │     │                  │     │                  │
│ GradientMode:     │     │ Supports:        │     │ IL-ETransformer  │
│ DETACHED (safe)   │     │ nn.Linear        │     │ style:            │
│ SELECTIVE (replay)│     │ nn.Conv2d        │     │ ContinualNorm     │
│ FULL (experimental)│     │                  │     │ ContinualNormLayer│
│                   │     │ Manager:         │     │                  │
│ Surprise detect:  │     │ batch EMA update │     │ Use in:          │
│ 1 - cos_sim(x,EMA)│     │ coordinated      │     │ streaming/online  │
│                   │     │ consolidation    │     │ scenarios        │
│ EMA target model  │     │                  │     │                  │
│ (MoCo-style)      │     │ Wei et al.       │     │ Never call        │
│                   │     │ WACV 2025        │     │ reset_running_    │
│ Titans NeurIPS    │     │                  │     │ stats() in        │
│ 2025 + MoCo       │     │                  │     │ continual setting │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

**Интеграция:** `OnlineLearner` подключается в `HDIMModel._forward_core()` после `_apply_memory()`. `OnlineLoRA` оборачивает слои модели через `wrap_with_online_lora()`. `ContinualNorm` заменяет BatchNorm/LayerNorm в streaming сценариях.

---

## Phase 30 (2026-03-26) — Bug Fixes

### MoEKernel _expert_bias → buffer

`_expert_bias` конвертирован из `nn.Parameter` в `register_buffer` — не участвует в оптимизации, корректно сохраняется в state_dict без градиентов.

### SoftMoERouter deadlock fix

В `forward()` объединён один `_ema_lock` для EMA-обновления и bias-обновления — устраняет возможный deadlock при multi-worker DataLoader.

### Session 14 (2026-03-26) — MoE + TitansMemory Smoke Test

5-epoch smoke test с MoE + TitansMemory (ep5=1.1508 ~ Run11 ep27 за 1/27 эпох):

- Нет NaN, нет OOM, GPU стабильна
- Рекомендован полный прогон 30 эпох

### ТЕКУЩИЙ ОПТИМАЛЬНЫЙ CONFIG


| Параметр              | Значение | Примечание                          |
| --------------------- | -------- | ----------------------------------- |
| `batch_size`          | 24       | Оптимум для RTX 3070 8GB            |
| `lr`                  | 5e-4     | Выше вызывает нестабильность        |
| `sbert_lr`            | 1e-5     | Осторожное размораживание SBERT     |
| `temperature`         | **0.10** | Рекорд Run 18; 0.12 даёт -0.0108    |
| `lambda_pair`         | **0.40** | Выше 0.40 не тестировалось          |
| `lambda_sts`          | **0.0**  | Любое значение > 0 подавляет margin |
| `infonce_temperature` | **0.10** | Оптимум Run 18                      |
| `patience`            | 15       | Нужно для multi-peak паттерна       |
| `epochs`              | 30       | Пик на ep13 с temp=0.10             |


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


| Паттерн                | Проблема              | Решение                          |
| ---------------------- | --------------------- | -------------------------------- |
| `ModularMoERouter`     | Нестабилен            | `SoftMoERouter`                  |
| `reset_memory('zero')` | Уничтожает знания     | `reset_memory('geometric')`      |
| `batch_size < 32`      | Мало негативов        | `batch_size >= 32`               |
| `temperature < 0.07`   | Gradient instability  | `temperature = 0.10` (оптимум)   |
| `lambda_z = 0`         | MoE collapse          | `lambda_z >= 0.01`               |
| `lambda_sts > 0`       | Подавляет pair_margin | `lambda_sts = 0.0` (обязательно) |
| `lambda_pair = 0.1`    | Недооптимизирован     | `lambda_pair = 0.40` (Run 18)    |
| Нет `reset_memory()`   | Memory drift          | epoch=1: hard, далее без сброса  |
| `DomainExpertPool`     | Удалён                | `MoEKernel` с named experts      |


---

## Ключевые ссылки

### Файлы

- `[src/core/hypercomplex.py](../src/core/hypercomplex.py)` — Алгебраическая база
- `[src/core/domain_operators.py](../src/core/domain_operators.py)` — Доменные операторы
- `[src/core/titans_memory.py](../src/core/titans_memory.py)` — Titans memory
- `[src/core/soft_moe_router.py](../src/core/soft_moe_router.py)` — Soft MoE router
- `[src/core/moe_kernel.py](../src/core/moe_kernel.py)` — MoEKernel (named domain experts)
- `[src/core/moe_interface.py](../src/core/moe_interface.py)` — MoERouter ABC
- `[src/core/moe_kernel_adapter.py](../src/core/moe_kernel_adapter.py)` — MoEKernel → MoERouter
- `[src/core/maxscore_router.py](../src/core/maxscore_router.py)` — MaxScore Router (ACL 2025)
- `[src/core/hbma_memory.py](../src/core/hbma_memory.py)` — HBMA 4-system memory
- `[src/core/memory_interface.py](../src/core/memory_interface.py)` — MemoryInterface ABC
- `[src/core/memory_persistence.py](../src/core/memory_persistence.py)` — Memory save/load
- `[src/core/msa_attention.py](../src/core/msa_attention.py)` — MSA sparse retrieval
- `[src/core/hallucination_detector.py](../src/core/hallucination_detector.py)` — 5-signal hallucination detection
- `[src/core/hallucination_feedback.py](../src/core/hallucination_feedback.py)` — Risk-based feedback loop
- `[src/core/semantic_entropy_probe.py](../src/core/semantic_entropy_probe.py)` — Uncertainty probe
- `[src/core/online_learner.py](../src/core/online_learner.py)` — Online continual learning
- `[src/core/online_lora.py](../src/core/online_lora.py)` — Online LoRA adaptation
- `[src/core/continual_norm.py](../src/core/continual_norm.py)` — Continual normalization
- `[src/core/transfer_engine.py](../src/core/transfer_engine.py)` — Transfer engine
- `[src/core/domain_encoder.py](../src/core/domain_encoder.py)` — Domain encoder
- `[src/core/invariant_processor.py](../src/core/invariant_processor.py)` — Invariant processor
- `[src/core/transfer_state.py](../src/core/transfer_state.py)` — Transfer state
- `[src/core/hdim_pipeline.py](../src/core/hdim_pipeline.py)` — Главный pipeline
- `[src/models/hdim_model.py](../src/models/hdim_model.py)` — HDIMModel, конфигурации
- `[src/models/text_hdim_model.py](../src/models/text_hdim_model.py)` — TextHDIMModel
- `[src/models/model_factory.py](../src/models/model_factory.py)` — Factory functions
- `[src/training/trainer.py](../src/training/trainer.py)` — HDIMTrainer, losses

---

*Документация сгенерирована на основе анализа исходного кода.*