# HDIM — Hypercomplex Domain Isomorphism Machine
## Implementable-first architecture specification
*Версия документа: 4.0 | Дата обновления: 2026-03-08 | Статус: MVP-oriented active development*

---

## 1. Purpose of this document
Этот документ фиксирует **реализуемую** спецификацию HDIM и намеренно отделяет:

1. **MVP-архитектуру**, уже привязанную к текущему Python-коду.
2. **Следующую инженерную фазу**, которая логично продолжает существующую реализацию.
3. **Долгосрочные research-компоненты**, которые остаются гипотезами и не должны восприниматься как уже реализованные возможности.

HDIM сохраняет научное ядро исходной идеи: представлять проблемы как гиперкомплексные структуры, извлекать доменно-инвариантное представление и переносить его между доменами через routing + memory. Но текущая система должна трактоваться как **research prototype / MVP transfer engine**, а не как завершённая машина автоматизированных научных открытий.

---

## 2. Core problem and scientific hypothesis
### 2.1 Knowledge blind spot
Современные LLM и стандартные RAG-системы обычно сопоставляют тексты по статистической близости токенов. Из-за этого они плохо переносят решения между дисциплинами, где словарь разный, а структурный конфликт одинаковый.

Пример: инженерная проблема кавитационного разрушения и стоматологическая задача удаления твёрдого налёта могут иметь похожий физический паттерн, хотя описываются разными терминами.

### 2.2 HDIM hypothesis
Рабочая гипотеза HDIM:

- проблема кодируется как гиперкомплексное представление;
- из представления снимается доменный отпечаток;
- получается приближённый структурный инвариант;
- invariant representation усиливается памятью и маршрутизацией;
- затем проецируется в целевой домен.

Важно: в текущей реализации **инвариант является инженерной аппроксимацией**, а не доказанным универсальным физическим законом. Это исследовательский объект, который оптимизируется и оценивается через proxy-метрики.

---

## 3. Implementable mathematical contract
### 3.1 Representation space
Текущий код реализует HDIM через quaternion/Clifford-inspired backbone на PyTorch.

Базовое представление:

$$
X \in Cl_{p,q,r}(\mathbb{R})
$$

На практике это означает, что внутренние состояния живут в `clifford_dim`, а часть слоёв использует quaternion-aware операции.

### 3.2 Domain abstraction
Для домена $A$ и его обучаемого оператора $R_A$ инвариант задаётся как:

$$
U_{inv} = R_A^{-1} \otimes G_A \otimes R_A
$$

В коде это соответствует `InvariantExtractor` и `DomainRotationOperator`.

### 3.3 Domain projection
Проекция инварианта в целевой домен $B$:

$$
G_B = R_B \otimes U \otimes R_B^{-1}
$$

Где $U$ в MVP — это не обязательно «чистый» инвариант, а чаще **processed invariant** после memory + router stages.

### 3.4 Routing stabilization approximation
Идея R3 сохраняется как приближение top-k replay-stabilized routing:

$$
g_{\mathrm{replay}, i} = \frac{I_{\mathrm{infer}, i} \exp(s_{\mathrm{train}, i})}{\sum_j I_{\mathrm{infer}, j} \exp(s_{\mathrm{train}, j})}
$$

Но в текущем MVP это следует понимать как **R3-inspired router stabilization**, а не как полную формальную реализацию rollout replay из отдельной RL-схемы.

### 3.5 Memory update approximation
Память следует Titans-подобной идее online adaptation:

$$
\mathcal{M}_t = (1 - \alpha_t)\mathcal{M}_{t-1} + S_t
$$

Однако текущий runtime-контракт — это прежде всего **stateful associative memory for invariant retrieval/update**, а не полноценная внешняя научная память со статьями, патентами и доказательной верификацией.

### 3.6 Isomorphism loss used in implementation
Целевая формула HDIM остаётся такой:

$$
\mathcal{L}_{\mathrm{iso}} = \sum_{(G_A, G_B) \in \mathcal{S}} \lVert R_A^{-1} G_A R_A - R_B^{-1} G_B R_B \rVert^2
$$

Для текущего MVP она реализуется в **попарном приближении** через paired dataset и MSE между инвариантными представлениями source/target пар.

Итоговый тренировочный контракт в MVP:

$$
L_{\mathrm{total}} = L_{\mathrm{recon}} + \lambda_{\mathrm{iso}} L_{\mathrm{iso}} + \lambda_{\mathrm{routing}} L_{\mathrm{routing}} + L_{\mathrm{memory}}
$$

Это и есть основной implementable objective, на который нужно опираться при развитии проекта.

---

## 4. MVP architecture
## 4.1 What is part of the MVP
MVP HDIM — это **pair-supervised, invariant-centered transfer pipeline** с five-stage path:

1. `encode source`
2. `extract invariant`
3. `augment with memory`
4. `route invariant`
5. `project to target and decode`

### 4.2 Canonical data flow
```text
x
→ encoder
→ G_source
→ invariant extractor
→ U_inv
→ Titans memory
→ U_mem
→ R3-inspired MoE router
→ U_route
→ domain projection
→ G_target
→ decoder
→ output
```

### 4.3 Current code mapping
Текущая MVP-архитектура уже опирается на существующие модули:

- `hypercoplexAI/src/core/hypercomplex.py`
 - `QuaternionLinear`
 - `QLayerNorm`
 - `CliffordAlgebra`
 - `PHMLinear`
- `hypercoplexAI/src/core/domain_operators.py`
 - `DomainRotationOperator`
 - `InvariantExtractor`
 - `sandwich_transfer`
- `hypercoplexAI/src/core/titans_memory.py`
 - `TitansMemoryModule`
- `hypercoplexAI/src/core/moe_router.py`
 - `R3MoERouter`
- `hypercoplexAI/src/core/hdim_pipeline.py`
 - canonical transfer pipeline and transfer state
- `hypercoplexAI/src/models/hdim_model.py`
 - integer-indexed domain interface
 - same-domain forward path
 - explicit paired transfer path
- `hypercoplexAI/src/training/dataset.py`
 - baseline dataset
 - paired demo dataset
- `hypercoplexAI/src/training/trainer.py`
 - reconstruction + iso + routing + memory loss integration
- `hypercoplexAI/src/models/metrics.py`
 - STS / DRS / AFR evaluation
- `hypercoplexAI/tests/test_hdim.py`
 - baseline and paired-contract tests
### 4.4 What the MVP can honestly claim
Текущая архитектура уже поддерживает:

- quaternion/Clifford-inspired representation path;
- обучаемые доменные операторы;
- извлечение invariant-like latent representation;
- stateful memory augmentation;
- R3-inspired MoE routing;
- explicit paired transfer training path;
- базовые метрики STS/DRS/AFR;
- синтетический paired dataset для обучения и валидации.

### 4.5 What the MVP must NOT claim
MVP не должен описываться как уже имеющий:

- доказанно универсальный физический инвариант;
- production-grade multi-domain scientific discovery engine;
- полноценный Scholar / PubMed / Patents memory ingestion pipeline;
- validated human-in-the-loop scientific review workflow;
- завершённый Q-Attention graph transformer;
- полноценную автоматическую интерпретацию ТРИЗ на runtime.

---

## 5. Realized components vs future components
## 5.1 Implemented or directly implementable now
### A. Hypercomplex backbone
Реализовано:
- quaternion-aware linear projection;
- Clifford algebra utilities;
- invariant extraction via sandwich-style transform.

### B. Domain transport layer
Реализовано:
- `DomainRotationOperator` для обучаемых доменных роторов;
- `InvariantExtractor`;
- `sandwich_transfer`.

### C. Memory layer
Реализовано:
- `TitansMemoryModule` как online-updated associative memory;
- memory retrieval/update inside pipeline;
- memory loss integration in trainer.

### D. Routing layer
Реализовано:
- `R3MoERouter`;
- top-k gating;
- router auxiliary loss;
- routing observability through transfer state.

### E. Training/evaluation scaffold
Реализовано:
- `DomainProblemDataset`;
- paired supervision via `pair_encoding` / `pair_domain_id`;
- `HDIMTrainer`;
- tests forward, paired transfer, validation, metrics.

## 5.2 Next-phase engineering components
Следующая фаза должна улучшать текущий код, а не подменять его новой абстрактной архитектурой.

Нужны:

1. более строгий paired dataset contract;
2. более явный lifecycle memory reset / retrieve / update;
3. алгебраические contract tests;
4. более прозрачный router state и debug surface;
5. alignment между training contract и evaluation contract;
6. прояснение роли `processed invariant` как основного объекта переноса.

## 5.3 Research / long-term components
Следующие элементы остаются исследовательскими и не входят в MVP-contract:

- Q-Attention как полноценный graph attention layer;
- TRIZ DSL как стандартный parser/executor для problem abstraction;
- avatars / human-in-the-loop multi-stage analyst workflow;
- external Scholar / Patent / PubMed ingestion;
- Bayesian quaternion operators;
- octonion or full generalized Clifford scaling;
- domain manifold clustering with large scientific taxonomies;
- external truth-validation layer for generated analogies.

---

## 6. Training contract
## 6.1 Baseline mode
Same-domain reconstruction mode:

- вход: `encoding`, `domain_id`
- путь: same-domain transfer / reconstruction
- цель: стабилизировать representation, memory, routing, decode path
## 6.2 Paired isomorphism mode
Cross-domain paired mode:

- вход: `encoding`, `domain_id`, `pair_encoding`, `pair_domain_id`
- путь: explicit transfer from source domain to paired target domain
- цель: приближать invariants между функционально связанными примерами
## 6.3 Loss composition
В implementable версии HDIM тренируется не на «чистой философской изоморфности», а на комбинации наблюдаемых loss-компонент:

- `L_recon` — reconstruction / paired reconstruction target
- `L_iso` — MSE между paired invariant representations
- `L_routing` — router auxiliary term
- `L_memory` — memory adaptation term
Это ключевой engineering contract текущей системы.

---

## 7. Evaluation contract
## 7.1 STS — Structural Transfer Score
В текущем проекте STS означает косинусное сходство между source и paired invariant representations.

Практически это **proxy for structure preservation**, а не финальное доказательство физической эквивалентности.

## 7.2 DRS — Domain Routing Stability
DRS измеряет стабильность router weights при повторных вызовах.

Это инженерная метрика устойчивости маршрутизации, а не формальная оценка корректности научного вывода.

## 7.3 AFR — Analogy Feasibility Rate
AFR в MVP — это threshold-based feasibility proxy, завязанный на STS/paired transfer signals.

Следовательно, AFR пока следует трактовать как **internal consistency metric**, а не как верификацию аналогии законами физики.

---

## 8. Optional conceptual modules kept as approximations
## 8.1 Q-Attention
Целевая формула сохраняется как conceptual extension:

$$
\alpha_{ij} = \mathrm{softmax}\left(\frac{\mathrm{Re}(q_i \otimes \bar{q}_j)}{\sqrt{d}}\right)
$$

Но в текущем HDIM это **не обязательный runtime-компонент MVP**. Его нужно трактовать как кандидат для следующей research-phase, а не как текущую основу пайплайна.

## 8.2 TRIZ DSL
ТРИЗ-формат полезен как интерфейс abstraction layer и может быть зафиксирован как внешний preprocessing contract:

```text
TRIZ primitive:
 Subject:   [entity under pressure]
 Object:    [entity causing conflict]
 Conflict: [need X but constrained by Y]
 Scale:     [nano / meso / macro]
 Form:      [accumulation / fracture / blockage / leakage]
```

Но в текущем коде это **ещё не реализованный parser/runtime layer**.

## 8.3 Human-in-the-loop / avatars
Схема режимов полезна как продуктовая надстройка:

| Mode | Interpretation |
|---|---|
| Auto | полностью автоматический MVP pipeline |
| Semi-auto | пользователь выбирает из candidate analogies |
| Manual | внешний исследовательский поиск и ручная фильтрация |

Но пока это следует понимать как **future interaction design**, а не как уже существующую часть runtime API.

## 8.4 Scholar API integration
External scientific retrieval остаётся долгосрочной интеграцией. Текущий `TitansMemoryModule` — это internal learnable memory, а не подключённая база Google Scholar / Semantic Scholar / PubMed.

## 8.5 MVP execution contract and implementation matrix
### 8.5.1 Minimal execution contract
Чтобы документ оставался синхронизированным с текущим кодом, MVP должен читаться как следующий исполнимый контракт:

1. входом служит уже готовый `encoding`, а не raw scientific corpus;
2. `HDIMModel` выполняет either same-domain reconstruction, либо explicit paired transfer;
3. `HDIMTrainer` оптимизирует наблюдаемые objective-компоненты `loss_recon`, `loss_iso`, `loss_routing`, `loss_memory`;
4. качество оценивается через `validate()` и `compute_all_metrics()`;
5. корректность MVP подтверждается тестами на forward / transfer / paired dataset / metrics.

### 8.5.2 Training loop tied to current code
Реалистичный цикл обучения для текущего MVP выглядит так:

```text
create_demo_dataset() or create_paired_demo_dataset()
→ DataLoader
→ HDIMModel.forward() or HDIMModel.transfer_pairs()
→ HDIMTrainer._compute_batch_losses()
→ loss_total = loss_recon + λ_iso·loss_iso + λ_routing·loss_routing + loss_memory
→ optimizer.step()
→ trainer.validate(...)
→ compute_all_metrics(...)
```

То есть основной путь проекта уже определён не whitepaper-обещаниями, а конкретным training contract в `src/training/trainer.py` и data contract в `src/training/dataset.py`.

### 8.5.3 Implementation matrix
| Component | MVP status | Code anchor | Practical note |
|---|---|---|---|
| Domain rotor / rotation operators | implemented | `hypercoplexAI/src/core/domain_operators.py` | основной перенос между доменами уже есть |
| Invariant extraction | implemented | `hypercoplexAI/src/core/domain_operators.py` | используется как базовый объект переноса |
| Sandwich transfer | implemented | `hypercoplexAI/src/core/domain_operators.py` | соответствует канонической формуле переноса |
| Memory augmentation | implemented approximation | `hypercoplexAI/src/core/titans_memory.py` | компактная differentiable memory, не external knowledge base |
| R3-style router | implemented approximation | `hypercoplexAI/src/core/moe_router.py` | top-k routing с auxiliary loss, не full RL replay stack |
| Batch-facing model API | implemented | `hypercoplexAI/src/models/hdim_model.py` | поддерживает same-domain и paired transfer path |
| Pair-supervised dataset | implemented | `hypercoplexAI/src/training/dataset.py` | synthetic paired supervision уже задаёт MVP data contract |
| Training loop | implemented | `hypercoplexAI/src/training/trainer.py` | total loss уже согласован с MVP objective |
| STS / DRS / AFR metrics | implemented | `hypercoplexAI/src/models/metrics.py` | это proxy-метрики, не scientific proof layer |
| Regression tests | implemented | `hypercoplexAI/tests/test_hdim.py` | покрывают forward, pairs, validate, metrics |
| Q-Attention | research only | not present in runtime code | оставить как post-MVP extension |
| TRIZ parser / avatars / Scholar ingestion | research only | not present in runtime code | внешний workflow, не ядро Python MVP |

### 8.5.4 MVP acceptance criteria
Документ следует считать честно выполненным относительно текущей архитектуры, если одновременно верны четыре условия:

- paired dataset может быть создан без нарушения cross-domain contract;
- `HDIMTrainer.train_step()` работает как для обычного, так и для paired batch;
- `validate()` возвращает агрегированные loss-метрики;
- `compute_all_metrics()` возвращает `STS`, `DRS`, `AFR` на том же model API.

Именно эти критерии делают HDIM implementable-first системой уже сейчас.

---

## 9. Roadmap
## 9.1 Phase 1 — MVP baseline
Цель: зафиксировать работающий, тестируемый transfer engine.

Входит:
- hypercomplex backbone;
- domain operators;
- memory augmentation;
- R3-inspired router;
- paired dataset;
- trainer;
- STS/DRS/AFR proxies;
- tests.

Статус: **частично реализовано и уже привязано к текущему коду**.

## 9.2 Phase 2 — next implementation phase
Цель: усилить реализуемость и экспериментальную достоверность.

Приоритеты:
1. улучшить paired supervision;
2. формализовать memory lifecycle;
3. расширить algebraic and transfer contract tests;
4. укрепить router observability;
5. выровнять training/eval semantics;
6. канонизировать invariant-state naming и API.

## 9.3 Phase 3 — long-term research
Цель: перейти от MVP transfer engine к исследовательской платформе.

Может включать:
- Q-Attention graph layers;
- TRIZ parser;
- avatars / human review loop;
- external scientific corpus ingestion;
- richer domain taxonomies;
- uncertainty-aware hypercomplex operators;
- benchmark against real cross-domain discovery tasks.

---

## 10. Engineering guidance for future edits
При дальнейшем редактировании архитектуры нужно соблюдать правило:

**любое сильное научное утверждение должно быть помечено как одно из трёх:**

- `implemented`
- `engineering next`
- `research hypothesis`

Это необходимо, чтобы HDIM.md больше не создавал ложного впечатления, будто весь vision-stack уже доступен в коде.

---

## 11. Practical status summary
### 11.1 What exists today
Сегодня HDIM — это:

- реализуемый PyTorch prototype;
- invariant-centered transfer architecture;
- pair-aware training scaffold;
- memory + routing + domain transport stack;
- система с уже существующими тестами и минимальным synthetic data loop.

### 11.2 What it is not yet
Сегодня HDIM — это ещё не:

- научный discovery engine с подключённой мировой литературой;
- доказательный cross-domain invention system;
- полноформатный graph reasoning platform;
- finished human-in-the-loop research workstation.

### 11.3 Honest one-line description
HDIM в текущем состоянии — это **MVP hypercomplex transfer prototype for learning and evaluating domain-invariant approximations under memory-augmented routing**, а не завершённая система автоматизации научных открытий.

---

## 12. File-level implementation index
Для навигации по текущей implementable architecture:

- `hypercoplexAI/HDIM.md` — this specification
- `hypercoplexAI/src/core/hypercomplex.py` — algebra and hypercomplex layers
- `hypercoplexAI/src/core/domain_operators.py` — domain transport primitives
- `hypercoplexAI/src/core/titans_memory.py` — memory module
- `hypercoplexAI/src/core/moe_router.py` — routing module
- `hypercoplexAI/src/core/hdim_pipeline.py` — orchestrated transfer pipeline
- `hypercoplexAI/src/models/hdim_model.py` — batch-facing model API
- `hypercoplexAI/src/models/metrics.py` — evaluation metrics
- `hypercoplexAI/src/training/dataset.py` — demo and paired datasets
- `hypercoplexAI/src/training/trainer.py` — training loop
- `hypercoplexAI/tests/test_hdim.py` — coverage for baseline and paired behavior
---

## 13. Final position
HDIM следует развивать как **implementable-first research system**:

- сначала выравнивать код, лоссы, метрики и contracts;
- затем усиливать экспериментальную строгость;
- и только потом добавлять большие research-надстройки вроде Scholar integration, human-in-the-loop orchestration и Q-Attention graph stack.

Это сохраняет научное ядро проекта и одновременно делает архитектурную документацию честной по отношению к текущему состоянию реализации.
