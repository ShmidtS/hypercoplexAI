# HDIM — Hypercomplex Domain Isomorphism Machine
## Implementable-first architecture specification
*Версия документа: 4.1 | Дата обновления: 2026-03-09 | Статус: MVP-oriented active development*

---

## 1. Purpose of this document
Этот документ фиксирует **реализуемую** спецификацию HDIM и намеренно отделяет:

1. **MVP-архитектуру**, уже привязанную к текущему Python-коду.
2. **Следующую инженерную фазу**, которая логично продолжает существующую реализацию.
3. **Долгосрочные research-компоненты**, которые остаются гипотезами и не должны восприниматься как уже реализованные возможности.

HDIM сохраняет научное ядро исходной идеи: представлять проблемы как гиперкомплексные структуры, извлекать доменно-инвариантное представление и переносить его между доменами через routing + memory. Текущий MVP уже допускает как embedding-first вход, так и minimal text-facing entry layer через `TextHDIMModel`, но эта текстовая поверхность остаётся thin wrapper над существующим HDIM core. Поэтому систему всё ещё нужно трактовать как **research prototype / MVP transfer engine**, а не как завершённую машину автоматизированных научных открытий или generative language model.

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

Где $U$ в MVP — это canonical transfer object после memory + router stages, то есть `exported_invariant`, а не абстрактный «processed invariant».

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
MVP HDIM — это **pair-supervised, invariant-centered transfer pipeline** с canonical transfer lifecycle:

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
- `hypercoplexAI/src/models/text_hdim_model.py`
 - thin text-entry wrapper around the HDIM core
 - `forward_texts(...)`, `transfer_texts(...)`, `transfer_text_pairs(...)`
 - structured retrieval/ranking artifact via `TextPairScoreResult`
- `hypercoplexAI/src/training/dataset.py`
 - baseline dataset
 - paired demo dataset
 - deterministic `texts_to_tensor(...)` scaffold for minimal text-mode experiments
- `hypercoplexAI/src/training/trainer.py`
 - reconstruction + iso + routing + memory loss integration
- `hypercoplexAI/src/training/train.py`
 - CLI training loop
 - optional machine-readable `results_json` run summary for orchestration/reporting flows
- `hypercoplexAI/src/models/metrics.py`
 - STS_exported / STS_training / DRS / AFR / pair_margin evaluation
- `hypercoplexAI/tests/test_hdim.py`
 - baseline and paired-contract tests
 - text-wrapper scoring coverage with retrieve-only memory semantics
### 4.4 What the MVP can honestly claim
Текущая архитектура уже поддерживает:

- quaternion/Clifford-inspired representation path;
- обучаемые доменные операторы;
- извлечение invariant-like latent representation;
- stateful memory augmentation;
- R3-inspired MoE routing;
- explicit paired transfer training path;
- minimal text-facing retrieval/ranking/transfer wrapper over the same HDIM core;
- structured text-pair scoring artifact with source/target runtime states for inspection/reporting;
- metric surface `STS_exported`, `STS_training`, `DRS`, `AFR`, `pair_margin`;
- синтетический paired dataset с `pair_relation_type`, `pair_family_id`, `pair_weight` и group-aware split для обучения и валидации.
- machine-readable run summary JSON from the training CLI for orchestration/reporting workflows.

### 4.5 What the MVP must NOT claim
MVP не должен описываться как уже имеющий:

- доказанно универсальный физический инвариант;
- production-grade multi-domain scientific discovery engine;
- полноценный Scholar / PubMed / Patents memory ingestion pipeline;
- validated human-in-the-loop scientific review workflow;
- завершённый Q-Attention graph transformer;
- полноценную автоматическую интерпретацию ТРИЗ на runtime;
- generative language model или pretrained semantic encoder;
- externally verified autoresearch runtime с доказательной литературной валидацией.

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
- pair metadata via `pair_group_id`, `pair_family_id`, `pair_relation_type`, `pair_relation_label`, `pair_weight`;
- `create_group_aware_split(...)` для leakage-aware group/family validation semantics без пересечения pair groups между train и validation.
- `HDIMTrainer`;
- CLI training loop with optional machine-readable run summary JSON;
- tests forward, paired transfer, validation, metrics.

### F. Minimal text-facing wrapper
Реализовано:
- `TextHDIMModel` как thin wrapper поверх `HDIMModel`;
- deterministic text-to-embedding entry via `texts_to_tensor(...)`;
- `forward_texts(...)`, `transfer_texts(...)`, `transfer_text_pairs(...)`;
- `score_text_pairs_with_state(...)` как retrieval/ranking surface, возвращающий scores и source/target runtime states;
- retrieve-only scoring contract (`memory_mode="retrieve"`, `update_memory=False`) для text-pair comparison без мутации памяти во время scoring.

Важно: этот слой предназначен для retrieval/ranking/transfer experiments и orchestration-friendly inspection, но не превращает HDIM в генеративную текстовую модель.

## 5.2 Next-phase engineering components
Следующая фаза должна улучшать текущий код, а не подменять его новой абстрактной архитектурой.

Нужны:

1. более строгий paired dataset contract;
2. более явный lifecycle memory reset / retrieve / update;
3. алгебраические contract tests;
4. более прозрачный router state и debug surface;
5. alignment между training contract и evaluation contract;
6. канонизировать `exported_invariant` / `training_invariant` naming surface вместо старого umbrella-термина `processed invariant`.

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

## 6. Canonical invariant and memory lifecycle
### 6.1 Frozen invariant naming
Текущий код фиксирует четыре разные стадии invariant lifecycle:

- `raw_invariant` — результат `InvariantExtractor` до памяти;
- `memory_augmented_invariant` — invariant после `TitansMemoryModule`;
- `exported_invariant` — canonical transfer object после router;
- `training_invariant` — projection `exported_invariant` в `hidden_dim` для `L_iso` и training-facing diagnostics.

Это означает, что прежний термин `processed invariant` больше не должен использоваться как универсальное имя для всех стадий сразу. В инженерном контракте MVP:

- основной объект переноса = `exported_invariant`;
- основной объект оптимизации = `training_invariant`.

### 6.2 Memory protocol
Memory lifecycle в MVP должен трактоваться как явный runtime contract с режимами:

- `memory_mode="update"` — train-path, retrieval + update памяти;
- `memory_mode="retrieve"` — eval/validation path без мутации памяти;
- `memory_mode="none"` — stateless ablation path без memory augmentation.

Флаг `update_memory` имеет смысл только вместе с `memory_mode="update"`; в остальных режимах мутация памяти отключается принудительно.

Минимальная инженерная гарантия текущего кода: repeated eval в `retrieve` режиме не должен менять memory state и router replay state.

### 6.3 Router observability contract
Публичный runtime state теперь должен восприниматься как contract, а не debug payload. Минимальный набор полей router observability:

- `routing_weights`
- `topk_idx`
- `topk_gate_weights`
- `train_scores_snapshot`
- `expert_usage`
- `routing_entropy`
- `router_loss`

Именно эти поля используются в trainer/tests и должны сохранять согласованную семантику между `forward`, `transfer` и `transfer_pairs`.

---

## 7. Evaluation contract
## 7.1 STS_exported — Structural Transfer Score in transfer space
В текущем проекте `STS_exported` означает косинусное сходство между aligned source/target `exported_invariant`.

Практически это **proxy for structure preservation in canonical transfer space**, а не финальное доказательство физической эквивалентности.

## 7.2 STS_training — Structural Transfer Score in optimization space
`STS_training` измеряет косинусное сходство между aligned source/target `training_invariant`.

Эта метрика нужна, чтобы evaluation surface не смешивал runtime transfer object и optimization projection.

## 7.3 DRS — Domain Routing Stability
DRS измеряет стабильность router weights при повторных вызовах.

Это инженерная метрика устойчивости маршрутизации, а не формальная оценка корректности научного вывода.

## 7.4 AFR — Analogy Feasibility Rate
AFR в MVP теперь трактуется как **margin-aware feasibility proxy**: aligned pair считается успешной, если одновременно выполняются similarity threshold и pair-margin threshold.

Следовательно, AFR остаётся **internal consistency metric**, но уже учитывает отличие aligned pair от mismatched negative pair.

## 7.5 Pair margin
`pair_margin` — это средний зазор между similarity aligned pairs и similarity mismatched pairs в пространстве `exported_invariant`.

Эта метрика нужна как минимальный honesty check, что paired evaluation различает корректное и некорректное сопоставление.

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

1. входом служит либо уже готовый `encoding`, либо raw text, который детерминированно переводится в embedding через minimal text-entry scaffold;
2. `HDIMModel` выполняет either same-domain reconstruction, либо explicit paired transfer;
3. `TextHDIMModel` добавляет thin text-facing entry layer для retrieval / ranking / transfer experiments поверх того же HDIM core;
4. text-pair scoring выполняется через cosine similarity в пространстве `exported_invariant` и может возвращать structured artifact со score и обеими runtime states;
5. `HDIMTrainer` оптимизирует наблюдаемые objective-компоненты `loss_recon`, `loss_iso`, `loss_routing`, `loss_memory`;
6. CLI training path может дополнительно записывать machine-readable run summary JSON для orchestration/reporting flows;
7. качество оценивается через `validate()` и `compute_all_metrics()`;
8. корректность MVP подтверждается тестами на forward / transfer / paired dataset / metrics и minimal text-wrapper surface.

### 8.5.2 Training loop tied to current code
Реалистичный цикл обучения для текущего MVP выглядит так:

```text
create_demo_dataset() or create_paired_demo_dataset()
→ create_group_aware_split(...)
→ DataLoader
→ HDIMModel.forward() or HDIMModel.transfer_pairs()
→ HDIMTrainer._compute_batch_losses()
→ loss_total = loss_recon + λ_iso·loss_iso + λ_routing·loss_routing + loss_memory
→ optimizer.step()
→ trainer.validate(...)
→ compute_all_metrics(...)
→ optional results_json writeout with config / validation / quality / checkpoint summary
```

Text-facing retrieval/ranking flow поверх того же core выглядит отдельно:

```text
raw texts
→ texts_to_tensor(...)
→ TextHDIMModel.forward_texts(...) or transfer_text_pairs(...)
→ exported_invariant
→ cosine similarity scoring
→ TextPairScoreResult for inspection/reporting
```

То есть основной путь проекта уже определён не whitepaper-обещаниями, а конкретным training contract в `src/training/trainer.py`, text-entry scaffold в `src/models/text_hdim_model.py` и data contract в `src/training/dataset.py`.

### 8.5.3 Implementation matrix
| Component | MVP status | Code anchor | Practical note |
|---|---|---|---|
| Domain rotor / rotation operators | implemented | `hypercoplexAI/src/core/domain_operators.py` | основной перенос между доменами уже есть |
| Invariant extraction | implemented | `hypercoplexAI/src/core/domain_operators.py` | используется как базовый объект переноса |
| Sandwich transfer | implemented | `hypercoplexAI/src/core/domain_operators.py` | соответствует канонической формуле переноса |
| Memory augmentation | implemented approximation | `hypercoplexAI/src/core/titans_memory.py` | компактная differentiable memory, не external knowledge base |
| R3-style router | implemented approximation | `hypercoplexAI/src/core/moe_router.py` | top-k routing с auxiliary loss, не full RL replay stack |
| Batch-facing model API | implemented | `hypercoplexAI/src/models/hdim_model.py` | поддерживает same-domain и paired transfer path |
| Text-facing model API | implemented | `hypercoplexAI/src/models/text_hdim_model.py` | thin wrapper для text entry, transfer и retrieval/ranking |
| Pair-supervised dataset | implemented | `hypercoplexAI/src/training/dataset.py` | synthetic paired supervision и deterministic text scaffold уже задают MVP data contract |
| Training loop | implemented | `hypercoplexAI/src/training/trainer.py` | total loss уже согласован с MVP objective |
| Training CLI run summary | implemented | `hypercoplexAI/src/training/train.py` | optional JSON summary для orchestration/reporting, не external evidence engine |
| STS / DRS / AFR metrics | implemented | `hypercoplexAI/src/models/metrics.py` | это proxy-метрики, не scientific proof layer |
| Regression tests | implemented | `hypercoplexAI/tests/test_hdim.py` | покрывают forward, pairs, validate, metrics и text-wrapper scoring contract |
| Q-Attention | research only | not present in runtime code | оставить как post-MVP extension |
| TRIZ parser / avatars / Scholar ingestion | research only | not present in runtime code | внешний workflow, не ядро Python MVP |

### 8.5.4 MVP acceptance criteria
Документ следует считать честно выполненным относительно текущей архитектуры, если одновременно верны шесть условий:

- paired dataset может быть создан без нарушения positive/negative cross-domain contract;
- train/validation split может быть построен через `create_group_aware_split(...)` без пересечения pair groups между split'ами;
- `HDIMTrainer.train_step()` работает как для обычного, так и для paired batch;
- `validate()` возвращает агрегированные loss-метрики;
- `compute_all_metrics()` возвращает `STS_exported`, `STS_training`, `DRS`, `AFR`, `pair_margin` на том же model API;
- text-wrapper scoring path возвращает retrieve-safe structured result без мутации памяти во время comparison.

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
- STS_exported / STS_training / DRS / AFR / pair_margin proxies;
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
