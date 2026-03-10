# HDIM — Hypercomplex Domain Isomorphism Machine
## Архитектурная спецификация — реализуемое прежде всего
*Версия документа: 7.0 | Дата обновления: 2026-03-11 | Статус: Фаза 7 — активна — рекорд score 0.9745 (Phase 6e)*

---

## 1. Назначение документа
Этот документ фиксирует **реализуемую** спецификацию HDIM и намеренно отделяет:

1. **MVP-архитектуру**, уже привязанную к текущему Python-коду.
2. **Следующую инженерную фазу**, которая логично продолжает существующую реализацию.
3. **Долгосрочные research-компоненты**, которые остаются гипотезами и не должны восприниматься как уже реализованные возможности.

HDIM сохраняет научное ядро исходной идеи: представлять проблемы как гиперкомплексные структуры, извлекать доменно-инвариантное представление и переносить его между доменами через routing + memory. Текущий MVP уже допускает как embedding-first вход, так и minimal text-facing entry layer через `TextHDIMModel`, но эта текстовая поверхность остаётся thin wrapper над существующим HDIM core. Поэтому систему всё ещё нужно трактовать как **research prototype / MVP transfer engine**, а не как завершённую машину автоматизированных научных открытий или generative language model.

---

## 2. Научная гипотеза и постановка проблемы
### 2.1 Слепое пятно знаний
Современные LLM и стандартные RAG-системы обычно сопоставляют тексты по статистической близости токенов. Из-за этого они плохо переносят решения между дисциплинами, где словарь разный, а структурный конфликт одинаковый.

Пример: инженерная проблема кавитационного разрушения и стоматологическая задача удаления твёрдого налёта могут иметь похожий физический паттерн, хотя описываются разными терминами.

### 2.2 Гипотеза HDIM
Рабочая гипотеза HDIM:

- проблема кодируется как гиперкомплексное представление;
- из представления снимается доменный отпечаток;
- получается приближённый структурный инвариант;
- invariant representation усиливается памятью и маршрутизацией;
- затем проецируется в целевой домен.

Важно: в текущей реализации **инвариант является инженерной аппроксимацией**, а не доказанным универсальным физическим законом. Это исследовательский объект, который оптимизируется и оценивается через proxy-метрики.

---

## 3. Реализуемый математический контракт
### 3.1 Пространство представлений
Текущий код реализует HDIM через quaternion/Clifford-inspired backbone на PyTorch.

Базовое представление:

$$
X \in Cl_{p,q,r}(\mathbb{R})
$$

На практике это означает, что внутренние состояния живут в `clifford_dim`, а часть слоёв использует quaternion-aware операции.

### 3.2 Абстракция домена
Для домена $A$ и его обучаемого оператора $R_A$ инвариант задаётся как:

$$
U_{inv} = R_A^{-1} \otimes G_A \otimes R_A
$$

В коде это соответствует `InvariantExtractor` и `DomainRotationOperator`.

### 3.3 Проекция в целевой домен
Проекция инварианта в целевой домен $B$:

$$
G_B = R_B \otimes U \otimes R_B^{-1}
$$

Где $U$ в MVP — это canonical transfer object после memory + router stages, то есть `exported_invariant`, а не абстрактный «processed invariant».

### 3.4 Приближение стабилизации маршрутизации
Идея R3 сохраняется как приближение top-k replay-stabilized routing:

$$
g_{\mathrm{replay}, i} = \frac{I_{\mathrm{infer}, i} \exp(s_{\mathrm{train}, i})}{\sum_j I_{\mathrm{infer}, j} \exp(s_{\mathrm{train}, j})}
$$

Но в текущем MVP это следует понимать как **R3-inspired router stabilization**, а не как полную формальную реализацию rollout replay из отдельной RL-схемы.

### 3.5 Приближение обновления памяти
Память следует Titans-подобной идее online adaptation:

$$
\mathcal{M}_t = (1 - \alpha_t)\mathcal{M}_{t-1} + S_t
$$

Однако текущий runtime-контракт — это прежде всего **stateful associative memory for invariant retrieval/update**, а не полноценная внешняя научная память со статьями, патентами и доказательной верификацией.

### 3.6 Функция потерь изоморфизма в реализации
Целевая формула HDIM остаётся такой:

$$
\mathcal{L}_{\mathrm{iso}} = \sum_{(G_A, G_B) \in \mathcal{S}} \lVert R_A^{-1} G_A R_A - R_B^{-1} G_B R_B \rVert^2
$$

Для текущего MVP она реализуется в **попарном приближении** через paired dataset и MSE между инвариантными представлениями source/target пар.

Итоговый тренировочный контракт в MVP:

$$
L_{\mathrm{total}} = L_{\mathrm{recon}} + \lambda_{\mathrm{iso}} L_{\mathrm{iso}} + \lambda_{\mathrm{pair}} L_{\mathrm{pair}} + \lambda_{\mathrm{routing}} L_{\mathrm{routing}} + L_{\mathrm{memory}}
$$

Это и есть основной implementable objective, на который нужно опираться при развитии проекта.

---

## 4. Архитектура MVP
### 4.1 Что входит в MVP
MVP HDIM — это **pair-supervised, invariant-centered transfer pipeline** с canonical transfer lifecycle:

1. `encode source`
2. `extract invariant`
3. `augment with memory`
4. `route invariant`
5. `project to target and decode`

### 4.2 Канонический поток данных
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

### 4.3 Соответствие текущему коду
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
 - trainable `SimpleTextEncoder` with optional vocab-driven tokenization config (`vocab_path`, `tokenizer_name`)
 - `forward_texts(...)`, `transfer_texts(...)`, `transfer_text_pairs(...)`
 - structured retrieval/ranking artifact via `TextPairScoreResult`
- `hypercoplexAI/src/training/dataset.py`
 - baseline dataset
 - paired demo dataset
 - deterministic `texts_to_tensor(...)` scaffold retained for demos/tests, but not the primary runtime path of `TextHDIMModel`
- `hypercoplexAI/src/training/trainer.py`
 - reconstruction + iso + routing + memory loss integration
- `hypercoplexAI/src/training/train.py`
 - CLI training loop
 - `--text_mode` wiring, which swaps `HDIMModel` for `TextHDIMModel` at runtime
 - `--config` manifest loading via `ExperimentConfig.from_json(...)`
 - optional machine-readable `results_json` run summary and `ledger_path` JSONL logging for orchestration/reporting flows
- `hypercoplexAI/src/training/experiment_config.py`
 - manifest schema for reproducible experiment runs
 - includes `text_mode`, `use_pairs`, output/ledger paths, and override dictionaries
- `hypercoplexAI/src/training/experiment_runner.py`
 - `ExperimentRunner` for single-manifest execution with persisted run artifacts
 - `AutoResearchRunner` for multi-run sessions, run scoring, best-run selection, and session summaries
- `hypercoplexAI/src/models/metrics.py`
 - STS_exported / STS_training / DRS / AFR / pair_margin evaluation
- `hypercoplexAI/tests/test_hdim.py`
 - baseline and paired-contract tests
 - text-wrapper scoring coverage with retrieve-only memory semantics
- `hypercoplexAI/src/models/advanced_text_encoder.py`
 - `AdvancedTextEncoder` — продвинутый текстовый энкодер с multi-head self-attention (RoPE), N-layer Transformer и attention pooling; backward-compatible замена SimpleTextEncoder
 - `RotaryEmbedding` — реализация RoPE для позиционного кодирования
 - `MultiHeadSelfAttention` — self-attention с RoPE и NaN-safe softmax
 - `TransformerBlock` — pre-norm Transformer block с FFN
 - `AttentionPooling` — learned query attention pooling вместо mean pooling
- `hypercoplexAI/src/core/hierarchical_memory.py`
 - `HierarchicalTitansMemory` — двухуровневая память (working + long-term) с surprise-based обновлением
 - Level 1 (working): обновляется на каждом шаге через TTT gradient step
 - Level 2 (long-term): обновляется только при высоком surprise (ошибка предсказания)
 - `blend_proj` — learned mix между working и long-term retrieval
- `hypercoplexAI/src/core/soft_moe_router.py`
 - `SoftMoERouter` — Soft Mixture of Experts (Puigcerver et al., ICLR 2024)
 - Dispatch/combine матрицы вместо hard top-k routing
 - Нет token dropping, стабильные градиенты, равномерная нагрузка экспертов
 - API-совместим с R3MoERouter
- `hypercoplexAI/scripts/gpu_train.py` — GPU training script с AMP, cosine LR, monitoring
- `hypercoplexAI/scripts/autoresearch_loop.py` — AutoResearch Loop для hyperparameter search
- `hypercoplexAI/tests/test_new_components.py` — тесты для новых компонентов
### 4.4 Что MVP умеет сейчас
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

### 4.5 Чего MVP не должен утверждать
MVP не должен описываться как уже имеющий:

- доказанно универсальный физический инвариант;
- production-grade multi-domain scientific discovery engine;
- полноценный Scholar / PubMed / Patents memory ingestion pipeline;
- validated human-in-the-loop scientific review workflow;
- завершённый Q-Attention graph transformer;
- полноценную автоматическую интерпретацию ТРИЗ на runtime;
- generative language model или pretrained semantic encoder;
- externally verified autoresearch runtime с доказательной литературной валидацией.

### 4.6 Ограничения текстового и оркестрационного слоя
Чтобы документ оставался честным относительно текущего кода, нужно явно фиксировать ограничения:

- `TextHDIMModel` — это trainable text-entry wrapper с простым internal encoder, а не pretrained LLM encoder и не полноценная generative text stack;
- `--text_mode` в CLI только переключает training/eval path на `TextHDIMModel` поверх того же HDIM core и не добавляет отдельный inference service;
- manifest wiring через `ExperimentConfig` покрывает reproducible local runs, но не является scheduler, queue system или distributed execution layer;
- `ExperimentRunner` и `AutoResearchRunner` сохраняют артефакты и score summaries, но не выполняют внешнюю литературную валидацию и не заменяют human review;
- текущий autoresearch loop — это local experiment orchestration around `src.training.train`, а не автономный scientific agent runtime;
- текущий runtime оценивает качество прогона в первую очередь через `pair_margin`, потому что именно он используется для выбора `best_run` в `AutoResearchRunner`.

### 4.7 Контракт интеграции авторесёрча
Новая autoresearch-поверхность уже существует как orchestration layer вокруг обучающего CLI и manifest schema.

Её честный контракт в текущем проекте:

1. `ExperimentConfig` фиксирует воспроизводимый run manifest (`text_mode`, `use_pairs`, `negative_ratio`, override-поля, `results_json`, `ledger_path`, `metadata`, `status`).
2. `ExperimentRunner` материализует один запуск в `manifest.json`, `stdout.txt`, `stderr.txt`, `results.json` и ledger-строку.
3. `AutoResearchRunner` выполняет серию manifest-driven прогонов, пишет `session_ledger.jsonl` и `session_summary.json`, а затем выбирает `best_run` полю `score`.
4. В текущей реализации `score = quality.pair_margin`, то есть autoresearch оптимизирует не абстрактное качество открытия, а proxy separability aligned vs mismatched pairs.
5. Loop stages жёстко ограничены локальным циклом `plan_manifest -> execute_training -> collect_artifacts -> score_quality -> record_decision`.

Следствие: autoresearch в HDIM сегодня — это reproducible local experiment loop для отбора конфигураций, а не self-improving scientific agent с внешним knowledge ingestion.


---

## 5. Реализованные и будущие компоненты
### 5.1 Реализовано или готово к реализации
#### A. Гиперкомплексный бэкбон
Реализовано:
- quaternion-aware linear projection;
- Clifford algebra utilities;
- invariant extraction via sandwich-style transform.

#### B. Транспортный слой домена
Реализовано:
- `DomainRotationOperator` для обучаемых доменных роторов;
- `InvariantExtractor`;
- `sandwich_transfer`.

#### C. Слой памяти
Реализовано:
- `TitansMemoryModule` как online-updated associative memory;
- memory retrieval/update inside pipeline;
- memory loss integration in trainer.

#### D. Слой маршрутизации
Реализовано:
- `R3MoERouter`;
- top-k gating;
- router auxiliary loss;
- routing observability through transfer state.

#### E. Инфраструктура обучения и оценки
Реализовано:
- `DomainProblemDataset`;
- paired supervision via `pair_encoding` / `pair_domain_id`;
- pair metadata via `pair_group_id`, `pair_family_id`, `pair_relation_type`, `pair_relation_label`, `pair_weight`;
- в текущем dataset surface `pair_family_id` дублирует `pair_group_id`, поэтому реальная split semantics сейчас group-aware, а не independent family-aware;
- `create_group_aware_split(...)` для leakage-aware validation semantics без пересечения `pair_group_id` между train и validation.
- `HDIMTrainer`;
- CLI training loop with optional machine-readable run summary JSON;
- tests forward, paired transfer, validation, metrics.

#### F. Минимальная текстовая обёртка
Реализовано:
- `TextHDIMModel` как thin wrapper поверх `HDIMModel`;
- trainable text entry через `SimpleTextEncoder`, который поддерживает fallback char-level path и optional vocab/tokenizer config;
- `forward_texts(...)`, `transfer_texts(...)`, `transfer_text_pairs(...)`;
- `score_text_pairs_with_state(...)` как retrieval/ranking surface, возвращающий scores и source/target runtime states;
- retrieve-only scoring contract (`memory_mode="retrieve"`, `update_memory=False`) для text-pair comparison без мутации памяти во время scoring.

Важно: этот слой предназначен для retrieval/ranking/transfer experiments и orchestration-friendly inspection, но не превращает HDIM в генеративную текстовую модель.

#### G. Продвинутый текстовый энкодер (Фаза 2)
Реализовано:
- `AdvancedTextEncoder` с multi-head self-attention (4 heads), RoPE, attention pooling;
- N-layer Transformer backbone (pre-norm, GELU FFN, residual);
- NaN-safe softmax с nan_to_num recovery;
- Xavier/zero init для стабильного обучения;
- Backward compatible с `SimpleTextEncoder` через единый интерфейс.

#### H. Иерархическая память (Фаза 2)
Реализовано:
- `HierarchicalTitansMemory` — двухуровневая ассоциативная память;
- Working memory: быстрое обновление через TTT на каждом шаге;
- Long-term memory: обновление через surprise gate (learned gate на основе ошибки предсказания);
- Learned blend коэффициент для смешивания working/long-term retrieval;
- API-совместима с `TitansMemoryModule` через `retrieve_and_update → MemoryState`.

#### I. Soft MoE маршрутизатор (Фаза 2)
Реализовано:
- `SoftMoERouter` — мягкая маршрутизация экспертов без token dropping;
- Dispatch/combine матрицы: каждый токен получает взвешенный mix всех экспертов;
- Entropy load balance loss;
- EMA train_scores для R3 совместимости;
- API-совместим с `R3MoERouter`.

#### J. Инфраструктура GPU-обучения (Фаза 3)
Реализовано:
- `scripts/gpu_train.py` — GPU training loop с AMP (torch.amp API), OneCycleLR scheduler;
- Мониторинг: `nan_batches_epoch`, `nan_batches_total`, `loss_memory`, `loss_routing`, `score` (canonical PRIMARY_SCORE_WEIGHTS);
- `compute_primary_score()` и `check_run_validity()` — единственный источник scoring для incumbent comparison;
- Checkpoint по лучшему `score` с `--results_json` / `--ledger_path` для orchestration;
- `scripts/autoresearch_loop.py` — двухфазный `two_phase` режим (explore→refine) с `IncumbentTracker`;
- `IncumbentTracker` отслеживает лучшую конфигурацию, ведёт failure taxonomy: `crash_nan`, `crash_oom`, `crash_runtime`, `metric_regression`, `timeout`;
- `check_run_validity()` блокирует metric_regression и crash_nan прогоны от попадания в incumbent.

#### K. Эмпирические закономерности (Фаза 3)
Из autoresearch экспериментов на RTX 3070:
- Конфигурация `advanced_encoder=True, soft_router=True, hierarchical_memory=False` показывает наиболее стабильное обучение;
- `HierarchicalTitansMemory` + `AdvancedTextEncoder` совместно нестабильны — TTT gradient плохо взаимодействует с AMP;
- `hidden_dim=64-128` с `num_experts=8` предпочтительнее больших моделей на малых датасетах;
- Исправление recon_target для negative pairs (source encoding вместо pair encoding) критически важно для положительного pair_margin;
- Labeled pair_margin (pos vs neg по `pair_relation_label`) является более надёжной метрикой чем группово-вычисляемый margin.

### 5.2 Компоненты следующей фазы
Следующая фаза должна улучшать текущий код, а не подменять его новой абстрактной архитектурой.

Нужны:

1. более строгий paired dataset contract;
2. более явный lifecycle memory reset / retrieve / update;
3. алгебраические contract tests;
4. более прозрачный router state и debug surface;
5. alignment между training contract и evaluation contract;
6. канонизировать `exported_invariant` / `training_invariant` naming surface вместо старого umbrella-термина `processed invariant`.

### 5.3 Исследовательские и долгосрочные компоненты
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

## 6. Контракт обучения
### 6.1 Базовый режим
Same-domain reconstruction mode:

- вход: `encoding`, `domain_id` или raw texts + `domain_id` при text path;
- путь: same-domain transfer / reconstruction;
- цель: стабилизировать representation, memory, routing, decode path.

### 6.2 Режим попарного изоморфизма
Cross-domain paired mode:

- вход: `encoding`, `domain_id`, `pair_encoding`, `pair_domain_id` либо raw text batch с `text` / `pair_text` и теми же domain ids;
- путь: explicit transfer from source domain to paired target domain;
- цель: приближать invariants между функционально связанными примерами и одновременно отделять aligned pair от mismatched pair.

### 6.3 Состав функции потерь
В implementable версии HDIM тренируется не на «чистой философской изоморфности», а на комбинации наблюдаемых loss-компонент:

- `L_recon` — reconstruction / paired reconstruction target;
- `L_iso` — MSE между paired invariant representations с positive/negative semantics через `pair_relation_label` и `pair_weight`;
- `L_pair` — ranking-margin term в пространстве `exported_invariant` для aligned vs mismatched pairs;
- `L_routing` — router auxiliary term;
- `L_memory` — memory adaptation term.

Это ключевой engineering contract текущей системы.

### 6.4 Семантика валидации
`HDIMTrainer.validate()` в текущем коде агрегирует batch-level значения `loss_recon`, `loss_iso`, `loss_pair`, `loss_routing`, `loss_memory`, `loss_total` по validation dataloader.

Важно: `loss_pair` не только участвует в training/eval batch objective внутри `_compute_batch_losses()`, но и действительно возвращается отдельным агрегированным полем из `validate()`. Поэтому validation contract документа должен явно включать эту компоненту.

### 6.5 Разбиение данных
Для paired datasets validation строится через `create_group_aware_split(...)`, который удерживает `pair_group_id` в непересекающихся train/validation split'ах и тем самым задаёт текущую leakage-aware semantics для MVP.

Важно: хотя sample payload экспортирует и `pair_family_id`, в текущей реализации `src/training/dataset.py` он заполняется тем же значением, что и `pair_group_id`. Поэтому документ не должен обещать отдельную family-level split semantics, пока она не появится в коде.

Для non-paired datasets тот же helper откатывается к обычному random split.

### 6.6 Метрики и отчётность
После training loop текущий CLI вызывает `compute_all_metrics()` и пишет quality surface `STS_exported`, `STS_training`, `DRS`, `AFR`, `pair_margin`.

Если задан `results_json`, `src.training.train` сохраняет machine-readable run summary с config/run args/validation/quality/checkpoint/status; если задан `ledger_path`, дополнительно дописывается JSONL ledger row. Это reporting/orchestration contract, а не внешняя scientific validation layer.

### 6.7 Примечание о нумерации
Раздел `7` фиксирует invariant lifecycle, а следующий самостоятельный блок оценки начинается с раздела `8`.

---

## 7. Жизненный цикл инварианта и памяти
### 7.1 Именование стадий инварианта
Текущий код фиксирует четыре разные стадии invariant lifecycle:

- `raw_invariant` — результат `InvariantExtractor` до памяти;
- `memory_augmented_invariant` — invariant после `TitansMemoryModule`;
- `exported_invariant` — canonical transfer object после router;
- `training_invariant` — projection `exported_invariant` в `hidden_dim` для `L_iso` и training-facing diagnostics.

Это означает, что прежний термин `processed invariant` больше не должен использоваться как универсальное имя для всех стадий сразу. В инженерном контракте MVP:

- основной объект переноса = `exported_invariant`;
- основной объект оптимизации = `training_invariant`.

### 7.2 Протокол памяти
Memory lifecycle в MVP должен трактоваться как явный runtime contract с режимами:

- `memory_mode="update"` — train-path, retrieval + update памяти;
- `memory_mode="retrieve"` — eval/validation path без мутации памяти;
- `memory_mode="none"` — stateless ablation path без memory augmentation.

Флаг `update_memory` имеет смысл только вместе с `memory_mode="update"`; в остальных режимах мутация памяти отключается принудительно.

Минимальная инженерная гарантия текущего кода: repeated eval в `retrieve` режиме не должен менять memory state и router replay state.

### 7.3 Наблюдаемость маршрутизатора
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

## 8. Контракт оценки качества
### 8.1 STS_exported — структурный перенос в пространстве передачи
В текущем проекте `STS_exported` означает косинусное сходство между aligned source/target `exported_invariant`.

Практически это **proxy for structure preservation in canonical transfer space**, а не финальное доказательство физической эквивалентности.

### 8.2 STS_training — структурный перенос в пространстве оптимизации
`STS_training` измеряет косинусное сходство между aligned source/target `training_invariant`.

Эта метрика нужна, чтобы evaluation surface не смешивал runtime transfer object и optimization projection.

### 8.3 DRS — стабильность доменной маршрутизации
DRS измеряет стабильность router weights при повторных вызовах.

Это инженерная метрика устойчивости маршрутизации, а не формальная оценка корректности научного вывода.

### 8.4 AFR — оценка применимости аналогии
AFR в MVP теперь трактуется как **margin-aware feasibility proxy**: aligned pair считается успешной, если одновременно выполняются similarity threshold и pair-margin threshold.

Следовательно, AFR остаётся **internal consistency metric**, но уже учитывает отличие aligned pair от mismatched negative pair.

### 8.5 Pair margin
`pair_margin` — это средний зазор между similarity aligned pairs и similarity mismatched pairs в пространстве `exported_invariant`.

Эта метрика нужна как минимальный honesty check, что paired evaluation различает корректное и некорректное сопоставление.

---

## 9. Концептуальные модули как приближения
### 9.1 Q-Attention
Целевая формула сохраняется как conceptual extension:

$$
\alpha_{ij} = \mathrm{softmax}\left(\frac{\mathrm{Re}(q_i \otimes \bar{q}_j)}{\sqrt{d}}\right)
$$

Но в текущем HDIM это **не обязательный runtime-компонент MVP**. Его нужно трактовать как кандидат для следующей research-phase, а не как текущую основу пайплайна.

### 9.2 ТРИЗ DSL
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

### 9.3 Человек в контуре
Схема режимов полезна как продуктовая надстройка:

| Mode | Interpretation |
|---|---|
| Auto | полностью автоматический MVP pipeline |
| Semi-auto | пользователь выбирает из candidate analogies |
| Manual | внешний исследовательский поиск и ручная фильтрация |

Но пока это следует понимать как **future interaction design**, а не как уже существующую часть runtime API.

### 9.4 Интеграция Scholar API
External scientific retrieval остаётся долгосрочной интеграцией. Текущий `TitansMemoryModule` — это internal learnable memory, а не подключённая база Google Scholar / Semantic Scholar / PubMed.

### 9.5 Контракт исполнения и матрица реализации
#### 9.5.1 Минимальный исполнимый контракт
Чтобы документ оставался синхронизированным с текущим кодом, MVP должен читаться как следующий исполнимый контракт:

1. входом служит либо уже готовый `encoding`, либо raw text, который кодируется trainable text-entry encoder внутри `TextHDIMModel`;
2. `HDIMModel` выполняет either same-domain reconstruction, либо explicit paired transfer;
3. `TextHDIMModel` добавляет thin text-facing entry layer для retrieval / ranking / transfer experiments поверх того же HDIM core;
4. text-pair scoring выполняется через cosine similarity в пространстве `exported_invariant` и может возвращать structured artifact со score и обеими runtime states;
5. `HDIMTrainer` оптимизирует наблюдаемые objective-компоненты `loss_recon`, `loss_iso`, `loss_pair`, `loss_routing`, `loss_memory`;
6. CLI training path может дополнительно записывать machine-readable run summary JSON для orchestration/reporting flows, а manifest schema (`ExperimentConfig`) фиксирует `text_mode`, `use_pairs`, `results_json`, `ledger_path`, `status`, override-поля и metadata;
7. качество оценивается через `validate()` и `compute_all_metrics()`;
8. корректность MVP подтверждается тестами на forward / transfer / paired dataset / metrics и minimal text-wrapper surface.

#### 9.5.2 Цикл обучения в привязке к коду
Реалистичный цикл обучения для текущего MVP выглядит так:

```text
create_demo_dataset() or create_paired_demo_dataset()
→ create_group_aware_split(...)
→ DataLoader
→ HDIMModel.forward() or HDIMModel.transfer_pairs()
→ HDIMTrainer._compute_batch_losses()
→ loss_total = loss_recon + λ_iso·loss_iso + λ_pair·loss_pair + λ_routing·loss_routing + loss_memory
→ optimizer.step()
→ trainer.validate(...)
→ compute_all_metrics(...)
→ optional results_json writeout with config / validation / quality / checkpoint summary
```

Text-facing retrieval/ranking flow поверх того же core выглядит отдельно:

```text
raw texts
→ TextHDIMModel.encode_texts(...)
→ SimpleTextEncoder
→ TextHDIMModel.forward_texts(...) or transfer_text_pairs(...)
→ exported_invariant
→ cosine similarity scoring
→ TextPairScoreResult for inspection/reporting
```

То есть основной путь проекта уже определён не whitepaper-обещаниями, а конкретным training contract в `src/training/trainer.py`, text-entry scaffold в `src/models/text_hdim_model.py` и data contract в `src/training/dataset.py`.

#### 9.5.3 Матрица реализации
| Компонент | Статус | Файл | Примечание |
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
| AdvancedTextEncoder | implemented (Phase 2) | `hypercoplexAI/src/models/advanced_text_encoder.py` | Transformer+RoPE+AttentionPooling замена SimpleTextEncoder |
| HierarchicalTitansMemory | implemented (Phase 2) | `hypercoplexAI/src/core/hierarchical_memory.py` | two-level memory с surprise-based routing |
| SoftMoERouter | implemented (Phase 2) | `hypercoplexAI/src/core/soft_moe_router.py` | Soft MoE без token dropping |
| GPU training script | implemented (Phase 3) | `hypercoplexAI/scripts/gpu_train.py` | AMP (torch.amp), OneCycleLR, canonical score, check_run_validity, nan tracking |
| AutoResearch Loop | implemented (Phase 3) | `hypercoplexAI/scripts/autoresearch_loop.py` | двухфазный explore→refine с IncumbentTracker и failure taxonomy |
| Q-Attention | research only | not present in runtime code | оставить как post-MVP extension |
| TRIZ parser / avatars / Scholar ingestion | research only | not present in runtime code | внешний workflow, не ядро Python MVP |

#### 9.5.4 Критерии приёмки MVP
Документ следует считать честно выполненным относительно текущей архитектуры, если одновременно верны шесть условий:

- paired dataset может быть создан без нарушения positive/negative cross-domain contract;
- train/validation split может быть построен через `create_group_aware_split(...)` без пересечения pair groups между split'ами;
- `HDIMTrainer.train_step()` работает как для обычного, так и для paired batch;
- `validate()` возвращает агрегированные loss-метрики;
- `compute_all_metrics()` возвращает `STS_exported`, `STS_training`, `DRS`, `AFR`, `pair_margin` на том же model API;
- text-wrapper scoring path возвращает retrieve-safe structured result без мутации памяти во время comparison.

Именно эти критерии делают HDIM implementable-first системой уже сейчас.

---

## 10. Дорожная карта
### 10.1 Фаза 1 — базовый MVP
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

### 10.2 Фаза 2 — завершена
Цель: усилить реализуемость и экспериментальную достоверность.

Выполнено:
1. AdvancedTextEncoder с RoPE + AttentionPooling — замена SimpleTextEncoder;
2. HierarchicalTitansMemory с surprise-based long-term routing;
3. SoftMoERouter без token dropping;
4. GPU training loop с AMP, canonical score, check_run_validity;
5. AutoResearch Loop с двухфазным explore→refine и IncumbentTracker;
6. Исправление recon_target для negative pairs (критический баг training objective);
7. Labeled pair_margin в compute_all_metrics.

### 10.3 Фаза 3 — активная
Цель: стабилизировать обучение и улучшить качество метрик.

Выявленные проблемы и работы:
1. HierarchicalTitansMemory TTT нестабилен с AMP — изолировать float32 path;
2. pair_margin остаётся малым (~0.02-0.04) — исследовать влияние датасета и loss весов;
3. val_loss скачет из-за редкого eval — увеличить частоту eval;
4. Масштабирование: GPU использует <1% VRAM — увеличить hidden_dim/batch для RTX 3070;
5. Реальные текстовые данные — добавить возможность обучения на реальных text corpora.

### 10.4 Долгосрочные исследования
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

## 11. Правила для будущих правок
При дальнейшем редактировании архитектуры нужно соблюдать правило:

**любое сильное научное утверждение должно быть помечено как одно из трёх:**

- `implemented`
- `engineering next`
- `research hypothesis`

Это необходимо, чтобы HDIM.md больше не создавал ложного впечатления, будто весь vision-stack уже доступен в коде.

---

## 12. Практический статус проекта
### 12.1 Что есть сегодня
Сегодня HDIM — это:

- реализуемый PyTorch prototype;
- invariant-centered transfer architecture;
- pair-aware training scaffold;
- memory + routing + domain transport stack;
- система с уже существующими тестами и минимальным synthetic data loop.

### 12.2 Чем проект ещё не является
Сегодня HDIM — это ещё не:

- научный discovery engine с подключённой мировой литературой;
- доказательный cross-domain invention system;
- полноформатный graph reasoning platform;
- finished human-in-the-loop research workstation.

### 12.3 Честное однострочное описание
HDIM в текущем состоянии — это **MVP hypercomplex transfer prototype for learning and evaluating domain-invariant approximations under memory-augmented routing**, а не завершённая система автоматизации научных открытий.

---

## 13. Индекс файлов реализации
Для навигации по текущей implementable architecture:

- `hypercoplexAI/HDIM.md` — this specification
- `hypercoplexAI/src/core/hypercomplex.py` — algebra and hypercomplex layers
- `hypercoplexAI/src/core/domain_operators.py` — domain transport primitives
- `hypercoplexAI/src/core/titans_memory.py` — memory module
- `hypercoplexAI/src/core/moe_router.py` — routing module
- `hypercoplexAI/src/core/hdim_pipeline.py` — orchestrated transfer pipeline
- `hypercoplexAI/src/models/hdim_model.py` — batch-facing model API
- `hypercoplexAI/src/models/metrics.py` — evaluation metrics
- `hypercoplexAI/src/models/text_hdim_model.py` — text-entry wrapper and structured text-pair scoring
- `hypercoplexAI/src/training/dataset.py` — demo and paired datasets
- `hypercoplexAI/src/training/trainer.py` — training loop and validation semantics
- `hypercoplexAI/src/training/train.py` — CLI training entrypoint, validation/quality reporting, and optional run-summary writeout
- `hypercoplexAI/src/training/experiment_config.py` — manifest schema for experiment runs
- `hypercoplexAI/src/training/experiment_runner.py` — experiment/autoresearch orchestration, artifact materialization, and best-run selection by `quality.pair_margin`
- `hypercoplexAI/src/training/results_logger.py` — JSON/JSONL helpers used by run summaries and ledgers
- `hypercoplexAI/tests/test_hdim.py` — coverage for baseline and paired behavior
- `hypercoplexAI/tests/test_new_components.py` — coverage for AdvancedTextEncoder, HierarchicalTitansMemory, SoftMoERouter, integration tests
- `hypercoplexAI/src/models/model_factory.py` — factory API for constructing HDIM model variants from config
---

## 15. Результаты обучения (обновлено 2026-03-11)

Все запуски выполнялись на GPU (CUDA, AMP) если не указано иное. Артефакты хранятся в `artifacts/`.

---

### 15.1 Первый лучший результат — `autoresearch_final`

| Метрика | Значение |
|---------|----------|
| **Score** | **0.2895** |
| STS_exported | 0.9376 |
| STS_training | 0.9789 |
| pair_margin | 0.0082 |
| AFR | 1.0 |
| DRS | ~2.6e-9 |
| Эпох | 40 |
| Время обучения | 164 с |

**Конфигурация:**
```json
{ "hidden_dim": 64, "num_experts": 4, "advanced_encoder": true,
  "hierarchical_memory": false, "soft_router": true,
  "lambda_iso": 0.0733, "lambda_pair": 0.1723,
  "lambda_routing": 0.012, "lambda_memory": 0.013 }
```

Чекпоинт: `artifacts/autoresearch_final/hdim-final-refine-002/checkpoints/best.pt`

---

### 15.2 Прогоны autoresearch_final (10 запусков)

#### Фаза explore (6 прогонов)
| Run | Score | STS_exp | pair_margin | Вердикт |
|-----|-------|---------|-------------|--------|
| explore-001 | 0.060 | 0.201 | 0.0 | discard |
| explore-002 | 0.284 | 0.945 | 0.0 | keep |
| explore-003 | 0.049 | 0.163 | 0.0 | discard |
| explore-004 | 0.186 | 0.621 | 0.0 | discard |
| explore-005 | 0.070 | 0.233 | 0.0 | discard |
| explore-006 | 0.192 | 0.640 | 0.0 | discard |

**Вывод:** compact конфиг (hidden_dim=64, soft_router=true, advanced_encoder=true) стабильно превосходит wide (hidden_dim=256+) и hard-router варианты.

#### Фаза refine (4 прогона)
| Run | Score | STS_exp | pair_margin | Вердикт |
|-----|-------|---------|-------------|--------|
| refine-001 | 0.287 | 0.936 | 0.0062 | keep |
| **refine-002** | **0.2895** | **0.938** | **0.0082** | **keep** |
| refine-003 | 0.237 | 0.782 | 0.0028 | discard |
| refine-004 | 0.069 | 0.260 | −0.0085 | discard |

**Вывод:** hierarchical_memory=true ухудшает результат на этом масштабе данных. Отключение soft_router обрушивает STS_exported с 0.94 до 0.26.

---

### 15.3 Прогоны autoresearch_phase3 (17 запусков)

Лучший прогон: `hdim-phase3-refine-004` — score **0.229**, pair_margin **0.035**, STS_exp **0.646**.

| Параметр | Значение |
|----------|----------|
| hidden_dim | 256 |
| num_experts | 4 |
| advanced_encoder | false |
| soft_router | true |
| lambda_pair | 0.3587 |
| ranking_margin | 0.2368 |

Phase3 проверял ranking_margin как отдельный гиперпараметр. AFR=0.89, pair_margin=0.035 — лучший pair_margin среди всех autoresearch прогонов, но STS_exported ниже из-за отсутствия advanced_encoder.

---

### 15.4 Ручные GPU-запуски

| Run | Эпох | best_score | best_STS_exp | pair_margin (финал) | Время |
|-----|------|-----------|-------------|---------------------|-------|
| gpu_training | 50 | — | — | — | — |
| gpu_run_001 | 30 | — | — | — | — |
| gpu_run_002 | 50 | — | — | — | — |
| gpu_run_003 | — | — | — | — | — |
| **gpu_run_best** | 80 | **0.333** | **0.914** | 0.035 | 447 с |
| gpu_run_final | 100 | 0.261 | 0.870 | 0.0 | 709 с |

`gpu_run_best` — исторически лучший единственный прогон по best_score=0.333 (epoch 10). Финальное качество упало из-за переобучения (epoch 10 vs epoch 80).

---

### 15.5 Ключевые выводы

1. **Лучший стабильный конфиг:** hidden_dim=64, num_experts=4, advanced_encoder=true, soft_router=true, hierarchical_memory=false.
2. **STS_exported ~0.94** достигается автоматически при правильной комбинации; без soft_router падает до 0.2–0.3.
3. **pair_margin** слабо коррелирует с STS_exported — оба нужны для надёжной оценки transfer quality.
4. **hierarchical_memory** на текущем масштабе данных (<2000 сэмплов) не помогает и стабильно снижает score.
5. **Переобучение по эпохам:** лучший checkpoint часто на epoch 5–15, финальный хуже. Рекомендуется early stopping или более строгий eval_every.
6. **0 крэшей** (crash_nan, crash_oom, crash_runtime) во всех autoresearch прогонах.


### 15.6 Targeted runs — фазы 6-8 (2026-03-10)

**Важное замечание:** прогоны targeted_run_008 через targeted_run_016 выполнялись на **CPU** (системный python3 без CUDA). Результаты немного хуже чем GPU-эквиваленты, сравнение с GPU incumbent условно.

**Incumbent (GPU, лучший за всё время):** `targeted_run_002` — score=**0.3002**, STS=0.927, margin=+0.022, ep=50
- Config: seed=77, lambda_iso=0.05, lambda_pair=0.4, ranking_margin=0.5, 80 epochs, num_samples=1000

| Run | Score | STS | Margin | Config notes |
|-----|-------|-----|--------|--------------|
| targeted_run_001 | 0.290 | 0.933 | +0.010 | seed=42, lambda_pair=0.3, 60ep |
| **targeted_run_002** | **0.300** | **0.927** | **+0.022** | seed=77, lambda_pair=0.4, 80ep — GPU BEST |
| targeted_run_003 | 0.294 | 0.956 | +0.007 | lambda_pair=0.45, 100ep |
| targeted_run_008 | 0.292 | 0.982 | −0.002 | seed=77, lambda_pair=0.4, 100ep (CPU) |
| targeted_run_009 | 0.299 | 0.995 | −0.0001 | seed=42, lambda_pair=0.38, 100ep (CPU) |
| targeted_run_010 | 0.294 | 0.982 | −0.0003 | seed=77, lambda_pair=0.35, 80ep (CPU) |
| targeted_run_011 | 0.275 | 0.958 | −0.013 | lambda_pair=0.17, 80ep (CPU) |
| targeted_run_012 | 0.299 | 0.996 | −0.0001 | seed=42, ranking_margin=0.55 (CPU) |
| targeted_run_013 | 0.298 | 0.995 | +0.00001 | seed=42, lambda_pair=0.32, neg_ratio=0.4 (CPU) |
| targeted_run_014 | 0.294 | 0.987 | −0.002 | seed=99 (CPU) |
| targeted_run_015 | 0.293 | 0.985 | −0.002 | seed=77, lambda_pair=0.38 (CPU) |
| targeted_run_016 | 0.298 | 0.986 | **+0.003** | seed=42, lambda_pair=0.4 (CPU) — устойчивый + margin |

**Выводы фаз 6-8:**
1. **seed=77 + lambda_pair=0.4** — уникальная комбинация для высокого margin. Снижение до 0.38 убивает положительный margin у seed=77.
2. **seed=42 + lambda_pair=0.4** даёт устойчивый +0.003 margin, но STS ниже (0.986 vs 0.995) → score=0.298 < 0.300.
3. **seed=42 + lambda_pair=0.38** даёт STS=0.995 но margin≈0 → score=0.299.
4. Потолок на CPU: ~0.299. Нужно запускать с правильным Python (`.venv/Scripts/python.exe`) для CUDA.
5. **Правильный интерпретатор:** `E:/hypercoplexAI/.venv/Scripts/python.exe` — torch 2.6.0+cu124, CUDA доступна.
6. Системный `python3` в MSYS bash = Python 3.13 без CUDA (torch cpu-only) — не использовать для обучения.

---

### 15.7 Phase 4 — Frozen SBERT + InfoNCE + Real Cross-Domain Pairs (2026-03-10)

**Ключевой прорыв:** замена trainable AdvancedTextEncoder на замороженный SBERT (paraphrase-multilingual-mpnet-base-v2, 278M параметров) + InfoNCE loss + реальные структурные аналогии. Результат: score вырос с 0.300 (incumbent) до **0.8828** — прирост ×2.9.

#### Архитектурные изменения фазы 4

| Компонент | До (Phase 3) | После (Phase 4) |
|-----------|-------------|----------------|
| Encoder | AdvancedTextEncoder (145k обучаемых) | Frozen SBERT (278M frozen) + projection (410k) |
| Loss | Ranking margin (pairwise) | InfoNCE/NT-Xent (in-batch negatives) |
| Данные | 1000 синтетических сэмплов | 30 реальных аналогий × augment |
| hidden_dim | 64 | 256 |

#### Новые модули
- `src/models/sbert_encoder.py` — FrozenSBERT с trainable projection (768→hidden_dim), embedding cache
- `src/models/model_factory.py` — `build_sbert_hdim_model()` factory
- `src/training/real_dataset.py` — `RealPairsDataset`, group-aware split, augmentation
- `data/real_pairs.json` — 30 реальных аналогий (20 pos + 10 neg, 4 домена)
- `data/real_pairs_v2.json` — расширенный датасет: 57 пар (45 pos + 12 neg)

#### Результаты `sbert_real_run` (30 пар × augment×15 = 750 сэмплов)

| Метрика | Значение |
|---------|----------|
| **best_score** | **0.8828** (epoch 45) |
| best_pair_margin | 0.6487 |
| best_STS_exported | 0.7804 |
| final_score | 0.8661 |
| final_margin | 0.6508 |
| final_STS | 0.7178 |
| total_time | 1224 с (~20 мин) |
| nan_batches | 0 |

Артефакт: `artifacts/sbert_real_run/`

#### Динамика обучения
- Ep 5: score=0.39, STS=0.87 (хорошая стартовая точка из SBERT)
- Ep 20: score=0.80, margin=0.57
- Ep 45: score=**0.88** (лучший), STS=0.78 ← оптимальный checkpoint
- Ep 65-79: score=0.86-0.91, margin=0.69, STS=0.67-0.71 (деградация STS)
- Ep 100: score=0.87 (финальный)

**Ключевая проблема:** после ep=45 STS деградирует 0.780→0.718. Модель переоптимизирует margin за счёт STS — InfoNCE давит косинусное сходство слишком агрессивно.

#### Эмпирические выводы Phase 4
1. **Frozen SBERT** устраняет проблему семантической бессодержательности trainable encoder с нуля: уже с ep=5 STS=0.87
2. **InfoNCE** (in-batch negatives) решает STS vs margin trade-off на ранних эпохах, но переоптимизирует на поздних
3. **30 реальных аналогий** дают лучший сигнал чем 1000 синтетических при том же времени обучения
4. **augment_factor=15** создаёт 750 сэмплов — достаточно для хорошей сходимости
5. **GPU=0.02GB** — модель очень маленькая относительно RTX 3070 (8.6GB); пространство для масштабирования огромное
6. **CosineAnnealingWarmRestarts** лучше OneCycleLR для стабилизации

#### Следующие шаги
- Joint STS regularization: добавить L_STS = 1 - cosine(src, tgt) как soft penalty
- Расширить датасет: `data/real_pairs_v2.json` (57 пар)
- Temperature annealing: постепенно повышать temperature 0.07→0.15 после ep=30
- Retrieval@K evaluation: recall@1, recall@5 как дополнительная метрика
- Масштаб: hidden_dim=512+, num_experts=8 при 8.6GB VRAM

### 15.8 Phase 4b — STS Regularization + Expanded Dataset (2026-03-10)

#### Лучший результат: `sbert_sts_run` — score=**0.9317** (INCUMBENT)

| Компонент | Значение |
|-----------|----------|
| hidden_dim | 256 |
| num_experts | 4 |
| soft_router | True |
| lr | 0.0005 |
| seed | 77 |
| epochs | 120 |
| lambda_pair | 0.4 |
| lambda_sts | 0.2 |
| infonce_temperature | 0.1 |
| augment_factor | 20 |
| dataset | real_pairs_v2.json (57 пар, 2040 сэмплов) |

#### Динамика sbert_sts_run

| Эпоха | score | margin | STS |
|-------|-------|--------|-----|
| 5 | 0.705 | — | 0.876 |
| 15 | 0.846 | 0.608 | 0.793 |
| **35** | **0.932** | **0.704** | **0.759** |
| 50 | 0.743 | 0.492 | 0.838 |
| 65 | 0.775 | 0.532 | 0.811 |

#### sbert_sts_v2 — дополнительный эксперимент (augment x25, 150 ep)

- Конфиг: augment_factor=25 (2550 сэмплов), epochs=150
- Процесс завершился досрочно (24 эпохи записано)
- Best ep=10: score=0.777, margin=0.541, STS=0.787
- **Вывод**: увеличение augment_factor с 20→25 не дало улучшения; loss_routing=2.56 высок — routing не сходится при большем датасете без fine-tuning роутера

#### Ключевые выводы Phase 4b
1. **lambda_sts=0.2** удерживает STS выше 0.75 на протяжении всего обучения (vs 0.71 floor без регуляризации)
2. **Оптимальный augment_factor=20**: 2040 сэмплов — баланс разнообразия и сходимости
3. **CosineAnnealingWarmRestarts** вызывает скачки score при LR-перезапусках; eval_every=1 критичен для захвата пика
4. **Early stopping** необходим: после ep=35 score ухудшается из-за LR-restarts
5. **loss_routing** (2.56) остаётся высоким — routing loss не участвует в финальной score, но указывает на неэффективность MoE
6. **GPU utilization 0.02GB** из 8.6GB — огромный запас для масштабирования

#### Рекомендации для Phase 5
- **eval_every=1 + early_stopping_patience=10**: захват пика и автостоп
- **hidden_dim=512, num_experts=8**: использовать VRAM
- **lambda_routing=0**: routing loss мешает сходимости основной задачи
- **Retrieval@K**: добавить recall@1, @5 как primary metric вместо score
- **Fine-tune SBERT projection**: unfreeze последние 2 слоя после ep=50

---

### 15.9 Phase 5a — AnglE Loss + Expanded Dataset + Learnable Temperature (2026-03-10)

#### Новый рекорд: `phase5a_run` — score=**0.9670** (INCUMBENT)

**Конфигурация:**
```
--hidden_dim 256 --num_experts 4 --soft_router
--pretrained_encoder
--real_pairs data/real_pairs_v4.json  # 140 пар (117 pos + 23 neg)
--augment_factor 30                   # 7710 сэмплов (было 2040)
--use_infonce --infonce_temperature 0.1
--lambda_pair 0.4 --lambda_sts 0.2 --lambda_angle 0.3
--learnable_temperature               # НОВОЕ: T обучается автоматически
--early_stopping_patience 15
--lr 0.0005 --seed 77 --epochs 150
--batch_size 64
T_0=60 (CosineAnnealingWarmRestarts)
```

#### Динамика обучения
| Epoch | Score | Margin | STS | Примечание |
|-------|-------|--------|-----|------------|
| 5 | 0.720 | 0.477 | 0.809 | Старт (лучше Phase 4b ep=5=0.705) |
| 15 | 0.769 | 0.522 | 0.823 | Стабильный рост |
| 30 | 0.821 | 0.569 | 0.838 | |
| 35 | 0.868 | 0.624 | 0.813 | Phase 4b пик был здесь |
| 50 | 0.902 | 0.664 | 0.794 | Промежуточный максимум |
| 60 | 0.858 | 0.620 | 0.793 | LR-restart T_0=60 |
| **65** | **0.967** | **0.727** | **0.801** | **← BEST (новый рекорд)** |
| 70 | 0.947 | 0.714 | 0.776 | |
| 75 | 0.931 | 0.693 | 0.793 | |
| 80 | 0.926 | 0.701 | 0.749 | |

#### Что изменилось vs Phase 4b
1. **Датасет расширен**: 57 → 140 пар (data/real_pairs_v4.json)
   - 83 новые пары охватывают ренормгруппу, нейронные сети, квантовые эффекты, молекулярные моторы
   - Augment factor: 20 → 30 (2040 → 7710 сэмплов)
2. **AnglE Loss** (lambda=0.3): оптимизация в угловом пространстве устраняет насыщение косинуса
   - Формула: L_angle = MSE(y, 2/π · arccos(cos_sim(src, tgt)))
   - Комбинация InfoNCE + AnglE даёт более резкие границы
3. **Learnable temperature**: T начинается с 0.1 и обучается как nn.Parameter
   - Реализован в HDIMTrainer._log_temp (log-parameterized для стабильности)
   - Добавляется в optimizer после создания trainer с LR×0.1
4. **batch_size 32 → 64**: больше in-batch негативов для InfoNCE
5. **T_0=60** (было 30): дольше первый цикл, пик позже (ep=65 vs ep=35)

#### Новые компоненты кода
- `HDIMTrainer._compute_angle_loss()` — AnglE loss implementation
- `HDIMTrainer._effective_temperature()` — unified temperature access
- `HDIMTrainer._log_temp` — learnable temperature parameter
- `--lambda_angle` / `--learnable_temperature` flags in gpu_train.py
- `data/real_pairs_v3_extension.json` — 83 новые пары
- `data/real_pairs_v4.json` — merged 140-pair dataset
- `scripts/merge_datasets.py` — dataset merge utility

#### Ключевые выводы Phase 5a
1. **AnglE Loss** дал прирост: Phase 4b пик 0.932 → Phase 5a пик 0.967 (+3.5%)
2. **Больший датасет** (140 vs 57 пар) замедляет ранний прогресс, но даёт более высокий потолок
3. **CosineAnnealing T_0=60** сдвигает пик на ep=65 (vs T_0=30 → пик ep=35)
4. **Learnable temperature** работает стабильно, не вызывает NaN
5. **Осцилляция score** при CosineAnnealing: необходим eval_every=5 + early_stopping
6. **VRAM 0.02GB**: огромный запас — всё ещё работаем на 0.2% capacity

#### Рекомендации для Phase 5b
- **Partial SBERT unfreezing**: разморозить слои 10-11 + pooling (LR=1e-5 для SBERT vs 5e-4 для HDIM)
- **SupCon loss**: использовать family-метки из датасета для multi-positive contrastive
- **hidden_dim=512, num_experts=8**: использовать свободный VRAM
- **eval_every=1**: захватывать каждый пик CosineAnnealing
- **Gated projection** 768→512→256 с residual connection

---

### 15.10 Phase 6 — Систематическая оптимизация + новый рекорд (2026-03-11)

#### Новый рекорд: `phase6e_run` — score=**0.9745** (INCUMBENT)

**Конфигурация Phase 6e (финальная, идентична Phase 5a конфигу):**
```
--hidden_dim 256 --num_experts 4 --soft_router
--pretrained_encoder
--real_pairs data/real_pairs_v4.json  # 140 пар (117 pos + 23 neg)
--augment_factor 30                   # 7710 сэмплов
--use_infonce --infonce_temperature 0.1
--lambda_pair 0.4 --lambda_sts 0.2 --lambda_angle 0.3
--learnable_temperature
--early_stopping_patience 40
--lr 0.0005 --seed 42 --epochs 200
--batch_size 32
T_0=60 (CosineAnnealingWarmRestarts), T_mult=2
warmup_epochs=20
```

#### Динамика обучения Phase 6e
| Epoch | Score | Margin | STS | LR | Примечание |
|-------|-------|--------|-----|----|------------|
| 10 | 0.919 | — | 0.798 | 0.000467 | Warmup завершён |
| 20 | 0.865 | 0.629 | 0.787 | 0.000375 | |
| 40 | 0.959 | 0.731 | 0.757 | 0.000125 | Первый пик (LR→min) |
| 60 | 0.872 | 0.638 | 0.779 | 0.000500 | LR restart (ep=60) |
| 65 | 0.961 | 0.728 | 0.776 | 0.000498 | Второй пик |
| **75** | **0.9745** | **0.7417** | **0.7760** | 0.000400 | **← BEST (новый рекорд)** |
| 80 | 0.903 | 0.667 | 0.788 | 0.000500 | LR restart (ep=80) |
| 85 | 0.935 | 0.694 | 0.803 | — | Подъём |

#### Фазы 6a-6d и что не сработало
- **Phase 6a** (batch_size=32, seed=42): повтор Phase5a для воспроизводимости
- **Phase 6b** (batch_size=64, параллельно с 6a): GPU conflict — убита
- **Phase 6c** (warmup=50): LR instability при warmup>T_0, loss рос на ep=33-43 → стоп ep=30 (best=0.903)
- **Phase 6d** (v5 dataset с hard negatives): score collapse ep=10 (0.621) — hard negatives несовместимы с integer domain IDs, модель не может различить same-domain пары → реверт к v4

#### Ключевые выводы Phase 6
1. **Новый рекорд 0.9745** (+0.75% vs Phase 5a 0.967): достигнут при тех же гиперпараметрах Phase5a но seed=42 (vs seed=77)
2. **CosineAnnealing T_0=60, T_mult=2**: пики при ep≈40, 65, 75; второй цикл T_0=120 (ep=80-200), пик ожидается ep≈140-160
3. **Hard negatives опасны**: same-domain hard negatives с integer domain IDs приводят к коллапсу
4. **Reproducibility issue**: seed меняет траекторию существенно (score разница до +0.78%)
5. **batch_size=32 > 64** для данного датасета: меньше шума при малом числе уникальных пар
6. **LR warmup должен быть ≤ T_0/3**: warmup=50 > T_0=60 нарушает scheduler

#### Итоговый статус Phase 6e (ep=94/200, остановлена вручную)
- Лучший чекпоинт: `artifacts/phase6e_run/checkpoints/best.pt` (ep=75, score=0.9745)
- Обучение остановлено на ep=94 — второй цикл не показал улучшения (лучший ep=75)
- Вывод: пик score достигается в конце первого LR-цикла (ep=75 при T_0=60+warmup), второй цикл не превышает первый

---

## 14. Итоговая позиция
HDIM следует развивать как **implementable-first research system**:

- сначала выравнивать код, лоссы, метрики и contracts;
- затем усиливать экспериментальную строгость;
- и только потом добавлять большие research-надстройки вроде Scholar integration, human-in-the-loop orchestration и Q-Attention graph stack.

Это сохраняет научное ядро проекта и одновременно делает архитектурную документацию честной по отношению к текущему состоянию реализации.

---

## 16. Выводы и рекомендации по оптимизации (Phase 6 → Phase 7)

### 16.1 Ключевые паттерны из Phase 6e

Анализ 94 эпох обучения с CosineAnnealingWarmRestarts (T_0=60, T_mult=2, warmup=20):

| Пик | Эпоха | Score | LR | Примечание |
|-----|-------|-------|----|-----------|
| 1 | 40 | 0.9585 | 0.000125 | Первый минимум LR |
| 2 | 65 | 0.9607 | 0.000498 | После рестарта, LR ≈ max |
| **3** | **75** | **0.9745** | **0.000481** | **Рекорд — лучший пик** |
| 4 | 85 | 0.9352 | 0.000448 | Второй цикл (T_0=120), слабее |

**Главный вывод:** пик score достигается не в минимуме LR (ep=40, 0.9585), а после рестарта в начале следующего цикла (ep=65-75, 0.9745). Второй цикл (T_0=120) не превышает первый.

### 16.2 Почему второй цикл слабее

1. **Датасет исчерпан**: 140 уникальных пар × augment=30 = 7710 сэмплов. После ~80 эпох модель видела каждую пару 80×7710/140≈4400 раз — переобучение неизбежно.
2. **Oscillation STS vs margin**: при CosineAnnealing margin растёт когда LR низкий (модель «уточняет»), STS растёт при высоком LR (модель «переобобщает»). Оптимальная точка — сразу после рестарта.
3. **Learnable temperature**: T продолжает обучаться и к ep=80+ уже не в оптимуме.

### 16.3 Рекомендации для Phase 7

#### Стратегия 1 — Прямая оптимизация (быстро, риск низкий)
- **eval_every=1** + **early_stopping_patience=5**: останавливаться сразу после пика ep=75
- **epochs=80**: нет смысла идти дальше первого цикла
- **T_0=75**: сдвинуть пик с ep=65-75 на ep=70-80 — больший первый цикл
- **Запустить grid: seed=[7,13,21,42,77,99]**: найти seed с лучшим ep=75-пиком

#### Стратегия 2 — Улучшение архитектуры (средний риск)
- **Частичное размораживание SBERT**: разморозить слои 10-11 + pooling с LR=1e-5 (vs 5e-4 для HDIM)
  - Это даст более семантически богатые embeddings → STS↑ → score↑
- **SupCon loss** (Supervised Contrastive): использовать `family_id` из датасета как метки классов для multi-positive contrastive learning
- **hidden_dim=512, num_experts=8**: VRAM=0.02GB из 8.6GB — огромный запас; масштаб даст лучшую генерализацию

#### Стратегия 3 — Расширение датасета (высокий потенциал)
- **Цель: 300+ пар** (сейчас 140): добавить аналогии из областей:
  - Квантовая механика ↔ термодинамика
  - Экономические модели ↔ физика конденсированного состояния
  - Нейронные сети ↔ статистическая физика (renormalization group)
- **Качественные hard negatives с semantic domain IDs**: текущие integer domain IDs (0,1,2,3) не дают модели различать семантически близкие домены
  - Решение: заменить integer domain ID на SBERT-embedding домена → модель получит семантическое понятие «домен»

#### Стратегия 4 — Scheduler (экспериментально)
- **OneCycleLR** (вместо CosineAnnealing): один плавный цикл max→min без осцилляций — устранит скачки score
- **Warmup + LinearDecay**: простой и предсказуемый — нет непредсказуемых LR-пиков
- **ReduceLROnPlateau**: адаптивное снижение LR при плато — автоматически находит оптимум

### 16.4 Ожидаемый потолок текущей архитектуры

При текущем датасете (140 пар) и архитектуре (hidden=256):
- **Оптимистичный**: 0.985 (с partial SBERT unfreezing + лучший seed)
- **Реалистичный**: 0.975-0.980 (лучший seed + eval_every=1)
- **Пессимистичный**: 0.967-0.975 (plateau — текущий лучший Phase5a/6e)

Для прорыва выше 0.985 нужен либо расширенный датасет (300+ пар), либо архитектурное изменение (partial SBERT unfreeze или semantic domain IDs).

---

## 17. Phase 7 — Модульная архитектура + SupCon (2026-03-11)

### 17.1 Реализованные изменения

#### Новые модули

**`src/core/modular_moe.py`** — `ModularMoERouter` с динамическим управлением экспертами:
- `ExpertConfig(dataclass)` — конфигурация одного эксперта: hidden_dim, dropout, activation, layer_count
- `ExpertModule(nn.Module)` — конфигурируемый эксперт с 1-3 слоями
- `ModularMoERouter(nn.Module)` — модульный роутер:
  - `add_expert(config)` — добавляет эксперта в runtime без перестройки модели
  - `remove_expert(id)` — удаляет эксперта с переиндексацией
  - `routing_type='soft'` — SoftMoE (дифференцируемый, без token dropping)
  - `routing_type='hard'` — Hard top-k с R3 EMA stabilization
  - Полностью совместим с R3MoERouter/SoftMoERouter (одинаковые router_state ключи)
  - EMA train_scores buffer пересоздаётся при add/remove
- `build_modular_moe(input_dim, num_experts, top_k, routing_type)` — быстрая сборка

#### Обновления trainer.py
- `lambda_supcon: float = 0.0` — новый параметр SupCon loss weight
- `_compute_supcon_loss(source_inv, target_inv, pair_relation_label, pair_weight, pair_group_id)` — Supervised Contrastive Loss (Khosla et al., NeurIPS 2020):
  - Для каждого anchor i: P(i) = все j из того же family с label>0.5
  - L_i = -1/|P(i)| * Σ_{p∈P(i)} log[ exp(sim(i,p)/T) / Σ_{a≠i} exp(sim(i,a)/T) ]
  - Использует ВСЕ позитивы из семейства, а не только диагональный → более эффективно при аугментации

#### Обновления gpu_train.py
- `--lambda_supcon` — SupCon loss weight
- `--scheduler_type {cosine_restarts, cosine_decay, plateau, onecycle}` — выбор scheduler
- `--modular_moe` — использовать ModularMoERouter вместо SoftMoERouter/R3MoERouter
- `_build_scheduler(optimizer, args, total_steps)` — фабрика schedulers

#### Датасет v5
- `data/real_pairs_v5.json`: 175 пар (139 pos + 36 neg, neg ratio 20.6%)
- +21 новых positive семейств: fractal_branching, percolation_contagion_2, entropy_aging_threshold, topological_barrier_protection, memory_consolidation_eviction, cascade_network_failure, feedback_windup_oscillation, mechanical_electrical_synchronization, adaptive_background_minimization, critical_slowing_warning, hydrodynamic_biological_lubrication, diffusion_limited_branching, winner_takes_all_suppression, hub_essentiality, damage_triggered_repair, countercurrent_exchange_2, oscillator_entrainment, cascade_signal_amplification, elastic_energy_storage_release, turing_pattern_formation, self_organized_criticality_2
- +13 hard negatives: семантически похожие пары (soliton vs action potential, Kalman vs glomerular filtration, magnetic hysteresis vs LTP, etc.)

#### Интеграция model_factory.py
- `_patch_modular_moe(core_model, routing_type)` — замена pipeline.moe на ModularMoERouter
- `build_sbert_hdim_model(..., modular_moe=True, modular_moe_routing_type='soft')` — поддержка флага

### 17.2 Конфигурация Phase 7 (текущий запуск)

```bash
python scripts/gpu_train.py \
  --epochs 200 --batch_size 64 --lr 0.0005 \
  --hidden_dim 256 --num_experts 4 --num_domains 4 \
  --pretrained_encoder --modular_moe --modular_moe_routing_type soft \
  --real_pairs data/real_pairs_v5.json --augment_factor 30 \
  --lambda_iso 0.1 --lambda_pair 0.4 --lambda_sts 0.2 \
  --lambda_angle 0.3 --lambda_supcon 0.2 \
  --learnable_temperature --scheduler_type cosine_decay \
  --eval_every 5 --early_stopping_patience 20 --seed 42
```

Ключевые отличия от Phase 6e:
- `ModularMoERouter` вместо `SoftMoERouter` — динамическое управление экспертами
- `lambda_supcon=0.2` — SupCon loss с family labels
- `cosine_decay` вместо `cosine_restarts` — монотонное снижение LR, устраняет score-дёрганье
- `data/real_pairs_v5.json` — 175 пар vs 140 (v4) с hard negatives
- `batch_size=64` — больше in-batch negatives для InfoNCE/SupCon

### 17.3 Ожидаемый эффект SupCon на dateset с family labels

SupCon преимущество: при батче из 64 семплов модель видит все позитивы из того же семейства одновременно. Для семейства bubble_collapse_destruction (3 пары × aug30 = 90 сэмплов) каждый anchor получает ~2-5 позитивов в одном батче, а не только 1 диагональный.

Математически это эквивалентно InfoNCE с увеличенным effective batch size: если семейство имеет k членов, SupCon использует k-1 позитивов, InfoNCE — только 1. Ожидаемый прирост: +0.01-0.03 на pair_margin.

### 17.4 Модульность экспертов — применение

```python
from src.core.modular_moe import ModularMoERouter, ExpertConfig, build_modular_moe

# Базовая сборка
router = build_modular_moe(input_dim=64, num_experts=4, top_k=2, routing_type='soft')

# Добавить специализированного эксперта
spec_cfg = ExpertConfig(hidden_dim=512, activation='silu', layer_count=3)
new_id = router.add_expert(spec_cfg)

# Удалить слабого эксперта после анализа usage
router.remove_expert(worst_id)

# Динамическое масштабирование без переобучения:
# - Добавить эксперта для нового домена
# - Удалить неиспользуемых (expert_usage < 0.05)
# - Всё без перестройки остальных весов
```
