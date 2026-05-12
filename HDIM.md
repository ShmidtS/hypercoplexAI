# HDIM — Hypercomplex Domain Isomorphism Machine
*Версия: 30.0+ | Дата: 2026-04-09 | **РЕКОРД: score=1.1814** (Run 18, ep13, temp=0.10, λ_pair=0.40, margin=1.0224) | Phase 30+: MoEKernel + Hallucination detection + Online learning + Memory persistence | Run pytest and numerical verification for current status*

---

> **Deprecated:** этот файл сохранён как historical pointer. Канонические и актуальные источники: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) для архитектуры и [`README.md`](README.md) для команд, статуса и quickstart.

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

### Проекция в целевой домен
$$G_B = R_B \otimes U \otimes R_B^{-1}$$

### Loss функция
$$L_{total} = L_{recon} + \lambda_{iso} L_{iso} + \lambda_{pair} L_{pair} + \lambda_{routing} L_{routing} + \lambda_z L_z + L_{memory} + \lambda_{expert\_ortho} L_{expert\_ortho} + \lambda_{dcl} L_{dcl} + \lambda_{uniformity} L_{uniformity}$$

Где $L_{pair}$ = Focal-InfoNCE/InfoNCE + AnglE + SupCon (при наличии family labels).
$L_z = (\log\sum_j e^{z_j})^2$ — Router Z-Loss для стабильности MoE (ST-MoE, Zoph et al. 2022).

### Similarity-Preserving Router Loss (Phase 21)
$$L_{balance} = -\sum_i \sum_j sim(x_i, x_j) \cdot sim(r_i, r_j)$$

Где $sim(x_i, x_j)$ — косинусная близость входных представлений, $sim(r_i, r_j)$ — косинусная близость routing-векторов. Loss штрафует роутер, если маршрутизация не отражает структурную близость входов.
