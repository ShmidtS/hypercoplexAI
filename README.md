# hypercoplexAI — HDIM: Hypercomplex Domain Isomorphism Machine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Research](https://img.shields.io/badge/status-research%20prototype-orange.svg)]()

> **Best score: 1.1370** (Phase 8e, ep45) — `pair_margin=0.906`, `STS=0.770`
> **Current:** Phase 19 running (ep11, score=0.489) | Phase 20 prepared (DCL + Uniformity)

---

## Overview

**HDIM** (Hypercomplex Domain Isomorphism Machine) is a research system for
**cross-domain structural analogy search** via hypercomplex invariants.

Unlike LLMs that compare texts by token proximity, HDIM finds **structural isomorphisms**
between problems from entirely different domains with different vocabularies but the same
deep structure.

**Canonical example:** Cavitation erosion (engineering) vs dental plaque removal (dentistry).
Different words, identical physics. Standard embedding models miss this; HDIM finds it via
a domain-invariant hypercomplex representation that strips domain vocabulary and preserves
structural topology.

---

## Key Features

- **Clifford Algebra Cl(3,1,0)** — degenerate algebra, `clifford_dim=16` multivectors,
  Cayley-table geometric product (precomputed, no runtime overhead)
- **Structural Invariant Extraction** — `U_inv = R_inv * G_A * R` (sandwich product)
  strips domain signature, exposing pure structural topology
- **Domain Transfer** — `G_B = R_B * U_inv * R_B_inv` reconstructs target-domain
  multivector from invariant
- **Titans Memory (TTT)** — test-time training associative memory with fp32-safe AMP path
- **Soft MoE Routing** — Puigcerver et al. ICLR 2024, differentiable dispatch without
  token dropping; Router Z-Loss prevents mode collapse
- **Focal-InfoNCE** — gamma applied only to denominator (Phase 17 fix C6)
- **DCL + Uniformity+Alignment** — Phase 20 additions (Yeh et al. 2022, Wang & Isola 2020)
- **Frozen SBERT + trainable MLP** — `paraphrase-multilingual-mpnet-base-v2` (768-dim)
  projected through `Linear(768->384->256)` into Clifford space

---

## Architecture

```
                    TEXT INPUT
                        |
           +------------v-----------+
           |   SBERT (frozen)       |
           |   paraphrase-multi-    |
           |   lingual-mpnet-       |
           |   base-v2 [768-dim]    |
           +------------+-----------+
                        |
           +------------v-----------+
           |   SimpleMLP (trainable)|
           |   768 -> 384 -> 256    |
           |   + LayerNorm + GELU   |
           +------------+-----------+
                        |
                   [256-dim]
                        |
           +------------v-----------+
           |  InvariantExtractor    |
           |  U_inv = R_inv*G_A*R   |
           |  DomainRotor per domain|
           +------+--------+--------+
                  |        |
       +----------v--+  +--v----------+
       | TitansMemory|  | SoftMoERouter|
       | (TTT, fp32) |  | 4 experts   |
       +----------+--+  +--+-----------+
                  |        |
                  +---+----+
                      |
           +----------v----------+
           |   DecoupledDecoder  |
           |   256 -> 384 -> 768 |
           +----------+----------+
                      |
                 OUTPUT VECTOR
```

### Mathematical Contract

| Operation | Formula | Description |
|-----------|---------|-------------|
| Encode A | `G_A = MLP(SBERT(text_A))` | Domain A multivector |
| Extract invariant | `U = R_inv x G_A x R` | Strip domain signature |
| Transfer to B | `G_B = R_B x U x R_B_inv` | Reconstruct in domain B |
| Isomorphism loss | `L_iso = MSE(G_B, G_B_target)` | Structural match penalty |
| PRIMARY score | `pair_margin * 1.0 + STS * 0.3` | Training objective |

---

## Quick Start

### Prerequisites

```bash
pip install torch>=2.0 sentence-transformers>=2.2 numpy scipy
```

### Installation

```bash
git clone https://github.com/your-org/hypercoplexAI.git
cd hypercoplexAI
pip install -r requirements.txt
```

### Demo

```bash
# Run full component demo
python hdim_demo.py

# GPU training (PRIMARY mode)
python scripts/gpu_train.py \
    --use_sbert \
    --sbert_model paraphrase-multilingual-mpnet-base-v2 \
    --epochs 200 \
    --batch_size 32 \
    --lr 3e-4
```

### Phase 20 Launch (DCL + Uniformity)

```bash
# Windows
scripts\phase20_train.bat

# Linux/Mac
bash scripts/phase20_train.sh
```

---

## Project Structure

```
hypercoplexAI/
|-- src/
|   |-- core/
|   |   |-- hypercomplex.py       # CliffordAlgebra, QuaternionLinear, PHMLinear
|   |   |-- domain_operators.py   # DomainRotor, InvariantExtractor, DomainRegistry
|   |   |-- hdim_pipeline.py      # HDIMPipeline orchestrator, TransferState
|   |   |-- titans_memory.py      # TitansMemoryModule (TTT)
|   |   |-- hierarchical_memory.py# HierarchicalTitansMemory (2-level)
|   |   |-- soft_moe_router.py    # SoftMoERouter (DEFAULT)
|   |   `-- modular_moe.py        # ModularMoERouter (experimental)
|   |-- models/
|   |   |-- hdim_model.py         # HDIMModel, HDIMConfig, HDIMAuxState
|   |   |-- advanced_text_encoder.py # RotaryEmb, TransformerBlock, AdvancedTextEncoder
|   |   |-- text_hdim_model.py    # TextHDIMModel (SBERT integration)
|   |   |-- sbert_encoder.py      # SBERTEncoder wrapper
|   |   |-- metrics.py            # STS, DRS, AFR, pair_margin metrics
|   |   `-- model_factory.py      # build_model() factory function
|   `-- training/
|       |-- trainer.py            # HDIMTrainer (1083 lines)
|       |-- train.py              # CLI entry point
|       |-- dataset.py            # HDIMDataset + collation
|       |-- real_dataset.py       # Real-world analogy pairs dataset
|       |-- experiment_config.py  # ExperimentConfig dataclass
|       |-- experiment_runner.py  # Auto-experiment runner
|       `-- results_logger.py     # CSV/JSON logging
|-- scripts/
|   |-- gpu_train.py              # PRIMARY training script (742 lines)
|   |-- auto_tune.py              # Hyperparameter search
|   |-- autoresearch_loop.py      # Autonomous research loop
|   |-- gen_dataset_v6.py         # Dataset generation v6
|   |-- phase20_train.bat         # Phase 20 launch config
|   `-- phase17_train.bat         # Phase 17 launch config
|-- tests/
|   |-- test_hdim.py              # Main test suite (962 lines)
|   `-- test_new_components.py    # Component tests (461 lines)
|-- hdim_demo.py                  # Full demo script
|-- test_phase17.py               # Phase 17 regression tests
|-- requirements.txt              # Python dependencies
|-- HDIM.md                       # Full technical specification
`-- README.md                     # This file
```

---

## Core Components

### CliffordAlgebra (`src/core/hypercomplex.py`)

Implements degenerate Clifford algebra `Cl(3,1,0)` with `clifford_dim=16`:

- `_build_cayley_table()` — precomputes 16x16 sign/index table for geometric product
- `geometric_product(a, b)` — batched multivector product using Cayley table
- `sandwich(R, x)` — computes `R * x * R_reverse` for invariant extraction
- `QuaternionLinear` — Hamilton product weight parameterization
- `PHMLinear` — Parameterized Hypercomplex Multiplication (Zhang et al. 2021)
- `QLayerNorm` — quaternion-aware layer normalization
- `DomainRotor` — learnable rotor with identity initialization

### InvariantExtractor (`src/core/domain_operators.py`)

Extracts structural invariants via sandwich product:

```python
# U_inv = R^{-1} * G_A * R  (strips domain signature)
U_inv = extractor.extract(G_A, domain="engineering")

# G_B = R_B * U_inv * R_B_inv
G_B = extractor.transfer(U_inv, target_domain="dentistry")
```

### TitansMemoryModule (`src/core/titans_memory.py`)

Test-Time Training associative memory:

- `update(k, v)` — gradient-based memory write with momentum
- `retrieve(k)` — associative lookup without modifying memory
- `retrieve_and_update(k, v)` — atomic read-write
- `reset_memory(strategy='geometric')` — smart reset preserving important patterns
- `stabilize_momentum()` — force-normalize momentum on LR restarts
- **AMP safe**: critical paths cast to `float32` before memory operations

### SoftMoERouter (`src/core/soft_moe_router.py`)

Default router (Puigcerver et al. ICLR 2024):

- Fully differentiable dispatch/combine — no token dropping
- Router Z-Loss: `z_loss = mean((logsumexp(logits))^2)` prevents mode collapse
- `SoftRouterState` dataclass tracks dispatch weights, entropy, load balance
- 4 experts by default; configurable via `HDIMConfig.moe_num_experts`

### HDIMModel (`src/models/hdim_model.py`)

Main model class with full lifecycle management:

- `forward(batch)` — full forward pass returning `HDIMAuxState`
- `transfer(text_A, domain_A, domain_B)` — cross-domain transfer inference
- `transfer_pairs(pairs)` — batch transfer for evaluation
- `add_domain(name)` / `remove_domain(name)` — dynamic domain registry
- `reset_memory(strategy)` — smart memory reset between epochs
- `HDIMConfig` — frozen dataclass with all hyperparameters

---

## Training

### Loss Suite (`src/training/trainer.py`)

The trainer implements a comprehensive multi-objective loss:

| Loss | Weight | Phase introduced | Description |
|------|--------|-----------------|-------------|
| `iso_loss` | 1.0 | Phase 1 | MSE isomorphism (structural match) |
| `recon_loss` | 0.5 | Phase 1 | Decoder reconstruction quality |
| `pair_margin` | 1.0 | Phase 3 | Margin-based pair ranking |
| `infonce` | 0.3 | Phase 5 | InfoNCE contrastive |
| `focal_infonce` | 0.3 | Phase 17 | Focal-InfoNCE (gamma=denom only) |
| `angle_loss` | 0.2 | Phase 11 | AnglE loss (Li & Li 2023) |
| `diversity` | 0.1 | Phase 7 | Entropy diversity regularizer |
| `router_z_loss` | 0.01 | Phase 9 | MoE Z-Loss anti-collapse |
| `dcl_loss` | 0.2 | Phase 20 | Decoupled Contrastive Learning |
| `uniformity` | 0.1 | Phase 20 | Uniformity+Alignment (Wang & Isola) |

### PRIMARY Score

```
PRIMARY_SCORE = pair_margin * 1.0 + STS_exported * 0.3
```

Best achieved: **1.1370** (Phase 8e, ep45, `pair_margin=0.906`, `STS=0.770`)

### Phase History

| Phase | Key Change | Best Score |
|-------|-----------|------------|
| 1-3 | Baseline, quaternion MLP, pair loss | 0.3xx |
| 4-6 | InfoNCE, diversity, MoE intro | 0.5xx |
| 7-8e | SoftMoE, angle loss, hard negatives | **1.1370** |
| 9-12 | Memory, hierarchical, scheduler | 0.7-0.9xx |
| 13-16 | Architecture search, encoder variants | 0.6-0.8xx |
| 17 | Critical fixes C1-C7 (P0 bugs) | 0.85x |
| 18-19 | Regularization, TTT improvements | 0.489 (ep11) |
| 20 | DCL + Uniformity+Alignment | In progress |

### Phase 17 Critical Fixes

Seven P0 bugs fixed that caused training instability:

- **C1** — Memory drift: clamp momentum_S norm to prevent explosion
- **C2** — MoE collapse: Z-Loss weight increased, entropy floor added
- **C3** — TTT AMP: cast to fp32 before memory gradient step
- **C4** — InfoNCE temperature: added `clamp(min=0.05)` for stability
- **C5** — Hard negative mining: fix off-by-one index in pair extraction
- **C6** — Focal-InfoNCE: apply gamma to denominator only (not numerator)
- **C7** — DomainRotor: normalize R before sandwich to prevent norm explosion

---

## Configuration

### HDIMConfig Parameters

```python
from src.models.hdim_model import HDIMConfig

config = HDIMConfig(
    input_dim=256,          # MLP output dimension
    clifford_dim=16,        # Cl(3,1,0) multivector dimension
    hidden_dim=128,         # Internal hidden dimension
    num_domains=4,          # Number of domain rotors
    moe_num_experts=4,      # SoftMoE expert count
    use_titans_memory=True, # Enable TTT memory
    use_soft_moe=True,      # Use SoftMoERouter (default)
    memory_key_dim=64,      # Titans key dimension
    memory_val_dim=128,     # Titans value dimension
)
```

### ExperimentConfig

```python
from src.training.experiment_config import ExperimentConfig

config = ExperimentConfig.from_json("configs/phase20.json")
hdim_kwargs = config.to_hdim_config_kwargs()
```

---

## Metrics

All metrics computed in `src/models/metrics.py`:

| Metric | Description | Target |
|--------|-------------|--------|
| `STS_exported` | Spearman correlation on STS benchmark (exported invariants) | > 0.7 |
| `STS_training` | STS on training-mode embeddings | > 0.75 |
| `pair_margin` | Mean margin between positive and negative pairs | > 0.8 |
| `DRS` | Domain Routing Stability (EMA of routing entropy) | > 0.6 |
| `AFR` | Analogy Feasibility Rate (transfer quality) | > 0.5 |
| `PRIMARY_SCORE` | `pair_margin + 0.3 * STS_exported` | > 1.0 |

---

## Requirements

### Python Packages

```
torch>=2.0
sentence-transformers>=2.2
numpy>=1.24
scipy>=1.10
transformers>=4.30
sklearn
tqdm
```

### System Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3080 or better recommended)
- **CUDA**: 11.8+ (for AMP / fp16 training)
- **RAM**: 16GB+ system RAM
- **Python**: 3.10+
- **OS**: Windows 10/11 or Linux (Ubuntu 20.04+)

### Implicit Dependencies

- `sentence_transformers` — SBERT encoder, downloaded on first use
- `huggingface_hub` — model weight download
- `torch.cuda.amp` — automatic mixed precision (requires CUDA)

---

## Development Status

### Production-Ready Components

- `CliffordAlgebra` — stable, fully tested
- `TitansMemoryModule` — stable with AMP-safe path
- `SoftMoERouter` — stable, default router
- `HDIMTrainer` — stable with full loss suite
- `SBERTEncoder` — stable, frozen weights

### Under Development

- `HierarchicalTitansMemory` — 2-level memory, experimental
- `ModularMoERouter` — dynamic expert add/remove, experimental
- `AdvancedTextEncoder` — custom transformer, not yet competitive with SBERT
- Phase 20 losses (DCL, Uniformity) — just added, tuning in progress

### Research Directions

- **Phase 21**: Explore learned Clifford basis (trainable `p,q,r` parameters)
- **Phase 22**: Multi-hop analogy chains (A->B->C transfer)
- **Phase 23**: Few-shot domain adaptation with 1-5 examples

---

## Known Issues

### Anti-Patterns (Do NOT)

- Do not use `ModularMoERouter` in production — use `SoftMoERouter`
- Do not disable Z-Loss (`moe_z_loss_weight=0`) — causes MoE collapse within 10 epochs
- Do not use `AdvancedTextEncoder` as primary encoder — SBERT outperforms it significantly
- Do not set `batch_size < 16` — InfoNCE requires sufficient negatives in batch
- Do not use `strategy='zero'` for memory reset — destroys learned associations

### Known Numerical Issues

- `TitansMemoryModule.momentum_S` can diverge if `stabilize_momentum()` not called after LR restart
- `CliffordAlgebra.norm()` returns scalar per sample; ensure not dividing by zero in normalization
- Focal-InfoNCE gamma > 3.0 causes gradient vanishing on easy pairs

### Platform Notes

- Windows: use `.bat` scripts; `\` path separators
- AMP (`--use_amp`) requires CUDA; falls back to fp32 on CPU
- `gen_dataset_v6.py` requires OpenAI API key in environment (`OPENAI_API_KEY`)

---

## Citation

If you use HDIM in research, please cite:

```bibtex
@software{hdim2026,
  title  = {HDIM: Hypercomplex Domain Isomorphism Machine},
  year   = {2026},
  url    = {https://github.com/your-org/hypercoplexAI}
}
```

**Key references:**
- Puigcerver et al. (2024) — Soft MoE: *From Sparse to Soft Mixtures of Experts*
- Gu & Dao (2023) — Titans Memory (Test-Time Training)
- Yeh et al. (2022) — Decoupled Contrastive Learning (DCL)
- Wang & Isola (2020) — Understanding Contrastive Representation Learning (Uniformity+Alignment)
- Zhang et al. (2021) — Parameterized Hypercomplex Multiplication (PHM)
- Li & Li (2023) — AnglE Loss for text embeddings

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Generated: 2026-03-13 | Research prototype — API may change between phases*