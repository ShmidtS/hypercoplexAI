# HDIM: Hypercomplex Domain Isomorphism Machine

[Python 3.10+](https://www.python.org/downloads/)
[PyTorch 2.0+](https://pytorch.org/)  
[Lean4](formalization/verify_numerical.py)
[Tests](tests/)
[![Project Status](https://img.shields.io/badge/status-active-success.svg)](https://github.com/hypercoplex/hdim/issues) • [Contributing](CONTRIBUTING.md)

> **Best Score:** 1.1814 (Run 18, epoch 13, temp=0.10, lambda_pair=0.40) — `pair_margin=1.0224`, `STS=0.537`
> **Phase 30+:** MoEKernel + Hallucination detection + Online learning
> **Numerical validation:** pytest suite and Lean4 formalization available; run verification commands below for current status
> **Features:** MoEKernel (math/language/code/science) + SharedExpert + AuxLossFree + ExpertOrtho + HallucinationDetector + OnlineLearner + OnlineLoRA

---

## Table of Contents

- [What is HDIM?](#what-is-hdim)
- [Why Hypercomplex Representations?](#why-hypercomplex-representations)
- [Core Innovation](#core-innovation-domain-invariant-structural-encoding)
- [Architecture](#architecture)
- [Key Components](#key-components)
- [Mathematical Foundation](#mathematical-foundation)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Verification](#verification)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [References](#references)
- [Citation](#citation)

---

## What is HDIM?

**HDIM (Hypercomplex Domain Isomorphism Machine)** is a research system that discovers **structural analogies between problems from completely different domains** by representing their underlying structure in a domain-invariant hypercomplex space.

### The Problem It Solves

Standard text embedding models (BERT, SBERT, etc.) compare texts by **token proximity and semantic similarity**. They cannot recognize that:

> *"Cavitation erosion in hydraulic turbines"* (engineering)
> and
> *"Dental plaque removal by ultrasonic scaling"* (dentistry)

...describe **the same physical phenomenon** — bubble collapse causing surface damage. Different vocabulary, identical structure.

HDIM solves this by:

1. **Stripping domain-specific vocabulary** through hypercomplex algebra
2. **Preserving structural topology** in a domain-invariant representation
3. **Finding isomorphisms** between problems with matching structure

### Canonical Example


| Domain A (Engineering)                                                            | Domain B (Dentistry)       |
| --------------------------------------------------------------------------------- | -------------------------- |
| Cavitation erosion                                                                | Ultrasonic plaque removal  |
| Bubble collapse dynamics                                                          | Micro-cavitation in fluids |
| Surface pitting                                                                   | Enamel erosion             |
| **Shared Physics:** Rapid pressure changes cause bubble collapse → surface damage |                            |


Standard embeddings miss this connection. HDIM finds it via the **sandwich product**:

```
U_inv = R⁻¹ ⊗ G_A ⊗ R   (extracts invariant structure)
G_B = R_B ⊗ U_inv ⊗ R_B⁻¹   (transfers to target domain)
```

---

## Why Hypercomplex Representations?

### Limitations of Standard Embeddings


| Approach                | What It Captures             | What It Misses            |
| ----------------------- | ---------------------------- | ------------------------- |
| **Word2Vec/FastText**   | Word co-occurrence           | Structural relationships  |
| **BERT/SBERT**          | Semantic similarity, context | Cross-domain isomorphisms |
| **Sentence Embeddings** | Meaning similarity           | Mathematical structure    |


### What HDIM Adds

**Clifford Algebra Cl(3,1,0)** provides:

- **Multivector representations** (16-dimensional): scalar + vectors + bivectors + trivectors + pseudoscalar
- **Geometric product**: Encodes both symmetric (inner) and antisymmetric (wedge) relationships
- **Rotor-based transformations**: Norm-preserving operations for domain transfer
- **Structural invariants**: Properties preserved across domain transformations

```
Multivector:  M = m₀ + m₁e₁ + m₂e₂ + m₃e₃ + m₁₂e₁₂ + m₁₃e₁₃ + m₂₃e₂₃ + m₁₂₃e₁₂₃ + ...
Components:   scalar + vector (3D) + bivector (3D) + trivector (3D) + pseudoscalar
Dimension:    1     + 3              + 3              + 3              + 1          = 16
```

---

## Core Innovation: Domain-Invariant Structural Encoding

### The Sandwich Product

The mathematical heart of HDIM is the **sandwich product** — a conjugation operation that extracts domain-invariant structure:

```
U_inv = R⁻¹ ⊗ G_A ⊗ R
```

Where:

- `G_A` = Multivector encoding of problem in domain A
- `R` = Domain rotor (learnable rotation in hypercomplex space)
- `U_inv` = Domain-invariant structural encoding

**Key Property:** If `R` is a unit rotor (||R|| = 1), then `||U_inv|| = ||G_A||` — norm is preserved.

### Domain Transfer

To transfer from domain A to domain B:

```
G_B = R_B ⊗ U_inv ⊗ R_B⁻¹
```

This reconstructs the target domain representation from the invariant.

### Verification

Mathematical properties are numerically verified in `formalization/verify_numerical.py` (tolerance-based Python checks):


| Category              | Theorems | Status         |
| --------------------- | -------- | -------------- |
| Clifford Algebra      | 67       | PASS           |
| Sandwich Product      | 17       | PASS           |
| Involutions           | 16       | PASS           |
| Domain Transfer       | 11       | PASS           |
| HBMA Memory           | 11       | PASS           |
| SoftMoE Router        | 10       | PASS           |
| Matryoshka Embeddings | 7        | PASS           |
| Quaternion Operations | 9        | PASS           |
| Training Losses       | 5        | PASS           |
| Memory Adapters       | 5        | PASS           |
| **MoEKernel (Ph.28)** | **11**   | PASS           |
| **Total**             | Run `python formalization/verify_numerical.py` for current status | |


---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HDIM ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: Text pairs from different domains                               │
│         ┌──────────────────┐    ┌──────────────────┐                    │
│         │  "Cavitation in  │    │ "Ultrasonic      │                    │
│         │   turbines..."   │    │  plaque..."      │                    │
│         └────────┬─────────┘    └────────┬─────────┘                    │
│                  │                       │                              │
│                  ▼                       ▼                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    ENCODING LAYER                                 │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │  │
│  │  │ SBERTEncoder│──▶│ MoEKernel   │───▶│ HDIMEncoder │            │  │
│  │  │ (frozen)    │    │ (4 experts) │    │ (MLP)       │            │  │
│  │  └─────────────┘    └─────────────┘    └──────┬──────┘            │  │
│  │                                               │                   │  │
│  │                                               ▼                   │  │
│  │                                    ┌─────────────────┐            │  │
│  │                                    │ Multivector G   │            │  │
│  │                                    │ (B × 16 dim)    │            │  │
│  │                                    └────────┬────────┘            │  │
│  └─────────────────────────────────────────────┼─────────────────────┘  │
│                                                │                        │
│                                                ▼                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                 STRUCTURAL EXTRACTION                             │  │
│  │  ┌────────────────────────────────────────────────────────────┐   │  │
│  │  │            InvariantExtractor                              │   │  │
│  │  │                                                            │   │  │
│  │  │   G ──▶ R⁻¹ ⊗ G ⊗ R ──▶ U_inv (domain-invariant)         │   │  │
│  │  │                                                            │   │  │
│  │  │   • Strips domain vocabulary                               │   │  │
│  │  │   • Preserves structural topology                          │   │  │
│  │  │   • Norm-preserving (||U|| = ||G||)                        │   │  │
│  │  └────────────────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────┬─────────────────────┘  │
│                                                │                        │
│                                                ▼                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    MEMORY LAYER (Titans)                          │  │
│  │  ┌────────────────────────────────────────────────────────────┐   │  │
│  │  │              TitansMemoryModule (TTT)                      │   │  │
│  │  │                                                            │   │  │
│  │  │   • Test-Time Training for adaptive memory                 │   │  │
│  │  │   • Working Memory (16 slots)                              │   │  │
│  │  │   • Episodic Memory (64 slots)                             │   │  │
│  │  │   • Semantic Memory (64 prototypes)                        │   │  │
│  │  │   • fp32-safe AMP path                                     │   │  │
│  │  └────────────────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────┬─────────────────────┘  │
│                                                │                        │
│                                                ▼                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                 MOE ROUTING LAYER                                 │  │
│  │  ┌────────────────────────────────────────────────────────────┐   │  │
│  │  │                 SoftMoERouter                              │   │  │
│  │  │                                                            │   │  │
│  │  │   • Soft routing (no token dropping)                       │   │  │
│  │  │   • Shared Expert (DeepSeek-V3 style)                      │   │  │
│  │  │   • Aux-loss-free load balancing                           │   │  │
│  │  │   • Expert orthogonalization                               │   │  │
│  │  │                                                            │   │  │
│  │  │   Expert 1    Expert 2    Expert 3    Expert 4   Shared    │   │  │
│  │  │     │           │           │           │          │       │   │  │
│  │  │     └───────────┴───────────┴───────────┴──────────┘       │   │  │
│  │  │                           │                                │   │  │
│  │  │                           ▼                                │   │  │
│  │  │                    [Weighted Combine]                      │   │  │
│  │  └────────────────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────┬─────────────────────┘  │
│                                                │                        │
│                                                ▼                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                   DECODING LAYER                                  │  │
│  │  ┌────────────────────────────────────────────────────────────┐   │  │
│  │  │   Domain Transfer: G_B = R_B ⊗ U_inv ⊗ R_B⁻¹              │   │  │
│  │  │                                                             │   │  │
│  │  │   HDIMDecoder ──▶ Output Embedding (B × output_dim)        │   │  │
│  │  └────────────────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────┬─────────────────────┘  │
│                                                │                        │
│                                                ▼                        │
│  OUTPUT: Domain-transferred embeddings for analogy matching             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### Core Layer


| Component                  | Description                                                  | File                                |
| -------------------------- | ------------------------------------------------------------ | ----------------------------------- |
| **CliffordAlgebra**        | Cl(3,1,0) implementation with Cayley-table geometric product | `src/core/hypercomplex.py`          |
| **DomainRotationOperator** | Learnable rotor for domain transformations                   | `src/core/domain_operators.py`      |
| **InvariantExtractor**     | Sandwich product: R⁻¹⊗G⊗R                                    | `src/core/domain_operators.py`      |
| **InvariantIndex**         | Invariant search index                                      | `src/core/invariant_index.py`     |
| **CoreEngineConfig**       | Core engine configuration                                   | `src/core/engine.py`              |
| **HDIMCoreEngine**         | Minimal encode/extract/match/transfer engine                | `src/core/engine.py`              |

### Extensions Layer

| Component                  | Description                                                  | File                                      |
| -------------------------- | ------------------------------------------------------------ | ----------------------------------------- |
| **TitansMemoryModule**     | Test-Time Training memory                                    | `src/extensions/memory/titans.py`         |
| **HBMAMemory**             | 4-system memory (Working, Episodic, Semantic, Procedural)    | `src/extensions/memory/hbma.py`           |
| **SoftMoERouter**          | Soft MoE routing                                             | `src/extensions/moe/soft_router.py`       |
| **MoEKernel**              | Full MoE kernel: math/language/code/science domain experts   | `src/extensions/moe/kernel.py`            |
| **CliffordInteraction**    | Clifford interaction layer                                   | `src/extensions/moe/clifford_interaction.py` |
| **InvariantProcessor**     | Memory-based invariant processing                            | `src/extensions/memory/invariant_processor.py` |
| **HallucinationDetector**  | 5-signal weighted risk detection                             | `src/extensions/hallucination/detector.py` |
| **HallucinationFeedbackLoop** | Risk-based feedback loop                                  | `src/extensions/hallucination/feedback.py` |
| **SemanticEntropyProbe**   | Linear probe for uncertainty quantification                  | `src/extensions/hallucination/semantic_entropy_probe.py` |
| **OnlineLoRA**             | Task-free low-rank adaptation                                | `src/extensions/lora/online_lora.py`      |


### Model Layer


| Component         | Description                                    | File                                |
| ----------------- | ---------------------------------------------- | ----------------------------------- |
| **HDIMModel**     | Core HDIM model wrapper                        | `src/models/hdim_model.py`          |
| **SBERTEncoder**  | Frozen SBERT encoder wrapper                   | `src/models/sbert_encoder.py`       |
| **ModernBertEncoder** | ModernBERT with Matryoshka multi-scale     | `src/models/modern_text_encoder.py` |


### Training Layer


| Component       | Description                        | File                         |
| --------------- | ---------------------------------- | ---------------------------- |
| **InvariantTrainer** | Active training loop | `src/training/invariant_trainer.py` |
| **HDIMTrainer** | Legacy training loop | `src/training/trainer.py` |


---

## Mathematical Foundation

### Clifford Algebra Cl(3,1,0)

```
Signature: 3 positive, 1 negative, 0 null basis vectors
Dimension: 2^(3+1+0) = 16

Basis vectors: {e₁, e₂, e₃, e₄}
Metric:        e₁² = e₂² = e₃² = +1, e₄² = -1

Geometric product rules:
  eᵢ ⊗ eᵢ = ±1 (metric-dependent)
  eᵢ ⊗ eⱼ = -eⱼ ⊗ eᵢ (i ≠ j) — anticommutativity
```

### Key Operations


| Operation          | Formula                        | Code                                  |
| ------------------ | ------------------------------ | ------------------------------------- |
| Geometric Product  | `a ⊗ b = a·b + a∧b`            | `CliffordAlgebra.geometric_product()`|
| Sandwich           | `R⁻¹ ⊗ M ⊗ R`                  | `InvariantExtractor.forward()`      |
| Rotor Exponential  | `exp(B) = cos(θ) + B·sin(θ/θ)` | `CliffordAlgebra.bivector_exp()`      |
| Reverse Involution | `rev(e₁₂₃) = e₃₂₁`             | `CliffordAlgebra.reverse()`           |


### Loss Functions


| Loss                 | Weight | Phase | Purpose                             |
| -------------------- | ------ | ----- | ----------------------------------- |
| Reconstruction       | 1.0    | 1     | MSE between input and output        |
| Isomorphism          | 0.0    | 1     | DISABLED: conflicts with pair_loss  |
| InfoNCE (pair)       | 0.40   | 3     | Pair discrimination (optimal)      |
| Routing Entropy      | 0.05   | 7     | Balanced expert utilization         |
| Z-Loss               | 0.01   | 9     | MoE collapse prevention             |
| Memory               | 0.05   | 6     | Titans memory consistency           |
| DCL                  | 0.2    | 20    | Decoupled Contrastive Learning      |
| Uniformity+Alignment | 0.1    | 20    | Representation quality              |
| Expert Ortho         | 0.01   | 26    | Expert diversity                    |
| STS                  | 0.0    | —     | DISABLED: duplicates InfoNCE        |


---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- PyTorch 2.0+

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/hypercoplexAI.git
cd hypercoplexAI

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0
sentence-transformers>=2.2
numpy
scipy
optuna  # for auto_tune
```

---

## Quick Start

### Minimal Example

```python
from src.models.hdim_model import HDIMConfig, HDIMModel
import torch

config = HDIMConfig(
    hidden_dim=256,
    num_domains=4,
    num_experts=4,
    top_k=2,
    memory_key_dim=32,
    clifford_p=3,  # Cl(3,1,0)
    clifford_q=1,
    clifford_r=0,
    domain_names=("physics", "chemistry", "biology", "engineering"),
)

model = HDIMModel(config).to("cuda")
encodings = torch.randn(2, config.hidden_dim, device="cuda")
domain_id = torch.tensor([0, 1], device="cuda")      # physics, chemistry
target_domain = torch.tensor([1, 0], device="cuda")  # chemistry, physics

result = model.transfer_pairs(encodings, domain_id, target_domain)

print(f"Output shape: {result.output.shape}")
print(f"Routing weights shape: {result.routing_weights.shape}")
print(f"Invariant norm: {result.invariant.norm()}")
```

---

## Training

### Basic Training

```bash
# SoftMoERouter (baseline)
python scripts/gpu_train.py \
    --pretrained_encoder --soft_router \
    --real_pairs data/real_pairs_v10.json \
    --amp --device cuda \
    --epochs 60 --batch_size 32

# MoEKernel (Phase 28) — domain-specialized experts
python scripts/gpu_train.py \
    --pretrained_encoder --moe_kernel \
    --real_pairs data/real_pairs_v10.json \
    --amp --device cuda \
    --lambda_z 0.01 --lambda_expert_ortho 0.01 \
    --epochs 60 --batch_size 32
```

### Project Record: Run 18 (MoE + TitansMemory)

Best result achieved on `data/real_pairs_v10.json`, RTX 3070 Laptop 8GB:

| Metric | Run 18 (Record) | Run 13 | Run 11 |
|---|---|---|---|
| **score** | **1.1814** | 1.1706 | 1.1528 |
| **pair_margin** | **1.0224** | 1.0073 | 0.9866 |
| best_epoch | 13 | 28 | 27 |
| temperature | **0.10** | 0.12 | 0.12 |
| lambda_pair | 0.40 | 0.40 | 0.35 |

**Session 14 smoke test** (5 epochs, MoE + TitansMemory): ep5=1.1508 ≈ Run11 ep27 (27x faster convergence).

**Optimal config:**
```bash
python scripts/gpu_train.py \
    --pretrained_encoder --moe_kernel \
    --real_pairs data/real_pairs_v10.json \
    --amp --device cuda \
    --epochs 30 --batch_size 24 \
    --lr 5e-4 --sbert_lr 1e-5 \
    --infonce_temperature 0.10 \
    --lambda_pair 0.40 --lambda_sts 0.0 \
    --lambda_z 0.01 --lambda_expert_ortho 0.01 \
    --patience 15
```

---

### Phase 28 Results: MoEKernel vs SoftMoERouter

Real training on `data/real_pairs_v10.json` (1036 pairs), SBERT encoder, RTX 3070:

| Metric | SoftMoERouter | MoEKernel | Improvement |
|---|---|---|---|
| **score** | 0.300 | **1.067** | +256% |
| **pair_margin** | 0.000 | **0.902** | ∞ |
| **STS** | 1.000 | 0.551 | — |
| **train_loss (ep5)** | 0.930 | **0.274** | -71% |
| **train_loss (ep10)** | — | **0.208** | — |
| **pair_margin (ep10)** | — | **0.930** | — |
| **score (ep10)** | — | **1.092** | — |
| params | 349K | 360K | +3% |

MoEKernel expert load (ep5): `[math=0.28, lang=0.21, code=0.23, sci=0.29]` — balanced, non-collapsed.
Training curve: ep1=0.697 → ep5=0.274 → ep10=0.208 (consistent decrease, no NaN).

### Hyperparameter Search (Optuna)

```bash
python scripts/auto_tune.py \
    --n_trials 100 \
    --study_name hdim_autotune_v27 \
    --epochs 20
```

### Key Hyperparameters


| Parameter     | Default | Range        | Effect                  |
| ------------- | ------- | ------------ | ----------------------- |
| `hidden_dim`  | 256     | [128, 512]   | Model capacity          |
| `num_experts` | 4       | [2, 8]       | MoE diversity           |
| `top_k`       | 2       | [1, 4]       | Experts per token       |
| `lambda_z`    | 0.01    | [0.001, 0.1] | MoE collapse prevention |
| `temperature` | 0.10    | [0.07, 0.5]  | InfoNCE contrast (optimal=0.10) |
| `focal_gamma` | 2.0     | [0.5, 3.0]   | Hard negative focus     |


---

## Verification

### Run All Tests

```bash
# Numerical verification
python formalization/verify_numerical.py

# pytest suite
python -m pytest tests/ -v

# Real-model verification (MoEKernel on SBERT + real_pairs_v10.json)
python scripts/verify_moe_kernel_real.py  # 14/14 checks PASS

# Import check
python -c "from src.core.hypercomplex import CliffordAlgebra; \
           from src.core.hbma_memory import HBMAMemory; \
           from src.core.hdim_pipeline import HDIMPipeline; \
           print('All imports OK')"
```

### Lean4 Categories

```python
# Run specific category
python formalization/verify_numerical.py --category sandwich

# Categories:
# - algebra         (associativity, distributivity, anticommutativity)
# - sandwich        (norm preservation, composition)
# - involution      (reverse, involute)
# - domain_transfer (isomorphism)
# - memory          (HBMA FIFO, EMA convergence)
# - moe             (load balance, z_loss)
```

---

## Project Structure

```
hypercoplexAI/
├── src/
│   ├── core/
│   │   ├── algebra.py                # CliffordAlgebra
│   │   ├── rotors.py                 # DomainRotationOperator
│   │   ├── invariants.py             # InvariantExtractor, sandwich_transfer
│   │   ├── invariant_index.py        # InvariantIndex
│   │   ├── engine.py                 # CoreEngineConfig, HDIMCoreEngine
│   │   ├── hypercomplex.py           # compatibility exports
│   │   ├── domain_operators.py       # compatibility exports
│   │   ├── hdim_pipeline.py          # compatibility wrapper
│   │   ├── domain_encoder.py         # compatibility wrapper
│   │   ├── transfer_engine.py        # compatibility wrapper
│   │   └── transfer_state.py         # TransferState dataclass
│   ├── extensions/
│   │   ├── memory/
│   │   │   ├── titans.py             # TitansMemoryModule
│   │   │   ├── hbma.py               # HBMAMemory
│   │   │   ├── interface.py          # MemoryInterface ABC
│   │   │   ├── invariant_processor.py # InvariantProcessor
│   │   │   ├── msa.py                # MSA memory
│   │   │   └── sparse_index.py       # Sparse retrieval index
│   │   ├── moe/
│   │   │   ├── kernel.py             # MoEKernel domain experts
│   │   │   ├── soft_router.py        # SoftMoERouter
│   │   │   ├── interface.py          # MoERouter ABC
│   │   │   └── clifford_interaction.py # CliffordInteractionLayer
│   │   ├── hallucination/
│   │   │   ├── detector.py           # HallucinationDetector
│   │   │   ├── feedback.py           # HallucinationFeedbackLoop
│   │   │   └── semantic_entropy_probe.py # SemanticEntropyProbe
│   │   └── lora/
│   │       ├── online_lora.py        # OnlineLoRA
│   │       └── per_domain_lora.py    # Per-domain LoRA
│   ├── models/
│   │   ├── hdim_model.py             # HDIMModel wrapper
│   │   ├── config.py                 # HDIMConfig
│   │   ├── results.py                # result containers
│   │   ├── sbert_encoder.py          # SBERTEncoder wrapper
│   │   ├── modern_text_encoder.py    # ModernBertEncoder, GatedMLPEncoder, HybridEncoder
│   │   └── metrics.py                # compute_all_metrics, analogy_feasibility_rate
│   ├── adapters/
│   │   └── text.py                   # TextAdapter, SimpleTextEncoder
│   └── training/
│       ├── train.py                  # active training entrypoint
│       ├── invariant_trainer.py      # InvariantTrainer
│       ├── trainer.py                # legacy trainer
│       ├── dataset.py                # DomainProblemDataset
│       └── real_dataset.py           # RealPairsDataset
├── scripts/
│   ├── gpu_train.py             # PRIMARY training script
│   ├── auto_tune.py             # Optuna hyperparameter search (v27)
│   └── autoresearch_loop.py     # Automated research loop
├── tests/
│   ├── test_hdim.py                  # Core model tests
│   ├── test_hdim_pipeline.py         # Pipeline integration
│   ├── test_moe_kernel.py            # MoEKernel (45 tests, Phase 28)
│   ├── test_titans_memory.py         # TitansMemory TTT
│   ├── test_titans_freeze.py         # RAG freeze API (Phase 29)
│   ├── test_clifford_interaction.py  # CAN layer
│   ├── test_can_integration.py       # CAN integration
│   ├── test_hbma_plugin.py           # HBMA 4-system memory
│   ├── test_msa_attention.py         # MSA sparse attention (tests prototype_memory, not msa_attention module)
│   ├── test_memory_interface.py      # Memory ABC
│   ├── test_memory_comparison.py     # Memory comparison
│   ├── test_memory_persistence.py   # MemoryPersistence (test file not yet created)
│   ├── test_augmentation.py          # Embedding augmentations
│   ├── test_auto_config.py           # AutoConfig
│   ├── test_nan_inf_forward.py       # NaN/Inf protection
│   ├── test_matryoshka_modernbert.py # Matryoshka + ModernBERT
│   ├── test_hallucination_detector.py # HallucinationDetector
│   ├── test_hallucination_feedback.py # HallucinationFeedbackLoop
│   ├── test_semantic_entropy_probe.py # SemanticEntropyProbe
│   ├── test_online_learner_gradient.py # OnlineLearner
│   ├── test_online_lora.py           # OnlineLoRA
│   ├── test_continual_norm.py       # ContinualNorm (test file not yet created)
│   ├── test_maxscore_router.py      # MaxScoreRouter (test file not yet created)
│   ├── test_production_benchmark.py  # Production benchmarks
│   ├── test_multi_worker.py         # Multi-worker training
│   ├── test_kernel_chat.py          # Interactive kernel chat
│   ├── test_router.py               # Router variants (test file not yet created)
│   ├── test_checkpoint_variants.py  # Checkpoint loading
│   ├── test_all_modules.py          # Full module smoke test
│   └── test_triton_performance.py    # Triton kernel benchmarks
│   # Run python -m pytest for current suite status
├── docs/
│   ├── ARCHITECTURE.md          # Full architecture documentation
│   └── DIAGRAMS.md              # Mermaid diagrams
├── formalization/
│   └── verify_numerical.py      # Numerical theorem verification
├── HDIM.md                      # Technical specification
└── README.md                    # This file
```

---

## Configuration

### HDIMConfig

```python
from src.models.hdim_model import HDIMConfig

config = HDIMConfig(
    # Dimensions
    hidden_dim=256,          # Input/output dimension
    clifford_dim=16,         # Multivector dimension (computed from p,q,r)

    # Clifford Algebra
    clifford_p=3,            # Positive basis vectors
    clifford_q=1,            # Negative basis vectors
    clifford_r=0,            # Null basis vectors

    # Domain
    num_domains=4,           # Number of domain rotors
    domain_names=["physics", "chemistry", "biology", "engineering"],

    # MoE
    num_experts=4,           # Expert count
    top_k=2,                 # Active experts per token

    # Memory
    memory_key_dim=32,       # Titans key dimension

    # Training
    dropout=0.1,
    layer_norm_eps=1e-5,
)
```

### Training Config

```python
training_config = {
    # Losses
    "loss_recon": 1.0,
    "loss_iso": 0.1,
    "loss_pair": 0.1,
    "loss_routing": 0.05,
    "router_z_loss": 0.01,
    "loss_memory": 0.05,
    "loss_dcl": 0.2,
    "loss_uniformity": 0.1,
    "loss_expert_ortho": 0.02,

    # Training
    "epochs": 60,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 100,

    # AMP
    "amp": True,
    "fp32_memory": True,  # Titans in fp32
}
```

---

## Documentation


| Document                                         | Description                                                |
| ------------------------------------------------ | ---------------------------------------------------------- |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Full architecture: layers, components, API, configurations |
| **[docs/DIAGRAMS.md](docs/DIAGRAMS.md)**         | Mermaid diagrams: data flow, sequence, class diagrams      |
| **[HDIM.md](HDIM.md)**                           | Technical specification: formulas, algorithms, phases      |


---

## References

### Key Papers


| Paper                    | Citation                      | Relevance                               |
| ------------------------ | ----------------------------- | --------------------------------------- |
| Soft MoE                 | Puigcerver et al. (ICLR 2024) | Soft routing without token dropping     |
| DeepSeek-V3              | DeepSeek (2024)               | Shared Expert + Aux-loss-free balancing |
| Titans Memory            | Gu & Dao (2023)               | Test-Time Training for memory           |
| DCL                      | Yeh et al. (2022)             | Decoupled Contrastive Learning          |
| Uniformity+Alignment     | Wang & Isola (2020)           | Contrastive representation quality      |
| Expert Orthogonalization | arXiv:2505.22323 (2025)       | Expert diversity via orthogonality      |


### Code References


| Component            | File                                  |
| -------------------- | ------------------------------------- |
| Geometric Product    | `src/core/hypercomplex.py`            |
| Sandwich Product     | `src/core/domain_operators.py`        |
| SoftMoE Router       | `src/core/soft_moe_router.py`         |
| MoEKernel            | `src/core/moe_kernel.py`              |
| Shared Expert        | `src/core/moe_kernel.py` (use_shared_expert) |
| Titans Memory        | `src/core/titans_memory.py`           |
| HBMA Adapter         | `src/core/hbma_memory.py`             |
| Hallucination Detect | `src/extensions/hallucination/detector.py` |
| Online Learning      | `src/core/online_learner.py`          |
| MoE Kernel           | `src/extensions/moe/kernel.py`        |
| Memory Extensions    | `src/extensions/memory/`              |


---

## Known Issues

### Anti-Patterns (Do NOT)

- ❌ `ModularMoERouter` — removed; use `MoEKernel` or `SoftMoERouter`
- ❌ `reset_memory('zero')` — use `reset_memory('geometric')`
- ❌ `batch_size < 32` — InfoNCE requires sufficient negatives
- ❌ `temperature < 0.07` — causes overconfidence, gradient instability (optimal: 0.10)
- ❌ `lambda_z = 0` — causes MoE collapse within 10 epochs

### Phase 17 Critical Fixes (C1-C7)


| Code | Issue                       | Fix                             |
| ---- | --------------------------- | ------------------------------- |
| C1   | SoftMoERouter guard for T=1 | Added epsilon in softmax        |
| C2   | Dynamic load-balance loss   | EMA scores for stability        |
| C3   | Out-of-place operations     | In-place tensor ops             |
| C4   | fp32 TTT path               | TitansMemory in fp32 during AMP |
| C5   | Memory drift                | `reset_memory()` per epoch      |
| C6   | Focal gamma                 | Applied to denominator only     |
| C7   | Non-leaf tensor fix         | `.clone()` for non-leaf tensors |


---

## Citation

If you use HDIM in your research, please cite:

```bibtex
@software{hdim2026,
  title = {HDIM: Hypercomplex Domain Isomorphism Machine},
  author = {HypercoplexAI Team},
  year = {2026},
  url = {https://github.com/your-org/hypercoplexAI},
  note = {Phase 30: MoEKernel buffer fix, SoftMoERouter deadlock fix; record score=1.1814}
}
```

---

## License

TBD

---

## Phase 29 (2026-03-19) — Architecture Refactoring

### CliffordInteractionExpert

CAN-style expert without GELU on multivectors — preserves geometric structure during domain routing.

```python
# Key difference: raw multivector processing (no GELU)
class CliffordInteractionExpert(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 16) multivector
        return self.clifford_transform(x)  # No GELU activation
```

### RAG-Compatible Freeze API

TitansMemory.freeze_memory() for deterministic embeddings in RAG pipelines.

```python
model = HDIMModel(config)
model.reset_memory()  # Compatibility no-op in the streamlined core
```

### Security Fix

torch.load RCE vulnerability patched — weights_only=True enforced.

### Package Setup

- pyproject.toml with CLI entry points
- SOTA benchmarks integration

---

*Last updated: 2026-04-09 | Phase 30+ | Research prototype — API may evolve*
