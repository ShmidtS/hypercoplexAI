# HDIM: Hypercomplex Domain Isomorphism Machine

[Python 3.10+](https://www.python.org/downloads/)
[PyTorch 2.0+](https://pytorch.org/)  
[Lean4](verify_lean4_numerical.py)
[Tests](tests/)
[Status]()

> **Best Score:** 1.1542 (Phase 26c, epoch 15) — `pair_margin=0.993`, `STS=0.537`
> **Verification:** 148/148 Lean4 theorems PASS | 123 pytest tests PASS
> **Features:** DomainExpertPool + SharedExpert + AuxLossFree + ExpertOrtho + auto_tune v27

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

### Verification: 148 Lean4 Theorems

All mathematical properties are numerically verified in `verify_lean4_numerical.py`:


| Category              | Theorems | Status         |
| --------------------- | -------- | -------------- |
| Clifford Algebra      | 67       | ✅ PASS         |
| Sandwich Product      | 17       | ✅ PASS         |
| Involutions           | 16       | ✅ PASS         |
| Domain Transfer       | 11       | ✅ PASS         |
| HBMA Memory           | 11       | ✅ PASS         |
| SoftMoE Router        | 10       | ✅ PASS         |
| Matryoshka Embeddings | 7        | ✅ PASS         |
| Quaternion Operations | 9        | ✅ PASS         |
| Training Losses       | 5        | ✅ PASS         |
| Memory Adapters       | 5        | ✅ PASS         |
| **Total**             | **148**  | **✅ ALL PASS** |


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
│  │  │ SBERTEncoder│───▶│ DomainExpert│───▶│ HDIMEncoder │            │  │
│  │  │ (frozen)    │    │ Pool (4x)   │    │ (MLP)       │            │  │
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
│  │  │   G ──▶ R⁻¹ ⊗ G ⊗ R ──▶ U_inv (domain-invariant)           │   │  │
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
│  │  │   Domain Transfer: G_B = R_B ⊗ U_inv ⊗ R_B⁻¹               │   │  │
│  │  │                                                            │   │  │
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
| **CliffordAlgebra**        | Cl(3,1,0) implementation with Cayley-table geometric product | `src/core/hypercomplex.py:20`       |
| **DomainRotationOperator** | Learnable rotor for domain transformations                   | `src/core/domain_operators.py:19`   |
| **InvariantExtractor**     | Sandwich product: R⁻¹⊗G⊗R                                    | `src/core/domain_operators.py:54`   |
| **TitansMemoryModule**     | Test-Time Training memory (fp32-safe)                        | `src/core/titans_memory.py:30`      |
| **HBMAMemoryAdapter**      | 4-system memory (Working, Episodic, Semantic, Procedural)    | `src/core/hbma_memory.py:20`        |
| **SoftMoERouter**          | Soft MoE with SharedExpert + AuxLossFree                     | `src/core/soft_moe_router.py:43`    |
| **DomainExpertPool**       | 4 frozen SBERT experts + trainable projections               | `src/core/domain_expert_pool.py:20` |
| **HDIMPipeline**           | End-to-end pipeline orchestrator                             | `src/core/hdim_pipeline.py:128`     |


### Model Layer


| Component         | Description                                    | File                                |
| ----------------- | ---------------------------------------------- | ----------------------------------- |
| **HDIMModel**     | Core HDIM model with configurable architecture | `src/models/hdim_model.py:117`      |
| **TextHDIMModel** | Text-to-embedding wrapper                      | `src/models/text_hdim_model.py:191` |
| **SBERTEncoder**  | Frozen SBERT encoder wrapper                   | `src/models/sbert_encoder.py:20`    |


### Training Layer


| Component       | Description                        | File                         |
| --------------- | ---------------------------------- | ---------------------------- |
| **HDIMTrainer** | Full training loop with all losses | `src/training/trainer.py:19` |


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
| Geometric Product  | `a ⊗ b = a·b + a∧b`            | `CliffordAlgebra.geometric_product()` |
| Sandwich           | `R⁻¹ ⊗ M ⊗ R`                  | `InvariantExtractor.forward()`        |
| Rotor Exponential  | `exp(B) = cos(θ) + B·sin(θ/θ)` | `CliffordAlgebra.bivector_exp()`      |
| Reverse Involution | `rev(e₁₂₃) = e₃₂₁`             | `CliffordAlgebra.reverse()`           |


### Loss Functions


| Loss                 | Weight | Phase | Purpose                             |
| -------------------- | ------ | ----- | ----------------------------------- |
| Reconstruction       | 1.0    | 1     | MSE between input and output        |
| Isomorphism          | 0.1    | 1     | MSE between transferred and target  |
| Focal-InfoNCE        | 0.1    | 3     | Pair discrimination with focal loss |
| Routing Entropy      | 0.05   | 7     | Balanced expert utilization         |
| Z-Loss               | 0.01   | 9     | MoE collapse prevention             |
| Memory               | 0.05   | 6     | Titans memory consistency           |
| DCL                  | 0.2    | 20    | Decoupled Contrastive Learning      |
| Uniformity+Alignment | 0.1    | 20    | Representation quality              |
| Expert Ortho         | 0.02   | 26    | Expert diversity                    |


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
from src.models.model_factory import build_sbert_hdim_model
from src.models.hdim_model import HDIMConfig
import torch

# Configure model
config = HDIMConfig(
    hidden_dim=256,
    num_domains=4,
    num_experts=4,
    top_k=2,
    memory_key_dim=32,
    clifford_p=3,  # Cl(3,1,0)
    clifford_q=1,
    clifford_r=0,
    domain_names=["physics", "chemistry", "biology", "engineering"],
)

# Build model
model = build_sbert_hdim_model(
    config,
    soft_router=True,      # Use SoftMoERouter (recommended)
    freeze_sbert=True,     # Freeze SBERT encoder (recommended)
    z_loss_weight=0.01,    # MoE anti-collapse
)

# Encode texts
texts = [
    "Cavitation erosion in hydraulic turbines causes surface damage",
    "Ultrasonic plaque removal uses cavitation bubbles",
]
encodings = model.encode_texts(texts, device="cuda")

# Cross-domain transfer
domain_id = torch.tensor([0, 1], device="cuda")      # physics, chemistry
target_domain = torch.tensor([1, 0], device="cuda")  # chemistry, physics

output, routing, invariant, state = model.transfer_text_pairs(
    texts, domain_id, target_domain
)

print(f"Output shape: {output.shape}")           # (2, hidden_dim)
print(f"Routing weights: {routing['weights']}")  # Expert assignments
print(f"Invariant norm: {invariant.norm()}")    # Should equal input norm
```

---

## Training

### Basic Training

```bash
python scripts/gpu_train.py \
    --use_pairs \
    --amp \
    --hidden_dim 256 \
    --num_experts 4 \
    --top_k 2 \
    --lambda_z 0.01 \
    --infonce_temperature 0.15 \
    --epochs 60 \
    --batch_size 32
```

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
| `temperature` | 0.15    | [0.1, 0.5]   | InfoNCE contrast        |
| `focal_gamma` | 2.0     | [0.5, 3.0]   | Hard negative focus     |


---

## Verification

### Run All Tests

```bash
# Lean4 numerical verification (148 theorems)
python verify_lean4_numerical.py

# pytest suite (123 tests)
python -m pytest tests/ -v

# Import check
python -c "from src.core.hypercomplex import CliffordAlgebra; \
           from src.core.hbma_memory import HBMAMemory; \
           from src.core.hdim_pipeline import HDIMPipeline; \
           print('All imports OK')"
```

### Lean4 Categories

```python
# Run specific category
python verify_lean4_numerical.py --category sandwich

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
│   │   ├── hypercomplex.py      # CliffordAlgebra, QuaternionLinear
│   │   ├── domain_operators.py  # DomainRotationOperator, InvariantExtractor
│   │   ├── hdim_pipeline.py     # HDIMPipeline orchestrator
│   │   ├── titans_memory.py     # TitansMemoryModule (TTT)
│   │   ├── hbma_memory.py       # HBMAMemory (4-system memory)
│   │   ├── soft_moe_router.py   # SoftMoERouter (Phase 26)
│   │   ├── domain_expert_pool.py # DomainExpertPool + SharedExpert
│   │   └── memory_interface.py  # MemoryInterface ABC
│   ├── models/
│   │   ├── hdim_model.py        # HDIMModel, HDIMConfig
│   │   ├── text_hdim_model.py   # TextHDIMModel
│   │   ├── sbert_encoder.py     # SBERTEncoder wrapper
│   │   └── model_factory.py     # build_*() functions
│   └── training/
│       ├── trainer.py           # HDIMTrainer (all losses)
│       ├── dataset.py           # DomainProblemDataset
│       └── real_dataset.py      # RealPairsDataset
├── scripts/
│   ├── gpu_train.py             # PRIMARY training script
│   ├── auto_tune.py             # Optuna hyperparameter search (v27)
│   └── autoresearch_loop.py     # Automated research loop
├── tests/
│   ├── test_hypercomplex.py
│   ├── test_domain_operators.py
│   ├── test_moe_router.py
│   └── ... (123 tests total)
├── docs/
│   ├── ARCHITECTURE.md          # Full architecture documentation
│   └── DIAGRAMS.md              # Mermaid diagrams
├── verify_lean4_numerical.py    # 148 theorem verification
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


| Component         | File                             | Key Lines |
| ----------------- | -------------------------------- | --------- |
| Geometric Product | `src/core/hypercomplex.py`       | L20-150   |
| Sandwich Product  | `src/core/domain_operators.py`   | L54-103   |
| SoftMoE Router    | `src/core/soft_moe_router.py`    | L43-250   |
| Shared Expert     | `src/core/domain_expert_pool.py` | L115-180  |
| Titans Memory     | `src/core/titans_memory.py`      | L30-200   |
| HBMA Adapter      | `src/core/hbma_memory.py`        | L20-150   |


---

## Known Issues

### Anti-Patterns (Do NOT)

- ❌ `ModularMoERouter` — removed; use `SoftMoERouter`
- ❌ `reset_memory('zero')` — use `reset_memory('geometric')`
- ❌ `batch_size < 32` — InfoNCE requires sufficient negatives
- ❌ `temperature < 0.15` — causes overconfidence
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
  note = {Phase 27: 148 Lean4-verified theorems, 123 tests passing}
}
```

---

## License

TBD

---

*Last updated: 2026-03-17 | Phase 27 Complete | Research prototype — API may evolve*