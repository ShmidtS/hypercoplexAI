# hypercoplexAI
HDIM (Hypercomplex Domain Isomorphism Machine) is a research-oriented Python/PyTorch project for cross-domain knowledge transfer using hypercomplex representations, routing, and memory modules.

## Installation
This project requires Python 3.10+.

### 1. Create and activate a virtual environment
Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install PyTorch
`requirements.txt` intentionally does not pin a specific PyTorch build because the correct wheel depends on your platform and whether you need CPU or CUDA support.

CPU example:

```bash
pip install torch
```

If you need CUDA, install the matching wheel from the official PyTorch selector first, then continue with the project dependencies below.

### 3. Install project dependencies
```bash
pip install -r requirements.txt
```

## Quick start
### Run training
Base run:

```bash
python -m src.training.train
```

Quick smoke run:

```bash
python -m src.training.train --epochs 1 --batch_size 8 --num_samples 20 --device cpu
```

Run with paired cross-domain supervision:

```bash
python -m src.training.train --use_pairs --epochs 1 --batch_size 8 --num_samples 20 --device cpu
```

### Run tests
Full test suite:

```bash
pytest
```

Short output:

```bash
pytest -q
```

### Run the demo
```bash
python hdim_demo.py
```

## Project layout
- `src/core/` - Clifford algebra, domain operators, routing, memory, and the HDIM pipeline
- `src/models/` - the HDIM model and quality metrics
- `src/training/` - dataset, trainer, and the training CLI entrypoint
- `tests/` - pytest coverage forward/transfer, dataset, and trainer contracts
- `checkpoints/` - output directory for saved training checkpoints
## Core commands
```bash
pip install torch
pip install -r requirements.txt
python -m src.training.train
python -m src.training.train --use_pairs --epochs 1 --batch_size 8 --num_samples 20 --device cpu
pytest -q
python hdim_demo.py
```
