"""Training package for HypercoplexAI — trainers, datasets, and utilities."""

from src.training.trainer import HDIMTrainer
from src.training.dataset import (
    DomainProblemDataset,
    create_demo_dataset,
    create_group_aware_split,
    create_paired_demo_dataset,
    texts_to_tensor,
)

__all__ = [
    "HDIMTrainer",
    "DomainProblemDataset",
    "create_demo_dataset",
    "create_group_aware_split",
    "create_paired_demo_dataset",
    "texts_to_tensor",
]
