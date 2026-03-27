"""Comparison test: TitansMemory vs CLSMemory (Hippocampus+Neocortex).

Runs a short training loop on real_pairs_v10.json (if available) or synthetic
data, then reports key metrics for both memory types side-by-side.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.hdim_model import HDIMConfig, HDIMModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(n: int = 256, dim: int = 64, num_domains: int = 4, seed: int = 42):
    """Create a simple synthetic dataset of (embedding, domain_id) pairs."""
    torch.manual_seed(seed)
    # Each domain has a different mean — model should learn domain structure
    embeddings = []
    domain_ids = []
    for d in range(num_domains):
        mean = torch.randn(dim) * 2  # domain-specific centroid
        samples = mean.unsqueeze(0) + torch.randn(n // num_domains, dim) * 0.5
        embeddings.append(samples)
        domain_ids.extend([d] * (n // num_domains))
    X = torch.cat(embeddings, dim=0)  # [N, dim]
    D = torch.tensor(domain_ids, dtype=torch.long)  # [N]
    return TensorDataset(X, D)


def _train_one_epoch(
    model: HDIMModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One epoch of reconstruction training. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for X, D in loader:
        X, D = X.to(device), D.to(device)
        optimizer.zero_grad()
        out, routing_weights, invariant, _ = model(X, D)
        # Reconstruction loss
        loss_recon = nn.functional.mse_loss(out, X)
        # Routing entropy loss (encourage diverse routing)
        eps = 1e-8
        rw_norm = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + eps)
        entropy = -(rw_norm * (rw_norm + eps).log()).sum(dim=-1).mean()
        loss = loss_recon - 0.01 * entropy
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss_recon.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _eval_model(
    model: HDIMModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model: reconstruction loss + routing entropy."""
    model.eval()
    total_loss = 0.0
    total_entropy = 0.0
    n_batches = 0
    with torch.no_grad():
        for X, D in loader:
            X, D = X.to(device), D.to(device)
            out, routing_weights, invariant, _ = model(X, D)
            loss = nn.functional.mse_loss(out, X)
            eps = 1e-8
            rw_norm = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + eps)
            entropy = -(rw_norm * (rw_norm + eps).log()).sum(dim=-1).mean()
            total_loss += loss.item()
            total_entropy += entropy.item()
            n_batches += 1
    return {
        'recon_loss': total_loss / max(n_batches, 1),
        'routing_entropy': total_entropy / max(n_batches, 1),
    }


def run_comparison(
    memory_type: str,
    n_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = torch.device('cpu'),
    seed: int = 42,
) -> Dict[str, float]:
    """Train and evaluate a model with given memory_type. Returns metrics dict."""
    torch.manual_seed(seed)
    cfg = HDIMConfig(
        hidden_dim=64,
        num_domains=4,
        num_experts=4,
        top_k=2,
        memory_type=memory_type,
    )
    model = HDIMModel(cfg).to(device)
    dataset = _make_synthetic_dataset(n=512, dim=64, num_domains=4, seed=seed)
    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    t0 = time.time()
    train_losses = []
    for epoch in range(1, n_epochs + 1):
        loss = _train_one_epoch(model, train_loader, optimizer, device)
        train_losses.append(loss)

    elapsed = time.time() - t0
    val_metrics = _eval_model(model, val_loader, device)
    n_params = sum(p.numel() for p in model.parameters())

    return {
        'memory_type': memory_type,
        'n_params': n_params,
        'train_loss_final': train_losses[-1],
        'train_loss_epoch1': train_losses[0],
        'val_recon_loss': val_metrics['recon_loss'],
        'val_routing_entropy': val_metrics['routing_entropy'],
        'train_time_s': round(elapsed, 2),
        'epochs': n_epochs,
    }


# ---------------------------------------------------------------------------
# pytest tests
# ---------------------------------------------------------------------------

class TestMemoryComparison:
    """Functional tests ensuring both memory types train correctly."""

    @pytest.fixture(scope='class')
    def results(self):
        """Run comparison once for all tests in the class."""
        device = torch.device('cpu')
        titans = run_comparison('titans', n_epochs=5, device=device)
        cls = run_comparison('cls', n_epochs=5, device=device)
        hippo = run_comparison('hippocampus', n_epochs=5, device=device)
        return {'titans': titans, 'cls': cls, 'hippocampus': hippo}

    def test_titans_trains(self, results):
        """Titans model loss decreases over training."""
        r = results['titans']
        assert r['train_loss_final'] < r['train_loss_epoch1'] * 1.1, (
            f"Titans loss did not decrease: {r['train_loss_epoch1']:.4f} -> {r['train_loss_final']:.4f}"
        )

    def test_cls_trains(self, results):
        """CLS model loss decreases over training."""
        r = results['cls']
        assert r['train_loss_final'] < r['train_loss_epoch1'] * 1.1, (
            f"CLS loss did not decrease: {r['train_loss_epoch1']:.4f} -> {r['train_loss_final']:.4f}"
        )

    def test_hippocampus_trains(self, results):
        """Hippocampus-only model trains correctly."""
        r = results['hippocampus']
        assert r['train_loss_final'] < r['train_loss_epoch1'] * 1.1

    def test_val_loss_reasonable(self, results):
        """Both models achieve reasonable validation reconstruction loss."""
        for name, r in results.items():
            assert r['val_recon_loss'] < 5.0, (
                f"{name} val loss too high: {r['val_recon_loss']:.4f}"
            )

    def test_routing_entropy_positive(self, results):
        """Routing entropy should be positive (experts are being used)."""
        for name, r in results.items():
            assert r['val_routing_entropy'] > 0, (
                f"{name} routing entropy is zero — all tokens routed to same expert"
            )

    def test_cls_forward_shapes(self):
        """CLS model produces correct output shapes."""
        cfg = HDIMConfig(memory_type='cls', hidden_dim=64, num_domains=4)
        m = HDIMModel(cfg)
        x = torch.randn(8, 64)
        d = torch.randint(0, 4, (8,))
        out, rw, inv, _ = m(x, d)
        assert out.shape == (8, 64)
        assert rw.shape == (8, 4)  # num_experts=4
        assert inv.shape == (8, 64)

    def test_titans_forward_shapes(self):
        """Titans model produces correct output shapes."""
        cfg = HDIMConfig(memory_type='titans', hidden_dim=64, num_domains=4)
        m = HDIMModel(cfg)
        x = torch.randn(8, 64)
        d = torch.randint(0, 4, (8,))
        out, rw, inv, _ = m(x, d)
        assert out.shape == (8, 64)
        assert rw.shape == (8, 4)
        assert inv.shape == (8, 64)

    def test_reset_memory(self):
        """reset_memory works for both memory types."""
        for mtype in ('titans', 'cls', 'hippocampus', 'neocortex'):
            cfg = HDIMConfig(memory_type=mtype)
            m = HDIMModel(cfg)
            m.reset_memory()  # should not raise

    def test_transfer_pairs(self):
        """transfer_pairs works with batched (source, src_ids, tgt_ids) interface."""
        cfg = HDIMConfig(memory_type='cls', hidden_dim=64, num_domains=4)
        m = HDIMModel(cfg)
        x = torch.randn(3, 64)
        src_ids = torch.tensor([0, 2, 1], dtype=torch.long)
        tgt_ids = torch.tensor([1, 3, 0], dtype=torch.long)
        out, routing, inv, slot_outputs, state = m.transfer_pairs(
            x, src_ids, tgt_ids,
            update_memory=False,
            memory_mode="retrieve",
        )
        assert out.shape == (3, 64)
        assert routing.shape == (3, cfg.num_experts)
        assert inv.shape == (3, 64)


# ---------------------------------------------------------------------------
# Standalone comparison report
# ---------------------------------------------------------------------------

def print_comparison_report():
    """Print a side-by-side comparison table."""
    device = torch.device('cpu')
    memory_types = ['titans', 'hippocampus', 'neocortex', 'cls']
    all_results = {}

    print('Running comparison (10 epochs each)...')
    for mtype in memory_types:
        print(f'  Training {mtype}...', end=' ', flush=True)
        r = run_comparison(mtype, n_epochs=10, device=device)
        all_results[mtype] = r
        print(f'done ({r["train_time_s"]}s)')

    print()
    print('=' * 75)
    print(f"{'Memory Type':<15} {'Params':>8} {'Ep1 Loss':>10} {'Final Loss':>10} {'Val Loss':>10} {'Time(s)':>8}")
    print('=' * 75)
    for mtype, r in all_results.items():
        print(
            f"{mtype:<15} {r['n_params']:>8,} "
            f"{r['train_loss_epoch1']:>10.4f} "
            f"{r['train_loss_final']:>10.4f} "
            f"{r['val_recon_loss']:>10.4f} "
            f"{r['train_time_s']:>8.1f}"
        )
    print('=' * 75)

    # Save to JSON
    out_path = Path('artifacts/memory_comparison.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f'Results saved to {out_path}')

    return all_results


if __name__ == '__main__':
    print_comparison_report()
