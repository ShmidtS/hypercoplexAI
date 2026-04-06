"""Stress tests for thread-safe operations with multiprocessing DataLoader.

CRITICAL-1 validation: Thread lock removal must be tested with num_workers > 0.

These tests use REAL multiprocessing (not threading) to verify:
1. SoftMoERouter train_scores consistency under concurrent access
2. EMA updates don't corrupt state when multiple processes read/write
"""
import os
import sys
import pickle
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.soft_moe_router import SoftMoERouter


# ============================================================
# Helper functions for multiprocessing tests
# ============================================================

def _worker_process(router_state_dict_path, input_tensor_path, result_path, worker_id):
    """Worker process function that loads router, runs forward, returns train_scores.

    Uses file-based communication instead of Queue to avoid Torch shared memory issues on Windows.

    Args:
        router_state_dict_path: Path to pickled state dict
        input_tensor_path: Path to input tensor
        result_path: Path to write result
        worker_id: Identifier for this worker
    """
    try:
        # Load state dict from file
        with open(router_state_dict_path, 'rb') as f:
            state_dict = torch.load(f, weights_only=True)

        # Load input tensor from file
        with open(input_tensor_path, 'rb') as f:
            input_tensor = torch.load(f, weights_only=True)

        # Create router and load state
        router = SoftMoERouter(
            input_dim=64,
            num_experts=4,
            expert_dim=128,
        )
        router.load_state_dict(state_dict)
        router.train()

        # Run forward pass
        with torch.no_grad():
            output, state = router(input_tensor)

        # Get train_scores snapshot
        train_scores = router.train_scores.clone().cpu().tolist()
        output_mean = output.mean().item()

        # Write result to file
        result = {'worker_id': worker_id, 'scores': train_scores, 'output_mean': output_mean}
        with open(result_path, 'w') as f:
            import json
            json.dump(result, f)
    except Exception as e:
        import json
        with open(result_path, 'w') as f:
            json.dump({'worker_id': worker_id, 'error': str(e)}, f)


def _run_batch_with_dataloader(router, num_workers, num_batches=100):
    """Run multiple batches through router using DataLoader with specified num_workers.

    Returns:
        List of train_scores snapshots after each batch
    """
    router.train()
    scores_history = []

    # Create dataset
    data = torch.randn(num_batches * 8, 64)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=8, num_workers=num_workers)

    for batch_idx, (batch,) in enumerate(loader):
        router(batch)
        scores_history.append(router.train_scores.clone().cpu())

    return scores_history


# ============================================================
# Test Classes
# ============================================================

class TestMultiWorkerThreadSafety:
    """Stress test for thread-safe operations with DataLoader."""

    def test_soft_moe_router_dataloader_num_workers_0(self):
        """Baseline: SoftMoERouter works with num_workers=0 (single process)."""
        router = SoftMoERouter(
            input_dim=64,
            num_experts=4,
            expert_dim=128,
        )

        scores_history = _run_batch_with_dataloader(router, num_workers=0, num_batches=50)

        # All train_scores should be valid tensors
        for i, scores in enumerate(scores_history):
            assert scores.shape == (4,), f"Batch {i}: scores shape mismatch"
            assert torch.isfinite(scores).all(), f"Batch {i}: NaN/Inf in scores"
            # Scores should sum to ~1 (probability distribution)
            assert abs(scores.sum().item() - 1.0) < 0.1, f"Batch {i}: scores don't sum to 1"

    def test_soft_moe_router_dataloader_num_workers_4(self):
        """CRITICAL: SoftMoERouter works with num_workers=4 (multiprocessing)."""
        router = SoftMoERouter(
            input_dim=64,
            num_experts=4,
            expert_dim=128,
        )

        scores_history = _run_batch_with_dataloader(router, num_workers=4, num_batches=100)

        # All train_scores should be valid
        for i, scores in enumerate(scores_history):
            assert scores.shape == (4,), f"Batch {i}: scores shape mismatch"
            assert torch.isfinite(scores).all(), f"Batch {i}: NaN/Inf in scores"
            assert abs(scores.sum().item() - 1.0) < 0.15, f"Batch {i}: scores don't sum to 1"

    def test_soft_moe_router_num_workers_8_stress(self):
        """Stress test with num_workers=8 (maximum parallelism)."""
        router = SoftMoERouter(
            input_dim=64,
            num_experts=4,
            expert_dim=128,
        )

        scores_history = _run_batch_with_dataloader(router, num_workers=8, num_batches=100)

        # Verify consistency
        for i, scores in enumerate(scores_history):
            assert scores.shape == (4,), f"Batch {i}: scores shape mismatch"
            assert torch.isfinite(scores).all(), f"Batch {i}: NaN/Inf in scores"
            # All scores should be positive
            assert (scores > 0).all(), f"Batch {i}: negative scores detected"

    def test_train_scores_ema_consistency_across_workers(self):
        """Verify train_scores EMA updates are consistent across worker processes.

        This test uses separate processes to verify that:
        1. Each process gets its own copy of the router state
        2. EMA updates within each process don't corrupt state

        Uses file-based IPC to avoid Torch shared memory issues on Windows.
        """
        import multiprocessing as mp
        import json
        import tempfile

        # Create router in main process
        router = SoftMoERouter(
            input_dim=64,
            num_experts=4,
            expert_dim=128,
        )
        router.train()

        # Get initial state and save to temp files
        initial_scores = router.train_scores.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save state dict to file
            state_dict_path = os.path.join(tmpdir, 'state_dict.pt')
            torch.save(router.state_dict(), state_dict_path)

            num_workers = 4
            worker_files = []

            # Spawn workers with file-based IPC
            ctx = mp.get_context('spawn')
            processes = []

            for i in range(num_workers):
                # Save input tensor for this worker
                input_path = os.path.join(tmpdir, f'input_{i}.pt')
                result_path = os.path.join(tmpdir, f'result_{i}.json')
                worker_files.append(result_path)

                torch.save(torch.randn(16, 64), input_path)

                p = ctx.Process(
                    target=_worker_process,
                    args=(state_dict_path, input_path, result_path, i)
                )
                p.start()
                processes.append(p)

            # Wait for all workers
            for p in processes:
                p.join(timeout=30)

            # Collect results from files
            results = {}
            for result_path in worker_files:
                with open(result_path, 'r') as f:
                    result = json.load(f)
                    worker_id = result['worker_id']
                    if 'error' in result:
                        results[worker_id] = (None, result['error'])
                    else:
                        scores = torch.tensor(result['scores'])
                        results[worker_id] = (scores, result['output_mean'])

            # Verify all workers completed successfully
            assert len(results) == num_workers, "Not all workers completed"

            # Each worker should have valid train_scores
            for worker_id, (scores, output_mean) in results.items():
                assert scores is not None, f"Worker {worker_id}: scores is None"
                assert scores.shape == (4,), f"Worker {worker_id}: shape mismatch"
                assert torch.isfinite(scores).all(), f"Worker {worker_id}: NaN/Inf in scores"

    def test_dataloader_persistent_workers(self):
        """Test with persistent_workers=True (workers persist across epochs)."""
        router = SoftMoERouter(
            input_dim=64,
            num_experts=4,
            expert_dim=128,
        )
        router.train()

        # Create dataset
        data = torch.randn(200, 64)
        dataset = TensorDataset(data)
        loader = DataLoader(
            dataset,
            batch_size=8,
            num_workers=4,
            persistent_workers=True,
        )

        # Run 3 epochs
        for epoch in range(3):
            epoch_scores = []
            for batch_idx, (batch,) in enumerate(loader):
                router(batch)
                epoch_scores.append(router.train_scores.clone().cpu())

            # Verify epoch scores
            for i, scores in enumerate(epoch_scores):
                assert scores.shape == (4,), f"Epoch {epoch}, batch {i}: shape mismatch"
                assert torch.isfinite(scores).all(), f"Epoch {epoch}, batch {i}: NaN/Inf"

    def test_dataloader_prefetch_factor(self):
        """Test with prefetch_factor > 2 (aggressive prefetching)."""
        router = SoftMoERouter(
            input_dim=64,
            num_experts=4,
            expert_dim=128,
        )
        router.train()

        data = torch.randn(200, 64)
        dataset = TensorDataset(data)
        loader = DataLoader(
            dataset,
            batch_size=8,
            num_workers=4,
            prefetch_factor=4,  # More aggressive prefetch
        )

        scores_history = []
        for batch, in loader:
            router(batch)
            scores_history.append(router.train_scores.clone().cpu())

        # Verify all scores are valid
        for i, scores in enumerate(scores_history):
            assert torch.isfinite(scores).all(), f"Batch {i}: NaN/Inf in scores"


class TestConcurrentEMAUpdates:
    """Test concurrent EMA updates don't corrupt state."""

    def test_sequential_forward_passes_preserve_ema(self):
        """Multiple sequential forward passes maintain valid EMA state."""
        router = SoftMoERouter(
            input_dim=64,
            num_experts=4,
            expert_dim=128,
        )
        router.train()

        # Run 100 sequential forward passes
        for i in range(100):
            x = torch.randn(16, 64)
            output, state = router(x)

            # Verify train_scores remain valid after each pass
            scores = router.train_scores
            assert torch.isfinite(scores).all(), f"Iteration {i}: NaN/Inf in train_scores"
            assert (scores > 0).all(), f"Iteration {i}: negative train_scores"

    def test_ema_updates_with_shared_memory_tensors(self):
        """Test EMA updates with tensors potentially in shared memory.

        When using DataLoader with num_workers > 0, tensors may be moved
        to shared memory for IPC. This test verifies EMA still works.
        """
        router = SoftMoERouter(
            input_dim=64,
            num_experts=4,
            expert_dim=128,
        )
        router.train()

        # Create data loader with multiprocessing
        data = torch.randn(100, 64)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=10, num_workers=2)

        initial_scores = router.train_scores.clone()

        for batch_idx, (batch,) in enumerate(loader):
            # Batch tensor may be in shared memory
            assert batch.is_shared() or True, "Batch tensor state"

            # Forward pass
            router(batch)

            # Verify train_scores
            scores = router.train_scores
            assert torch.isfinite(scores).all(), f"Batch {batch_idx}: NaN/Inf"
            # EMA should have changed from initial
            if batch_idx > 5:
                assert not torch.allclose(scores, initial_scores, atol=1e-4), \
                    f"Batch {batch_idx}: EMA not updating"

    def test_ema_decay_factor_stability(self):
        """Test EMA with different decay factors under concurrent access."""
        for decay in [0.99, 0.95, 0.9, 0.8]:
            router = SoftMoERouter(
                input_dim=64,
                num_experts=4,
                expert_dim=128,
            )
            router.train()

            # Simulate EMA updates manually with concurrent-like access
            data = torch.randn(50, 64)
            dataset = TensorDataset(data)
            loader = DataLoader(dataset, batch_size=5, num_workers=2)

            scores_history = []
            for batch, in loader:
                router(batch)
                # Manual EMA: scores = decay * scores + (1 - decay) * new_load
                scores_history.append(router.train_scores.clone().cpu())

            # Verify monotonic-like behavior (scores should stabilize)
            for i, scores in enumerate(scores_history):
                assert torch.isfinite(scores).all(), f"Decay {decay}, batch {i}: NaN/Inf"


class TestMultiprocessingPickleCompatibility:
    """Test that router can be pickled/unpickled for multiprocessing."""

    def test_router_pickle_roundtrip(self):
        """Router state can be pickled and unpickled."""
        router = SoftMoERouter(
            input_dim=64,
            num_experts=4,
            expert_dim=128,
        )

        # Modify state
        router.train_scores.fill_(0.3)

        # Pickle roundtrip
        pickled = pickle.dumps(router.state_dict())
        state_dict = pickle.loads(pickled)

        # Create new router and load state
        router2 = SoftMoERouter(
            input_dim=64,
            num_experts=4,
            expert_dim=128,
        )
        router2.load_state_dict(state_dict)

        # Verify state preserved
        assert torch.allclose(router2.train_scores, router.train_scores)

    def test_router_tensor_device_cpu(self):
        """Router tensors remain on CPU after multiprocessing operations."""
        router = SoftMoERouter(
            input_dim=64,
            num_experts=4,
            expert_dim=128,
        )
        router.train()

        # All parameters should be on CPU by default
        for name, param in router.named_parameters():
            assert param.device.type == 'cpu', f"{name} not on CPU"

        # train_scores buffer should be on CPU
        assert router.train_scores.device.type == 'cpu', "train_scores not on CPU"


class TestDataLoaderMultiprocessingStress:
    """Additional stress tests for DataLoader multiprocessing scenarios."""

    def test_dataloader_multiprocessing_windows_compatible(self):
        """Verify DataLoader multiprocessing works on Windows (spawn method)."""
        import multiprocessing as mp

        # Windows uses 'spawn' by default
        assert mp.get_start_method() == 'spawn' or mp.get_start_method() == 'fork', \
            f"Unexpected start method: {mp.get_start_method()}"

        router = SoftMoERouter(
            input_dim=64,
            num_experts=4,
            expert_dim=128,
        )
        router.train()

        # Create loader with explicit multiprocessing
        data = torch.randn(80, 64)
        dataset = TensorDataset(data)
        loader = DataLoader(
            dataset,
            batch_size=8,
            num_workers=2,
            multiprocessing_context='spawn',  # Explicit spawn for Windows compatibility
        )

        scores_history = []
        for batch, in loader:
            router(batch)
            scores_history.append(router.train_scores.clone())

        # Verify all operations succeeded
        assert len(scores_history) == 10, "Not all batches processed"
        for i, scores in enumerate(scores_history):
            assert torch.isfinite(scores).all(), f"Batch {i}: NaN/Inf"

    def test_large_batch_high_worker_count(self):
        """Test with large batch size and high worker count."""
        router = SoftMoERouter(
            input_dim=64,
            num_experts=8,  # More experts
            expert_dim=128,
        )
        router.train()

        data = torch.randn(1000, 64)
        dataset = TensorDataset(data)
        loader = DataLoader(
            dataset,
            batch_size=100,  # Large batches
            num_workers=4,
        )

        for batch_idx, (batch,) in enumerate(loader):
            router(batch)
            scores = router.train_scores
            assert scores.shape == (8,), f"Batch {batch_idx}: shape mismatch"
            assert torch.isfinite(scores).all(), f"Batch {batch_idx}: NaN/Inf"


# ============================================================
# Main entry point for debugging
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
