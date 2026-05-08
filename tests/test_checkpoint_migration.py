import torch

from src.training.trainer import HDIMTrainer


def test_moe_kernel_checkpoint_migration():
    """Verify old MoEKernelAdapter checkpoint keys migrate after MoE consolidation."""
    old_state = {
        "pipeline.moe.kernel.router_proj.weight": torch.randn(4, 4),
        "pipeline.moe.kernel.experts.0.net.0.weight": torch.randn(4, 4),
        "pipeline.moe.some_other": torch.randn(4, 4),
    }

    migrated = HDIMTrainer._migrate_checkpoint_state_dict(old_state)

    assert "pipeline.moe.router_proj.weight" in migrated
    assert "pipeline.moe.experts.0.net.0.weight" in migrated
    assert "pipeline.moe.some_other" in migrated
    assert "pipeline.moe.kernel.router_proj.weight" not in migrated
    assert "pipeline.moe.kernel.experts.0.net.0.weight" not in migrated
    assert migrated["pipeline.moe.some_other"] is old_state["pipeline.moe.some_other"]
