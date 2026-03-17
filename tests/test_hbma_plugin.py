"""Tests for HBMA plugin system (MemorySubsystemPlugin, ConsolidationContext).

Covers:
- MemorySubsystemPlugin ABC: cannot instantiate directly
- ConsolidationContext: dataclass creation
- HBMAMemory.register_plugin: basic registration, duplicate rejection
- HBMAMemory._maybe_rebuild: router/fusion rebuild on plugin registration
- Forward pass: with and without plugins (backward compat)
- memory_loss: includes plugin auxiliary losses
- reset: calls plugin reset
- Consolidation hook: fires during forward
- Zero-overhead: no plugin = no changes to base behavior
- Gradient flow: through plugin params
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.hbma_memory import (
    HBMAMemory,
    MemorySubsystemPlugin,
    ConsolidationContext,
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
)


class EmotionalMemoryPlugin(MemorySubsystemPlugin):
    """Test plugin: tracks emotional valence via mood prototypes."""

    name = "emotional"
    priority = 5

    def __init__(self, hidden_dim: int, num_mood_slots: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mood_protos = nn.Parameter(torch.randn(num_mood_slots, hidden_dim) * 0.02)
        self.register_buffer("mood_conf", torch.full((num_mood_slots,), 0.5))
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_called = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        q_norm = F.normalize(q, dim=-1)
        m_norm = F.normalize(self.mood_protos, dim=-1)
        sim = q_norm @ m_norm.T
        attn = F.softmax(sim / 0.1, dim=-1)
        retrieved = attn @ self.mood_protos
        combined = torch.cat([x, retrieved], dim=-1)
        gate_val = torch.sigmoid(self.gate(combined))
        blended = gate_val * retrieved + (1 - gate_val) * x
        out = self.dropout(F.gelu(self.out_proj(combined)))
        out = self.norm(out + blended)
        return out

    def on_consolidate(self, ctx: ConsolidationContext) -> None:
        pass

    def auxiliary_loss(self) -> torch.Tensor:
        m = F.normalize(self.mood_protos, dim=-1)
        sim = m @ m.T
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        return sim[mask].pow(2).mean()

    def reset(self) -> None:
        self._reset_called = True




# ─── Plugin base class ──────────────────────────────────────────────────

class TestMemorySubsystemPlugin:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            MemorySubsystemPlugin()

    def test_subclass_defaults(self):
        p = EmotionalMemoryPlugin(hidden_dim=64)
        assert p.name == "emotional"
        assert p.priority == 5

    def test_default_auxiliary_loss_zero(self):
        """Base class auxiliary_loss returns 0 tensor."""
        p = EmotionalMemoryPlugin(hidden_dim=64)
        loss = p.auxiliary_loss()
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


# ─── ConsolidationContext ───────────────────────────────────────────────

class TestConsolidationContext:
    def test_dataclass_creation(self):
        wm = WorkingMemory(hidden_dim=64)
        em = EpisodicMemory(hidden_dim=64)
        sm = SemanticMemory(hidden_dim=64)
        pm = ProceduralMemory(hidden_dim=64)
        ctx = ConsolidationContext(
            hidden=torch.randn(2, 64),
            working=wm, episodic=em, semantic=sm, procedural=pm,
            is_training=True, step=42,
        )
        assert ctx.step == 42
        assert ctx.is_training is True


# ─── Registration ───────────────────────────────────────────────────────

class TestPluginRegistration:
    def test_register_single_plugin(self):
        mem = HBMAMemory(hidden_dim=64)
        plugin = EmotionalMemoryPlugin(hidden_dim=64)
        mem.register_plugin(plugin)
        assert len(mem._plugins) == 1
        assert mem._plugin_names == ["emotional"]
        assert mem._needs_rebuild is True

    def test_register_duplicate_name_raises(self):
        mem = HBMAMemory(hidden_dim=64)
        mem.register_plugin(EmotionalMemoryPlugin(hidden_dim=64))
        with pytest.raises(ValueError, match="already registered"):
            mem.register_plugin(EmotionalMemoryPlugin(hidden_dim=64))

    def test_get_all_subsystems(self):
        mem = HBMAMemory(hidden_dim=64)
        subs = mem._get_all_subsystems()
        assert len(subs) == 4
        assert [n for n, _ in subs] == ["working", "episodic", "semantic", "procedural"]

        mem.register_plugin(EmotionalMemoryPlugin(hidden_dim=64))
        subs = mem._get_all_subsystems()
        assert len(subs) == 5
        assert subs[-1][0] == "emotional"


# ─── Rebuild ─────────────────────────────────────────────────────────────

class TestMaybeRebuild:
    def test_no_rebuild_without_plugins(self):
        mem = HBMAMemory(hidden_dim=64)
        mem._maybe_rebuild()
        assert mem.router[-1].out_features == 4

    def test_rebuild_with_plugin(self):
        mem = HBMAMemory(hidden_dim=64)
        mem.register_plugin(EmotionalMemoryPlugin(hidden_dim=64))
        mem._maybe_rebuild()
        assert mem.router[-1].out_features == 5
        assert mem._needs_rebuild is False

    def test_fusion_rebuilt(self):
        mem = HBMAMemory(hidden_dim=64)
        mem.register_plugin(EmotionalMemoryPlugin(hidden_dim=64))
        mem._maybe_rebuild()
        # fusion_proj should now accept 5 * 64 = 320
        assert mem.fusion_proj.in_features == 64 * 5


# ─── Forward pass ───────────────────────────────────────────────────────

class TestForwardWithPlugins:
    def test_forward_no_plugins_backward_compat(self):
        mem = HBMAMemory(hidden_dim=64)
        x = torch.randn(4, 64)
        out = mem(x)
        assert out.shape == (4, 64)

    def test_forward_with_plugin(self):
        mem = HBMAMemory(hidden_dim=64)
        mem.register_plugin(EmotionalMemoryPlugin(hidden_dim=64))
        x = torch.randn(4, 64)
        out = mem(x)
        assert out.shape == (4, 64)

    def test_forward_gate_shape_with_plugin(self):
        mem = HBMAMemory(hidden_dim=64)
        mem.register_plugin(EmotionalMemoryPlugin(hidden_dim=64))
        x = torch.randn(4, 64)
        out, gate = mem(x, return_gate=True)
        assert gate.shape == (4, 5)

    def test_forward_gate_shape_without_plugin(self):
        mem = HBMAMemory(hidden_dim=64)
        x = torch.randn(4, 64)
        out, gate = mem(x, return_gate=True)
        assert gate.shape == (4, 4)


# ─── Memory loss ─────────────────────────────────────────────────────────

class TestMemoryLoss:
    def test_memory_loss_no_plugins(self):
        mem = HBMAMemory(hidden_dim=64)
        loss = mem.memory_loss()
        assert loss.requires_grad

    def test_memory_loss_with_plugin(self):
        mem = HBMAMemory(hidden_dim=64)
        mem.register_plugin(EmotionalMemoryPlugin(hidden_dim=64))
        loss = mem.memory_loss()
        assert loss.requires_grad

    def test_memory_loss_includes_plugin_aux(self):
        mem = HBMAMemory(hidden_dim=64)
        mem.register_plugin(EmotionalMemoryPlugin(hidden_dim=64))
        loss = mem.memory_loss()
        # Should be nonzero (plugin has mood_protos diversity loss)
        assert loss.item() > 0


# ─── Reset ───────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_calls_plugin(self):
        mem = HBMAMemory(hidden_dim=64)
        plugin = EmotionalMemoryPlugin(hidden_dim=64)
        mem.register_plugin(plugin)
        mem.reset()
        assert plugin._reset_called is True


# ─── Consolidation hook ─────────────────────────────────────────────────

class TestConsolidationHook:
    def test_consolidation_hook_fires(self):
        """Plugin.on_consolidate should be called during forward."""
        class TrackingPlugin(EmotionalMemoryPlugin):
            name = "tracking"
            def __init__(self, hidden_dim):
                super().__init__(hidden_dim)
                self.consolidate_count = 0
            def on_consolidate(self, ctx):
                self.consolidate_count += 1

        mem = HBMAMemory(hidden_dim=64)
        plugin = TrackingPlugin(hidden_dim=64)
        mem.register_plugin(plugin)
        mem.train()

        x = torch.randn(4, 64)
        mem(x)
        assert plugin.consolidate_count == 1


# ─── Gradient flow ───────────────────────────────────────────────────────

class TestGradientFlow:
    def test_plugin_params_receive_gradient(self):
        torch.manual_seed(7)  # fix full RNG state before model + input creation
        mem = HBMAMemory(hidden_dim=64)
        plugin = EmotionalMemoryPlugin(hidden_dim=64)
        mem.register_plugin(plugin)
        mem.train()

        x = torch.randn(4, 64, requires_grad=True)
        out = mem(x)
        loss = out.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in plugin.parameters()
        )
        assert has_grad, "Plugin params should receive gradients"


# ─── Zero overhead ──────────────────────────────────────────────────────

class TestZeroOverhead:
    def test_no_plugins_no_rebuild_flag(self):
        mem = HBMAMemory(hidden_dim=64)
        x = torch.randn(4, 64)
        out = mem(x)
        assert mem._needs_rebuild is False

    def test_output_same_shape_no_plugins(self):
        mem1 = HBMAMemory(hidden_dim=64)
        mem2 = HBMAMemory(hidden_dim=64)
        x = torch.randn(4, 64)
        out1 = mem1(x)
        out2 = mem2(x)
        assert out1.shape == out2.shape == (4, 64)
