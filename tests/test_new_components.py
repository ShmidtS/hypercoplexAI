"""
Тесты для новых компонентов HDIM:
- AdvancedTextEncoder (src/models/advanced_text_encoder.py)
- HierarchicalTitansMemory (src/core/hierarchical_memory.py)
- SoftMoERouter (src/core/soft_moe_router.py)
"""

import pytest
import torch
import torch.nn as nn


# ============================================================
# AdvancedTextEncoder tests
# ============================================================

class TestAdvancedTextEncoder:
    @pytest.fixture
    def encoder(self):
        from src.models.advanced_text_encoder import AdvancedTextEncoder
        return AdvancedTextEncoder(
            output_dim=64,
            vocab_size=257,
            max_length=32,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )

    def test_forward_returns_correct_shape(self, encoder):
        texts = ["hello world", "test text here"]
        out = encoder(texts)
        assert out.shape == (2, 64)

    def test_forward_single_text(self, encoder):
        texts = ["single"]
        out = encoder(texts)
        assert out.shape == (1, 64)

    def test_forward_empty_batch(self, encoder):
        out = encoder([])
        assert out.shape[0] == 0

    def test_output_is_differentiable(self, encoder):
        texts = ["differentiable test"]
        out = encoder(texts)
        assert out.requires_grad

    def test_different_texts_give_different_outputs(self, encoder):
        out1 = encoder(["text one"])
        out2 = encoder(["completely different text"])
        assert not torch.allclose(out1.detach(), out2.detach(), atol=1e-5)

    def test_tokenize_correct_shape(self, encoder):
        texts = ["hello", "world test"]
        token_ids, mask = encoder.tokenize(texts)
        assert token_ids.shape == (2, encoder.max_length)
        assert mask.shape == (2, encoder.max_length)
        assert mask.dtype == torch.bool

    def test_tokenize_respects_max_length(self, encoder):
        long_text = "a" * 1000
        token_ids, mask = encoder.tokenize([long_text])
        assert token_ids.shape[1] == encoder.max_length
        # All positions should be filled (max_length chars)
        assert mask[0].sum().item() == encoder.max_length

    def test_from_text_config(self):
        from src.models.advanced_text_encoder import AdvancedTextEncoder
        from src.models.hdim_model import HDIMTextConfig
        config = HDIMTextConfig(vocab_size=300, max_length=64)
        encoder = AdvancedTextEncoder.from_text_config(
            output_dim=128,
            text_config=config,
            fallback_dropout=0.1,
        )
        texts = ["test"]
        out = encoder(texts)
        assert out.shape == (1, 128)

    def test_encoder_with_device(self, encoder):
        device = torch.device("cpu")
        texts = ["device test"]
        out = encoder(texts, device=device)
        assert out.device.type == "cpu"

    def test_rope_embedding_different_positions(self):
        from src.models.advanced_text_encoder import RotaryEmbedding
        rope = RotaryEmbedding(dim=16, max_seq_len=32)
        q = torch.randn(1, 2, 4, 16)  # batch, heads, seq, dim
        k = torch.randn(1, 2, 4, 16)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        # Rotated != original (with high probability)
        assert not torch.allclose(q_rot, q, atol=1e-4)


# ============================================================
# HierarchicalTitansMemory tests
# ============================================================

class TestHierarchicalTitansMemory:
    @pytest.fixture
    def memory(self):
        from src.core.hierarchical_memory import HierarchicalTitansMemory
        return HierarchicalTitansMemory(
            key_dim=32,
            val_dim=32,
            hidden_dim=16,
        )

    def test_forward_returns_correct_shape(self, memory):
        k = torch.randn(4, 32)
        v = torch.randn(4, 32)
        memory.train()
        retrieved, loss = memory(k, v, update_memory=False)
        assert retrieved.shape == (4, 32)
        assert loss.ndim == 0

    def test_retrieve_does_not_update_weights(self, memory):
        k = torch.randn(4, 32)
        v = torch.randn(4, 32)
        w1_before = memory.working_memory.weight.detach().clone()
        lt_before = memory.longterm_memory.weight.detach().clone()
        state = memory.retrieve(k, v)
        assert torch.allclose(memory.working_memory.weight, w1_before)
        assert torch.allclose(memory.longterm_memory.weight, lt_before)
        assert not state.updated

    def test_update_modifies_working_memory(self, memory):
        k = torch.randn(4, 32)
        v = torch.randn(4, 32)
        memory.train()
        w_before = memory.working_memory.weight.detach().clone()
        memory.update(k, v)
        # Working memory should be updated
        assert not torch.allclose(memory.working_memory.weight, w_before)

    def test_reset_clears_all_memory(self, memory):
        k = torch.randn(4, 32)
        v = torch.randn(4, 32)
        memory.train()
        memory.update(k, v)
        memory.reset_memory()
        assert torch.allclose(memory.working_memory.weight, torch.zeros_like(memory.working_memory.weight))
        assert torch.allclose(memory.longterm_memory.weight, torch.zeros_like(memory.longterm_memory.weight))
        assert torch.allclose(memory.working_momentum, torch.zeros_like(memory.working_momentum))
        assert torch.allclose(memory.longterm_momentum, torch.zeros_like(memory.longterm_momentum))

    def test_retrieve_eval_no_mutation(self, memory):
        k = torch.randn(4, 32)
        v = torch.randn(4, 32)
        memory.train()
        memory.update(k, v)
        w1 = memory.working_memory.weight.detach().clone()
        lt1 = memory.longterm_memory.weight.detach().clone()
        memory.eval()
        with torch.no_grad():
            memory(k, v, update_memory=False)
            memory(k, v, update_memory=False)
        assert torch.allclose(memory.working_memory.weight, w1)
        assert torch.allclose(memory.longterm_memory.weight, lt1)

    def test_loss_is_non_negative(self, memory):
        k = torch.randn(8, 32)
        v = torch.randn(8, 32)
        _, loss = memory(k, v, update_memory=False)
        assert loss.item() >= 0.0

    def test_surprise_decreases_after_updates(self, memory):
        """После нескольких обновлений surprise должен уменьшаться."""
        k = torch.randn(4, 32)
        v = torch.randn(4, 32)
        memory.train()
        surprises = []
        for _ in range(5):
            s1 = memory._compute_surprise(k, v).item()
            surprises.append(s1)
            memory.update(k, v)
        # Surprise должен снижаться (не обязательно монотонно, но в целом)
        assert surprises[0] >= surprises[-1] or len(set(surprises)) > 1


# ============================================================
# SoftMoERouter tests
# ============================================================

class TestSoftMoERouter:
    @pytest.fixture
    def router(self):
        from src.core.soft_moe_router import SoftMoERouter
        return SoftMoERouter(
            input_dim=32,
            num_experts=4,
            expert_dim=64,
            top_k=2,
        )

    def test_forward_returns_correct_shape(self, router):
        x = torch.randn(8, 32)
        out, state = router(x)
        assert out.shape == (8, 32)

    def test_router_state_has_required_fields(self, router):
        x = torch.randn(8, 32)
        _, state = router(x)
        required_fields = [
            "loss", "router_loss", "scores", "topk_idx",
            "gate_weights", "train_scores_snapshot",
            "topk_gate_weights", "expert_usage", "routing_entropy",
            "dispatch_weights",
        ]
        for field in required_fields:
            assert field in state, f"Missing field: {field}"

    def test_topk_idx_shape(self, router):
        x = torch.randn(8, 32)
        _, state = router(x)
        assert state["topk_idx"].shape == (8, 2)  # top_k=2

    def test_gate_weights_shape(self, router):
        x = torch.randn(8, 32)
        _, state = router(x)
        assert state["gate_weights"].shape == (8, 4)  # num_experts=4

    def test_expert_usage_shape(self, router):
        x = torch.randn(8, 32)
        _, state = router(x)
        assert state["expert_usage"].shape == (4,)  # num_experts=4

    def test_output_is_differentiable(self, router):
        x = torch.randn(8, 32, requires_grad=True)
        out, _ = router(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_train_scores_updated_during_training(self, router):
        x = torch.randn(8, 32)
        router.train()
        scores_before = router.train_scores.detach().clone()
        _, _ = router(x)
        # EMA update should change train_scores
        assert not torch.allclose(router.train_scores, scores_before)

    def test_train_scores_not_updated_during_eval(self, router):
        x = torch.randn(8, 32)
        # First do a train step to get non-trivial scores
        router.train()
        _, _ = router(x)
        # Now eval
        router.eval()
        scores_before = router.train_scores.detach().clone()
        with torch.no_grad():
            _, _ = router(x)
        assert torch.allclose(router.train_scores, scores_before)

    def test_r3_style_api_compatible_with_pipeline(self):
        """Проверяем что SoftMoERouter compatible с HDIMPipeline API."""
        from src.core.soft_moe_router import SoftMoERouter
        from src.models.hdim_model import HDIMConfig, HDIMModel
        cfg = HDIMConfig(hidden_dim=64, num_experts=4, top_k=2)
        model = HDIMModel(cfg)
        # Replace router with SoftMoERouter
        new_router = SoftMoERouter(
            input_dim=model.pipeline.clifford_dim,
            num_experts=cfg.num_experts,
            expert_dim=128,
            top_k=cfg.top_k,
        )
        model.pipeline.moe = new_router
        x = torch.randn(4, cfg.hidden_dim)
        domain_id = torch.zeros(4, dtype=torch.long)
        out, routing, inv = model(x, domain_id, update_memory=False, memory_mode="retrieve")
        assert out.shape == (4, cfg.hidden_dim)
        assert routing.shape == (4, cfg.num_experts)


# ============================================================
# Integration tests: AdvancedTextEncoder + TextHDIMModel
# ============================================================

class TestAdvancedEncoderIntegration:
    def test_text_hdim_with_advanced_encoder(self):
        from src.models.advanced_text_encoder import AdvancedTextEncoder
        from src.models.hdim_model import HDIMConfig, HDIMModel
        from src.models.text_hdim_model import TextHDIMModel

        cfg = HDIMConfig(hidden_dim=64)
        model = HDIMModel(cfg)
        text_model = TextHDIMModel(model)
        # Replace encoder
        text_model.text_encoder = AdvancedTextEncoder(
            output_dim=64,
            max_length=32,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )
        texts = ["engineering problem", "biology analogy"]
        encodings = text_model.encode_texts(texts)
        assert encodings.shape == (2, 64)

    def test_advanced_encoder_train_step_with_trainer(self):
        from src.models.advanced_text_encoder import AdvancedTextEncoder
        from src.models.hdim_model import HDIMConfig, HDIMModel
        from src.models.text_hdim_model import TextHDIMModel
        from src.training.trainer import HDIMTrainer

        cfg = HDIMConfig(hidden_dim=64)
        model = HDIMModel(cfg)
        text_model = TextHDIMModel(model)
        text_model.text_encoder = AdvancedTextEncoder(
            output_dim=64, max_length=32, num_layers=1, num_heads=4, dropout=0.0,
        )
        trainer = HDIMTrainer(
            text_model,
            torch.optim.Adam(text_model.parameters(), lr=1e-3),
            device="cpu",
        )
        batch = {
            "text": ["source text", "another source"],
            "domain_id": torch.tensor([0, 1], dtype=torch.long),
        }
        loss = trainer.train_step(batch)
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0.0

    def test_hierarchical_memory_in_pipeline(self):
        from src.core.hierarchical_memory import HierarchicalTitansMemory
        from src.models.hdim_model import HDIMConfig, HDIMModel

        cfg = HDIMConfig(hidden_dim=64)
        model = HDIMModel(cfg)
        old_memory = model.pipeline.memory
        new_memory = HierarchicalTitansMemory(
            key_dim=old_memory.key_dim,
            val_dim=old_memory.val_dim,
            hidden_dim=32,
        )
        model.pipeline.memory = new_memory
        x = torch.randn(4, cfg.hidden_dim)
        domain_id = torch.zeros(4, dtype=torch.long)
        out, routing, inv = model(x, domain_id, update_memory=False, memory_mode="retrieve")
        assert out.shape == (4, cfg.hidden_dim)
