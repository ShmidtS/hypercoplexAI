"""MoE gradient flow and loss tests."""
import pytest
import torch

from src.extensions.moe import MoEKernel, MoEKernelConfig


@pytest.fixture
def config():
    return MoEKernelConfig(
        input_dim=64,
        expert_hidden_dim=128,
        num_experts=4,
        slots_per_expert=1,
        temperature=1.0,
        z_loss_weight=0.01,
        ortho_loss_weight=0.01,
        use_shared_expert=True,
        use_aux_loss_free=True,
        use_expert_ortho=True,
        expert_names=["math", "language", "code", "science"],
    )


@pytest.fixture
def kernel_train(config):
    k = MoEKernel(config)
    k.train()
    return k


class TestGradientFlow:
    def test_input_gradient(self, kernel_train):
        x = torch.randn(8, 64, requires_grad=True)
        out, state = kernel_train(x)
        loss = out.mean() + state["total_loss"]
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_expert_gradients_all(self, kernel_train):
        x = torch.randn(8, 64)
        out, state = kernel_train(x)
        loss = out.mean() + state["total_loss"]
        loss.backward()
        for expert in kernel_train.experts:
            for p in expert.parameters():
                assert p.grad is not None
                assert not torch.isnan(p.grad).any()

    def test_router_proj_gradient(self, kernel_train):
        x = torch.randn(8, 64)
        out, state = kernel_train(x)
        loss = out.mean() + state["router_loss"]
        loss.backward()
        assert kernel_train.router_proj.weight.grad is not None
        assert not torch.isnan(kernel_train.router_proj.weight.grad).any()

    def test_shared_expert_gradient(self, kernel_train):
        x = torch.randn(8, 64)
        out, state = kernel_train(x)
        loss = out.mean()
        loss.backward()
        for p in kernel_train.shared_expert.parameters():
            assert p.grad is not None


class TestLosses:
    def test_router_loss_nonneg(self, kernel_train):
        x = torch.randn(8, 64)
        _, state = kernel_train(x)
        assert state["router_loss"].item() >= 0

    def test_z_loss_nonneg(self, kernel_train):
        x = torch.randn(8, 64)
        _, state = kernel_train(x)
        assert state["z_loss"].item() >= 0

    def test_ortho_loss_nonneg(self, kernel_train):
        x = torch.randn(8, 64)
        _, state = kernel_train(x)
        assert state["ortho_loss"].item() >= 0

    def test_total_loss_finite(self, kernel_train):
        x = torch.randn(8, 64)
        _, state = kernel_train(x)
        total = state["total_loss"]
        assert torch.isfinite(total)

    def test_z_loss_weight_zero_gives_zero(self, config):
        cfg = MoEKernelConfig(
            input_dim=64, expert_hidden_dim=128, num_experts=4,
            z_loss_weight=0.0,
            expert_names=["math", "language", "code", "science"]
        )
        k = MoEKernel(cfg)
        k.train()
        x = torch.randn(8, 64)
        _, state = k(x)
        assert state["z_loss"].item() == 0.0

    def test_ortho_loss_disabled(self, config):
        cfg = MoEKernelConfig(
            input_dim=64, expert_hidden_dim=128, num_experts=4,
            use_expert_ortho=False,
            expert_names=["math", "language", "code", "science"]
        )
        k = MoEKernel(cfg)
        k.train()
        x = torch.randn(8, 64)
        _, state = k(x)
        assert state["ortho_loss"].item() == 0.0

    def test_expert_ortho_loss_positive(self, kernel_train):
        # expert_orthogonalization_loss должен быть >= 0 (Gram-matrix deviation)
        loss = kernel_train.expert_orthogonalization_loss()
        assert loss.item() >= 0

    def test_router_similarity_loss_positive(self, kernel_train):
        loss = kernel_train.router_similarity_loss()
        assert loss.item() >= 0
