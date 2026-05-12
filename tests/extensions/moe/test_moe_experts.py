"""MoE expert creation and behavior tests."""
import torch

from src.extensions.moe import (
    MoEKernel, MoEKernelConfig,
    MLPExpert, EXPERT_CONFIGS,
    _create_mlp_expert,
)


class TestMLPExperts:
    def test_math_expert_forward(self):
        expert = MLPExpert(input_dim=64, hidden_dim=128, name="math", config=EXPERT_CONFIGS["math"])
        x = torch.randn(8, 64)
        out = expert(x)
        assert out.shape == (8, 64)
        assert not torch.isnan(out).any()

    def test_language_expert_forward(self):
        expert = MLPExpert(input_dim=64, hidden_dim=128, name="language", config=EXPERT_CONFIGS["language"])
        x = torch.randn(8, 64)
        out = expert(x)
        assert out.shape == (8, 64)
        assert not torch.isnan(out).any()

    def test_code_expert_forward(self):
        expert = MLPExpert(input_dim=64, hidden_dim=128, name="code", config=EXPERT_CONFIGS["code"])
        x = torch.randn(8, 64)
        out = expert(x)
        assert out.shape == (8, 64)
        assert not torch.isnan(out).any()

    def test_science_expert_forward(self):
        expert = MLPExpert(input_dim=64, hidden_dim=128, name="science", config=EXPERT_CONFIGS["science"])
        x = torch.randn(8, 64)
        out = expert(x)
        assert out.shape == (8, 64)
        assert not torch.isnan(out).any()

    def test_create_mlp_expert_by_name(self):
        for name in ["math", "language", "code", "science"]:
            e = _create_mlp_expert(name, input_dim=32, hidden_dim=64, dropout=0.0)
            assert e.name == name
            assert isinstance(e, MLPExpert)

    def test_create_mlp_expert_unknown_falls_back(self):
        e = _create_mlp_expert("unknown_domain", input_dim=32, hidden_dim=64, dropout=0.0)
        assert isinstance(e, MLPExpert)
        assert e.name == "unknown_domain"

    def test_expert_configs_keys(self):
        assert set(EXPERT_CONFIGS.keys()) == {"math", "language", "code", "science"}


class TestExpertConfigs:
    """Tests for EXPERT_CONFIGS dict (replaces registry tests)."""

    def test_expert_configs_has_builtin_domains(self):
        """EXPERT_CONFIGS contains built-in domain configs."""
        assert "math" in EXPERT_CONFIGS
        assert "language" in EXPERT_CONFIGS
        assert "code" in EXPERT_CONFIGS
        assert "science" in EXPERT_CONFIGS

    def test_create_mlp_expert_with_custom_name(self):
        """_create_mlp_expert works with custom names (no registry needed)."""
        e = _create_mlp_expert("custom_domain", input_dim=32, hidden_dim=64, dropout=0.0)
        assert isinstance(e, MLPExpert)
        assert e.name == "custom_domain"

    def test_moe_kernel_with_custom_expert_names(self):
        """MoEKernel works with custom expert names (no registry needed)."""
        cfg = MoEKernelConfig(
            input_dim=64,
            num_experts=2,
            expert_names=["custom_a", "custom_b"],
        )
        kernel = MoEKernel(cfg)
        kernel.eval()
        assert kernel.experts[0].name == "custom_a"
        assert kernel.experts[1].name == "custom_b"

    def test_expert_configs_override(self):
        """Custom expert config can be added to EXPERT_CONFIGS."""
        EXPERT_CONFIGS["test_domain"] = {"activation": "relu"}
        e = _create_mlp_expert("test_domain", input_dim=32, hidden_dim=64, dropout=0.0)
        assert isinstance(e, MLPExpert)
        assert e.name == "test_domain"
        # Cleanup
        del EXPERT_CONFIGS["test_domain"]


class TestCANExperts:
    """Tests for CAN-style experts using CliffordInteractionLayer."""

    def test_domain_expert_use_can_flag(self):
        """MLPExpert with use_can=True uses CliffordInteractionLayer as pre_hook."""
        expert = MLPExpert(input_dim=16, hidden_dim=32, use_can=True)
        assert expert.use_can is True
        assert hasattr(expert, 'pre_hook')
        assert expert.pre_hook is not None

    def test_domain_expert_use_can_false(self):
        """MLPExpert with use_can=False uses standard FFN."""
        expert = MLPExpert(input_dim=64, hidden_dim=128, use_can=False)
        assert expert.use_can is False
        assert hasattr(expert, 'net')

    def test_domain_expert_forward_can(self):
        """Forward pass works with CAN-enabled expert."""
        expert = MLPExpert(input_dim=16, hidden_dim=32, use_can=True)
        expert.eval()
        x = torch.randn(4, 16)
        out = expert(x)
        assert out.shape == (4, 16)
        assert not torch.isnan(out).any()

    def test_domain_expert_forward_ffn(self):
        """Forward pass works with standard FFN expert."""
        expert = MLPExpert(input_dim=64, hidden_dim=128, use_can=False)
        expert.eval()
        x = torch.randn(4, 64)
        out = expert(x)
        assert out.shape == (4, 64)
        assert not torch.isnan(out).any()

    def test_create_expert_with_can(self):
        """_create_mlp_expert function supports use_can parameter."""
        expert = _create_mlp_expert(
            name="custom",
            input_dim=16,
            hidden_dim=32,
            dropout=0.1,
            use_can=True,
        )
        assert expert.use_can is True
        assert expert.input_dim == 16

    def test_moe_kernel_config_use_can_experts(self):
        """MoEKernelConfig has use_can_experts parameter."""
        cfg = MoEKernelConfig(
            input_dim=16,
            expert_hidden_dim=32,
            num_experts=2,
            use_can_experts=True,
        )
        assert cfg.use_can_experts is True

    def test_moe_kernel_with_can_experts(self):
        """MoEKernel works with CAN-enabled experts."""
        cfg = MoEKernelConfig(
            input_dim=16,
            expert_hidden_dim=32,
            num_experts=2,
            expert_names=["custom_0", "custom_1"],
            use_can_experts=True,
        )
        kernel = MoEKernel(cfg)
        kernel.eval()

        # Verify experts use CAN
        for expert in kernel.experts:
            assert expert.use_can is True

        # Forward pass
        x = torch.randn(8, 16)
        out, state = kernel(x)
        assert out.shape == (8, 16)
        assert not torch.isnan(out).any()

    def test_moe_kernel_backward_compatibility(self):
        """MoEKernel with use_can_experts=False uses standard FFN."""
        cfg = MoEKernelConfig(
            input_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            expert_names=["math", "language", "code", "science"],
            use_can_experts=False,
        )
        kernel = MoEKernel(cfg)
        kernel.eval()

        # Verify experts use FFN
        for expert in kernel.experts:
            assert expert.use_can is False

        # Forward pass
        x = torch.randn(8, 64)
        out, state = kernel(x)
        assert out.shape == (8, 64)
        assert not torch.isnan(out).any()

    def test_can_experts_gradient_flow(self):
        """Gradients flow correctly through CAN experts."""
        cfg = MoEKernelConfig(
            input_dim=16,
            expert_hidden_dim=32,
            num_experts=2,
            expert_names=["a", "b"],
            use_can_experts=True,
        )
        kernel = MoEKernel(cfg)
        kernel.train()

        x = torch.randn(4, 16, requires_grad=True)
        out, state = kernel(x)
        loss = out.mean() + state["total_loss"]
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_can_experts_with_sequence_input(self):
        """CAN experts handle sequence input (B, T, D)."""
        cfg = MoEKernelConfig(
            input_dim=16,
            expert_hidden_dim=32,
            num_experts=2,
            expert_names=["a", "b"],
            use_can_experts=True,
        )
        kernel = MoEKernel(cfg)
        kernel.eval()

        # Sequence input: (batch=2, seq=4, dim=16)
        x = torch.randn(2, 4, 16)
        out, state = kernel(x)
        assert out.shape == (2, 4, 16)
        assert not torch.isnan(out).any()

    def test_mixed_expert_types_not_supported(self):
        """Specialized experts (MathExpert etc.) now use MLPExpert with config."""
        # MathExpert is now MLPExpert(name="math", config=EXPERT_CONFIGS["math"])
        math_expert = MLPExpert(input_dim=64, hidden_dim=128, name="math", config=EXPERT_CONFIGS["math"])
        # These experts use FFN by default (no CAN)
        assert hasattr(math_expert, 'net')
