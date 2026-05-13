"""Tests for the current HDIM model/core engine API.

Coverage:
- HDIMModel initialization from HDIMConfig
- Forward and transfer paths
- Core engine domain rotor operations
- Gradient flow through forward
- Encoder/decoder shape checks
"""

import pytest
import torch

from src.core.engine import CoreEngineConfig
from src.core.engine import HDIMCoreEngine
from src.core.rotors import DomainRotationOperator
from src.models.config import HDIMConfig
from src.models.hdim_model import HDIMModel


@pytest.fixture
def config():
    """Fixture: compact HDIM config with named domains."""
    return HDIMConfig(
        hidden_dim=64,
        num_domains=2,
        domain_names=("source", "target"),
        clifford_p=3,
        clifford_q=1,
        clifford_r=0,
        dropout=0.0,
        expert_names=("source_expert", "target_expert"),
        memory_type="titans",
    )


@pytest.fixture
def model(config):
    """Fixture: HDIM model over the core engine."""
    return HDIMModel(config)


@pytest.fixture
def engine(config):
    """Fixture: core engine matching the model dimensions."""
    return HDIMCoreEngine(
        CoreEngineConfig(
            input_dim=config.hidden_dim,
            clifford_p=config.clifford_p,
            clifford_q=config.clifford_q,
            clifford_r=config.clifford_r,
            domain_names=tuple(config.get_domain_names()),
            dropout=config.dropout,
        )
    )


@pytest.fixture
def sample_input():
    """Fixture: test input tensor."""
    return torch.randn(4, 64)


class TestModelInit:
    """Tests for HDIMModel initialization."""

    def test_model_init(self, model, config):
        """Model wires config into the core engine."""
        assert model.config is config
        assert isinstance(model.engine, HDIMCoreEngine)
        assert model.clifford_dim == model.engine.algebra.dim
        assert model.project_in is model.engine.encoder
        assert set(model.domain_rotors.keys()) == {"source", "target"}
        assert model.config.memory_type == "titans"
        assert model.config.num_experts == 2

    def test_model_init_custom_params(self):
        """Custom config controls domains and Clifford algebra size."""
        config = HDIMConfig(
            hidden_dim=128,
            num_domains=3,
            domain_names=("a", "b", "c"),
            clifford_p=2,
            clifford_q=2,
            dropout=0.0,
            expert_names=("e0", "e1", "e2"),
        )
        model = HDIMModel(config)

        assert model.engine.config.input_dim == 128
        assert model.clifford_dim == 16
        assert set(model.domain_rotors.keys()) == {"a", "b", "c"}
        assert model.config.num_experts == 3

    def test_model_init_expert_names(self):
        """expert_names drives the MoE expert count in config."""
        config = HDIMConfig(hidden_dim=64, num_domains=2, expert_names=("expert_0", "expert_1", "expert_2"))

        assert config.num_experts == 3


class TestModelForward:
    """Tests for HDIMModel.forward."""

    def test_model_forward(self, model, sample_input):
        """Forward returns output, routing weights, and training invariant."""
        domain_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        result = model(sample_input, domain_id, return_state=True, update_memory=False, memory_mode="retrieve")

        assert result.output.shape == (4, 64)
        assert result.routing_weights.shape == (4, model.config.num_experts)
        assert result.invariant.shape == (4, 64)
        assert result.aux_state.raw_invariant.shape == (4, model.clifford_dim)
        assert result.aux_state.exported_invariant.shape == (4, model.clifford_dim)
        assert result.aux_state.memory_mode == "retrieve"
        assert result.aux_state.update_memory is False
        assert not torch.isnan(result.output).any()

    def test_model_forward_rejects_invalid_domain_id(self, model, sample_input):
        """Domain ids must be inside config.num_domains."""
        domain_id = torch.tensor([0, 1, 2, 0], dtype=torch.long)

        with pytest.raises(IndexError):
            model(sample_input, domain_id, update_memory=False, memory_mode="retrieve")

    def test_model_forward_rejects_bad_memory_mode(self, model, sample_input):
        """Unsupported memory modes fail before execution."""
        domain_id = torch.zeros(4, dtype=torch.long)

        with pytest.raises(ValueError, match="Unsupported memory_mode"):
            model(sample_input, domain_id, update_memory=False, memory_mode="invalid")


class TestModelTransfer:
    """Tests for transfer behavior."""

    def test_model_transfer(self, model, sample_input):
        """Transfer decodes source domain input into target domain output."""
        output, state = model.transfer(
            sample_input,
            source_domain=0,
            target_domain=1,
            return_state=True,
            update_memory=False,
            memory_mode="retrieve",
        )

        assert output.shape == (4, 64)
        assert state.raw_invariant.shape == (4, model.clifford_dim)
        assert state.exported_invariant.shape == (4, model.clifford_dim)
        assert state.memory_mode == "retrieve"
        assert state.memory_updated is False
        assert not torch.isnan(output).any()

    def test_model_transfer_pairs(self, model, sample_input):
        """Batched source/target domain ids are supported."""
        source_domain_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        target_domain_id = torch.tensor([1, 1, 0, 0], dtype=torch.long)

        result = model.transfer_pairs(
            sample_input,
            source_domain_id,
            target_domain_id,
            update_memory=False,
            memory_mode="retrieve",
        )

        assert result.output.shape == (4, 64)
        assert result.invariant.shape == (4, 64)
        assert result.aux_state.raw_invariant.shape == (4, model.clifford_dim)
        assert result.aux_state.exported_invariant.shape == (4, model.clifford_dim)
        assert len(result.aux_state.matches) == 4

    def test_core_engine_extract_and_transfer(self, engine, sample_input):
        """Core engine exposes encode/extract/transfer building blocks."""
        encoded = engine.encode(sample_input)
        invariant = engine.extract(encoded, "source")
        transferred = engine.transfer(invariant, "target")

        assert encoded.shape == (4, engine.algebra.dim)
        assert invariant.shape == (4, engine.algebra.dim)
        assert transferred.shape == (4, engine.algebra.dim)
        assert not torch.isnan(transferred).any()


class TestModelGradientFlow:
    """Tests for gradient flow."""

    def test_model_gradient_flow(self, model):
        """Gradients pass through HDIMModel.forward."""
        model.train()
        x = torch.randn(2, 64, requires_grad=True)
        domain_id = torch.tensor([0, 1], dtype=torch.long)

        result = model(x, domain_id, update_memory=False, memory_mode="retrieve")
        loss = result.output.sum() + result.invariant.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert x.grad.abs().sum() > 0


class TestCoreEngineDomainOperations:
    """Tests for domain operations through engine.domain_rotors."""

    def test_engine_domain_rotors_present(self, engine):
        """Configured domains create rotor modules."""
        assert set(engine.domain_rotors.keys()) == {"source", "target"}
        assert isinstance(engine.domain_rotors["source"], DomainRotationOperator)
        assert isinstance(engine.domain_rotors["target"], DomainRotationOperator)

    def test_engine_domain_rotor_forward(self, engine):
        """Domain rotor forward preserves Clifford dimension."""
        x = torch.randn(4, engine.algebra.dim)

        output = engine.domain_rotors["source"](x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_engine_add_domain_rotor(self, engine):
        """New domain rotors can be registered on the engine ModuleDict."""
        engine.domain_rotors["new_domain"] = DomainRotationOperator(engine.algebra, domain_name="new_domain")

        assert "new_domain" in engine.domain_rotors
        assert isinstance(engine.domain_rotors["new_domain"], DomainRotationOperator)


class TestModelEncoderDecoder:
    """Tests for encoder and decoder shapes."""

    def test_encoder_output_shape(self, model, sample_input):
        """Encoder maps hidden_dim into Clifford dimension."""
        output = model.project_in(sample_input)

        assert output.shape == (4, model.clifford_dim)
        assert not torch.isnan(output).any()

    def test_decoder_output_shape(self, model):
        """Decoder maps Clifford dimension back into hidden_dim."""
        clifford_input = torch.randn(4, model.clifford_dim)

        output = model.decoder(clifford_input)

        assert output.shape == (4, model.config.hidden_dim)
        assert not torch.isnan(output).any()

    def test_encoder_decoder_no_nan(self, model, sample_input):
        """Encoder and decoder produce finite tensors."""
        encoded = model.project_in(sample_input)
        decoded = model.decoder(encoded)

        assert not torch.isnan(encoded).any()
        assert not torch.isnan(decoded).any()
