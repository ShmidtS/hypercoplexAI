"""Тесты для HDIMPipeline — главного пайплайна кроссдоменного переноса.

Покрытие:
- Инициализация pipeline
- Кодирование домена (encode_domain)
- Трансфер между доменами
- Режимы памяти (none/retrieve/update)
- Градиенты проходят через forward
"""

import pytest
import torch

from src.core.hdim_pipeline import HDIMPipeline, HDIMEncoder, HDIMDecoder, TransferState


@pytest.fixture
def pipeline():
    """Фикстура: HDIM pipeline со стандартными параметрами."""
    return HDIMPipeline(
        input_dim=64,
        output_dim=64,
        clifford_p=3,
        clifford_q=1,
        clifford_r=0,
        domain_names=["source", "target"],
        num_experts=4,
        top_k=2,
        memory_key_dim=32,
        memory_type="titans",
    )


@pytest.fixture
def sample_input():
    """Фикстура: тестовый входной тензор."""
    return torch.randn(4, 64)


class TestPipelineInit:
    """Тесты инициализации HDIMPipeline."""

    def test_pipeline_init(self, pipeline):
        """Проверка корректной инициализации pipeline."""
        # Проверка размерностей
        assert pipeline.clifford_dim == pipeline.algebra.dim
        assert pipeline.memory_type == "titans"

        # Проверка наличия компонентов
        assert hasattr(pipeline, 'encoder')
        assert hasattr(pipeline, 'decoder')
        assert hasattr(pipeline, 'domain_rotors')
        assert hasattr(pipeline, 'invariant_extractor')
        assert hasattr(pipeline, 'moe')
        assert hasattr(pipeline, 'memory')

        # Проверка доменов
        assert len(pipeline.domain_names) == 2
        assert "source" in pipeline.domain_rotors
        assert "target" in pipeline.domain_rotors

    def test_pipeline_init_custom_params(self):
        """Инициализация с кастомными параметрами."""
        pipeline = HDIMPipeline(
            input_dim=128,
            output_dim=256,
            clifford_p=2,
            clifford_q=2,
            domain_names=["a", "b", "c"],
            num_experts=8,
            top_k=3,
            memory_type="hbma",
        )

        assert pipeline.memory_type == "hbma"
        assert len(pipeline.domain_names) == 3
        assert len(pipeline.domain_rotors) == 3

    def test_pipeline_init_expert_names(self):
        """Инициализация с expert_names вместо num_experts."""
        pipeline = HDIMPipeline(
            input_dim=64,
            output_dim=64,
            expert_names=["expert_0", "expert_1", "expert_2"],
        )

        # num_experts должен вычисляться из expert_names
        assert pipeline.moe.num_experts == 3


class TestPipelineEncodeDomain:
    """Тесты кодирования домена."""

    def test_pipeline_encode_domain(self, pipeline, sample_input):
        """Кодирование входа в мультивектор и инвариант."""
        g_source, u_inv = pipeline.encode_domain(sample_input, "source")

        # Проверка размерностей
        assert g_source.shape == (4, pipeline.clifford_dim)
        assert u_inv.shape == (4, pipeline.clifford_dim)

        # Проверка, что нет NaN
        assert not torch.isnan(g_source).any()
        assert not torch.isnan(u_inv).any()

    def test_pipeline_encode_different_domains(self, pipeline, sample_input):
        """Кодирование одного входа в разных доменах."""
        g_a, u_a = pipeline.encode_domain(sample_input, "source")
        g_b, u_b = pipeline.encode_domain(sample_input, "target")

        # Инварианты должны быть разными для разных доменов
        # (разные роторы применяются)
        assert g_a.shape == g_b.shape
        assert u_a.shape == u_b.shape


class TestPipelineTransfer:
    """Тесты трансфера между доменами."""

    def test_pipeline_transfer(self, pipeline, sample_input):
        """Полный трансфер source → target."""
        pipeline.train()
        output, state = pipeline.transfer(
            sample_input,
            source_domain="source",
            target_domain="target",
            update_memory=False,
            memory_mode="retrieve",
        )

        # Проверка размерностей выхода
        assert output.shape == (4, 64)

        # Проверка структуры state
        assert isinstance(state, dict)
        assert "output" in state
        assert "u_inv" in state
        assert "u_mem" in state
        assert "u_route" in state
        assert "g_target" in state

        # Проверка размерностей в state
        assert state["output"].shape == (4, 64)
        assert state["g_source"].shape == (4, pipeline.clifford_dim)
        assert state["u_inv"].shape == (4, pipeline.clifford_dim)
        assert state["u_mem"].shape == (4, pipeline.clifford_dim)
        assert state["u_route"].shape == (4, pipeline.clifford_dim)
        assert state["g_target"].shape == (4, pipeline.clifford_dim)

        # Контракт alias-полей TransferState
        assert torch.allclose(state["raw_invariant"], state["u_inv"])
        assert torch.allclose(state["memory_augmented_invariant"], state["u_mem"])
        assert torch.allclose(state["exported_invariant"], state["u_route"])
        assert torch.allclose(state["invariant"], state["u_route"])

    def test_pipeline_transfer_input_is_invariant(self, pipeline):
        """Трансфер когда вход уже является инвариантом."""
        invariant_input = torch.randn(4, pipeline.clifford_dim)

        pipeline.eval()
        output, state = pipeline.transfer(
            invariant_input,
            source_domain="source",
            target_domain="target",
            update_memory=False,
            memory_mode="retrieve",
            input_is_invariant=True,
        )

        assert output.shape == (4, 64)
        assert state["input_is_invariant"] is True
        assert state["g_source"] is None
        assert state["u_inv"].shape == invariant_input.shape
        assert state["u_mem"].shape == invariant_input.shape
        assert state["u_route"].shape == invariant_input.shape
        assert state["g_target"].shape == invariant_input.shape
        assert torch.allclose(state["u_inv"], invariant_input)
        assert not torch.isnan(state["g_target"]).any()
        assert not torch.isnan(output).any()


class TestPipelineMemoryModes:
    """Тесты режимов памяти."""

    def test_pipeline_memory_modes_none(self, pipeline, sample_input):
        """Режим памяти 'none' — память не используется."""
        pipeline.eval()
        output, state = pipeline.transfer(
            sample_input,
            source_domain="source",
            target_domain="target",
            memory_mode="none",
        )

        # u_mem должен быть равен u_inv
        assert torch.allclose(state["u_mem"], state["u_inv"], atol=1e-5)
        assert state["memory_mode"] == "none"

    def test_pipeline_memory_modes_retrieve(self, pipeline, sample_input):
        """Режим памяти 'retrieve' — только чтение."""
        pipeline.eval()
        output, state = pipeline.transfer(
            sample_input,
            source_domain="source",
            target_domain="target",
            memory_mode="retrieve",
            update_memory=False,
        )

        # Память не должна обновиться
        assert state["memory_updated"] is False
        assert state["memory_mode"] == "retrieve"

    def test_pipeline_memory_modes_update(self, pipeline, sample_input):
        """Режим памяти 'update' — чтение и запись."""
        pipeline.train()
        output, state = pipeline.transfer(
            sample_input,
            source_domain="source",
            target_domain="target",
            memory_mode="update",
            update_memory=True,
        )

        # Память должна обновиться
        assert state["memory_updated"] is True
        assert state["memory_mode"] == "update"

    def test_pipeline_memory_modes_invalid(self, pipeline, sample_input):
        """Неверный режим памяти вызывает ошибку."""
        with pytest.raises(ValueError, match="Unsupported memory_mode"):
            pipeline.transfer(
                sample_input,
                source_domain="source",
                target_domain="target",
                memory_mode="invalid_mode",
            )


class TestPipelineGradientFlow:
    """Тесты градиентного потока."""

    def test_pipeline_gradient_flow(self, pipeline):
        """Градиенты проходят через весь pipeline."""
        pipeline.train()
        x = torch.randn(2, 64, requires_grad=True)

        output, state = pipeline(x, update_memory=False, memory_mode="retrieve")

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Проверяем градиенты входа
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_pipeline_gradient_flow_with_memory(self, pipeline):
        """Градиенты проходят с обновлением памяти."""
        pipeline.train()
        x = torch.randn(2, 64, requires_grad=True)

        output, state = pipeline(x, update_memory=True, memory_mode="update")

        loss = output.sum() + state["memory_loss"]
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestPipelineDomainOperations:
    """Тесты операций с доменами."""

    def test_pipeline_add_domain(self, pipeline):
        """Добавление нового домена в runtime."""
        initial_count = len(pipeline.domain_names)

        pipeline.add_domain("new_domain")

        assert len(pipeline.domain_names) == initial_count + 1
        assert "new_domain" in pipeline.domain_rotors

    def test_pipeline_add_duplicate_domain(self, pipeline):
        """Добавление дубликата домена вызывает ошибку."""
        with pytest.raises(ValueError, match="already exists"):
            pipeline.add_domain("source")

    def test_pipeline_reset_memory(self, pipeline):
        """Сброс памяти pipeline."""
        pipeline.train()
        x = torch.randn(2, 64)
        _ = pipeline(x, update_memory=True, memory_mode="update")

        # Сброс памяти
        pipeline.reset_memory(strategy="hard")

        # Проверяем, что память сброшена (для titans)
        if pipeline.memory_type == "titans":
            # TitansAdapter хранит модуль в self.titans
            assert torch.all(pipeline.memory.titans.memory.weight == 0)


class TestPipelineEncoderDecoder:
    """Тесты encoder и decoder."""

    def test_encoder_output_shape(self, pipeline, sample_input):
        """Encoder выдаёт корректную размерность."""
        output = pipeline.encoder(sample_input)
        assert output.shape == (4, pipeline.clifford_dim)

    def test_decoder_output_shape(self, pipeline):
        """Decoder выдаёт корректную размерность."""
        clifford_input = torch.randn(4, pipeline.clifford_dim)
        output = pipeline.decoder(clifford_input)
        assert output.shape == (4, 64)

    def test_encoder_decoder_no_nan(self, pipeline, sample_input):
        """Encoder/Decoder не производят NaN."""
        encoded = pipeline.encoder(sample_input)
        decoded = pipeline.decoder(encoded)

        assert not torch.isnan(encoded).any()
        assert not torch.isnan(decoded).any()


class TestPipelineIsomorphismLoss:
    """Тесты потери изоморфизма."""

    def test_isomorphism_loss(self, pipeline, sample_input):
        """Вычисление потери изоморфизма."""
        domain_pairs = [
            (sample_input, "source", "target"),
            (sample_input[:2], "source", "target"),
        ]

        loss = pipeline.compute_isomorphism_loss(domain_pairs)

        # Проверка, что loss — скаляр
        assert loss.ndim == 0
        assert loss >= 0
        assert not torch.isnan(loss)
