import pytest

from src.models.config import HDIMConfig


def test_minimal_config_constructs_core():
    cfg = HDIMConfig(
        hidden_dim=128,
        num_domains=2,
        domain_names=("source", "target"),
    )

    assert cfg.hidden_dim == 128
    assert cfg.num_domains == 2
    assert cfg.domain_names == ("source", "target")
    assert cfg.extensions == {}


def test_old_memory_type_emits_warning():
    with pytest.warns(DeprecationWarning, match="memory_type is deprecated"):
        cfg = HDIMConfig(memory_type="titans")

    assert cfg.extensions["memory"]["memory_type"] == "titans"


def test_old_num_experts_emits_warning():
    with pytest.warns(DeprecationWarning, match="num_experts is deprecated"):
        cfg = HDIMConfig(num_experts=4)

    assert cfg.extensions["moe"]["num_experts"] == 4


def test_domain_names_length_equals_num_domains():
    with pytest.raises(ValueError, match="must equal num_domains"):
        HDIMConfig(num_domains=2, domain_names=("source",))


def test_extension_config_roundtrip():
    extensions = {
        "moe": {"num_experts": 8, "top_k": 3},
        "memory": {"memory_type": "msa"},
        "custom": {"enabled": True},
    }

    cfg = HDIMConfig(extensions=extensions)

    assert cfg.extensions == extensions
