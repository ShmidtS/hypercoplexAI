import importlib
import sys


def test_importing_core_does_not_import_optional_layers():
    for module_name in list(sys.modules):
        if module_name == "src.core" or module_name.startswith("src.core."):
            sys.modules.pop(module_name)
        elif module_name in {"src.models", "src.training", "src.extensions"}:
            sys.modules.pop(module_name)
        elif module_name.startswith(("src.models.", "src.training.", "src.extensions.")):
            sys.modules.pop(module_name)

    importlib.import_module("src.core")

    assert "src.models" not in sys.modules
    assert "src.training" not in sys.modules
    assert "src.extensions" not in sys.modules


def test_core_all_exports_only_public_core_api():
    core = importlib.import_module("src.core")

    assert core.__all__ == [
        "CliffordAlgebra",
        "DomainRotationOperator",
        "InvariantExtractor",
        "sandwich_transfer",
        "InvariantIndex",
        "CoreEngineConfig",
        "HDIMCoreEngine",
    ]
