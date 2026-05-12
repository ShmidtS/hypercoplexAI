import torch
import torch.nn as nn

from src.core.domain_operators import DomainRotationOperator
from src.core.hypercomplex import CliffordAlgebra
from src.core.transfer_engine import TransferEngine


class ScalingRouter(nn.Module):
    num_experts = 1
    top_k = 1

    def forward(self, x: torch.Tensor):
        return x * 2, {"routed": True}


def _make_transfer_engine(router=None):
    algebra = CliffordAlgebra(p=3, q=0, r=0)
    engine = TransferEngine(
        clifford_dim=algebra.dim,
        output_dim=4,
        algebra=algebra,
        router=router,
    )
    engine.eval()
    source_rotor = DomainRotationOperator(algebra=algebra, domain_name="source")
    target_rotor = DomainRotationOperator(algebra=algebra, domain_name="target")
    return engine, source_rotor, target_rotor, algebra.dim


def test_transfer_engine_without_router_uses_identity_routing():
    engine, source_rotor, target_rotor, clifford_dim = _make_transfer_engine(router=None)
    u_mem = torch.randn(2, clifford_dim)

    with torch.no_grad():
        output, state = engine.transfer(
            u_mem=u_mem,
            source_rotor=source_rotor,
            target_rotor=target_rotor,
            input_is_invariant=True,
        )

    assert output.shape == (2, 4)
    assert torch.equal(state["u_route"], u_mem)
    assert "routed" not in state


def test_transfer_engine_with_router_uses_router_output():
    engine, source_rotor, target_rotor, clifford_dim = _make_transfer_engine(router=ScalingRouter())
    u_mem = torch.randn(2, clifford_dim)

    with torch.no_grad():
        output, state = engine.transfer(
            u_mem=u_mem,
            source_rotor=source_rotor,
            target_rotor=target_rotor,
            input_is_invariant=True,
        )

    assert output.shape == (2, 4)
    assert torch.equal(state["u_route"], u_mem * 2)
    assert state["routed"] is True


def test_moe_extension_classes_load():
    from src.extensions.moe import MoEKernel, MoEKernelConfig, MoERouter, SoftMoERouter

    assert issubclass(MoEKernel, MoERouter)
    assert issubclass(SoftMoERouter, MoERouter)
    cfg = MoEKernelConfig(input_dim=8, expert_hidden_dim=16, num_experts=2)
    kernel = MoEKernel(cfg)
    router = SoftMoERouter(input_dim=8, num_experts=2, expert_dim=16)
    assert kernel.num_experts == 2
    assert router.num_experts == 2
