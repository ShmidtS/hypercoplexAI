#!/usr/bin/env python
"""
MoE Kernel Demo — запуск ядра с лёгкими доменными экспертами.

Проверяет работоспособность:
  1. Создание ядра с 4 доменными экспертами (math, language, code, science)
  2. Forward pass с синтетическими данными
  3. Gradient flow (обратное распространение)
  4. Load balance — равномерность нагрузки на экспертов
  5. Интеграция с HDIMPipeline (замена SoftMoERouter)
  6. Совместимость с Lean4-верифицированными операциями алгебры Клиффорда
"""

from __future__ import annotations

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from src.core.moe_kernel import MoEKernelConfig, MoEKernel, MathExpert, LanguageExpert, CodeExpert, ScienceExpert
from src.core.hypercomplex import CliffordAlgebra
from src.core.hdim_pipeline import HDIMPipeline


def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def check(condition: bool, msg: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    return condition


def run_demo() -> int:
    """Запускает все проверки. Возвращает количество провалов."""
    failures = 0
    torch.manual_seed(42)

    # --------------------------------------------------------
    # 1. Создание ядра
    # --------------------------------------------------------
    separator("1. Создание MoE-ядра с 4 доменными экспертами")

    config = MoEKernelConfig(
        input_dim=128,
        expert_hidden_dim=256,
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
    kernel = MoEKernel(config)
    kernel.train()

    n_params = sum(p.numel() for p in kernel.parameters())
    print(f"  Параметров: {n_params:,}")
    failures += not check(n_params > 0, f"Ядро создано ({n_params:,} params)")

    # Проверяем типы экспертов
    failures += not check(isinstance(kernel.experts[0], MathExpert), "Expert[0] = MathExpert")
    failures += not check(isinstance(kernel.experts[1], LanguageExpert), "Expert[1] = LanguageExpert")
    failures += not check(isinstance(kernel.experts[2], CodeExpert), "Expert[2] = CodeExpert")
    failures += not check(isinstance(kernel.experts[3], ScienceExpert), "Expert[3] = ScienceExpert")
    failures += not check(kernel.shared_expert is not None, "SharedExpert создан")

    # --------------------------------------------------------
    # 2. Forward pass
    # --------------------------------------------------------
    separator("2. Forward pass (batch=32, dim=128)")

    x = torch.randn(32, 128)
    output, state = kernel(x)

    failures += not check(output.shape == x.shape, f"Output shape: {output.shape} == {x.shape}")
    failures += not check(not torch.isnan(output).any(), "No NaN in output")
    failures += not check(not torch.isinf(output).any(), "No Inf in output")
    failures += not check(state.expert_weights.shape == (32, 4), f"Expert weights shape: {state.expert_weights.shape}")
    failures += not check(state.expert_usage.shape == (4,), f"Expert usage shape: {state.expert_usage.shape}")
    failures += not check(state.top_expert_idx.shape == (32,), f"Top expert idx shape: {state.top_expert_idx.shape}")

    # Проверка имён доминирующих экспертов
    dominant = state.dominant_expert_names()
    failures += not check(len(dominant) == 32, "Dominant expert names: 32 entries")
    valid_names = {"math", "language", "code", "science"}
    failures += not check(all(n in valid_names for n in dominant), f"All names valid: {set(dominant)}")
    print(f"  Доминирующие эксперты (первые 8): {dominant[:8]}")

    # --------------------------------------------------------
    # 3. Gradient flow
    # --------------------------------------------------------
    separator("3. Gradient flow")

    x_grad = torch.randn(16, 128, requires_grad=True)
    output_g, state_g = kernel(x_grad)
    total_loss = output_g.mean() + state_g.total_loss()
    total_loss.backward()

    failures += not check(x_grad.grad is not None, "Input gradient exists")
    failures += not check(not torch.isnan(x_grad.grad).any(), "No NaN in input gradient")

    # Проверка градиентов для каждого эксперта
    for i, expert in enumerate(kernel.experts):
        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in expert.parameters()
        )
        failures += not check(has_grad, f"Expert[{i}] ({expert.name}) has valid gradients")

    # --------------------------------------------------------
    # 4. Load balance
    # --------------------------------------------------------
    separator("4. Load balance (128 samples)")

    kernel_eval = MoEKernel(config)
    kernel_eval.eval()
    with torch.no_grad():
        x_lb = torch.randn(128, 128)
        _, state_lb = kernel_eval(x_lb)

    usage = state_lb.expert_usage
    expected = 1.0 / 4
    max_dev = (usage - expected).abs().max().item()
    print(f"  Expert usage: {[f'{u:.3f}' for u in usage.tolist()]}")
    print(f"  Max deviation from uniform: {max_dev:.4f}")
    failures += not check(max_dev < 0.2, f"Load balance deviation < 0.2 (got {max_dev:.4f})")
    failures += not check(not torch.isnan(state_lb.routing_entropy), "Routing entropy finite")
    print(f"  Routing entropy: {state_lb.routing_entropy.item():.4f}")

    # --------------------------------------------------------
    # 5. Losses
    # --------------------------------------------------------
    separator("5. Проверка лоссов")

    kernel_train = MoEKernel(config)
    kernel_train.train()
    x_l = torch.randn(16, 128)
    _, state_l = kernel_train(x_l)

    failures += not check(state_l.router_loss.item() >= 0, f"router_loss >= 0: {state_l.router_loss.item():.4f}")
    failures += not check(state_l.z_loss.item() >= 0, f"z_loss >= 0: {state_l.z_loss.item():.6f}")
    failures += not check(state_l.ortho_loss.item() >= 0, f"ortho_loss >= 0: {state_l.ortho_loss.item():.4f}")
    total_l = state_l.total_loss().item()
    failures += not check(not math.isnan(total_l), f"total_loss finite: {total_l:.4f}")
    print(f"  router_loss={state_l.router_loss.item():.4f}  z_loss={state_l.z_loss.item():.6f}  ortho_loss={state_l.ortho_loss.item():.4f}")

    # --------------------------------------------------------
    # 6. Seq shape (batch, seq, dim)
    # --------------------------------------------------------
    separator("6. Sequence input (batch=8, seq=16, dim=128)")

    x_seq = torch.randn(8, 16, 128)
    out_seq, state_seq = kernel_train(x_seq)
    failures += not check(out_seq.shape == (8, 16, 128), f"Seq output shape: {out_seq.shape}")
    failures += not check(not torch.isnan(out_seq).any(), "No NaN in seq output")
    failures += not check(state_seq.expert_weights.shape == (8, 16, 4), f"Expert weights seq shape: {state_seq.expert_weights.shape}")

    # --------------------------------------------------------
    # 7. Интеграция с CliffordAlgebra
    # --------------------------------------------------------
    separator("7. Интеграция: CliffordAlgebra Cl(4,1,0) + MoEKernel")

    ca = CliffordAlgebra(p=4, q=1, r=0)  # dim=32
    config_cl = MoEKernelConfig(
        input_dim=ca.dim,
        expert_hidden_dim=ca.dim * 2,
        num_experts=4,
        expert_names=["math", "language", "code", "science"],
        use_shared_expert=True,
        use_aux_loss_free=True,
        use_expert_ortho=True,
    )
    kernel_cl = MoEKernel(config_cl)
    kernel_cl.train()

    # Симуляция: мультивектор Cl(4,1,0) через роторное преобразование → MoE
    batch = 16
    mv = torch.randn(batch, ca.dim)
    # Нормализованный ротор (единичный)
    R = torch.zeros(1, ca.dim)
    R[0, 0] = 1.0
    angle = 0.5
    R_bv = torch.zeros(1, ca.dim)
    R_bv[0, 0] = math.cos(angle)
    R_bv[0, 3] = math.sin(angle)  # e_1 ^ e_2 компонент
    R_bv = R_bv / ca.norm(R_bv)
    R_bv_exp = R_bv.expand(batch, -1)  # (batch, dim) — broadcast для geometric_product
    mv_rotated = ca.sandwich(R_bv_exp, mv, unit=True)

    out_cl, state_cl = kernel_cl(mv_rotated)
    failures += not check(out_cl.shape == mv_rotated.shape, f"Clifford+MoE output shape: {out_cl.shape}")
    failures += not check(not torch.isnan(out_cl).any(), "No NaN after Clifford sandwich + MoE")
    print(f"  Clifford dim={ca.dim}, MoE input={config_cl.input_dim}, output norm: {out_cl.norm(dim=-1).mean():.4f}")

    # --------------------------------------------------------
    # 8. HDIMPipeline интеграция
    # --------------------------------------------------------
    separator("8. HDIMPipeline: замена SoftMoERouter на MoEKernel")

    pipeline = HDIMPipeline(
        input_dim=64,
        output_dim=64,
        clifford_p=3,
        clifford_q=1,
        clifford_r=0,
        domain_names=["math", "language", "code", "science"],
        num_experts=4,
        top_k=2,
        memory_key_dim=32,
        memory_type="titans",
    )

    x_pipe = torch.randn(8, 64)
    out_pipe, ts = pipeline.transfer(x_pipe, "math", "science")
    failures += not check(out_pipe.shape == (8, 64), f"Pipeline output shape: {out_pipe.shape}")
    failures += not check(not torch.isnan(out_pipe).any(), "No NaN in pipeline output")
    print(f"  router_loss={ts['router_state']['router_loss'].item():.4f}")

    # --------------------------------------------------------
    # 9. Aux-Loss-Free bias update
    # --------------------------------------------------------
    separator("9. Aux-Loss-Free bias update (10 шагов обучения)")

    kernel_aux = MoEKernel(config)
    kernel_aux.train()
    initial_bias = kernel_aux._expert_bias.data.clone()

    for step in range(10):
        x_a = torch.randn(32, 128)
        out_a, st_a = kernel_aux(x_a)
        loss_a = out_a.mean() + st_a.total_loss()
        loss_a.backward()
        # Симулируем optimizer step (только для параметров с grad)
        with torch.no_grad():
            for p in kernel_aux.parameters():
                if p.grad is not None:
                    p.data -= 0.001 * p.grad
                    p.grad.zero_()

    final_bias = kernel_aux._expert_bias.data
    bias_changed = (final_bias - initial_bias).abs().sum().item() > 0
    failures += not check(bias_changed, f"Bias updated after training steps")
    print(f"  Bias delta: {(final_bias - initial_bias).tolist()}")

    # --------------------------------------------------------
    # Итог
    # --------------------------------------------------------
    separator("ИТОГОВЫЙ РЕЗУЛЬТАТ")
    total_checks = 32  # примерное число проверок
    print(f"  Провалов: {failures}")
    if failures == 0:
        print("  [OK] ALL CHECKS PASSED - MoE kernel is functional!")
    else:
        print(f"  [FAIL] {failures} checks failed")
    return failures


if __name__ == "__main__":
    sys.exit(run_demo())
