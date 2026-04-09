#!/usr/bin/env python
"""Test HDIM as MoE Router Coordinator."""

import torch
from src.models.hdim_model import HDIMModel, HDIMConfig
from src.models.model_factory import _patch_moe_kernel

print('=== HDIM MoE ROUTER TEST ===\n')

config = HDIMConfig(
    hidden_dim=64,
    num_domains=4,
    num_experts=4,
    memory_type='hbma',
    online_learning=True,
    hallucination_detection=True,
)

model = HDIMModel(config)
_patch_moe_kernel(model, expert_names=["math", "language", "code", "science"])
model.eval()

print('Router Architecture:')
print(f'  Experts: {model.pipeline.moe.expert_names}')
print(f'  Domains: {model._domain_names}')
print(f'  Memory: HBMA (4-level hierarchy)\n')

print('Routing Tests:')
for i in range(5):
    x = torch.randn(1, 64)
    domain_id = torch.tensor([i % 4])

    with torch.no_grad():
        output, weights, inv, slot, aux = model(x, domain_id, return_state=True)

    expert_idx = aux.topk_idx[0, 0].item()
    expert_name = model.pipeline.moe.expert_names[expert_idx]

    print(f'  Input {i+1}: Domain={domain_id.item()} -> Expert={expert_name}')
    print(f'           Risk: {aux.hallucination_risk:.2%}, Memory: {aux.memory_updated}')

print('\n=== ROUTER FUNCTIONS ===')
print('1. MoE Routing: Routes inputs to specialized experts')
print('2. Domain Transfer: Transforms between domain representations')
print('3. Memory Integration: HBMA stores/retrieves patterns')
print('4. Hallucination Detection: Monitors routing confidence')
print('5. Online Learning: Adapts via TTT-style updates')
print('\nHDIM = COORDINATION KERNEL (needs external LLM for text)')
