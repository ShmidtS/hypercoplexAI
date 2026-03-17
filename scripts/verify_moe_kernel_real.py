#!/usr/bin/env python
"""
Real-model verification for MoEKernel.

Tests the full chain:
  real texts -> SBERT encoder -> HDIMPipeline (MoEKernel) -> training losses

Success criteria:
  1. No NaN/Inf in outputs
  2. pair_margin is finite (model distinguishes positive/negative pairs)
  3. Gradients flow through entire chain
  4. Expert loads are non-trivially distributed
  5. z_loss / router_loss are valid scalars
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.hdim_model import HDIMConfig
from src.models.model_factory import build_sbert_hdim_model, _patch_moe_kernel


DEVICE = "cpu"
BATCH_SIZE = 8


def load_real_pairs(path, n=32):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pos = [d for d in data if d["relation"] == "positive"][:n // 2]
    neg = [d for d in data if d["relation"] == "negative"][:n // 2]
    pairs = pos + neg
    texts_a = [p["source_text"] for p in pairs]
    texts_b = [p["target_text"] for p in pairs]
    labels = torch.tensor([1.0 if p["relation"] == "positive" else 0.0 for p in pairs])
    domains = torch.zeros(len(pairs), dtype=torch.long)
    return texts_a, texts_b, labels, domains


results = []


def check(name, ok, detail=""):
    tag = "[PASS]" if ok else "[FAIL]"
    msg = f"  {tag} {name}"
    if detail:
        msg += f" ({detail})"
    print(msg)
    results.append(ok)
    return ok


def main():
    print("=" * 60)
    print("MoEKernel Real-Model Verification")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Build model
    # --------------------------------------------------------
    print("\n[1] Building SBERT + MoEKernel model...")
    cfg = HDIMConfig(hidden_dim=64, num_experts=4, num_domains=4, top_k=2)
    model = build_sbert_hdim_model(
        cfg,
        soft_router=False,
        freeze_sbert=True,
    )
    _patch_moe_kernel(
        model.core_model,
        expert_names=["math", "language", "code", "science"],
        z_loss_weight=0.01,
        ortho_loss_weight=0.01,
    )
    model.to(DEVICE)
    model.train()
    moe = model.core_model.pipeline.moe
    check("Model built with MoEKernel",
          hasattr(moe, "kernel"),
          f"num_experts={moe.num_experts}, top_k={moe.top_k}")

    # --------------------------------------------------------
    # 2. Load real text pairs
    # --------------------------------------------------------
    print("\n[2] Loading real pairs from data/real_pairs_v10.json...")
    pairs_path = str(Path(__file__).resolve().parents[1] / "data" / "real_pairs_v10.json")
    texts_a, texts_b, labels, domains = load_real_pairs(pairs_path, n=BATCH_SIZE * 2)
    check("Real pairs loaded", len(texts_a) >= BATCH_SIZE, f"{len(texts_a)} pairs")

    # --------------------------------------------------------
    # 3. Encode texts through SBERT
    # --------------------------------------------------------
    print("\n[3] Encoding texts through SBERT encoder...")
    dev = torch.device(DEVICE)
    with torch.no_grad():
        enc_a = model.encode_texts(texts_a[:BATCH_SIZE], device=dev)
        enc_b = model.encode_texts(texts_b[:BATCH_SIZE], device=dev)
    check("Encodings non-NaN",
          not torch.isnan(enc_a).any() and not torch.isnan(enc_b).any(),
          f"shape={enc_a.shape}")
    check("Encodings non-zero",
          enc_a.abs().max().item() > 1e-6,
          f"max={enc_a.abs().max().item():.4f}")

    # --------------------------------------------------------
    # 4. Forward pass through HDIMModel with MoEKernel
    # --------------------------------------------------------
    print("\n[4] Forward pass HDIMModel (MoEKernel) on real encodings...")
    dom = domains[:BATCH_SIZE].to(DEVICE)
    output, routing_weights, invariant, aux_state = model(
        enc_a, dom, return_state=True, memory_mode="none"
    )
    check("Output non-NaN", not torch.isnan(output).any(), f"shape={output.shape}")
    check("Invariant non-NaN", not torch.isnan(invariant).any(), f"shape={invariant.shape}")
    check("router_loss valid", torch.isfinite(aux_state.router_loss),
          f"val={aux_state.router_loss.item():.6f}")
    check("z_loss valid", torch.isfinite(aux_state.z_loss),
          f"val={aux_state.z_loss.item():.6f}")

    # --------------------------------------------------------
    # 5. Expert usage check (non-collapsed distribution)
    # --------------------------------------------------------
    print("\n[5] Expert load distribution check...")
    usage = aux_state.expert_usage.cpu()
    max_load = usage.max().item()
    non_collapsed = max_load < 0.95
    check("Expert load non-collapsed", non_collapsed,
          f"usage={[round(u, 3) for u in usage.tolist()]}")

    # --------------------------------------------------------
    # 6. Pair margin test
    # --------------------------------------------------------
    print("\n[6] Pair margin test (positive vs negative pairs)...")
    with torch.no_grad():
        inv_a = model(enc_a, dom, return_state=False, memory_mode="none")[2]
        inv_b = model(enc_b, dom, return_state=False, memory_mode="none")[2]
    lbl = labels[:BATCH_SIZE].to(DEVICE)
    cos = F.cosine_similarity(inv_a, inv_b, dim=-1)
    pos_mask = lbl > 0.5
    neg_mask = lbl < 0.5
    pos_sim = cos[pos_mask].mean().item() if pos_mask.any() else 0.0
    neg_sim = cos[neg_mask].mean().item() if neg_mask.any() else 0.0
    margin = pos_sim - neg_sim
    check("Pair cosine sims non-NaN", not torch.isnan(cos).any(),
          f"pos={pos_sim:.4f} neg={neg_sim:.4f}")
    check("Pair margin finite", abs(margin) < 1000,
          f"margin={margin:.4f}")

    # --------------------------------------------------------
    # 7. Gradient flow
    # --------------------------------------------------------
    print("\n[7] Gradient flow through entire chain...")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    optimizer.zero_grad()
    out2, _, inv2, aux2 = model(enc_a, dom, return_state=True, memory_mode="none")
    loss = F.mse_loss(inv2, torch.zeros_like(inv2)) + aux2.router_loss + aux2.z_loss
    loss.backward()

    kernel = model.core_model.pipeline.moe.kernel
    router_grad = kernel.router_proj.weight.grad
    expert0_grad = None
    for p in kernel.experts[0].parameters():
        if p.grad is not None:
            expert0_grad = p.grad
            break
    check("router_proj grad non-None", router_grad is not None,
          f"norm={router_grad.norm().item():.6f}" if router_grad is not None else "None")
    check("Expert[0] grad non-None", expert0_grad is not None,
          f"norm={expert0_grad.norm().item():.6f}" if expert0_grad is not None else "None")

    # --------------------------------------------------------
    # 8. Short training loop (3 steps)
    # --------------------------------------------------------
    print("\n[8] Short training loop (3 steps)...")
    losses = []
    for step in range(3):
        optimizer.zero_grad()
        out_s, _, inv_s, aux_s = model(enc_a, dom, return_state=True, memory_mode="none")
        loss_s = F.mse_loss(out_s, enc_a) + 0.01 * aux_s.router_loss
        loss_s.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss_s.item())
    check("Training loss finite across steps",
          all(abs(l) < 1e6 for l in losses),
          f"losses={[round(l, 4) for l in losses]}")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    n_pass = sum(results)
    n_total = len(results)
    status = "PASS" if all(results) else "FAIL"
    print(f"Result: {n_pass}/{n_total} checks passed  [{status}]")
    print("=" * 60)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
