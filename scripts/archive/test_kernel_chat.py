#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test HDIM Kernel Chat programmatically."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from scripts.interactive_kernel_chat import HDIMKernelChat

def test_kernel_chat():
    """Test kernel chat with checkpoint."""
    print("=" * 60)
    print("HDIM Kernel Chat Test")
    print("=" * 60)

    # Initialize with best checkpoint
    chat = HDIMKernelChat(
        checkpoint_path="artifacts/run_018/checkpoints/best.pt",
        demo_mode=False
    )

    # Test queries
    test_queries = [
        "What is Fourier transform?",
        "Explain DNA replication",
        "How does gradient descent work?",
        "Compare recursion and iteration",
    ]

    results = []
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("-" * 60)

        result = chat.encode(query)

        # Correct field access: routing is nested dict
        domain_id = result['routing']['dominant_expert']
        print(f"Domain: {chat.expert_names[domain_id]}")
        print(f"Routing weights: {result['routing']['weights']}")
        print(f"Embedding shape: {result['embedding'].shape}")

        # Generate response (finds analogy)
        response = chat.generate_response(result, query)
        print(f"\nResponse:\n{response}")

        results.append({
            "query": query,
            "domain": chat.expert_names[domain_id],
            "routing": result['routing'],
        })

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        weights_str = ", ".join([f"{chat.expert_names[i]}={w:.1%}" for i, w in enumerate(r['routing']['weights'])])
        print(f"{r['domain']:10} | {weights_str}")

    return results

if __name__ == "__main__":
    test_kernel_chat()
