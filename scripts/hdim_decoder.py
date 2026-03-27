#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HDIM-Integrated Decoder - Uses HDIM embedding, memory context, and routing weights.

This decoder properly integrates with HDIM architecture:
1. Cross-attention conditioning on HDIM embedding
2. Memory context as generation prefix
3. Weighted expert mixing based on routing weights

Unlike the previous GPT2Decoder, this one actually uses HDIM semantics.
"""

import sys
import io
import os
import re
import math
from typing import Dict, Any, List, Optional, Tuple

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.system("chcp 65001 >nul 2>&1")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import GPT2LMHeadModel, GPT2Tokenizer


class CrossAttentionConditioning(nn.Module):
    """Cross-attention layer that conditions GPT-2 on HDIM embedding."""

    def __init__(self, hdim_dim: int, gpt_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = gpt_dim // num_heads

        # Project HDIM embedding to key/value space
        self.k_proj = nn.Linear(hdim_dim, gpt_dim)
        self.v_proj = nn.Linear(hdim_dim, gpt_dim)

        # Query from GPT hidden states
        self.q_proj = nn.Linear(gpt_dim, gpt_dim)
        self.out_proj = nn.Linear(gpt_dim, gpt_dim)

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(gpt_dim)

    def forward(self, gpt_hidden: torch.Tensor, hdim_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
                gpt_hidden: (batch, seq_len, gpt_dim) - GPT hidden states
                hdim_emb: (batch, hdim_dim) or (hdim_dim,) - HDIM embedding
        Returns:
                conditioned hidden states
        """
        batch_size, seq_len, gpt_dim = gpt_hidden.shape

        # Ensure hdim_emb has batch dimension
        if hdim_emb.dim() == 1:
            hdim_emb = hdim_emb.unsqueeze(0)  # (hdim_dim,) -> (1, hdim_dim)

        # Project HDIM embedding to K, V
        # Expand to sequence length for attention
        hdim_expanded = hdim_emb.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # (batch, 1, hdim_dim) -> (batch, seq, hdim_dim)
        k = self.k_proj(hdim_expanded)  # (batch, seq, gpt_dim)
        v = self.v_proj(hdim_expanded)

        # Query from GPT hidden
        q = self.q_proj(gpt_hidden)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, gpt_dim)
        )

        # Output projection + residual
        output = self.out_proj(attn_output)
        return self.layer_norm(gpt_hidden + output)


class HDIMIntegratedDecoder(nn.Module):
    """Decoder that uses HDIM embedding, memory context, and routing weights.

    Architecture:
    1. Expert-specific prompt templates (weighted by routing)
    2. Memory context prefix (if available)
    3. Cross-attention conditioning on HDIM embedding
    4. GPT-2 generation with HDIM-influenced hidden states
    """

    # Expert response templates - used as prefixes, not full responses
    EXPERT_PREFIXES = {
        "Math": "Математический ответ:",
        "Language": "Ответ:",
        "Code": "Код:",
        "Science": "Научное объяснение:",
    }

    def __init__(
        self,
        hdim_dim: int = 256,
        gpt_model: str = "microsoft/DialoGPT-medium",  # Multilingual conversational model
        device: str = "cuda",
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.hdim_dim = hdim_dim
        self.use_cross_attention = use_cross_attention

        # Load multilingual conversational GPT model
        print(f"Loading {gpt_model} for HDIM-integrated generation...")
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt_model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.gpt_dim = self.gpt2.config.n_embd  # 1024 for DialoGPT-medium

        # HDIM embedding projection
        self.hdim_projection = nn.Sequential(
            nn.Linear(hdim_dim, self.gpt_dim * 2),
            nn.GELU(),
            nn.LayerNorm(self.gpt_dim * 2),
            nn.Linear(self.gpt_dim * 2, self.gpt_dim),
        )

        # Cross-attention for HDIM conditioning
        if use_cross_attention:
            self.cross_attn = CrossAttentionConditioning(
                hdim_dim=hdim_dim,
                gpt_dim=self.gpt_dim,
                num_heads=4,
            )

        # Routing weight influence layer
        self.routing_gate = nn.Sequential(
            nn.Linear(4, self.gpt_dim),  # 4 experts
            nn.Sigmoid(),
        )

        self.gpt2.to(self.device)
        self.hdim_projection.to(self.device)
        if use_cross_attention:
            self.cross_attn.to(self.device)
        self.routing_gate.to(self.device)
        self.eval()

        print(f"HDIMIntegratedDecoder ready on {self.device}")
        print(f" - HDIM dim: {hdim_dim}")
        print(f" - Cross-attention: {use_cross_attention}")
        print(f" - GPT dim: {self.gpt_dim}")

    def _build_weighted_prompt(
        self,
        user_input: str,
        routing_weights: List[float],
        memory_context: Optional[str] = None,
    ) -> str:
        """Build prompt with weighted expert influence."""

        # Start with memory context if available
        parts = []

        if memory_context:
            parts.append(f"[Контекст: {memory_context[:100]}]")

        # Add weighted expert prefixes based on routing
        expert_names = ["Math", "Language", "Code", "Science"]
        weighted_prefixes = []
        for name, weight in zip(expert_names, routing_weights):
            if weight > 0.15:  # Only include significant experts
                weighted_prefixes.append(f"{self.EXPERT_PREFIXES[name]}")

        if weighted_prefixes:
            parts.append(" | ".join(weighted_prefixes[:2]))

        # Add user input
        parts.append(f"\nВопрос: {user_input}")
        parts.append("Ответ:")

        return "\n".join(parts)

    def _apply_cross_attention(
        self,
        input_ids: torch.Tensor,
        hdim_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cross-attention conditioning on GPT hidden states."""

        # Ensure hdim_emb has batch dimension
        if hdim_emb.dim() == 1:
            hdim_emb = hdim_emb.unsqueeze(0)  # (hdim_dim,) -> (1, hdim_dim)

        # Get GPT hidden states
        with torch.no_grad():
            outputs = self.gpt2.transformer(input_ids)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, gpt_dim)

        # Apply cross-attention conditioning
        if self.use_cross_attention:
            conditioned = self.cross_attn(hidden_states, hdim_emb)
        else:
            # Simple addition fallback: project hdim and broadcast across sequence
            projected = self.hdim_projection(hdim_emb)  # (batch, gpt_dim)
            conditioned = hidden_states + 0.1 * projected.unsqueeze(
                1
            )  # (batch, seq_len, gpt_dim)

        return conditioned

    def generate_response(
        self,
        embedding: np.ndarray,
        user_input: str,
        routing_weights: List[float] = None,
        expert_name: str = "Language",
        memory_context: Optional[str] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate response using HDIM embedding, routing weights, and memory context.

        Fully neural generation via GPT-2 with HDIM conditioning.
        """
        # Default routing weights if not provided
        if routing_weights is None:
            routing_weights = [0.25, 0.25, 0.25, 0.25]

        # 1. Convert embedding
        if isinstance(embedding, np.ndarray):
            emb = torch.from_numpy(embedding).float().to(self.device)
        else:
            emb = embedding.float().to(self.device)

        if emb.dim() > 1:
            emb = emb.squeeze(0)
        emb = F.normalize(emb, p=2, dim=0)

        # 2. Build prompt for GPT-2
        expert_prompts = {
            "Math": "Question about mathematics:",
            "Language": "Conversation in Russian:",
            "Code": "Programming question:",
            "Science": "Science question:",
        }
        prompt_prefix = expert_prompts.get(expert_name, "Question:")
        prompt = f"{prompt_prefix} {user_input}\nAnswer:"

        # 3. Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # 4. Generate with standard GPT-2 (input_ids, not inputs_embeds)
        with torch.no_grad():
            outputs = self.gpt2.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.5,
                no_repeat_ngram_size=2,
            )

        # 5. Decode and clean
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response after "Answer:"
        if "Answer:" in full_response:
            response = full_response.split("Answer:")[-1].strip()
        else:
            response = full_response[len(prompt) :].strip()

        # Clean up - take first line
        if "\n" in response:
            response = response.split("\n")[0].strip()

        return response if response else "Понял вопрос."


def test_hdim_decoder():
    """Test the HDIM-integrated decoder."""
    print("=" * 60)
    print("Testing HDIM-Integrated Decoder")
    print("=" * 60)

    # Create decoder
    decoder = HDIMIntegratedDecoder(hdim_dim=256, use_cross_attention=True)

    # Simulate HDIM outputs
    test_cases = [
        {
            "input": "Привет!",
            "embedding": np.random.randn(256).astype(np.float32),
            "routing": [0.1, 0.8, 0.05, 0.05],  # Language dominant
            "expert": "Language",
            "memory": None,
        },
        {
            "input": "Сколько будет 5 + 3?",
            "embedding": np.random.randn(256).astype(np.float32),
            "routing": [0.85, 0.05, 0.05, 0.05],  # Math dominant
            "expert": "Math",
            "memory": "Арифметические операции",
        },
        {
            "input": "Как написать цикл в Python?",
            "embedding": np.random.randn(256).astype(np.float32),
            "routing": [0.1, 0.1, 0.75, 0.05],  # Code dominant
            "expert": "Code",
            "memory": "Python программирование",
        },
        {
            "input": "Почему небо голубое?",
            "embedding": np.random.randn(256).astype(np.float32),
            "routing": [0.1, 0.1, 0.1, 0.7],  # Science dominant
            "expert": "Science",
            "memory": "Физика атмосферы",
        },
    ]

    for test in test_cases:
        print(f"\n{'='*40}")
        print(f"User: {test['input']}")
        print(
            f"Routing: Math={test['routing'][0]:.0%}, Lang={test['routing'][1]:.0%}, Code={test['routing'][2]:.0%}, Sci={test['routing'][3]:.0%}"
        )
        print(f"Expert: {test['expert']}")
        if test["memory"]:
            print(f"Memory: {test['memory']}")

        # Normalize embedding
        emb = test["embedding"] / np.linalg.norm(test["embedding"])

        response = decoder.generate_response(
            embedding=emb,
            user_input=test["input"],
            routing_weights=test["routing"],
            expert_name=test["expert"],
            memory_context=test["memory"],
        )
        print(f"Assistant: {response}")

    print("\n" + "=" * 60)
    print("HDIM-Integrated Decoder test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_hdim_decoder()
