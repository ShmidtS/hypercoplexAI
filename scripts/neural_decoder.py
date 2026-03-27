#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Neural Decoder for HDIM Kernel - generates text from expert outputs.

Полностью нейросетевая генерация без захардкоженных шаблонов.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer


class NeuralDecoder(nn.Module):
    """Генерирует текст из expert output embeddings через SBERT."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        input_dim: int = 768,
        device: str = "cuda"
    ):
        super().__init__()
        self.sbert = SentenceTransformer(model_name)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        sbert_dim = self.sbert.get_sentence_embedding_dimension()
        self.projection = nn.Linear(input_dim, sbert_dim)
        
        self.sbert.to(self.device)
        self.projection.to(self.device)
        
        self.vocab = None
        self.vocab_embeddings = None
        self._build_vocab_cache()
    
    def _build_vocab_cache(self, top_k: int = 15000):
        """Build vocabulary embedding cache."""
        tokenizer = self.sbert.tokenizer
        vocab_size = min(len(tokenizer), top_k)
        
        common_tokens = []
        for i in range(vocab_size):
            token = tokenizer.convert_ids_to_tokens(i)
            if token and not token.startswith('[') and not token.startswith('<') and len(token) > 1:
                common_tokens.append(token)
        
        if common_tokens:
            with torch.no_grad():
                token_embs = self.sbert.encode(common_tokens, convert_to_tensor=True, show_progress_bar=False)
                self.vocab = common_tokens
                self.vocab_embeddings = token_embs.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
    
    def decode(
        self,
        expert_output: np.ndarray,
        max_length: int = 15,
        temperature: float = 0.7,
        top_p: float = 0.92,
    ) -> str:
        """Generate text purely from neural network - no templates."""
        if isinstance(expert_output, np.ndarray):
            target_emb = torch.from_numpy(expert_output).float().to(self.device)
        else:
            target_emb = expert_output.float().to(self.device)
        
        if target_emb.dim() == 1:
            target_emb = target_emb.unsqueeze(0)
        
        with torch.no_grad():
            projected = self.projection(target_emb.squeeze(0))
        
        projected = F.normalize(projected, p=2, dim=0)
        
        generated_tokens = []
        
        for _ in range(max_length):
            if self.vocab_embeddings is not None:
                similarities = F.cosine_similarity(
                    projected.unsqueeze(0),
                    self.vocab_embeddings,
                    dim=1
                )
                
                probs = F.softmax(similarities / temperature, dim=0)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=0)
                nucleus_mask = cumsum <= top_p
                nucleus_mask[0] = True
                
                nucleus_probs = sorted_probs[nucleus_mask]
                nucleus_indices = sorted_indices[nucleus_mask]
                
                if len(nucleus_indices) > 0:
                    idx = torch.multinomial(nucleus_probs, 1)
                    token_idx = nucleus_indices[idx].item()
                    token = self.vocab[token_idx]
                    
                    if token not in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']:
                        if len(generated_tokens) == 0 or token != generated_tokens[-1]:
                            generated_tokens.append(token)
            
            if len(generated_tokens) >= 8:
                break
        
        if generated_tokens:
            text = ' '.join(generated_tokens)
            text = text.replace(' ##', '').replace('##', '')
            return text
        return "..."  # Minimal fallback


if __name__ == "__main__":
    decoder = NeuralDecoder(input_dim=768)
    
    test_emb = np.random.randn(768).astype(np.float32)
    test_emb = test_emb / np.linalg.norm(test_emb)
    
    response = decoder.decode(test_emb, max_length=10)
    print(f"Generated: {response}")
