#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HDIM Kernel Interactive Chat - Neural generation from expert outputs.

Полностью нейросетевая генерация ответов через HDIMIntegratedDecoder.
"""

import sys
import io
import os

if sys.platform == "win32":
	sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
	sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
	os.system("chcp 65001 >nul 2>&1")

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from src.models.hdim_model import HDIMConfig, HDIMModel
from src.models.model_factory import build_sbert_hdim_model, _patch_moe_kernel
from src.core.titans_memory import TitansMemoryModule


class HDIMKernelChat:
	"""Интерактивный интерфейс с нейросетевой генерацией ответов."""

	def __init__(self, checkpoint_path: Optional[str] = None, demo_mode: bool = True):
		self.demo_mode = demo_mode
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(f"Device: {self.device}")

		self.config = HDIMConfig(
			hidden_dim=256,
			num_experts=4,
			num_domains=4,
			memory_type="titans",
			top_k=2,
		)

		print("Building HDIM model...")
		if demo_mode:
			self.model = build_sbert_hdim_model(
				self.config,
				soft_router=True,
				freeze_bottom_frac=0.8,
			)
			_patch_moe_kernel(
				self.model.core_model,
				expert_names=["math", "language", "code", "science"],
				z_loss_weight=0.01,
			)
			print("Demo mode: random weights")
		else:
			self.model = self._load_checkpoint(checkpoint_path)

		self.model.to(self.device)
		self.model.eval()

		self.memory = TitansMemoryModule(
			key_dim=self.config.hidden_dim,
			val_dim=self.config.hidden_dim,
			hidden_dim=self.config.hidden_dim,
		).to(self.device)

		self.expert_names = ["Math", "Language", "Code", "Science"]
		self._init_knowledge_base()

		print("HDIM Kernel initialized!")
		print(f" Experts: {self.config.num_experts} ({', '.join(self.expert_names)})")
		print(f" Knowledge base loaded")
		print()

	def _init_knowledge_base(self):
		"""Load knowledge base from real_pairs and encode with SBERT for retrieval."""
		kb_path = Path(__file__).resolve().parents[1] / "data" / "real_pairs_v10.json"
		try:
			raw = kb_path.read_bytes()
			pairs = json.loads(raw.decode("utf-8"))
		except Exception as e:
			print(f"Warning: could not load knowledge base: {e}")
			self.kb_texts = []
			self.kb_embeddings = None
			return

		# Keep only positive cross-domain pairs with readable text
		import re
		def _readable(t):
			return bool(re.search(r'[a-zA-Zа-яА-ЯёЁ]{4}', t))

		pos = [
			p for p in pairs
			if p.get("relation") == "positive"
			and _readable(p["source_text"])
			and _readable(p["target_text"])
		]

		# Store pairs for retrieval: encode source side with SBERT
		self.kb_pairs = pos
		self.kb_source_texts = [p["source_text"] for p in pos]
		self.kb_target_texts = [p["target_text"] for p in pos]
		self.kb_families = [p.get("family", "?") for p in pos]

		print(f"Encoding {len(pos)} knowledge pairs with SBERT...")
		with torch.no_grad():
			self.kb_embeddings = self.model.text_encoder(
				self.kb_source_texts, device=self.device
			)  # (N, D)
		print(f"Knowledge base ready: {len(pos)} pairs")

	def _load_checkpoint(self, path: str) -> HDIMModel:
		"""Загружает чекпоинт с автоопределением hidden_dim."""
		print(f"Loading checkpoint: {path}")
		import io
		with open(path, "rb") as _f:
			_data = _f.read()
		checkpoint = torch.load(io.BytesIO(_data), map_location=self.device, weights_only=True)

		proj_bias = checkpoint["model_state_dict"].get("text_encoder.projection.4.bias")
		if proj_bias is not None:
			hidden_dim = proj_bias.shape[0]
			print(f"Detected hidden_dim={hidden_dim} from checkpoint")
		else:
			hidden_dim = 256

		config = HDIMConfig(
			hidden_dim=hidden_dim,
			num_experts=4,
			num_domains=4,
			memory_type="titans",
			top_k=2,
		)

		model = build_sbert_hdim_model(config, soft_router=True)
		_patch_moe_kernel(
			model.core_model,
			expert_names=["math", "language", "code", "science"],
			z_loss_weight=0.01,
		)
		print("Loading weights...")
		model.load_state_dict(checkpoint["model_state_dict"])
		self.config = config
		return model

	def encode(self, text: str) -> Dict[str, Any]:
		"""Кодирует текст через HDIM pipeline."""
		with torch.no_grad():
			text_emb = self.model.text_encoder([text], device=self.device)
			domain_id = torch.zeros(1, dtype=torch.long, device=self.device)

			result = self.model.core_model(text_emb, domain_id=domain_id, return_state=True)

			if len(result) == 5:
				output, routing_weights, invariant, slot_outputs, aux_state = result
			else:
				output, routing_weights, invariant, slot_outputs = result[:4]
				aux_state = result[-1] if len(result) > 4 else None

			if isinstance(output, tuple):
				embeddings = output[0]
			else:
				embeddings = output

			if isinstance(embeddings, torch.Tensor):
				embedding = embeddings[0].cpu().numpy()
			else:
				embedding = np.array(embeddings[0])

			if aux_state is not None and hasattr(aux_state, "expert_usage"):
				weights_arr = aux_state.expert_usage.cpu().numpy()
			elif routing_weights is not None:
				weights_arr = routing_weights[0].cpu().numpy() if routing_weights.dim() > 1 else routing_weights.cpu().numpy()
			else:
				weights_arr = np.zeros(self.config.num_experts)

			routing_info = {
				"weights": weights_arr.tolist() if isinstance(weights_arr, np.ndarray) else weights_arr,
				"dominant_expert": int(np.argmax(weights_arr)) if isinstance(weights_arr, np.ndarray) else 0,
			}

			slot_out = None
			if slot_outputs is not None and isinstance(slot_outputs, torch.Tensor):
				slot_out = slot_outputs

			return {
				"embedding": embedding,
				"norm": float(np.linalg.norm(embedding)),
				"routing": routing_info,
				"slot_outputs": slot_out,
			}

	def generate_response(self, result: Dict[str, Any], user_input: str = "") -> str:
		"""Ответ через HDIM-retrieval: ищем ближайшие аналогии в базе знаний."""
		embedding = result["embedding"]  # np.ndarray (D,)
		expert_idx = result["routing"]["dominant_expert"]
		expert_name = self.expert_names[expert_idx]
		weights = result["routing"]["weights"]

		# --- Retrieval via cosine similarity ---
		if self.kb_embeddings is not None and len(self.kb_source_texts) > 0:
			emb_t = torch.from_numpy(embedding).float().to(self.device)  # (D,)
			emb_t = torch.nn.functional.normalize(emb_t.unsqueeze(0), dim=-1)  # (1, D)
			kb_norm = torch.nn.functional.normalize(self.kb_embeddings, dim=-1)  # (N, D)
			sims = (kb_norm @ emb_t.T).squeeze(-1)  # (N,)
			top3 = sims.topk(min(3, len(sims))).indices.tolist()

			best_idx = top3[0]
			best_sim = sims[best_idx].item()
			src = self.kb_source_texts[best_idx]
			tgt = self.kb_target_texts[best_idx]
			family = self.kb_families[best_idx]

			# Format the analogy response
			lines = [
				f"Домен: {expert_name} (уверенность: {max(weights):.1%})",
				f"Аналогия [{family.replace('_', ' ')}] (схожесть: {best_sim:.3f}):",
				f"  А: {src[:120]}",
				f"  Б: {tgt[:120]}",
			]
			if len(top3) > 1:
				i2 = top3[1]
				sim2 = sims[i2].item()
				fam2 = self.kb_families[i2].replace('_', ' ')
				lines.append(f"Также близко [{fam2}] (схожесть: {sim2:.3f}):")
				lines.append(f"  А: {self.kb_source_texts[i2][:100]}")
			return "\n".join(lines)
		else:
			return f"[{expert_name}] Embedding norm: {result['norm']:.4f}. База знаний не загружена."

	def chat_loop(self):
		"""Основной цикл чата."""
		print("\n" + "=" * 50)
		print("HDIM Kernel Neural Chat")
		print("HDIM-integrated generation with routing weights")
		print("Введите 'quit' для выхода")
		print("=" * 50 + "\n")

		while True:
			try:
				text = input("> ").strip()
				if not text:
					continue
				if text.lower() in ["quit", "exit", "q"]:
					print("Выход...")
					break

				result = self.encode(text)
				response = self.generate_response(result, text)
				print(f"\n{response}\n")

			except KeyboardInterrupt:
				print("\nВыход...")
				break
			except Exception as e:
				print(f"Error: {e}")
				continue


def main():
	import argparse
	parser = argparse.ArgumentParser(description="HDIM Kernel Neural Chat")
	parser.add_argument(
		"--checkpoint",
		type=str,
		default=r"E:\hypercoplexAI\artifacts\run_018\checkpoints\best.pt",
		help="Path to model checkpoint",
	)
	parser.add_argument("--demo", action="store_true", help="Run in demo mode")
	args = parser.parse_args()

	if args.demo:
		chat = HDIMKernelChat(demo_mode=True)
	else:
		chat = HDIMKernelChat(checkpoint_path=args.checkpoint, demo_mode=False)

	chat.chat_loop()


if __name__ == "__main__":
	main()
