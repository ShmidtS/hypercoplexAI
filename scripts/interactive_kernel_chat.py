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
			hidden_dim=768,
			num_experts=4,
			num_domains=4,
			memory_type="titans",
			top_k=2,
			clifford_p=4,
			clifford_q=1,
			clifford_r=0,
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
				ortho_loss_weight=0.01,
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
		self.domain_keywords = {
			0: ("math", "алгеб", "геометр", "интеграл", "дифф", "формул", "теор", "матриц", "вектор", "уравнен", "числ", "fourier", "фурье"),
			1: ("language", "язык", "текст", "слово", "граммат", "перевод", "семан", "синтакс", "литерат", "метафор", "sentence", "translation"),
			2: ("code", "код", "программ", "python", "java", "javascript", "bug", "debug", "алгоритм", "функц", "класс", "api", "script", "git", "repo"),
			3: ("science", "физ", "хим", "био", "dna", "ген", "клет", "тепл", "терм", "энерг", "квант", "молек", "эксперимент", "наук", "кавитац"),
		}
		self._init_knowledge_base()

		print("HDIM Kernel initialized!")
		print(f" Experts: {self.config.num_experts} ({', '.join(self.expert_names)})")
		print(f" Knowledge base loaded")
		print()

	def _normalize_family(self, family: str) -> str:
		base_family = family or "?"
		for suffix in ("_aug_permute", "_aug_expand", "_aug_noise"):
			base_family = base_family.replace(suffix, "")
		return base_family.replace("_", " ").strip()

	def _infer_domain_id(self, text: str) -> int:
		text_lower = text.lower()
		scores = {
			domain_id: sum(text_lower.count(keyword) for keyword in keywords)
			for domain_id, keywords in self.domain_keywords.items()
		}
		best_domain, best_score = max(scores.items(), key=lambda item: item[1])
		sorted_scores = sorted(scores.values(), reverse=True)
		if best_score <= 0:
			return 0
		if len(sorted_scores) > 1 and best_score == sorted_scores[1]:
			return 0
		return best_domain

	def _infer_kb_domain(self, source_text: str, target_text: str, family: str) -> int:
		return self._infer_domain_id(f"{family} {source_text} {target_text}")

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
			self.kb_domains = []
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
		self.kb_domains = [
			self._infer_kb_domain(p["source_text"], p["target_text"], p.get("family", "?"))
			for p in pos
		]

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
			hidden_dim = 768

		# Auto-detect clifford signature from rotor dimensions
		rotor_key = None
		for prefix in ["pipeline.domain_rotors.domain_0.R", "pipeline.domain_encoder.domain_rotors.domain_0.R"]:
			if prefix in checkpoint["model_state_dict"]:
				rotor_key = prefix
				break
		if rotor_key is not None:
			clifford_dim = checkpoint["model_state_dict"][rotor_key].shape[0]
			import math as _math
			n_gen = int(_math.log2(clifford_dim))
			clifford_p, clifford_q, clifford_r = n_gen - 1, 1, 0
			print(f"Detected Cl({clifford_p},{clifford_q},{clifford_r}) dim={clifford_dim} from checkpoint")
		else:
			clifford_p, clifford_q, clifford_r = 3, 1, 0

		config = HDIMConfig(
			hidden_dim=hidden_dim,
			num_experts=4,
			num_domains=4,
			memory_type="titans",
			top_k=2,
			clifford_p=clifford_p,
			clifford_q=clifford_q,
			clifford_r=clifford_r,
		)

		model = build_sbert_hdim_model(config, soft_router=True)
		_patch_moe_kernel(
			model.core_model,
			expert_names=["math", "language", "code", "science"],
			z_loss_weight=0.01,
			ortho_loss_weight=0.01,
		)
		print("Loading weights...")
		sd = checkpoint["model_state_dict"]
		if not any(k.startswith("core_model.") for k in sd.keys()):
			sd = {"core_model." + k if not k.startswith("text_encoder.") and not k.startswith("_log_temp") else k: v for k, v in sd.items()}
		model.load_state_dict(sd, strict=False)
		self.config = config
		return model

	def encode(self, text: str) -> Dict[str, Any]:
		"""Кодирует текст через HDIM pipeline."""
		with torch.no_grad():
			text_emb = self.model.text_encoder([text], device=self.device)
			pred_domain = self._infer_domain_id(text)
			domain_id = torch.tensor([pred_domain], dtype=torch.long, device=self.device)

			result = self.model.core_model(text_emb, domain_id=domain_id, return_state=True)

			if len(result) == 5:
				output, routing_weights, invariant, slot_outputs, aux_state = result
			else:
				output, routing_weights, invariant, slot_outputs, aux_state = result

			if isinstance(output, tuple):
				embeddings = output[0]
			else:
				embeddings = output

			if isinstance(embeddings, torch.Tensor):
				embedding = embeddings[0].cpu().numpy()
			else:
				embedding = np.array(embeddings[0])

			dominant_expert = pred_domain
			if aux_state is not None and hasattr(aux_state, "expert_usage"):
				expert_weights = aux_state.expert_usage
				if isinstance(expert_weights, torch.Tensor):
					expert_weights = expert_weights.reshape(-1, expert_weights.shape[-1])
					weights_arr = expert_weights[0].detach().cpu().numpy()
				else:
					weights_arr = np.array(expert_weights)
			elif routing_weights is not None:
				weights_arr = routing_weights[0].detach().cpu().numpy() if routing_weights.dim() > 1 else routing_weights.detach().cpu().numpy()
			else:
				weights_arr = np.zeros(self.config.num_experts, dtype=np.float32)

			if aux_state is not None and hasattr(aux_state, "topk_idx"):
				top_expert_idx = aux_state.topk_idx
				if isinstance(top_expert_idx, torch.Tensor):
					dominant_expert = int(top_expert_idx.reshape(-1)[0].item())
			elif isinstance(weights_arr, np.ndarray) and weights_arr.size > 0:
				dominant_expert = int(np.argmax(weights_arr))

			routing_info = {
				"weights": weights_arr.tolist() if isinstance(weights_arr, np.ndarray) else weights_arr,
				"dominant_expert": dominant_expert,
				"predicted_domain": pred_domain,
			}

			slot_out = None
			if slot_outputs is not None and isinstance(slot_outputs, torch.Tensor):
				slot_out = slot_outputs

			sbert_emb = text_emb[0].cpu().numpy()  # SBERT space = same as KB

			return {
				"embedding": embedding,
				"sbert_embedding": sbert_emb,
				"norm": float(np.linalg.norm(embedding)),
				"routing": routing_info,
				"slot_outputs": slot_out,
			}

	def generate_response(self, result: Dict[str, Any], user_input: str = "") -> str:
		"""Ответ через HDIM-retrieval с объяснением структурной аналогии."""
		embedding = result.get("sbert_embedding", result["embedding"])  # SBERT space
		expert_idx = result["routing"]["dominant_expert"]
		pred_domain = result["routing"].get("predicted_domain", expert_idx)
		predicted_name = self.expert_names[pred_domain]
		expert_name = self.expert_names[expert_idx]
		weights = result["routing"]["weights"]

		expert_weights = [
			f"{self.expert_names[i]}={w:.1%}"
			for i, w in enumerate(weights)
		]

		if self.kb_embeddings is None or len(self.kb_source_texts) == 0:
			return f"[{expert_name}] База знаний не загружена."

		emb_t = torch.from_numpy(embedding).float().to(self.device)
		emb_t = F.normalize(emb_t.unsqueeze(0), dim=-1)  # (1, D)
		kb_norm = F.normalize(self.kb_embeddings, dim=-1)  # (N, D)
		sims = (kb_norm @ emb_t.T).squeeze(-1)  # (N,)

		preferred_domains = {pred_domain, expert_idx}
		domain_bonus = 0.05
		scores = sims.clone()
		if self.kb_domains:
			for idx, kb_domain in enumerate(self.kb_domains):
				if kb_domain in preferred_domains:
					scores[idx] += domain_bonus

		reranked_idx = scores.argsort(descending=True).tolist()
		domain_matched = [idx for idx in reranked_idx if self.kb_domains[idx] in preferred_domains]
		fallback_idx = [idx for idx in reranked_idx if self.kb_domains[idx] not in preferred_domains]
		ranked_idx = domain_matched + fallback_idx

		selected = []
		seen_families: set = set()
		for idx in ranked_idx:
			base_fam = self._normalize_family(self.kb_families[idx])
			if base_fam not in seen_families:
				seen_families.add(base_fam)
				selected.append(idx)
			if len(selected) >= 3:
				break

		lines = []
		lines.append(
			f"Предсказанный домен: {predicted_name} | Routed expert: {expert_name} | "
			f"Маршрутизация: {', '.join(expert_weights)}"
		)
		lines.append("")

		if selected:
			best_idx = selected[0]
			best_sim = sims[best_idx].item()
			best_score = scores[best_idx].item()
			best_src = self.kb_source_texts[best_idx]
			best_tgt = self.kb_target_texts[best_idx]
			best_fam = self._normalize_family(self.kb_families[best_idx])
			principle = best_fam.split()[0] if best_fam else "общий"
			match_note = "domain-match" if self.kb_domains[best_idx] in preferred_domains else "fallback"

			lines.append(f"Структурная аналогия [{best_fam}] (cos={best_sim:.3f}, score={best_score:.3f}, {match_note}):")
			lines.append(f"  Область A: {best_src[:160]}")
			lines.append(f"  Область B: {best_tgt[:160]}")
			lines.append(f"  Общий принцип: оба явления описываются одной структурой — '{principle}' паттерн")

		if len(selected) > 1:
			lines.append("")
			lines.append("Смежные аналогии:")
			for rank, idx in enumerate(selected[1:], 2):
				sim = sims[idx].item()
				score = scores[idx].item()
				fam = self._normalize_family(self.kb_families[idx])
				src = self.kb_source_texts[idx]
				tgt = self.kb_target_texts[idx]
				match_note = "domain-match" if self.kb_domains[idx] in preferred_domains else "fallback"
				lines.append(f"  {rank}. [{fam}] (cos={sim:.3f}, score={score:.3f}, {match_note})")
				lines.append(f"     {src[:100]}")
				lines.append(f"     ≈ {tgt[:100]}")

		return "\n".join(lines)

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
		default=None,
		help="Path to model checkpoint (auto-detect if not specified)",
	)
	parser.add_argument("--demo", action="store_true", help="Run in demo mode")
	args = parser.parse_args()

	if args.demo:
		chat = HDIMKernelChat(demo_mode=True)
	else:
		if args.checkpoint is None:
			import glob
			artifact_dirs = sorted(glob.glob(r"E:\hypercoplexAI\artifacts\run_*"))
			if artifact_dirs:
				latest = artifact_dirs[-1]
				best_pt = os.path.join(latest, "checkpoints", "best.pt")
				final_pt = os.path.join(latest, "checkpoints", "epoch_0030.pt")
				if os.path.exists(best_pt):
					args.checkpoint = best_pt
				elif os.path.exists(final_pt):
					args.checkpoint = final_pt
				else:
					pts = glob.glob(os.path.join(latest, "checkpoints", "*.pt"))
					if pts:
						args.checkpoint = sorted(pts)[-1]
			if args.checkpoint is None:
				print("No checkpoint found. Use --demo or --checkpoint PATH")
				sys.exit(1)
		chat = HDIMKernelChat(checkpoint_path=args.checkpoint, demo_mode=False)

	chat.chat_loop()


if __name__ == "__main__":
	main()
