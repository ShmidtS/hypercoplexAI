#!/usr/bin/env python
"""Online HDIM training with LLM-generated synthetic pairs.

Generates cross-domain text analogies via a local OpenAI-compatible endpoint,
converts them to deterministic synthetic embeddings, and trains an HDIM model
end-to-end using InvariantTrainer.

Usage:
    python scripts/online_train_with_llm.py --num_pairs 40 --epochs 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

# Ensure project root is on path for src.* imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.models.hdim_model import HDIMConfig
from src.models.hdim_model import HDIMModel
from src.training.invariant_trainer import InvariantTrainer

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Env helper
# ------------------------------------------------------------------ #

def _parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip("'\"\"")
    return values


# ------------------------------------------------------------------ #
# LLM client (OpenAI-compatible, zero external deps)
# ------------------------------------------------------------------ #

class _LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str = "zai/glm-5.1") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        choice = result["choices"][0]
        content = choice.get("message", {}).get("content", "")
        if not content:
            content = choice.get("message", {}).get("reasoning_content", "")
        return content.strip()


# ------------------------------------------------------------------ #
# Pair generation
# ------------------------------------------------------------------ #

_SYSTEM = (
    "You are a synthetic data generator for cross-domain semantic transfer learning.\n"
    "Output ONLY a raw JSON array. Do NOT wrap it in markdown code blocks.\n"
    "Each element must have these exact keys:\n"
    '  "source_text" (string),\n'
    '  "target_text" (string),\n'
    '  "source_domain" (integer index),\n'
    '  "target_domain" (integer index),\n'
    '  "relation": "positive",\n'
    '  "group_id" (integer >= 1).\n'
    "Generate meaningful analogies / translations / paraphrases across the requested domains.\n"
    "group_id should group semantically related items."
)


def _build_user_prompt(domains: list[str], num_pairs: int) -> str:
    domain_list = ", ".join(f"{i}={name}" for i, name in enumerate(domains))
    return (
        f"Generate {num_pairs} cross-domain semantic pairs.\n"
        f"Domains: {domain_list}.\n"
        f"Use domain indices (0..{len(domains) - 1}) for source_domain and target_domain.\n"
        f"Ensure at least {max(1, num_pairs // len(domains))} pairs involve each domain.\n"
        f"Return only the JSON array."
    )


def generate_pairs(
    client: _LLMClient,
    domains: list[str],
    num_pairs: int,
    retries: int = 2,
) -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _build_user_prompt(domains, num_pairs)},
    ]
    for attempt in range(retries + 1):
        try:
            raw = client.chat(messages, max_tokens=8192, temperature=0.7)
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            if raw.endswith("```"):
                raw = raw.rsplit("\n", 1)[0] if "\n" in raw else raw
            raw = raw.strip()
            pairs = json.loads(raw)
            if isinstance(pairs, list) and len(pairs) > 0:
                for i, p in enumerate(pairs):
                    p.setdefault("source_domain", 0)
                    p.setdefault("target_domain", 1)
                    p.setdefault("relation", "positive")
                    p.setdefault("group_id", (i % 10) + 1)
                logger.info(f"Generated {len(pairs)} pairs from LLM")
                return pairs
        except Exception as exc:
            logger.warning(f"LLM generation attempt {attempt + 1} failed: {exc}")
    raise RuntimeError("Failed to generate pairs from LLM after retries")


# ------------------------------------------------------------------ #
# Synthetic embedding from text (deterministic, lightweight)
# ------------------------------------------------------------------ #

def _text_to_tensor(text: str, dim: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Deterministic pseudo-embedding from a string."""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    gen = torch.Generator().manual_seed(seed)
    t = torch.randn(dim, generator=gen, dtype=dtype)
    return t / (t.norm() + 1e-6)  # unit-normalised


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class _SyntheticPairDataset(Dataset):
    def __init__(self, pairs: list[dict[str, Any]], hidden_dim: int, device: torch.device) -> None:
        self.pairs = pairs
        self.hidden_dim = hidden_dim
        self.device = device

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        p = self.pairs[idx]
        return {
            "encoding": _text_to_tensor(p["source_text"], self.hidden_dim, self.device),
            "pair_encoding": _text_to_tensor(p["target_text"], self.hidden_dim, self.device),
            "domain_id": torch.tensor(p["source_domain"], dtype=torch.long),
            "pair_domain_id": torch.tensor(p["target_domain"], dtype=torch.long),
            "pair_relation_label": torch.tensor(1.0 if p.get("relation") == "positive" else 0.0, dtype=torch.float32),
            "pair_group_id": torch.tensor(p.get("group_id", 1), dtype=torch.long),
            "pair_weight": torch.tensor(1.0, dtype=torch.float32),
        }


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Online HDIM training with LLM-generated pairs")
    parser.add_argument("--env", type=Path, default=Path(".env"), help="Path to .env file")
    parser.add_argument("--domains", nargs="+", default=["math", "science", "language", "code"], help="Domain names")
    parser.add_argument("--num_pairs", type=int, default=40, help="Number of synthetic pairs to generate")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_domains", type=int, default=None, help="Defaults to len(domains)")
    parser.add_argument("--device", default=None, help="cuda / cpu / mps; auto-detect if omitted")
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_generation", action="store_true", help="Skip LLM generation if JSON exists")
    args = parser.parse_args(argv)

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    env = _parse_env(args.env)
    base_url = env.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
    api_key = env.get("OPENAI_AUTH_TOKEN", "")
    if not api_key:
        logger.error("OPENAI_AUTH_TOKEN not found in .env")
        raise SystemExit(1)

    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    num_domains = args.num_domains if args.num_domains is not None else len(args.domains)
    torch.manual_seed(args.seed)

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    data_path = data_dir / "synthetic_pairs.json"

    if not (args.skip_generation and data_path.exists()):
        client = _LLMClient(base_url, api_key)
        pairs = generate_pairs(client, args.domains, args.num_pairs)
        data_path.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Wrote {len(pairs)} pairs to {data_path}")
    else:
        pairs = json.loads(data_path.read_text(encoding="utf-8"))
        logger.info(f"Loaded {len(pairs)} existing pairs from {data_path}")

    cfg = HDIMConfig(hidden_dim=args.hidden_dim, num_domains=num_domains)
    model = HDIMModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    trainer = InvariantTrainer(model, optimizer, device=device)

    dataset = _SyntheticPairDataset(pairs, args.hidden_dim, device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        trainer.set_epoch(epoch)
        metrics = trainer.train_epoch(loader)
        logger.info(f"Epoch {epoch}/{args.epochs} | loss={metrics.get('loss_total', 0.0):.4f}")

    ckpt_path = args.output_dir / "online_llm_checkpoint.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(ckpt_path))
    logger.info(f"Saved checkpoint to {ckpt_path}")
    logger.info("Online training complete.")


if __name__ == "__main__":
    main()
