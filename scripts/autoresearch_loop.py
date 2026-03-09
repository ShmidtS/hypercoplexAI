#!/usr/bin/env python
"""
HDIM AutoResearch Loop
Итеративный поиск лучшей архитектуры через автоматизированные эксперименты.

Запуск:
  python scripts/autoresearch_loop.py --iterations 5 --device auto
  python scripts/autoresearch_loop.py --search random --n_configs 10 --device cuda
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRID_SEARCH_SPACE = {
    "hidden_dim": [64, 128, 256],
    "num_experts": [4, 8],
    "lambda_iso": [0.05, 0.1, 0.2],
    "lambda_pair": [0.05, 0.1, 0.2],
    "advanced_encoder": [False, True],
    "hierarchical_memory": [False, True],
    "soft_router": [False, True],
}

RANDOM_SEARCH_RANGES = {
    "hidden_dim": [64, 128, 256],
    "num_experts": [4, 8],
    "lambda_iso": (0.01, 0.3),
    "lambda_pair": (0.01, 0.3),
    "advanced_encoder": [False, True],
    "hierarchical_memory": [False, True],
    "soft_router": [False, True],
}


def generate_grid_configs(max_configs: int = 20) -> list[dict]:
    keys = list(GRID_SEARCH_SPACE.keys())
    values = list(GRID_SEARCH_SPACE.values())
    configs = []
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        configs.append(cfg)
        if len(configs) >= max_configs:
            break
    return configs


def generate_random_configs(n: int = 10, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    configs = []
    for _ in range(n):
        cfg: dict[str, Any] = {}
        for key, space in RANDOM_SEARCH_RANGES.items():
            if isinstance(space, list):
                cfg[key] = rng.choice(space)
            elif isinstance(space, tuple) and len(space) == 2:
                cfg[key] = round(rng.uniform(space[0], space[1]), 4)
        configs.append(cfg)
    return configs


def generate_refinement_configs(best_config: dict, n: int = 5, seed: int = 123) -> list[dict]:
    rng = random.Random(seed)
    configs = []
    perturbable_floats = ["lambda_iso", "lambda_pair"]
    for i in range(n):
        cfg = dict(best_config)
        for key in perturbable_floats:
            if key in cfg and isinstance(cfg[key], float):
                delta = cfg[key] * 0.3
                cfg[key] = round(max(0.01, cfg[key] + rng.uniform(-delta, delta)), 4)
        for key in ["advanced_encoder", "hierarchical_memory", "soft_router"]:
            if key in cfg and rng.random() < 0.3:
                cfg[key] = not cfg[key]
        configs.append(cfg)
    return configs


class HDIMAutoResearchLoop:
    def __init__(self, repo_root: Path, output_dir: Path, device: str = "auto",
                 epochs: int = 20, num_samples: int = 200, batch_size: int = 32, seed: int = 42):
        self.repo_root = repo_root
        self.output_dir = output_dir
        self.device = device
        self.epochs = epochs
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed
        self.results: list[dict] = []
        output_dir.mkdir(parents=True, exist_ok=True)

    def _build_command(self, cfg: dict, run_dir: Path) -> list[str]:
        cmd = [
            sys.executable,
            str(self.repo_root / "scripts" / "gpu_train.py"),
            "--device", self.device,
            "--epochs", str(self.epochs),
            "--num_samples", str(self.num_samples),
            "--batch_size", str(self.batch_size),
            "--seed", str(self.seed),
            "--output_dir", str(run_dir),
            "--eval_every", "5",
            "--save_every", str(max(1, self.epochs)),
            "--use_pairs",
            "--text_mode",
            "--hidden_dim", str(cfg.get("hidden_dim", 128)),
            "--num_experts", str(cfg.get("num_experts", 4)),
            "--lambda_iso", str(cfg.get("lambda_iso", 0.1)),
            "--lambda_pair", str(cfg.get("lambda_pair", 0.1)),
        ]
        if cfg.get("advanced_encoder"):
            cmd.append("--advanced_encoder")
        if cfg.get("hierarchical_memory"):
            cmd.append("--hierarchical_memory")
        if cfg.get("soft_router"):
            cmd.append("--soft_router")
        return cmd

    def run_single(self, cfg: dict, run_id: str) -> dict[str, Any]:
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        config_path.write_text(json.dumps({"run_id": run_id, **cfg}, indent=2), encoding="utf-8")
        cmd = self._build_command(cfg, run_dir)
        print(f"\n[{run_id}] Starting...")
        start_t = time.time()
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        try:
            completed = subprocess.run(cmd, cwd=self.repo_root, capture_output=True, text=True, timeout=3600)
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            elapsed = time.time() - start_t
            if completed.returncode != 0:
                print(f"[{run_id}] FAILED: {completed.stderr[-300:]}")
                return {"run_id": run_id, "status": "failed", "config": cfg, "score": float("-inf"), "elapsed_s": round(elapsed, 1)}
            results_path = run_dir / "results.json"
            if results_path.exists():
                results = json.loads(results_path.read_text(encoding="utf-8"))
                quality = results.get("quality", {})
                score = quality.get("pair_margin", 0.0) + 0.3 * quality.get("STS_exported", 0.0)
                print(f"[{run_id}] OK | margin={quality.get('pair_margin',0):.4f} | STS={quality.get('STS_exported',0):.4f} | score={score:.4f} | {elapsed:.0f}s")
                return {"run_id": run_id, "status": "ok", "config": cfg, "quality": quality, "score": score,
                        "checkpoint": results.get("best_checkpoint"), "elapsed_s": round(elapsed, 1)}
            return {"run_id": run_id, "status": "no_results", "config": cfg, "score": float("-inf"), "elapsed_s": round(elapsed, 1)}
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_t
            return {"run_id": run_id, "status": "timeout", "config": cfg, "score": float("-inf"), "elapsed_s": round(elapsed, 1)}

    def run_session(self, configs: list[dict], session_name: str = "autoresearch") -> dict[str, Any]:
        print(f"\nAutoResearch Session: {session_name} | {len(configs)} configs")
        session_results = []
        for i, cfg in enumerate(configs, start=1):
            run_id = f"{session_name}-{i:03d}"
            result = self.run_single(cfg, run_id)
            session_results.append(result)
            self._save_session_summary(session_results, session_name)
        successful = [r for r in session_results if r["status"] == "ok"]
        best = max(successful, key=lambda x: x["score"]) if successful else None
        summary = {"session_name": session_name, "total_configs": len(configs),
                   "successful": len(successful), "best_run": best, "all_runs": session_results}
        if best:
            print(f"\nBest: {best['run_id']} | score={best['score']:.4f} | config={best['config']}")
        self._save_session_summary(session_results, session_name)
        return summary

    def _save_session_summary(self, results: list[dict], session_name: str) -> None:
        summary_path = self.output_dir / f"{session_name}_summary.json"
        data = {"session_name": session_name, "runs": results,
                "best_score": max((r["score"] for r in results if r["status"] == "ok"), default=None)}
        summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="HDIM AutoResearch Loop")
    parser.add_argument("--search", choices=["grid", "random", "refinement"], default="random")
    parser.add_argument("--n_configs", type=int, default=6)
    parser.add_argument("--max_grid_configs", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/autoresearch"))
    parser.add_argument("--session_name", default="hdim-autoresearch")
    parser.add_argument("--best_config", type=Path, default=None)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    loop = HDIMAutoResearchLoop(repo_root=repo_root, output_dir=args.output_dir, device=args.device,
                                epochs=args.epochs, num_samples=args.num_samples,
                                batch_size=args.batch_size, seed=args.seed)
    if args.search == "grid":
        configs = generate_grid_configs(max_configs=args.max_grid_configs)
    elif args.search == "random":
        configs = generate_random_configs(n=args.n_configs, seed=args.seed)
    elif args.search == "refinement":
        if args.best_config is None:
            parser.error("--best_config required for refinement mode")
        best_cfg = json.loads(args.best_config.read_text(encoding="utf-8"))
        configs = generate_refinement_configs(best_cfg, n=args.n_configs, seed=args.seed)
    else:
        configs = generate_random_configs(n=args.n_configs, seed=args.seed)
    summary = loop.run_session(configs, session_name=args.session_name)
    print(f"\nDone: {summary['successful']}/{summary['total_configs']} succeeded")
    if summary["best_run"]:
        best = summary["best_run"]
        best_config_path = args.output_dir / "best_config.json"
        best_config_path.write_text(json.dumps(best["config"], indent=2), encoding="utf-8")
        print(f"Best config saved to {best_config_path}")


if __name__ == "__main__":
    main()
