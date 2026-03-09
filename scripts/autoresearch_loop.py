#!/usr/bin/env python
"""
HDIM AutoResearch Loop
Итеративный поиск лучшей архитектуры через автоматизированные эксперименты.

Новые возможности:
  - IncumbentTracker: отслеживание лучшей конфигурации с failure taxonomy
  - Двухфазный режим: explore (широкий поиск) → refine (уточнение вокруг лучшей)
  - Таксономия ошибок: crash_nan, crash_oom, crash_runtime, metric_regression
  - Consistent score formula через PRIMARY_SCORE_WEIGHTS из gpu_train

Запуск:
  python scripts/autoresearch_loop.py --iterations 5 --device auto
  python scripts/autoresearch_loop.py --search two_phase --n_configs 10 --device cuda
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Canonical score formula — импортируется из gpu_train для согласованности
from scripts.gpu_train import PRIMARY_SCORE_WEIGHTS, compute_primary_score, check_run_validity

# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
GRID_SEARCH_SPACE = {
    "hidden_dim": [64, 128, 256],
    "num_experts": [4, 8],
    "lambda_iso": [0.05, 0.1, 0.2],
    "lambda_pair": [0.1, 0.2, 0.3],
    "advanced_encoder": [False, True],
    "hierarchical_memory": [False, True],
    "soft_router": [False, True],
}

RANDOM_SEARCH_RANGES = {
    "hidden_dim": [64, 128, 256],
    "num_experts": [4, 8],
    "lambda_iso": (0.01, 0.2),
    "lambda_pair": (0.1, 0.5),
    "lambda_routing": (0.005, 0.05),
    "lambda_memory": (0.005, 0.03),
    "ranking_margin": (0.1, 0.6),
    "advanced_encoder": [False, True],
    "hierarchical_memory": [False, True],
    "soft_router": [False, True],
}


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------
FAILURE_CODES = {
    "crash_nan": "Too many NaN batches during training",
    "crash_oom": "CUDA out of memory",
    "crash_runtime": "Runtime error (non-zero exit code)",
    "metric_regression": "All metrics near zero or NaN",
    "timeout": "Exceeded time budget",
}


def classify_failure(result: dict) -> str:
    """Classify failure type from run result."""
    status = result.get("status", "")
    if status == "timeout":
        return "timeout"
    if status == "failed":
        stderr = result.get("stderr", "")
        if "out of memory" in stderr.lower() or "cuda" in stderr.lower():
            return "crash_oom"
        if "nan" in stderr.lower():
            return "crash_nan"
        return "crash_runtime"
    if status == "no_results":
        return "crash_runtime"
    # Delegate to check_run_validity for metric-based failures
    if status == "ok":
        quality = result.get("quality", {})
        nan_total = result.get("nan_batches_total", 0)
        training_summary = result.get("training_summary", {})
        is_valid, reason = check_run_validity(
            {"quality": quality, "nan_batches_total": nan_total,
             "training_summary": training_summary}
        )
        if not is_valid:
            return reason
    return ""


# ---------------------------------------------------------------------------
# IncumbentTracker
# ---------------------------------------------------------------------------
@dataclass
class IncumbentState:
    run_id: str
    config: dict
    score: float
    quality: dict
    checkpoint: Optional[str] = None
    epoch: int = 0


class IncumbentTracker:
    """Отслеживает лучшую конфигурацию (incumbent) и историю попыток.

    Ведёт статистику по failure taxonomy и решает keep/discard для
    каждого нового прогона.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.incumbent: Optional[IncumbentState] = None
        self.history: list[dict] = []
        self.failure_counts: dict[str, int] = {k: 0 for k in FAILURE_CODES}
        self.failure_counts["crash_runtime"] = 0
        self._path = output_dir / "incumbent.json"
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if data.get("incumbent"):
                    self.incumbent = IncumbentState(**data["incumbent"])
                self.history = data.get("history", [])
                self.failure_counts = data.get("failure_counts", self.failure_counts)
            except Exception:
                pass

    def _save(self) -> None:
        data = {
            "incumbent": asdict(self.incumbent) if self.incumbent else None,
            "history": self.history,
            "failure_counts": self.failure_counts,
        }
        self._path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def update(self, result: dict) -> tuple[str, bool]:
        """Update tracker with new run result.

        Returns
        -------
        (verdict, is_new_incumbent)
            verdict: 'keep' | 'discard' | 'failed:<code>'
            is_new_incumbent: True if this run is the new best
        """
        run_id = result.get("run_id", "unknown")
        status = result.get("status", "failed")

        # Record in history
        self.history.append({
            "run_id": run_id,
            "status": status,
            "score": result.get("score", float("-inf")),
            "config": result.get("config", {}),
        })

        if status != "ok":
            failure_code = classify_failure(result)
            self.failure_counts[failure_code] = self.failure_counts.get(failure_code, 0) + 1
            self._save()
            return f"failed:{failure_code}", False

        # Check metric-based validity
        failure_code = classify_failure(result)
        if failure_code:
            self.failure_counts[failure_code] = self.failure_counts.get(failure_code, 0) + 1
            self._save()
            return f"failed:{failure_code}", False

        # Compute score
        quality = result.get("quality", {})
        score = compute_primary_score(quality)
        result["score"] = score

        is_new_incumbent = False
        if self.incumbent is None or score > self.incumbent.score:
            self.incumbent = IncumbentState(
                run_id=run_id,
                config=result.get("config", {}),
                score=score,
                quality=quality,
                checkpoint=result.get("checkpoint"),
            )
            is_new_incumbent = True
            verdict = "keep"
            print(f"  [Incumbent] New best: {run_id} score={score:.4f}")
        else:
            verdict = "discard"

        self._save()
        return verdict, is_new_incumbent

    def best_config(self) -> Optional[dict]:
        return self.incumbent.config if self.incumbent else None

    def best_score(self) -> float:
        return self.incumbent.score if self.incumbent else float("-inf")

    def failure_summary(self) -> str:
        parts = [f"{k}={v}" for k, v in self.failure_counts.items() if v > 0]
        return ", ".join(parts) if parts else "none"


# ---------------------------------------------------------------------------
# Config generators
# ---------------------------------------------------------------------------

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
    perturbable_floats = ["lambda_iso", "lambda_pair", "lambda_routing", "lambda_memory", "ranking_margin"]
    for i in range(n):
        cfg = dict(best_config)
        for key in perturbable_floats:
            if key in cfg and isinstance(cfg[key], float):
                delta = cfg[key] * 0.25
                cfg[key] = round(max(0.005, cfg[key] + rng.uniform(-delta, delta)), 4)
        for key in ["advanced_encoder", "hierarchical_memory", "soft_router"]:
            if key in cfg and rng.random() < 0.2:
                cfg[key] = not cfg[key]
        configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# HDIMAutoResearchLoop
# ---------------------------------------------------------------------------

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
        self.tracker = IncumbentTracker(output_dir)

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
            "--lambda_pair", str(cfg.get("lambda_pair", 0.2)),
            "--lambda_routing", str(cfg.get("lambda_routing", 0.05)),
            "--lambda_memory", str(cfg.get("lambda_memory", 0.01)),
            "--ranking_margin", str(cfg.get("ranking_margin", 0.3)),
            "--negative_ratio", "0.4",
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
        print(f"\n[{run_id}] Starting: {cfg}")
        start_t = time.time()
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        try:
            completed = subprocess.run(
                cmd, cwd=self.repo_root, capture_output=True, text=True, timeout=3600
            )
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            elapsed = time.time() - start_t

            if completed.returncode != 0:
                print(f"[{run_id}] FAILED (rc={completed.returncode}): {completed.stderr[-400:]}")
                result = {
                    "run_id": run_id, "status": "failed", "config": cfg,
                    "score": float("-inf"), "elapsed_s": round(elapsed, 1),
                    "stderr": completed.stderr[-400:],
                }
                verdict, _ = self.tracker.update(result)
                result["verdict"] = verdict
                return result

            results_path = run_dir / "results.json"
            if results_path.exists():
                results = json.loads(results_path.read_text(encoding="utf-8"))
                quality = results.get("quality", {})
                score = compute_primary_score(quality)
                nan_total = results.get("nan_batches_total", 0)
                training_summary = results.get("training_summary", {})
                print(
                    f"[{run_id}] OK | margin={quality.get('pair_margin', 0):.4f} "
                    f"| STS={quality.get('STS_exported', 0):.4f} "
                    f"| score={score:.4f} | NaN={nan_total} | {elapsed:.0f}s"
                )
                result = {
                    "run_id": run_id, "status": "ok", "config": cfg,
                    "quality": quality, "score": score,
                    "nan_batches_total": nan_total,
                    "training_summary": training_summary,
                    "checkpoint": results.get("best_checkpoint"),
                    "elapsed_s": round(elapsed, 1),
                }
                verdict, is_new_best = self.tracker.update(result)
                result["verdict"] = verdict
                if is_new_best:
                    print(f"  *** NEW INCUMBENT: {run_id} score={score:.4f} ***")
                return result

            result = {
                "run_id": run_id, "status": "no_results", "config": cfg,
                "score": float("-inf"), "elapsed_s": round(elapsed, 1),
            }
            verdict, _ = self.tracker.update(result)
            result["verdict"] = verdict
            return result

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_t
            result = {
                "run_id": run_id, "status": "timeout", "config": cfg,
                "score": float("-inf"), "elapsed_s": round(elapsed, 1),
            }
            verdict, _ = self.tracker.update(result)
            result["verdict"] = verdict
            return result

    def run_session(
        self, configs: list[dict], session_name: str = "autoresearch"
    ) -> dict[str, Any]:
        print(f"\nAutoResearch Session: {session_name} | {len(configs)} configs")
        session_results = []
        for i, cfg in enumerate(configs, start=1):
            run_id = f"{session_name}-{i:03d}"
            result = self.run_single(cfg, run_id)
            session_results.append(result)
            self._save_session_summary(session_results, session_name)
        successful = [r for r in session_results if r["status"] == "ok"]
        best = max(successful, key=lambda x: x["score"]) if successful else None
        print(f"\nFailures: {self.tracker.failure_summary()}")
        summary = {
            "session_name": session_name,
            "total_configs": len(configs),
            "successful": len(successful),
            "best_run": best,
            "incumbent": self.tracker.incumbent.run_id if self.tracker.incumbent else None,
            "incumbent_score": self.tracker.best_score(),
            "failure_counts": self.tracker.failure_counts,
            "all_runs": session_results,
        }
        if best:
            print(f"\nBest: {best['run_id']} | score={best['score']:.4f} | config={best['config']}")
        self._save_session_summary(session_results, session_name)
        return summary

    def _save_session_summary(self, results: list[dict], session_name: str) -> None:
        summary_path = self.output_dir / f"{session_name}_summary.json"
        data = {
            "session_name": session_name,
            "runs": results,
            "incumbent_score": self.tracker.best_score(),
            "best_score": max((r["score"] for r in results if r["status"] == "ok"), default=None),
        }
        summary_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def run_two_phase(
        self,
        n_explore: int = 8,
        n_refine: int = 5,
        session_name: str = "two-phase",
        seed: int = 42,
    ) -> dict[str, Any]:
        """Phase 1: broad random explore. Phase 2: refine around best incumbent."""
        print(f"\n=== PHASE 1: EXPLORE ({n_explore} configs) ===")
        explore_configs = generate_random_configs(n=n_explore, seed=seed)
        explore_summary = self.run_session(explore_configs, session_name=f"{session_name}-explore")

        print(f"\n=== PHASE 2: REFINE ({n_refine} configs) ===")
        best_cfg = self.tracker.best_config()
        if best_cfg is None:
            print("No valid incumbent found in explore phase, using random refine")
            refine_configs = generate_random_configs(n=n_refine, seed=seed + 1)
        else:
            print(f"Refining around: {best_cfg}")
            refine_configs = generate_refinement_configs(best_cfg, n=n_refine, seed=seed + 1)

        refine_summary = self.run_session(refine_configs, session_name=f"{session_name}-refine")

        all_runs = explore_summary["all_runs"] + refine_summary["all_runs"]
        successful = [r for r in all_runs if r["status"] == "ok"]
        best = max(successful, key=lambda x: x["score"]) if successful else None

        return {
            "session_name": session_name,
            "explore_summary": explore_summary,
            "refine_summary": refine_summary,
            "best_run": best,
            "incumbent_score": self.tracker.best_score(),
            "failure_counts": self.tracker.failure_counts,
        }


def main():
    parser = argparse.ArgumentParser(description="HDIM AutoResearch Loop")
    parser.add_argument("--search", choices=["grid", "random", "refinement", "two_phase"], default="random")
    parser.add_argument("--n_configs", type=int, default=6)
    parser.add_argument("--n_explore", type=int, default=6, help="Explore phase configs (two_phase mode)")
    parser.add_argument("--n_refine", type=int, default=4, help="Refine phase configs (two_phase mode)")
    parser.add_argument("--max_grid_configs", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/autoresearch"))
    parser.add_argument("--session_name", default="hdim-autoresearch")
    parser.add_argument("--best_config", type=Path, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    loop = HDIMAutoResearchLoop(
        repo_root=repo_root,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    if args.search == "grid":
        configs = generate_grid_configs(max_configs=args.max_grid_configs)
        summary = loop.run_session(configs, session_name=args.session_name)
    elif args.search == "random":
        configs = generate_random_configs(n=args.n_configs, seed=args.seed)
        summary = loop.run_session(configs, session_name=args.session_name)
    elif args.search == "refinement":
        if args.best_config is None:
            parser.error("--best_config required for refinement mode")
        best_cfg = json.loads(args.best_config.read_text(encoding="utf-8"))
        configs = generate_refinement_configs(best_cfg, n=args.n_configs, seed=args.seed)
        summary = loop.run_session(configs, session_name=args.session_name)
    elif args.search == "two_phase":
        summary = loop.run_two_phase(
            n_explore=args.n_explore,
            n_refine=args.n_refine,
            session_name=args.session_name,
            seed=args.seed,
        )
    else:
        configs = generate_random_configs(n=args.n_configs, seed=args.seed)
        summary = loop.run_session(configs, session_name=args.session_name)

    successful = summary.get("successful", 0)
    total = summary.get("total_configs", args.n_configs)
    print(f"\nDone: {successful}/{total} succeeded")
    print(f"Incumbent score: {loop.tracker.best_score():.4f}")
    print(f"Failures: {loop.tracker.failure_summary()}")

    best_run = summary.get("best_run")
    if best_run:
        best_config_path = args.output_dir / "best_config.json"
        best_config_path.write_text(
            json.dumps(best_run["config"], indent=2, default=str), encoding="utf-8"
        )
        print(f"Best config saved to {best_config_path}")


if __name__ == "__main__":
    main()
