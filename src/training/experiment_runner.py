from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.training.experiment_config import ExperimentConfig
from src.training.results_logger import append_ledger_row


class ExperimentRunner:
    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root)

    def run(self, config: ExperimentConfig) -> dict[str, Any]:
        output_dir = Path(config.output_dir) if config.output_dir else self.repo_root / "artifacts" / "experiments"
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = config.metadata.get("run_id") or f"hdim-{config.config_hash()}"
        results_json = Path(config.results_json) if config.results_json else output_dir / f"{run_id}.json"

        command = [
            sys.executable,
            "-m",
            "src.training.train",
            "--epochs",
            str(config.epochs),
            "--batch_size",
            str(config.batch_size),
            "--lr",
            str(config.lr),
            "--device",
            config.device,
            "--num_samples",
            str(config.num_samples),
            "--negative_ratio",
            str(config.negative_ratio),
            "--train_fraction",
            str(config.train_fraction),
            "--seed",
            str(config.seed),
            "--description",
            config.description,
            "--results_json",
            str(results_json),
        ]
        if config.use_pairs:
            command.append("--use_pairs")
        if config.ledger_path:
            command.extend(["--ledger_path", str(config.ledger_path)])
        for key, value in config.model_overrides.items():
            command.extend(["--model_override", f"{key}={value}"])
        for key, value in config.trainer_overrides.items():
            command.extend(["--trainer_override", f"{key}={value}"])

        completed = subprocess.run(
            command,
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(results_json.read_text(encoding="utf-8"))
        payload["runner"] = {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "run_id": run_id,
        }
        if config.ledger_path:
            append_ledger_row(
                config.ledger_path,
                {
                    "run_id": run_id,
                    "status": payload.get("status", config.status),
                    "config_hash": config.config_hash(),
                    "results_json": results_json.as_posix(),
                    "quality": payload.get("quality", {}),
                },
            )
        return payload
