from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.training.experiment_config import ExperimentConfig
from src.training.results_logger import append_ledger_row, read_jsonl, write_json


class ExperimentRunner:
    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root)

    def run(self, config: ExperimentConfig) -> dict[str, Any]:
        run_id = config.metadata.get("run_id") or f"hdim-{config.config_hash()}"
        run_dir = self._resolve_run_dir(config, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = run_dir / "manifest.json"
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        results_json = Path(config.results_json) if config.results_json else run_dir / "results.json"
        ledger_path = Path(config.ledger_path) if config.ledger_path else run_dir / "ledger.jsonl"
        effective_metadata = dict(config.metadata)
        effective_metadata["run_id"] = run_id
        effective_config = ExperimentConfig(
            **{
                **config.to_dict(),
                "output_dir": run_dir.as_posix(),
                "results_json": results_json.as_posix(),
                "ledger_path": ledger_path.as_posix(),
                "metadata": effective_metadata,
            }
        )

        write_json(manifest_path, effective_config.to_dict())
        command = self._build_command(manifest_path)

        started_at = datetime.now(UTC)
        completed = subprocess.run(
            command,
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        finished_at = datetime.now(UTC)

        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")

        payload = json.loads(results_json.read_text(encoding="utf-8"))
        payload["runner"] = {
            "run_id": run_id,
            "command": command,
            "run_dir": run_dir.as_posix(),
            "manifest_path": manifest_path.as_posix(),
            "results_json": results_json.as_posix(),
            "ledger_path": ledger_path.as_posix(),
            "stdout_path": stdout_path.as_posix(),
            "stderr_path": stderr_path.as_posix(),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "returncode": completed.returncode,
        }
        payload["run_id"] = run_id
        payload["config_hash"] = effective_config.config_hash()
        payload["manifest"] = {
            "run_id": run_id,
            "path": manifest_path.as_posix(),
            "config_hash": effective_config.config_hash(),
        }
        payload["artifacts"] = {
            **payload.get("artifacts", {}),
            "checkpoint": payload.get("checkpoint"),
            "manifest": manifest_path.as_posix(),
            "results_json": results_json.as_posix(),
            "ledger_path": ledger_path.as_posix(),
            "stdout": stdout_path.as_posix(),
            "stderr": stderr_path.as_posix(),
        }
        payload["runner"]["artifacts"] = payload["artifacts"]
        payload["runner"]["config_hash"] = payload["config_hash"]
        write_json(results_json, payload)
        return payload

    def _resolve_run_dir(self, config: ExperimentConfig, run_id: str) -> Path:
        output_dir = Path(config.output_dir) if config.output_dir else self.repo_root / "artifacts" / "experiments"
        return output_dir / run_id

    def _build_command(self, manifest_path: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "src.training.train",
            "--config",
            str(manifest_path),
        ]


class AutoResearchRunner:
    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root)
        self.runner = ExperimentRunner(repo_root)

    def run_many(
        self,
        configs: list[ExperimentConfig],
        *,
        session_name: str = "autoresearch",
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        session_dir = self._resolve_session_dir(session_name, output_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        session_ledger = session_dir / "session_ledger.jsonl"

        runs: list[dict[str, Any]] = []
        for index, config in enumerate(configs, start=1):
            run_id = config.metadata.get("run_id") or f"{session_name}-{index:03d}-{config.config_hash()}"
            merged_metadata = dict(config.metadata)
            merged_metadata.update({"run_id": run_id, "session_name": session_name, "run_index": index})
            run_config = ExperimentConfig(
                **{
                    **config.to_dict(),
                    "output_dir": session_dir.as_posix(),
                    "results_json": None,
                    "ledger_path": None,
                    "metadata": merged_metadata,
                }
            )
            payload = self.runner.run(run_config)
            payload["runner"]["session_ledger"] = session_ledger.as_posix()
            run_summary = self._summarise_run(payload)
            append_ledger_row(session_ledger, run_summary)
            runs.append(run_summary)

        best_run = self._select_best_run(runs)
        session_summary = {
            "session_name": session_name,
            "session_dir": session_dir.as_posix(),
            "run_count": len(runs),
            "runs": runs,
            "best_run": best_run,
            "loop_stages": [
                "plan_manifest",
                "execute_training",
                "collect_artifacts",
                "score_quality",
                "record_decision",
            ],
            "artifacts": {
                "session_ledger": session_ledger.as_posix(),
                "session_summary": (session_dir / "session_summary.json").as_posix(),
            },
        }
        write_json(session_dir / "session_summary.json", session_summary)
        return session_summary

    def _resolve_session_dir(self, session_name: str, output_dir: str | Path | None) -> Path:
        base_dir = Path(output_dir) if output_dir else self.repo_root / "artifacts" / "autoresearch"
        return base_dir / session_name

    def _summarise_run(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        quality = payload.get("quality", {})
        run_summary = {
            "run_id": payload.get("runner", {}).get("run_id"),
            "status": payload.get("status", "keep"),
            "decision": payload.get("status", "keep"),
            "config_hash": payload.get("config_hash"),
            "results_json": payload.get("runner", {}).get("results_json"),
            "checkpoint": payload.get("artifacts", {}).get("checkpoint") or payload.get("checkpoint"),
            "manifest_path": payload.get("manifest", {}).get("path") or payload.get("runner", {}).get("manifest_path"),
            "run_dir": payload.get("runner", {}).get("run_dir"),
            "quality": quality,
            "validation": payload.get("validation", {}),
            "score": quality.get("pair_margin", 0.0),
        }
        return run_summary
    def load_session_summary(self, session_dir: str | Path) -> dict[str, Any]:
        summary_path = Path(session_dir) / "session_summary.json"
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def load_session_ledger(self, session_dir: str | Path) -> list[dict[str, Any]]:
        return read_jsonl(Path(session_dir) / "session_ledger.jsonl")

    def plan_from_manifest_paths(
        self,
        manifest_paths: list[str | Path],
        *,
        session_name: str = "autoresearch",
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        configs = [ExperimentConfig.from_json(path) for path in manifest_paths]
        return self.run_many(configs, session_name=session_name, output_dir=output_dir)

    def load_run_artifacts(self, run_dir: str | Path) -> dict[str, Any]:
        path = Path(run_dir)
        return {
            "manifest": json.loads((path / "manifest.json").read_text(encoding="utf-8")),
            "results": json.loads((path / "results.json").read_text(encoding="utf-8")),
            "stdout": (path / "stdout.txt").read_text(encoding="utf-8"),
            "stderr": (path / "stderr.txt").read_text(encoding="utf-8"),
        }

    def describe_loop(self) -> dict[str, Any]:
        return {
            "stages": [
                "plan_manifest",
                "execute_training",
                "collect_artifacts",
                "score_quality",
                "record_decision",
            ],
            "artifacts": [
                "manifest.json",
                "results.json",
                "stdout.txt",
                "stderr.txt",
                "session_ledger.jsonl",
                "session_summary.json",
            ],
        }

    def run_from_manifest_paths(
        self,
        manifest_paths: list[str | Path],
        *,
        session_name: str = "autoresearch",
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return self.plan_from_manifest_paths(
            manifest_paths,
            session_name=session_name,
            output_dir=output_dir,
        )

    def run_single_manifest(self, manifest_path: str | Path) -> dict[str, Any]:
        return self.runner.run(ExperimentConfig.from_json(manifest_path))

    def list_run_dirs(self, session_dir: str | Path) -> list[str]:
        base = Path(session_dir)
        if not base.exists():
            return []
        return sorted(child.as_posix() for child in base.iterdir() if child.is_dir())

    def get_best_run(self, session_dir: str | Path) -> dict[str, Any] | None:
        summary = self.load_session_summary(session_dir)
        return summary.get("best_run")

    def get_session_scoreboard(self, session_dir: str | Path) -> list[dict[str, Any]]:
        summary = self.load_session_summary(session_dir)
        return summary.get("runs", [])

    def get_run_decision(self, session_dir: str | Path, run_id: str) -> dict[str, Any] | None:
        for row in self.load_session_ledger(session_dir):
            if row.get("run_id") == run_id and "decision" in row:
                return row
        return None
    def mark_run_decision(
        self,
        session_dir: str | Path,
        *,
        run_id: str,
        decision: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        row = {"run_id": run_id, "decision": decision}
        if note is not None:
            row["note"] = note
        ledger_path = Path(session_dir) / "session_ledger.jsonl"
        append_ledger_row(ledger_path, row)
        return row
    def rebuild_session_summary(self, session_dir: str | Path) -> dict[str, Any]:
        session_path = Path(session_dir)
        ledger_rows = self.load_session_ledger(session_path)
        runs = [row for row in ledger_rows if "quality" in row and "run_dir" in row]
        summary = {
            "session_name": session_path.name,
            "session_dir": session_path.as_posix(),
            "run_count": len(runs),
            "runs": runs,
            "best_run": self._select_best_run(runs),
            "loop_stages": self.describe_loop()["stages"],
            "artifacts": {
                "session_ledger": (session_path / "session_ledger.jsonl").as_posix(),
                "session_summary": (session_path / "session_summary.json").as_posix(),
            },
        }
        write_json(session_path / "session_summary.json", summary)
        return summary
    def run_and_rebuild(
        self,
        configs: list[ExperimentConfig],
        *,
        session_name: str = "autoresearch",
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        summary = self.run_many(configs, session_name=session_name, output_dir=output_dir)
        return self.rebuild_session_summary(summary["session_dir"])

    def get_session_manifest_paths(self, session_dir: str | Path) -> list[str]:
        return [str(Path(run_dir) / "manifest.json") for run_dir in self.list_run_dirs(session_dir)]

    def get_session_results_paths(self, session_dir: str | Path) -> list[str]:
        return [str(Path(run_dir) / "results.json") for run_dir in self.list_run_dirs(session_dir)]

    def get_session_stdout_paths(self, session_dir: str | Path) -> list[str]:
        return [str(Path(run_dir) / "stdout.txt") for run_dir in self.list_run_dirs(session_dir)]

    def get_session_stderr_paths(self, session_dir: str | Path) -> list[str]:
        return [str(Path(run_dir) / "stderr.txt") for run_dir in self.list_run_dirs(session_dir)]

    def get_session_artifact_index(self, session_dir: str | Path) -> dict[str, list[str]]:
        return {
            "manifests": self.get_session_manifest_paths(session_dir),
            "results": self.get_session_results_paths(session_dir),
            "stdout": self.get_session_stdout_paths(session_dir),
            "stderr": self.get_session_stderr_paths(session_dir),
        }

    def get_session_overview(self, session_dir: str | Path) -> dict[str, Any]:
        summary = self.load_session_summary(session_dir)
        summary["artifact_index"] = self.get_session_artifact_index(session_dir)
        return summary

    def _select_best_run(self, runs: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not runs:
            return None
        return max(runs, key=lambda run: run.get("score", float("-inf")))
