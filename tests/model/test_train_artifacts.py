import json
import subprocess
import sys

import pytest

from src.training.experiment_config import ExperimentConfig


def test_experiment_config_round_trips_text_mode_manifest_fields(tmp_path):
    results_path = tmp_path / "results" / "text_mode_summary.json"
    ledger_path = tmp_path / "results" / "text_mode_ledger.jsonl"
    config_path = tmp_path / "text_mode_experiment.json"
    config = ExperimentConfig(
        description="text-mode manifest contract",
        text_mode=True,
        use_pairs=True,
        negative_ratio=0.25,
        results_json=str(results_path),
        ledger_path=str(ledger_path),
        metadata={"run_id": "text-mode-smoke"},
    )
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

    loaded = ExperimentConfig.from_json(config_path)

    assert loaded.text_mode is True
    assert loaded.use_pairs is True
    assert loaded.negative_ratio == pytest.approx(0.25)
    assert loaded.results_json == str(results_path)
    assert loaded.ledger_path == str(ledger_path)
    assert loaded.metadata["run_id"] == "text-mode-smoke"


def test_train_script_writes_results_json(tmp_path):
    results_path = tmp_path / "results" / "hdim_run_summary.json"
    command = [
        sys.executable,
        "-m",
        "src.training.train",
        "--epochs",
        "1",
        "--batch_size",
        "4",
        "--num_samples",
        "16",
        "--device",
        "cpu",
        "--text_mode",
        "--results_json",
        str(results_path),
    ]
    completed = subprocess.run(
        command,
        cwd="E:/hypercoplexAI",
        check=True,
        capture_output=True,
        text=True,
    )

    assert results_path.exists()
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["run_args"]["epochs"] == 1
    assert payload["run_args"]["batch_size"] == 4
    assert payload["run_args"]["text_mode"] is True
    assert payload["validation"]["loss_total"] >= 0.0
    assert set(payload["quality"]) == {"STS_exported", "STS_training", "DRS", "AFR", "pair_margin"}
    assert payload["checkpoint"] == (results_path.parent / "checkpoints" / "hdim_final.pt").as_posix()
    assert payload["run_id"] is None
    assert payload["status"] == "keep"
    assert "Wrote run summary" in completed.stdout


def test_train_script_manifest_preserves_run_identity_and_artifacts(tmp_path):
    results_path = tmp_path / "results" / "manifest_run.json"
    ledger_path = tmp_path / "results" / "ledger.jsonl"
    config_path = tmp_path / "experiment.json"
    config = ExperimentConfig(
        description="manifest identity",
        epochs=1,
        batch_size=4,
        num_samples=16,
        text_mode=True,
        results_json=str(results_path),
        ledger_path=str(ledger_path),
        metadata={"run_id": "manifest-run-001", "tag": "smoke"},
    )
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

    subprocess.run(
        [sys.executable, "-m", "src.training.train", "--config", str(config_path)],
        cwd="E:/hypercoplexAI",
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    ledger_rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert payload["run_id"] == "manifest-run-001"
    assert payload["config_hash"] == config.config_hash()
    assert payload["checkpoint"] == (results_path.parent / "checkpoints" / "hdim_final.pt").as_posix()
    assert ledger_rows[-1]["run_id"] == "manifest-run-001"
    assert ledger_rows[-1]["config_hash"] == config.config_hash()
    assert ledger_rows[-1]["checkpoint"] == payload["checkpoint"]


def test_train_script_supports_manifest_and_ledger(tmp_path):
    results_path = tmp_path / "results" / "manifest_run.json"
    ledger_path = tmp_path / "results" / "ledger.jsonl"
    config_path = tmp_path / "experiment.json"
    config = ExperimentConfig(
        description="manifest smoke",
        epochs=1,
        batch_size=4,
        num_samples=16,
        use_pairs=True,
        negative_ratio=0.25,
        text_mode=True,
        results_json=str(results_path),
        ledger_path=str(ledger_path),
    )
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "src.training.train", "--config", str(config_path)],
        cwd="E:/hypercoplexAI",
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    ledger_rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert payload["run_args"]["use_pairs"] is True
    assert payload["run_args"]["negative_ratio"] == pytest.approx(0.25)
    assert payload["run_args"]["text_mode"] is True
    assert payload["run_args"]["description"] == "manifest smoke"
    assert ledger_rows
    assert ledger_rows[-1]["status"] == "keep"
    assert "Appended run ledger row" in completed.stdout
