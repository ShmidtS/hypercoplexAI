import json
import subprocess
import sys


def test_train_script_writes_output_dir_artifacts(tmp_path):
    output_dir = tmp_path / "run"
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
        "--output_dir",
        str(output_dir),
    ]
    completed = subprocess.run(
        command,
        cwd="E:/hypercoplexAI",
        check=True,
        capture_output=True,
        text=True,
    )

    results_path = output_dir / "run_summary.json"
    checkpoint_path = output_dir / "checkpoints" / "hdim_final.pt"
    assert results_path.exists()
    assert checkpoint_path.exists()

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["run_args"]["epochs"] == 1
    assert payload["run_args"]["batch_size"] == 4
    assert payload["run_args"]["data_path"] is None
    assert payload["validation"]["loss_total"] >= 0.0
    assert set(payload["quality"]) == {"STS_exported", "STS_training", "DRS", "AFR", "pair_margin"}
    assert payload["checkpoint"] == checkpoint_path.as_posix()
    assert "Wrote run summary" in completed.stdout
