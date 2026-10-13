"""Exercise the actual CLI from immutable export through checkpoint and reporting."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_experiments import make_training_data, spec


def invoke(*arguments: str) -> str:
    """Run the application as a fresh process, preserving the test import roots."""
    root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(root / "projects/corl2026_cg/src")]
    )
    return subprocess.check_output(
        [sys.executable, "-m", "corl2026_cg", *arguments],
        env=environment,
        text=True,
        stderr=subprocess.STDOUT,
    )


def test_help_does_not_import_learning_or_simulation() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from corl2026_cg.cli import main
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
assert not {'torch', 'h5py', 'robosuite', 'mujoco', 'wandb'} & sys.modules.keys()
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_prepare_generate_train_loss_report_in_fresh_processes(tmp_path: Path) -> None:
    make_training_data(tmp_path / "original", prepared=False)
    invoke(
        "prepare", "--source", str(tmp_path / "original"), "--destination", str(tmp_path / "data")
    )
    original = spec(tmp_path).model_copy(update={"seeds": (11,)})
    template = tmp_path / "spec.json"
    template.write_text(original.model_dump_json())
    jobs = json.loads(invoke("generate", "--config", str(template), "--dry-run"))
    assert len(jobs) == 1
    directory = Path(jobs[0]["directory"])
    assert not directory.exists()
    invoke("generate", "--config", str(template))
    manifest = directory / "experiment.json"
    invoke("run", "--jobs", str(manifest), "--skip-eval")
    assert Path(jobs[0]["checkpoint"]).is_file()
    assert json.loads((directory / "train_status.json").read_text())["status"] == "complete"
    with pytest.raises(subprocess.CalledProcessError) as failed:
        invoke("train", "--job", str(manifest))
    assert "Checkpoint already exists" in failed.value.output
    invoke(
        "evaluate-loss",
        "--job",
        str(manifest),
        "--samples",
        "5",
        "--batch-size",
        "3",
        "--device",
        "cpu",
        "--local-only",
    )
    result = json.loads((directory / "losses.json").read_text())
    assert result["splits"]["trained_tasks"]["num_samples"] == 5
    assert result["splits"]["unseen_tasks"]["num_samples"] == 5
    output = tmp_path / "report"
    invoke("report", "--jobs", str(directory), "--output", str(output))
    report = json.loads((output / "report.json").read_text())
    assert report["summaries"]
    assert any(
        item["status"] == "missing" and item["kind"] == "rollout" for item in report["results"]
    )
    assert (output / "report.md").is_file()
