"""Report comparisons follow actual measurements rather than declared job labels."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from corl2026_cg.experiments.report import write_report


def _job(root: Path, seed: int) -> tuple[Path, dict[str, Any]]:
    """Write an independent training seed with shared scientific configuration."""
    directory = root / f"seed{seed}"
    directory.mkdir()
    job = {
        "name": f"unit_seed{seed}",
        "group": "unit",
        "seed": seed,
        "family": "l4",
        "instruction": "full",
        "task_set": {
            "train_slugs": ["cross_into_bin"],
            "eval_slugs": ["cross_into_bin", "cross_into_mug"],
        },
        "evaluation": {
            "horizon": 400,
            "n_execute": 8,
            "n_trials_per_task": 25,
            "seed": 0,
            "simulator_revision": "pinned",
            "simulator_root": f"/machine{seed}/simulator",
            "device": f"cuda:{seed}",
            "logging": {"wandb": {"name": f"unit_seed{seed}"}},
        },
    }
    (directory / "experiment.json").write_text(json.dumps(job))
    return directory, job


def _rollout(directory: Path, job: dict[str, Any], *, recorded: bool = True) -> dict[str, Any]:
    """Create a result and the actual settings used to produce it."""
    evaluation = directory / "evaluation"
    evaluation.mkdir()
    (evaluation / "eval_summary.json").write_text(
        json.dumps({"checkpoint_step": 100, "aggregated": {"pc_success": 50}})
    )
    actual = {
        key: copy.deepcopy(job[key]) for key in ("family", "instruction", "task_set", "evaluation")
    }
    actual.update(checkpoint=str(directory / "train/latest.pt"), checkpoint_step=100)
    if recorded:
        (evaluation / "eval_config_used.json").write_text(json.dumps(actual))
    return actual


def _loss(directory: Path, seed: int) -> dict[str, Any]:
    """Create a loss summary with explicit sampling and checkpoint provenance."""
    result = {
        "metric_seed": 0,
        "sampling_seed": 0,
        "requested_samples": 500,
        "batch_size": 50,
        "family": "l4",
        "instruction": "full",
        "checkpoint_step": 100,
        "training_seed": seed,
        "checkpoint": str(directory / "train/latest.pt"),
        "elapsed_seconds": float(seed),
        "splits": {"trained_tasks": {"objective_loss": 0.5}},
    }
    (directory / "losses.json").write_text(json.dumps(result))
    return result


def test_rollout_groups_match_across_training_seeds_and_paths(tmp_path: Path) -> None:
    directories = []
    for seed in (11, 22):
        directory, job = _job(tmp_path, seed)
        _rollout(directory, job)
        directories.append(directory)
    report = write_report(directories, tmp_path / "report")
    assert len(report["summaries"]) == 1
    assert report["summaries"][0]["group"] == "unit"
    assert report["summaries"][0]["seeds"] == 2
    assert report["summaries"][0]["evaluation_identity"].startswith("rollout-")
    assert all(
        result["provenance"]["status"] == "recorded"
        for result in report["results"]
        if result["status"] == "complete"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("horizon", 800), ("n_execute", 1), ("seed", 7), ("simulator_revision", "other")],
)
def test_actual_rollout_overrides_separate_comparisons(
    tmp_path: Path, field: str, value: Any
) -> None:
    first, first_job = _job(tmp_path, 11)
    second, second_job = _job(tmp_path, 22)
    _rollout(first, first_job)
    actual = _rollout(second, second_job)
    actual["evaluation"][field] = value
    (second / "evaluation/eval_config_used.json").write_text(json.dumps(actual))
    report = write_report([first, second], tmp_path / "report")
    assert len(report["summaries"]) == 2
    assert len({row["evaluation_identity"] for row in report["summaries"]}) == 2
    assert {row["seeds"] for row in report["summaries"]} == {1}


@pytest.mark.parametrize("field", ["family", "instruction", "task_set", "checkpoint_step"])
def test_task_semantics_and_checkpoint_steps_separate_rollouts(tmp_path: Path, field: str) -> None:
    first, first_job = _job(tmp_path, 11)
    second, second_job = _job(tmp_path, 22)
    _rollout(first, first_job)
    actual = _rollout(second, second_job)
    actual[field] = {
        "family": "l5",
        "instruction": "object_only",
        "task_set": {"eval_slugs": ["cube_into_bin"]},
        "checkpoint_step": 50,
    }[field]
    (second / "evaluation/eval_config_used.json").write_text(json.dumps(actual))
    if field == "checkpoint_step":
        (second / "evaluation/eval_summary.json").write_text(
            json.dumps({"checkpoint_step": 50, "aggregated": {"pc_success": 50}})
        )
    report = write_report([first, second], tmp_path / "report")
    assert len(report["summaries"]) == 2


def test_missing_actual_rollout_config_is_explicit_and_not_merged(tmp_path: Path) -> None:
    first, first_job = _job(tmp_path, 11)
    second, second_job = _job(tmp_path, 22)
    _rollout(first, first_job)
    _rollout(second, second_job, recorded=False)
    report = write_report([first, second], tmp_path / "report")
    assert len(report["summaries"]) == 2
    fallback = next(
        result
        for result in report["results"]
        if result["name"] == second_job["name"] and result["kind"] == "rollout"
    )
    assert fallback["provenance"]["status"] == "declared_only"
    assert fallback["provenance"]["source"] == "experiment.json fallback"
    assert "Incomplete evaluation provenance" in (tmp_path / "report/report.md").read_text()


def test_disagreeing_rollout_artifacts_fail_clearly(tmp_path: Path) -> None:
    directory, job = _job(tmp_path, 11)
    actual = _rollout(directory, job)
    actual["checkpoint_step"] = 200
    (directory / "evaluation/eval_config_used.json").write_text(json.dumps(actual))
    with pytest.raises(ValueError, match="Checkpoint step differs"):
        write_report([directory], tmp_path / "report")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metric_seed", 1),
        ("sampling_seed", 1),
        ("requested_samples", 1000),
        ("batch_size", 25),
        ("family", "l5"),
        ("instruction", "object_only"),
        ("checkpoint_step", 50),
    ],
)
def test_loss_sampling_settings_separate_comparisons(
    tmp_path: Path, field: str, value: Any
) -> None:
    first, _ = _job(tmp_path, 11)
    second, _ = _job(tmp_path, 22)
    _loss(first, 11)
    result = _loss(second, 22)
    result[field] = value
    (second / "losses.json").write_text(json.dumps(result))
    report = write_report([first, second], tmp_path / "report")
    assert len(report["summaries"]) == 2
    assert {row["seeds"] for row in report["summaries"]} == {1}


def test_loss_training_seed_paths_and_runtime_do_not_split_comparisons(tmp_path: Path) -> None:
    first, _ = _job(tmp_path, 11)
    second, _ = _job(tmp_path, 22)
    _loss(first, 11)
    _loss(second, 22)
    report = write_report([first, second], tmp_path / "report")
    assert len(report["summaries"]) == 1
    assert report["summaries"][0]["seeds"] == 2


def test_older_loss_metadata_remains_readable_with_explicit_missing_fields(tmp_path: Path) -> None:
    directory, _ = _job(tmp_path, 11)
    result = _loss(directory, 11)
    del result["batch_size"]
    (directory / "losses.json").write_text(json.dumps(result))
    report = write_report([directory], tmp_path / "report")
    loss = next(result for result in report["results"] if result["kind"] == "loss")
    assert loss["provenance"]["status"] == "incomplete"
    assert loss["provenance"]["missing_fields"] == ["batch_size"]


@pytest.mark.parametrize("same_data", [True, False])
def test_loss_dataset_identity_uses_content_fingerprints_over_locations(
    tmp_path: Path, same_data: bool
) -> None:
    first, _ = _job(tmp_path, 11)
    second, _ = _job(tmp_path, 22)
    for index, directory in enumerate((first, second)):
        result = _loss(directory, index)
        result["dataset_root"] = str(tmp_path / f"data-copy{index}")
        fingerprint = "same" if same_data else f"content{index}"
        result["dataset_fingerprints"] = {"trained_tasks": fingerprint}
        (directory / "losses.json").write_text(json.dumps(result))
    report = write_report([first, second], tmp_path / "report")
    assert len(report["summaries"]) == (1 if same_data else 2)


def test_loss_without_fingerprints_keeps_dataset_root_in_comparison(tmp_path: Path) -> None:
    directories = []
    for seed in (11, 22):
        directory, _ = _job(tmp_path, seed)
        result = _loss(directory, seed)
        result["dataset_root"] = str(tmp_path / f"data{seed}")
        (directory / "losses.json").write_text(json.dumps(result))
        directories.append(directory)
    report = write_report(directories, tmp_path / "report")
    assert len(report["summaries"]) == 2
