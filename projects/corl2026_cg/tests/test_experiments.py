"""Experiment identities, safe orchestration, and real checkpoint loss evaluation."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pytest
from robopolicy import RunConfig, Trainer, read_checkpoint

from corl2026_cg.experiments import runner
from corl2026_cg.experiments.config import (
    ExecutionConfig,
    ExperimentSpec,
    load_experiment,
    resolve_jobs,
    set_nested,
    write_jobs,
)
from corl2026_cg.experiments.losses import evaluate_losses
from corl2026_cg.experiments.report import write_report
from corl2026_cg.tasks import (
    build_task_set,
    instruction_fields,
    parse_task,
    semantic_values,
    validate_conditioning,
    vocab_sizes,
)


def native_config(
    root: Path, algorithm: str = "flow_matching", instruction: str = "full"
) -> RunConfig:
    """Build a tiny indexed state policy with explicit native configuration."""
    return RunConfig.model_validate(
        {
            "features": {
                "features": [
                    {"key": "state", "shape": [2], "paths": ["state"], "normalization": "standard"},
                    {
                        "key": "actions",
                        "shape": [2],
                        "paths": ["actions"],
                        "future": 1,
                        "normalization": "standard",
                    },
                    {
                        "key": "task_indices",
                        "shape": [len(instruction_fields("l4", instruction))],
                        "paths": [
                            f"cg/task/{field}" for field in instruction_fields("l4", instruction)
                        ],
                        "dtype": "int64",
                    },
                ]
            },
            "dataset": {"root": str(root / "data"), "selection": {"val_fraction": 0}},
            "loader": {"batch_size": 3},
            "policy": {
                "state_dim": 2,
                "action_dim": 2,
                "action_steps": 2,
                "conditioning": {"type": "indexed", "vocab_sizes": vocab_sizes("l4", instruction)},
                "fusion": {"dim": 8, "heads": 2, "layers": 1, "ff_dim": 16},
                "head": {
                    "type": "transformer",
                    "dim": 8,
                    "heads": 2,
                    "layers": 1,
                    "ff_dim": 16,
                    "time_dim": 8,
                },
                "algorithm": {"type": algorithm, "num_inference_steps": 1},
            },
            "training": {"max_updates": 2, "checkpoint_every": 1, "log_every": 1},
            "logging": {"out_dir": str(root / "train"), "wandb": {"enabled": False}},
        }
    )


def make_training_data(root: Path, *, prepared: bool = True) -> None:
    """Write all sixteen L4 tasks so trained and unseen task losses are evaluable."""
    tasks = []
    for index, slug in enumerate(build_task_set("l4", "full").train_slugs):
        directory = root / slug
        directory.mkdir(parents=True)
        with h5py.File(directory / "shard.h5", "w") as handle:
            group = handle.create_group("data/episode")
            state = np.arange(8, dtype=np.float32).reshape(4, 2) / 8 + index / 16
            group["state"] = state
            group["actions"] = state * 0.5
            if prepared:
                for field, value in semantic_values(parse_task(slug)).items():
                    group[f"cg/task/{field}"] = np.full((4, 1), value, dtype=np.int64)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "shards": [
                        {
                            "shard_id": 0,
                            "hdf5": "shard.h5",
                            "demos": [{"demo_key": "episode", "length": 4}],
                        }
                    ],
                }
            )
        )
        tasks.append({"task_slug": slug, "manifest": f"{slug}/manifest.json"})
    (root / "dataset_manifest.json").write_text(json.dumps({"tasks": tasks}))


def spec(root: Path) -> ExperimentSpec:
    """Declare two seeds without depending on datasets or a simulator."""
    return ExperimentSpec(
        name="unit",
        family="l4",
        sets=("diag_cor",),
        seeds=(11, 22),
        run=native_config(root),
        output_root=str(root / "runs"),
    )


def test_group_identity_omits_training_seed_but_retains_data_and_model_settings(
    tmp_path: Path,
) -> None:
    original = spec(tmp_path)
    jobs = resolve_jobs(original)
    assert len(jobs) == 2 and jobs[0]["group"] == jobs[1]["group"]
    assert jobs[0]["name"] != jobs[1]["name"]
    moved = original.model_copy(update={"seeds": (77,), "output_root": str(tmp_path / "elsewhere")})
    assert resolve_jobs(moved)[0]["group"] == jobs[0]["group"]
    for path, value in (
        ("dataset.selection.subset_seed", 17),
        ("dataset.sampling.seed", 17),
        ("policy.head.layers", 2),
    ):
        altered = original.model_copy(update={"matrix": {path: [value]}})
        assert resolve_jobs(altered)[0]["group"] != jobs[0]["group"]
    matrix = original.model_copy(update={"matrix": {"policy.head.layers": [1, 2]}})
    assert len({job["group"] for job in resolve_jobs(matrix)}) == 2
    assert len(resolve_jobs(matrix)) == 4
    with pytest.raises(ValueError, match="Duplicate resolved"):
        resolve_jobs(original.model_copy(update={"matrix": {"policy.head.layers": [1, 1]}}))


def test_overrides_are_strict_and_accept_existing_list_indices(tmp_path: Path) -> None:
    path = tmp_path / "experiment.json"
    path.write_text(spec(tmp_path).model_dump_json())
    changed = load_experiment(path, ["seeds.0=33", "run.policy.head.layers=2"])
    assert changed.seeds == (33, 22) and changed.run.policy.head.layers == 2
    for override in ("missing=1", "seeds.-1=3", "seeds.9=3", "run..seed=1", "run.seed"):
        with pytest.raises(ValueError):
            load_experiment(path, [override])
    with pytest.raises(ValueError):
        load_experiment(path, ["run.policy.head.layers=0"])
    with pytest.raises(ValueError):
        set_nested({"list": [1]}, "list.bad", 2)
    with pytest.raises(ValueError, match="disjoint"):
        ExecutionConfig(overlap=True, train_gpus="0,1", eval_gpus="1")
    assert ExecutionConfig(overlap=True, train_gpus="0", eval_gpus="1").overlap


def test_write_jobs_preflights_all_collisions_before_publishing(tmp_path: Path) -> None:
    jobs = resolve_jobs(spec(tmp_path))
    later = Path(jobs[1]["directory"])
    later.mkdir(parents=True)
    (later / "run.json").write_text("preserve me")
    with pytest.raises(FileExistsError):
        write_jobs(jobs)
    assert not Path(jobs[0]["directory"]).exists()
    assert (later / "run.json").read_text() == "preserve me"
    duplicate = copy.deepcopy(jobs[1])
    duplicate["directory"] = jobs[0]["directory"]
    with pytest.raises(ValueError, match="Duplicate output"):
        write_jobs([jobs[0], duplicate], overwrite=True)
    assert not Path(jobs[0]["directory"]).exists()
    paths = write_jobs(jobs, overwrite=True)
    assert len(paths) == 2 and all(path.is_file() for path in paths)


def test_process_failure_is_recorded_and_propagated(tmp_path: Path) -> None:
    executable = tmp_path / "failing-python"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$CUDA_VISIBLE_DEVICES"\nexit 9\n')
    executable.chmod(0o755)
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps(
            {
                "execution": {
                    "train_python": str(executable),
                    "eval_python": sys.executable,
                    "train_gpus": "6",
                    "eval_gpus": "7",
                }
            }
        )
    )
    with pytest.raises(subprocess.CalledProcessError) as error:
        runner.run_stage(path, "train")
    assert error.value.returncode == 9
    assert json.loads((tmp_path / "train_status.json").read_text())["status"] == "failed"
    assert (tmp_path / "train.log").read_text().strip() == "6"


def test_overlap_bounds_evaluation_backlog_and_finishes_every_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [Path(str(index)) for index in range(5)]
    release = threading.Event()
    started = threading.Event()
    trained_three = threading.Event()
    lock = threading.Lock()
    trained: list[Path] = []
    evaluated: list[Path] = []
    active = 0
    peak = 0

    def stage(path: Path, name: str) -> None:
        """Hold the first evaluation so the orchestrator reaches its backlog bound."""
        nonlocal active, peak
        if name == "train":
            with lock:
                trained.append(path)
                if len(trained) == 3:
                    trained_three.set()
            return
        with lock:
            active += 1
            peak = max(peak, active)
        if path == paths[0]:
            started.set()
            assert release.wait(5)
        with lock:
            evaluated.append(path)
            active -= 1

    monkeypatch.setattr(runner, "run_stage", stage)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(runner.run_jobs, paths, overlap=True)
        try:
            assert started.wait(5) and trained_three.wait(5)
            assert trained == paths[:3] and not future.done()
        finally:
            release.set()
        future.result(timeout=5)
    assert trained == evaluated == paths and peak == 1


def test_failed_background_evaluation_stops_subsequent_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [Path(str(index)) for index in range(4)]
    failed = threading.Event()
    trained: list[Path] = []

    def stage(path: Path, name: str) -> None:
        """Fail evaluation while the next permitted training job is finishing."""
        if name == "evaluate":
            failed.set()
            raise RuntimeError("evaluation failed")
        trained.append(path)
        if path != paths[0]:
            assert failed.wait(5)

    monkeypatch.setattr(runner, "run_stage", stage)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        runner.run_jobs(paths, overlap=True)
    assert trained in (paths[:1], paths[:2])


def test_report_keeps_id_ood_sample_counts_and_missing_results(tmp_path: Path) -> None:
    jobs = resolve_jobs(spec(tmp_path))
    paths = write_jobs(jobs)
    directories = [path.parent for path in paths]
    for directory, success in zip(directories, (40, 60)):
        (directory / "evaluation").mkdir()
        (directory / "evaluation/eval_summary.json").write_text(
            json.dumps(
                {
                    "aggregated": {"pc_success": success},
                    "per_task": {"cross_into_bin": {"pc_success": success, "n_episodes": 2}},
                    "splits": {
                        "id": {"n_episodes": 2, "pc_success": success},
                        "ood": {"n_episodes": 0, "pc_success": None},
                    },
                }
            )
        )
    (directories[0] / "losses.json").write_text(
        json.dumps(
            {
                "splits": {
                    "trained_tasks": {
                        "status": "complete",
                        "num_samples": 3,
                        "objective_loss": 0.25,
                    },
                    "unseen_tasks": {"status": "not_applicable", "num_samples": 0},
                }
            }
        )
    )
    report = write_report(directories, tmp_path / "report")
    metrics = {row["metric"]: row for row in report["summaries"]}
    assert metrics["splits/id/pc_success"]["mean"] == 50
    assert metrics["splits/id/pc_success"]["seeds"] == 2
    assert metrics["per_task/cross_into_bin/pc_success"]["mean"] == 50
    assert metrics["per_task/cross_into_bin/pc_success"]["seeds"] == 2
    assert "splits/ood/pc_success" not in metrics
    assert metrics["splits/trained_tasks/num_samples"]["mean"] == 3
    assert metrics["splits/trained_tasks/objective_loss"]["seeds"] == 1
    assert [result["kind"] for result in report["results"] if result["status"] == "missing"] == [
        "loss"
    ]
    loss = next(
        result
        for result in report["results"]
        if result["kind"] == "loss" and result["status"] == "complete"
    )
    assert loss["splits"]["unseen_tasks"]["status"] == "not_applicable"
    assert "Missing results" in (tmp_path / "report/report.md").read_text()
    with pytest.raises(ValueError, match="Duplicate report"):
        write_report([directories[0], directories[0]], tmp_path / "duplicate")


def test_report_with_no_finished_results_is_explicitly_empty(tmp_path: Path) -> None:
    path = write_jobs(resolve_jobs(spec(tmp_path)))[0]
    report = write_report([path.parent], tmp_path / "report")
    assert report["summaries"] == []
    assert all(result["status"] == "missing" for result in report["results"])
    assert "No completed evaluation metrics" in (tmp_path / "report/report.md").read_text()


@pytest.mark.parametrize("algorithm", ["flow_matching", "improved_mean_flow"])
def test_native_training_checkpoint_and_offline_task_losses(tmp_path: Path, algorithm: str) -> None:
    make_training_data(tmp_path / "data")
    values = native_config(tmp_path, algorithm).model_dump(mode="json")
    trained = build_task_set("l4", "diag_cor").train_slugs
    values["dataset"]["selection"]["include_tasks"] = trained
    config = RunConfig.model_validate(values)
    Trainer(config).fit()
    checkpoint = tmp_path / "train/latest.pt"
    assert read_checkpoint(checkpoint)["completed_updates"] == 2
    result = evaluate_losses(checkpoint, family="l4", samples=100, batch_size=5, seed=19)
    assert result["algorithm"] == algorithm
    assert result["training_seed"] == config.seed
    assert result["family"] == "l4" and result["instruction"] == "full"
    assert result["training_task_slugs"] == sorted(trained)
    assert result["splits"]["trained_tasks"]["num_samples"] == len(trained) * 3
    assert result["splits"]["unseen_tasks"]["num_samples"] == (16 - len(trained)) * 3
    for split in result["splits"].values():
        assert all(
            math.isfinite(split[key])
            for key in ("objective_loss", "action_mse_normalized", "action_mse_raw")
        )
    repeated = evaluate_losses(checkpoint, family="l4", samples=100, batch_size=5, seed=19)
    for name, split in result["splits"].items():
        assert split["objective_loss"] == repeated["splits"][name]["objective_loss"]
        assert split["action_mse_raw"] == repeated["splits"][name]["action_mse_raw"]


def test_full_training_set_has_no_unseen_task_loss(tmp_path: Path) -> None:
    make_training_data(tmp_path / "data")
    Trainer(native_config(tmp_path)).fit()
    result = evaluate_losses(tmp_path / "train/latest.pt", family="l4", samples=2, batch_size=3)
    assert result["splits"]["trained_tasks"]["num_samples"] == 2
    assert result["splits"]["unseen_tasks"] == {"status": "not_applicable", "num_samples": 0}
    with pytest.raises(ValueError, match="positive"):
        evaluate_losses(tmp_path / "train/latest.pt", family="l4", samples=0)


def test_same_width_instruction_mismatch_fails_before_model_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_training_data(tmp_path / "data")
    Trainer(native_config(tmp_path, instruction="object_only")).fit()

    def forbidden_load(*args: object, **kwargs: object) -> None:
        """Require semantic validation before allocating a policy."""
        pytest.fail("Policy construction must follow conditioning validation")

    monkeypatch.setattr("corl2026_cg.experiments.losses.load_policy", forbidden_load)
    with pytest.raises(ValueError, match="task_indices paths"):
        evaluate_losses(tmp_path / "train/latest.pt", family="l4", instruction="container_only")
    with pytest.raises(ValueError, match="vocabulary sizes"):
        validate_conditioning("l4", "object_only", ("cg/task/object0",), (5,))


def test_loss_rejects_checkpoint_task_filters_from_another_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_training_data(tmp_path / "data")
    Trainer(native_config(tmp_path)).fit()
    payload = read_checkpoint(tmp_path / "train/latest.pt")
    payload["run_config"]["dataset"]["selection"]["include_tasks"] = ["cross_into_bin_press_red"]
    monkeypatch.setattr("corl2026_cg.experiments.losses.read_checkpoint", lambda path: payload)
    with pytest.raises(ValueError, match="does not match family"):
        evaluate_losses(tmp_path / "train/latest.pt", family="l4")
