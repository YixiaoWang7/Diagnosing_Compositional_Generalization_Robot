"""Native preset coverage, budgets, and fine-tuning compatibility regressions."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from corl2026_cg.experiments.config import load_experiment, resolve_jobs

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PRESERVED_COUNTS = {
    "l4_full": 3,
    "l4_object_only": 3,
    "l4_container_only": 3,
    "l4_container_latents": 2,
    "l4_infer_cont0_full": 2,
    "l4_infer_cont0_orthogonal": 2,
    "l4_infer_cont0_cov9_B16": 1,
    "l5_demo_budgets": 20,
    "l5_orthogonal_smoke_30updates": 3,
    "l5_finetune_full": 6,
    "l5_finetune_random_B16": 15,
    "l5_finetune_random_B32": 15,
}
RELEASE_BASELINES = {
    "release_pp_50k": "L4 single stage",
    "release_ppp_50k": "L5",
    "release_2s_pp_100k": "L4 two stage infer_cont0",
    "release_2s_full_100k": "L4 two stage full",
}


def _scientific_settings(job: dict[str, Any]) -> dict[str, Any]:
    """Remove destinations, generated identities, and remote checkpoint storage.

    Args:
        job: Resolved manifest from a preset or the archived measured run.

    Returns:
        Independent manifest retaining all scientific and execution settings.
    """
    settings = copy.deepcopy(job)
    for key in ("name", "group", "checkpoint", "directory"):
        settings.pop(key)
    settings["run"]["dataset"].pop("root")
    settings["run"]["logging"].pop("out_dir")
    for key in ("name", "group"):
        settings["run"]["logging"]["wandb"].pop(key)
    # Upload transport does not change training or evaluation. Historical
    # evidence predates this opt-in field; keep those original records intact.
    settings["run"]["logging"]["wandb"].pop("upload_checkpoints", None)
    settings["evaluation"]["logging"]["wandb"].pop("upload_checkpoints", None)
    settings["evaluation"].pop("simulator_root")
    settings["evaluation"]["logging"].pop("out_dir")
    for key in ("train_python", "eval_python"):
        settings["execution"].pop(key)
    return settings


@pytest.mark.parametrize("preset,label", RELEASE_BASELINES.items())
def test_release_baselines_match_recorded_experiments(preset: str, label: str) -> None:
    """Keep release task semantics and measurements tied to the completed runs.

    Args:
        preset: Portable release preset filename without its extension.
        label: Corresponding measured run label in the committed evidence.
    """
    evidence_path = (
        CONFIGS.parents[2] / "docs/validation/cg_training_evaluation_setups_20261001.json"
    )
    evidence = json.loads(evidence_path.read_text())
    record = next(run for run in evidence["runs"] if run["label"] == label)
    spec = load_experiment(CONFIGS / f"{preset}.json")
    jobs = resolve_jobs(spec)
    assert len(jobs) == 1
    job = jobs[0]
    assert _scientific_settings(job) == _scientific_settings(record["resolved_experiment"])
    assert job["seed"] == 11
    assert job["finetune_from"] is None
    summary = record["evaluation_summary"]
    assert job["run"]["training"]["max_updates"] == summary["checkpoint_step"]
    assert (
        len(job["task_set"]["eval_slugs"]) * job["evaluation"]["n_trials_per_task"]
        == summary["aggregated"]["n_episodes"]
    )
    # Portable templates must never retain the author's filesystem or interpreter.
    for location in (
        spec.output_root,
        spec.run.dataset.root,
        spec.run.logging.out_dir,
        spec.evaluation.simulator_root,
        spec.evaluation.logging.out_dir,
    ):
        assert not Path(location).is_absolute()
    assert spec.execution.train_python == spec.execution.eval_python == sys.executable


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.json")), ids=lambda path: path.stem)
def test_all_checked_in_presets_resolve_native_jobs(path: Path) -> None:
    """Resolve without datasets, DINO downloads, W&B sessions, or GPU initialization."""
    spec = load_experiment(path)
    jobs = resolve_jobs(spec)
    assert jobs
    assert len({job["name"] for job in jobs}) == len(jobs)
    for job in jobs:
        run = job["run"]
        assert run["logging"]["wandb"]["project"] == "corl2026_cg"
        assert run["logging"]["wandb"]["enabled"]
        assert run["dataset"]["selection"]["include_tasks"] == job["task_set"]["train_slugs"]
        assert run["policy"]["action_steps"] == 16
        assert run["policy"]["state_dim"] == 9


def test_supported_local_preservation_snapshot_has_exactly_75_jobs() -> None:
    """Compact matrices must neither omit nor multiply the supported local recipes."""
    counts = {
        name: len(resolve_jobs(load_experiment(CONFIGS / f"{name}.json")))
        for name in PRESERVED_COUNTS
    }
    assert counts == PRESERVED_COUNTS
    assert sum(counts.values()) == 75


def test_local_l4_query_and_training_budgets() -> None:
    """Constant query slots and orthogonal update counts survive config translation."""
    constant = resolve_jobs(load_experiment(CONFIGS / "l4_container_latents.json"))
    assert {job["run"]["training"]["max_updates"] for job in constant} == {50000, 100000}
    for job in constant:
        assert job["run"]["policy"]["conditioning"]["vocab_sizes"] == [4, 1, 1]
        feature = next(
            value for value in job["run"]["features"]["features"] if value["key"] == "task_indices"
        )
        assert feature["paths"] == ["cg/task/container0", "cg/task/constant", "cg/task/constant"]
    orthogonal = resolve_jobs(load_experiment(CONFIGS / "l4_infer_cont0_orthogonal.json"))
    assert {job["seed"] for job in orthogonal} == {3, 4}
    assert {job["run"]["training"]["max_updates"] for job in orthogonal} == {150000}
    assert {len(job["task_set"]["train_slugs"]) for job in orthogonal} == {48}


def test_l5_budget_matrix_and_smoke_are_explicit() -> None:
    """Separate 30-update smoke jobs from full 50k-update budget comparisons."""
    jobs = resolve_jobs(load_experiment(CONFIGS / "l5_demo_budgets.json"))
    expected = {
        (name, seed, budget)
        for name in ("full", "orthogonal")
        for seed in (44, 55)
        for budget in (64, 128, 256, 512, 1024)
    }
    assert {
        (job["set_name"], job["seed"], job["run"]["dataset"]["selection"]["max_demos"])
        for job in jobs
    } == expected
    assert all(job["run"]["dataset"]["selection"]["balance_demos_across_tasks"] for job in jobs)
    assert all(job["run"]["training"]["max_updates"] == 50000 for job in jobs)
    smoke = load_experiment(CONFIGS / "l5_orthogonal_smoke_30updates.json")
    assert smoke.run.training.max_updates == 30
    assert "smoke_30updates" in smoke.name


@pytest.mark.parametrize(
    "target,source",
    [
        ("l5_finetune_full", "l5_pretrain_final_B10"),
        ("l5_finetune_random_B16", "l5_pretrain_diag_B4"),
        ("l5_finetune_random_B32", "l5_pretrain_diag_B4"),
    ],
)
def test_finetune_sources_resolve_to_compatible_native_pretraining(target, source) -> None:
    """Fine-tuning must point at companion native checkpoints with the same model schema."""
    source_job = resolve_jobs(load_experiment(CONFIGS / f"{source}.json"))[0]
    targets = resolve_jobs(load_experiment(CONFIGS / f"{target}.json"))
    for job in targets:
        assert Path(job["finetune_from"]).resolve() == Path(source_job["checkpoint"])
        assert Path(job["finetune_from"]).name == "latest.pt"
        assert job["run"]["policy"] == source_job["run"]["policy"]
        assert job["run"]["features"] == source_job["run"]["features"]
        assert job["run"]["training"]["max_updates"] == 5000


def test_state_improved_mean_flow_uses_only_selected_explicit_entities() -> None:
    """The native state baseline does not accidentally retain camera features."""
    config = load_experiment(CONFIGS / "example_l4_state_improved_mean_flow.json").run
    assert config.policy.cameras == 0
    assert config.policy.vision is None
    assert config.policy.algorithm.type == "improved_mean_flow"
    assert config.policy.algorithm.num_inference_steps == 1
    assert config.features["entities"].shape == (8, 7)
    assert config.features["entity_indices"].paths == ("cg/entity/object0", "cg/entity/container0")
    assert "images" not in {feature.key for feature in config.features.features}
