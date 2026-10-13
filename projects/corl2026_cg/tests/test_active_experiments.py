"""Application contracts distinguish reactive tokens from composition identity."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from robopolicy import RunConfig
from test_experiments import spec
from test_report_identity import _job, _rollout

from corl2026_cg.experiments.config import ExperimentSpec, resolve_jobs
from corl2026_cg.experiments.report import write_report
from corl2026_cg.instructions import get_profile
from corl2026_cg.instructions.contracts import (
    policy_profile,
    validate_finetune_conditioning,
    validate_policy_conditioning,
)


def active_spec(root: Path, *, family: str = "l5") -> ExperimentSpec:
    """Reuse a tiny native architecture while changing only application semantics."""
    values = spec(root).model_dump(mode="json")
    values.update(
        family=family,
        instruction="full" if family == "l5" else "infer_cont0",
        conditioning="active_v1",
        sets=["diag_B4" if family == "l5" else "orthogonal"],
        seeds=[11],
        eval_set="full",
    )
    return ExperimentSpec.model_validate(values)


def test_active_jobs_use_one_token_without_collapsing_composition_membership(
    tmp_path: Path,
) -> None:
    job = resolve_jobs(active_spec(tmp_path))[0]
    config = RunConfig.model_validate(job["run"])
    assert config.features["task_indices"].paths == ("cg/active_instruction/l5_v1",)
    assert config.policy.conditioning.vocab_sizes == (12,)
    assert len(job["task_set"]["train_slugs"]) == 4
    assert len(job["task_set"]["id_slugs"]) == 4
    assert len(job["task_set"]["ood_slugs"]) == 60
    assert policy_profile(config) == get_profile("l5_v1")
    assert job["active_profile"] == get_profile("l5_v1").to_dict()


def test_evaluation_population_does_not_change_training_group(tmp_path: Path) -> None:
    original = active_spec(tmp_path)
    full = resolve_jobs(original)[0]
    subset = resolve_jobs(original.model_copy(update={"eval_set": "diag_B4"}))[0]
    assert full["group"] == subset["group"]
    assert full["run"] == subset["run"]
    assert len(subset["task_set"]["eval_slugs"]) == 4
    assert subset["task_set"]["ood_slugs"] == []


def test_active_mode_changes_training_identity(tmp_path: Path) -> None:
    active = active_spec(tmp_path)
    static = active.model_copy(update={"conditioning": "static"})
    assert resolve_jobs(active)[0]["group"] != resolve_jobs(static)[0]["group"]


def test_active_mode_requires_supported_task_semantics(tmp_path: Path) -> None:
    values = active_spec(tmp_path).model_dump(mode="json")
    values.update(family="l4", sets=["full"])
    with pytest.raises(ValueError):
        ExperimentSpec.model_validate(values)


def test_equal_vocabulary_sizes_do_not_allow_cross_family_finetuning(tmp_path: Path) -> None:
    l5 = RunConfig.model_validate(resolve_jobs(active_spec(tmp_path))[0]["run"])
    l4 = RunConfig.model_validate(resolve_jobs(active_spec(tmp_path, family="l4_2s"))[0]["run"])
    assert l5.policy.conditioning == l4.policy.conditioning
    with pytest.raises(ValueError, match="matching active instruction profiles"):
        validate_finetune_conditioning(l4, l5)
    validate_finetune_conditioning(l5, l5)
    with pytest.raises(ValueError, match="Active conditioning requires"):
        validate_policy_conditioning(l5, "l4_2s", "infer_cont0", "active_v1")


def test_unknown_profile_versions_are_rejected(tmp_path: Path) -> None:
    values = resolve_jobs(active_spec(tmp_path))[0]["run"]
    for feature in values["features"]["features"]:
        if feature["key"] == "task_indices":
            feature["paths"] = ["cg/active_instruction/l5_v999"]
    with pytest.raises(ValueError):
        policy_profile(RunConfig.model_validate(values))


def test_report_never_combines_different_active_profiles(tmp_path: Path) -> None:
    directories = []
    for seed in (11, 22):
        directory, job = _job(tmp_path, seed)
        actual = _rollout(directory, job)
        actual["conditioning"] = "active_v1"
        actual["active_profile"] = copy.deepcopy(get_profile("l5_v1").to_dict())
        if seed == 22:
            actual["active_profile"]["name"] = "future_profile"
        (directory / "evaluation/eval_config_used.json").write_text(json.dumps(actual))
        directories.append(directory)
    report = write_report(directories, tmp_path / "report")
    assert len(report["summaries"]) == 2
