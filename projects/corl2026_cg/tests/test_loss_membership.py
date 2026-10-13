"""Offline results distinguish selected training tasks and incomplete datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from robopolicy import RunConfig, Trainer, read_checkpoint
from test_experiments import make_training_data, native_config
from test_report_identity import _job

from corl2026_cg.experiments.losses import evaluate_losses
from corl2026_cg.experiments.report import write_report
from corl2026_cg.tasks import build_task_set


def _remove_manifest_tasks(root: Path, removed: set[str]) -> None:
    """Simulate an unavailable task while leaving selected training files unchanged."""
    path = root / "dataset_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["tasks"] = [task for task in manifest["tasks"] if task["task_slug"] not in removed]
    path.write_text(json.dumps(manifest))


def test_loss_membership_uses_actual_budgeted_checkpoint_selection(tmp_path: Path) -> None:
    """A requested full design with one selected demo has fifteen unseen tasks."""
    make_training_data(tmp_path / "data")
    values = native_config(tmp_path).model_dump(mode="json")
    values["dataset"]["selection"].update(
        include_tasks=list(build_task_set("l4", "full").train_slugs), max_demos=1
    )
    trainer = Trainer(RunConfig.model_validate(values))
    try:
        trainer.fit()
    finally:
        trainer.close()
    checkpoint = tmp_path / "train/latest.pt"
    selected = {identity[0] for identity in read_checkpoint(checkpoint)["selection"]}
    assert len(selected) == 1
    result = evaluate_losses(checkpoint, family="l4", samples=3, batch_size=3)
    assert set(result["training_task_slugs"]) == selected
    assert set(result["splits"]["trained_tasks"]["task_slugs"]) == selected
    assert result["splits"]["unseen_tasks"]["requested_task_count"] == 15
    assert not selected.intersection(result["splits"]["unseen_tasks"]["task_slugs"])


@pytest.mark.parametrize("remove_all_unseen", [False, True])
def test_missing_unseen_tasks_are_reported_without_failing_available_losses(
    tmp_path: Path, remove_all_unseen: bool
) -> None:
    """Missing demonstrations remain explicit in JSON and the readable report."""
    make_training_data(tmp_path / "data")
    trained = build_task_set("l4", "diag").train_slugs
    values = native_config(tmp_path).model_dump(mode="json")
    values["dataset"]["selection"]["include_tasks"] = trained
    trainer = Trainer(RunConfig.model_validate(values))
    try:
        trainer.fit()
    finally:
        trainer.close()
    unseen = set(build_task_set("l4", "full").train_slugs) - set(trained)
    missing = unseen if remove_all_unseen else {"cross_into_mug"}
    _remove_manifest_tasks(tmp_path / "data", missing)
    result = evaluate_losses(tmp_path / "train/latest.pt", family="l4", samples=3, batch_size=3)
    split = result["splits"]["unseen_tasks"]
    assert set(split["missing_task_slugs"]) == missing
    assert result["splits"]["trained_tasks"]["status"] == "complete"
    if remove_all_unseen:
        assert split["status"] == "unavailable"
        assert split["num_samples"] == 0
    else:
        assert split["status"] == "complete"
        assert split["requested_task_count"] == 12
        assert split["available_task_count"] == 11
        assert set(split["task_slugs"]) == unseen - missing
    directory, _ = _job(tmp_path, 11)
    (directory / "losses.json").write_text(json.dumps(result))
    report = write_report([directory], tmp_path / "report")
    reported = next(record for record in report["results"] if record["kind"] == "loss")
    assert reported["splits"]["unseen_tasks"]["missing_task_slugs"] == sorted(missing)
    markdown = (tmp_path / "report/report.md").read_text()
    assert "Tasks unavailable in the offline dataset" in markdown
    assert all(task in markdown for task in missing)


def test_missing_trained_tasks_fail_before_allocating_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evaluation cannot silently replace checkpoint training membership."""
    make_training_data(tmp_path / "data")
    trainer = Trainer(native_config(tmp_path))
    try:
        trainer.fit()
    finally:
        trainer.close()
    _remove_manifest_tasks(tmp_path / "data", {"cross_into_bin"})

    def forbidden_load(*args: object, **kwargs: object) -> None:
        """Missing-data validation must happen before model allocation."""
        pytest.fail("Policy loaded before checking selected training membership")

    monkeypatch.setattr("corl2026_cg.experiments.losses.load_policy", forbidden_load)
    with pytest.raises(ValueError, match="missing trained tasks"):
        evaluate_losses(tmp_path / "train/latest.pt", family="l4")
