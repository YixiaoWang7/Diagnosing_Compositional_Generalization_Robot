"""Same-run strict W&B recovery and append-only overlapping update axes."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from robopolicy.config.run import RunConfig
from robopolicy.training.logging import (
    LoggingConfig,
    MetricsLogger,
    RunIdentity,
    WandbConfig,
    log_evaluation,
)
from robopolicy.training.trainer import Trainer
from tests.core.test_training import run_config


class FakeWandb:
    """Small service substitute retaining histories and artifacts by run identity."""

    def __init__(self):
        self.runs = {}
        self.calls = []
        self.axes = []
        self.artifacts = []

    class Settings:
        def __init__(self, **kwargs):
            self.values = kwargs

    class Artifact:
        def __init__(self, **kwargs):
            self.values = kwargs
            self.files = []

        def add_file(self, path):
            self.files.append(path)

    def init(self, **kwargs):
        self.calls.append(kwargs)
        identity = (kwargs.get("entity") or "team", kwargs["project"], kwargs["id"])
        exists = identity in self.runs
        if kwargs["resume"] == "must" and not exists:
            raise ValueError("run does not exist")
        history = self.runs.setdefault(identity, [])
        return SimpleNamespace(
            id=identity[2],
            entity=identity[0],
            project=identity[1],
            resumed=exists,
            url=f"https://wandb.ai/{identity[0]}/{identity[1]}/runs/{identity[2]}",
            define_metric=lambda *a, **k: self.axes.append((a, k)),
            log=lambda event, **kwargs: history.append((event, kwargs)),
            log_artifact=lambda artifact, **kwargs: self._log_artifact(artifact, kwargs),
            finish=lambda: None,
        )

    def _log_artifact(self, artifact, kwargs):
        self.artifacts.append((artifact, kwargs))
        return SimpleNamespace(wait=lambda: None)

    def Api(self, **kwargs):
        service = self

        class Api:
            def run(self, path):
                entity, project, run_id = path.split("/")
                identity = (entity, project, run_id)
                if identity not in service.runs:
                    raise ValueError("run does not exist")
                return SimpleNamespace(id=run_id, entity=entity, project=project)

        return Api()


@pytest.mark.parametrize("upload_checkpoints", [None, False, True])
def test_same_identity_overlapping_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upload_checkpoints: bool | None
) -> None:
    """Preserve local recovery and metrics while making checkpoint uploads opt-in."""
    service = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", service)
    original = run_config(tmp_path)
    config = RunConfig.model_validate(
        {
            **original.model_dump(),
            "logging": {
                "out_dir": str(tmp_path / "run"),
                "wandb": {
                    "enabled": True,
                    "entity": "team",
                    "project": "pilot",
                    "group": "seeds",
                    **(
                        {"upload_checkpoints": upload_checkpoints}
                        if upload_checkpoints is not None
                        else {}
                    ),
                },
            },
        }
    )
    trainer = Trainer(config)
    trainer.fit()
    saved_identity = trainer.logger.identity
    resumed = Trainer.resume(tmp_path / "run/step-3.pt")
    resumed.fit()
    assert resumed.logger.identity == saved_identity
    assert service.calls[-1]["resume"] == "must"
    assert service.calls[-1]["id"] == service.calls[0]["id"]
    history = service.runs[("team", "pilot", saved_identity.id)]
    assert all(kwargs == {} for _, kwargs in history), "Never pass a W&B history step"
    training = [event["train/step"] for event, _ in history if event["event"] == "train"]
    assert training == [1, 2, 3, 4, 5, 6, 4, 5, 6]
    segments = {event["resume/segment"] for event, _ in history}
    assert len(segments) == 2
    boundary = [event for event, _ in history if event["event"] == "resume"][0]
    assert boundary["resume/checkpoint_step"] == 3
    assert any(kwargs.get("step_metric") == "train/step" for _, kwargs in service.axes)
    artifact_types = [artifact.values["type"] for artifact, _ in service.artifacts]
    assert artifact_types.count("provenance") == 2
    assert artifact_types.count("checkpoint") == (3 if upload_checkpoints else 0)
    assert (tmp_path / "run/latest.pt").is_file()
    assert resumed.config.logging.wandb.upload_checkpoints is bool(upload_checkpoints)
    for artifact, kwargs in service.artifacts:
        if artifact.values["type"] == "checkpoint":
            step = artifact.values["metadata"]["completed_updates"]
            assert kwargs["aliases"] == ["latest", f"step-{step}"]
    assert all(call["entity"] == "team" for call in service.calls)
    assert all(call["mode"] == "online" for call in service.calls)
    assert all(call["group"] == "seeds" for call in service.calls)
    local = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert local == [event for event, _ in history]


def test_malformed_run_url_fails_startup(tmp_path, monkeypatch):
    service = FakeWandb()
    initialize = service.init

    def malformed_init(**kwargs):
        run = initialize(**kwargs)
        run.url = "https://wandb.ai/wrong/project/runs/wrong"
        return run

    service.init = malformed_init
    monkeypatch.setitem(sys.modules, "wandb", service)
    with pytest.raises(RuntimeError, match="malformed run URL"):
        MetricsLogger(
            LoggingConfig(
                out_dir=str(tmp_path),
                wandb={"enabled": True, "entity": "team", "project": "pilot"},
            ),
            {},
        )


def test_missing_remote_run_fails_strictly(tmp_path, monkeypatch):
    service = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", service)
    with pytest.raises(RuntimeError, match="strict resume"):
        MetricsLogger(
            LoggingConfig(out_dir=str(tmp_path), wandb={"enabled": True}),
            {},
            RunIdentity(id="missing", entity="team", project="pilot"),
            10,
        )
    assert service.calls[0]["resume"] == "must" and not service.runs


def test_logging_disabled_never_initializes_wandb(tmp_path, monkeypatch):
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError("W&B must remain optional")

    monkeypatch.setitem(sys.modules, "wandb", Forbidden())
    logger = MetricsLogger(LoggingConfig(out_dir=str(tmp_path), wandb={"enabled": False}), {})
    logger.log({"train/step": 1, "train/loss": 0.5})
    logger.close()
    assert logger.identity is None


def test_defaults_use_explicit_personal_entity_and_online_project():
    config = WandbConfig()
    assert config.enabled
    assert not config.upload_checkpoints
    assert config.entity == "yixiao_wang"
    assert config.project == "robopolicy"
    assert config.group is None
    with pytest.raises(ValueError):
        WandbConfig(entity="")
    with pytest.raises(ValueError):
        WandbConfig(project="")


def test_evaluation_uses_separate_run_and_uploads_result(tmp_path, monkeypatch):
    service = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", service)
    result_path = tmp_path / "evaluation.json"
    result_path.write_text('{"ok": true}\n')
    training_identity = RunIdentity(id="training", entity="team", project="pilot")
    identity = log_evaluation(
        LoggingConfig(
            out_dir=str(tmp_path),
            wandb={"enabled": True, "entity": "team", "project": "pilot", "group": "seeds"},
        ),
        {"seed": 0},
        {"val/loss": 0.25, "val/samples": 4.0},
        result_path,
        training_identity=training_identity,
        checkpoint_step=6,
    )
    assert identity is not None and identity.id != training_identity.id
    assert service.calls[-1]["job_type"] == "evaluation"
    assert service.calls[-1]["group"] == "seeds"
    artifact, kwargs = service.artifacts[-1]
    assert artifact.values["type"] == "evaluation-result"
    assert artifact.files == [str(result_path)]
    assert kwargs["aliases"] == ["latest", "step-6"]
    history = service.runs[("team", "pilot", identity.id)]
    assert history[-1][0]["val/loss"] == 0.25
