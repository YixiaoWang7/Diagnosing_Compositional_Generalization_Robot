"""Structured local metrics and verified online W&B tracking."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from robopolicy.contracts.schema import StrictConfig


class WandbConfig(StrictConfig):
    """Online W&B settings for a stable checked-in project."""

    enabled: bool = True
    upload_checkpoints: bool = Field(
        default=False, description="Upload saved training checkpoints when W&B is enabled."
    )
    entity: str = Field(default="yixiao_wang", min_length=1)
    project: str = Field(default="robopolicy", min_length=1)
    name: str | None = None
    group: str | None = None


class LoggingConfig(StrictConfig):
    """Run output directory and online tracking configuration."""

    out_dir: str = "runs/example"
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class RunIdentity(StrictConfig):
    """The server-resolved W&B identity persisted in metadata and checkpoints."""

    id: str
    entity: str
    project: str

    @property
    def url(self) -> str:
        """Return the canonical browser URL for this run."""
        return f"https://wandb.ai/{self.entity}/{self.project}/runs/{self.id}"


def _verify_remote_run(wandb_module: Any, identity: RunIdentity) -> None:
    """Require the authenticated API to resolve the exact initialized run."""
    remote: Any = None
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            remote = wandb_module.Api(timeout=20).run(
                f"{identity.entity}/{identity.project}/{identity.id}"
            )
            break
        except Exception as error:  # pragma: no cover - retries depend on the service
            last_error = error
            if attempt < 2:
                time.sleep(1)
    if remote is None:
        raise RuntimeError(f"W&B API could not verify {identity.url}: {last_error}")
    remote_identity = RunIdentity(
        id=str(remote.id), entity=str(remote.entity), project=str(remote.project)
    )
    if remote_identity != identity:
        raise RuntimeError(f"W&B API returned {remote_identity.url} while verifying {identity.url}")


def _start_online_run(
    config: WandbConfig,
    out_dir: Path,
    resolved_config: dict[str, Any],
    *,
    identity: RunIdentity | None = None,
    job_type: str,
    name: str | None = None,
) -> tuple[Any, Any, RunIdentity]:
    """Initialize one explicit online run and verify its URL through the API."""
    import wandb

    run_id = identity.id if identity is not None else uuid.uuid4().hex[:8]
    kwargs: dict[str, Any] = {
        "mode": "online",
        "dir": str(out_dir),
        "config": resolved_config,
        "id": run_id,
        "entity": identity.entity if identity is not None else config.entity,
        "project": identity.project if identity is not None else config.project,
        "name": name if name is not None else config.name,
        "job_type": job_type,
        "resume": "must" if identity is not None else "never",
        # W&B 0.24.2 can fail while rendering an account banner whose server
        # flags are null. Silent settings skip that banner without disabling uploads.
        "settings": wandb.Settings(silent=True),
    }
    if config.group is not None:
        kwargs["group"] = config.group
    try:
        run = wandb.init(**kwargs)
    except Exception as error:
        operation = "strict resume" if identity is not None else "initialization"
        raise RuntimeError(f"W&B {operation} failed: {error}") from error
    if run is None:
        raise RuntimeError("W&B initialization returned no run")
    try:
        resolved_identity = RunIdentity(
            id=str(run.id), entity=str(run.entity), project=str(run.project)
        )
        if identity is not None and not bool(run.resumed):
            raise RuntimeError("W&B did not resume the required existing run")
        if identity is not None and resolved_identity != identity:
            raise RuntimeError("W&B returned a different run identity on strict resume")
        if str(run.url).rstrip("/") != resolved_identity.url:
            raise RuntimeError(
                f"W&B returned malformed run URL {run.url!r}; expected {resolved_identity.url}"
            )
        _verify_remote_run(wandb, resolved_identity)
    except BaseException:
        run.finish()
        raise
    print(f"W&B run: {resolved_identity.url}", flush=True)
    return wandb, run, resolved_identity


def _upload_file(
    wandb_module: Any,
    run: Any,
    path: Path,
    *,
    name: str,
    artifact_type: str,
    aliases: Sequence[str],
    metadata: dict[str, Any],
) -> None:
    """Upload one file and wait until W&B confirms the artifact."""
    artifact = wandb_module.Artifact(name=name, type=artifact_type, metadata=metadata)
    artifact.add_file(str(path))
    uploaded = run.log_artifact(artifact, aliases=list(aliases))
    uploaded.wait()


class MetricsLogger:
    """Append local and W&B events without setting W&B's internal history step."""

    def __init__(
        self,
        config: LoggingConfig,
        resolved_config: dict[str, Any],
        identity: RunIdentity | None = None,
        checkpoint_step: int | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        """Start a fresh run or strictly resume its saved identity.

        Recovery appends a distinct segment, allowing overlapping train/step
        values when server history is newer than the recovered checkpoint.
        """
        self.config = config
        self.segment = uuid.uuid4().hex
        self.identity = identity
        self.run: Any = None
        self._wandb: Any = None
        self._uploaded_checkpoint_steps: set[int] = set()
        self.path = Path(config.out_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self._metrics = (self.path / "metrics.jsonl").open("a", encoding="utf-8")
        try:
            if config.wandb.enabled:
                if checkpoint_step is not None and identity is None:
                    raise ValueError("W&B recovery requires the saved run identity")
                self._wandb, self.run, self.identity = _start_online_run(
                    config.wandb,
                    self.path,
                    resolved_config,
                    identity=identity,
                    job_type="training",
                )
                self.run.define_metric("train/step")
                self.run.define_metric("train/*", step_metric="train/step")
                self.run.define_metric("val/*", step_metric="train/step")
            elif identity is not None:
                raise ValueError("Cannot disable W&B while resuming a W&B training state")
            metadata = {
                "wandb": self.identity.model_dump() if self.identity else None,
                "segment": self.segment,
                "checkpoint_step": checkpoint_step,
                "config": resolved_config,
                "provenance": provenance,
            }
            segment_path = self.path / f"segment-{self.segment}.json"
            segment_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            if self.run is not None:
                assert self.identity is not None
                _upload_file(
                    self._wandb,
                    self.run,
                    segment_path,
                    name=f"robopolicy-provenance-{self.identity.id}",
                    artifact_type="provenance",
                    aliases=("latest", f"segment-{self.segment}"),
                    metadata={"segment": self.segment, "checkpoint_step": checkpoint_step},
                )
            self.log(
                {
                    "event": "resume" if checkpoint_step is not None else "start",
                    "train/step": checkpoint_step or 0,
                    "resume/checkpoint_step": checkpoint_step,
                }
            )
        except BaseException:
            self.close()
            raise

    def log(self, metrics: dict[str, Any]) -> None:
        """Append an event to local JSONL and optionally the same W&B run."""
        event = {**metrics, "resume/segment": self.segment}
        self._metrics.write(json.dumps(event, allow_nan=False) + "\n")
        self._metrics.flush()
        if self.run is not None:
            self.run.log(event)

    def log_checkpoint(self, path: str | Path, step: int) -> None:
        """Upload opted-in checkpoints once per completed update in this segment."""
        if (
            not self.config.wandb.upload_checkpoints
            or self.run is None
            or step in self._uploaded_checkpoint_steps
        ):
            return
        checkpoint = Path(path)
        assert self.identity is not None
        _upload_file(
            self._wandb,
            self.run,
            checkpoint,
            name=f"robopolicy-checkpoint-{self.identity.id}",
            artifact_type="checkpoint",
            aliases=("latest", f"step-{step}"),
            metadata={"completed_updates": step, "schema_version": 1},
        )
        self._uploaded_checkpoint_steps.add(step)

    def close(self) -> None:
        """Flush local output and finish the optional remote segment once."""
        if self.run is not None:
            run, self.run = self.run, None
            run.finish()
        if not self._metrics.closed:
            self._metrics.close()


def log_evaluation(
    config: LoggingConfig,
    resolved_config: dict[str, Any],
    metrics: dict[str, float],
    result_path: str | Path,
    *,
    training_identity: RunIdentity | None,
    checkpoint_step: int,
) -> RunIdentity | None:
    """Record aggregate evaluation metrics in a separate run and result artifact."""
    output = Path(result_path)
    if not output.is_file():
        raise FileNotFoundError(f"Evaluation result does not exist: {output}")
    event: dict[str, Any] = {
        "event": "evaluation",
        "evaluation/checkpoint_step": checkpoint_step,
        **metrics,
    }
    local_path = Path(config.out_dir) / "evaluation-metrics.jsonl"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
    if not config.wandb.enabled:
        return None
    evaluation_config = {
        "run": resolved_config,
        "source_training_run": training_identity.model_dump() if training_identity else None,
        "checkpoint_step": checkpoint_step,
    }
    name = f"{config.wandb.name}-evaluation" if config.wandb.name else "evaluation"
    wandb_module, run, identity = _start_online_run(
        config.wandb,
        Path(config.out_dir),
        evaluation_config,
        job_type="evaluation",
        name=name,
    )
    try:
        run.log(event)
        _upload_file(
            wandb_module,
            run,
            output,
            name=f"robopolicy-evaluation-{identity.id}",
            artifact_type="evaluation-result",
            aliases=("latest", f"step-{checkpoint_step}"),
            metadata={
                "checkpoint_step": checkpoint_step,
                "source_training_run": training_identity.id if training_identity else None,
            },
        )
    finally:
        run.finish()
    return identity
