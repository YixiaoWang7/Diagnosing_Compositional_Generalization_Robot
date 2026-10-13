"""Application settings compose native RoboPolicy configuration without duplicating it."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from robopolicy import RunConfig
from typing_extensions import Self

from corl2026_cg.evaluation import EvaluationConfig
from corl2026_cg.instructions import resolve_profile
from corl2026_cg.tasks import build_task_set, instruction_fields, vocab_sizes


class ExecutionConfig(BaseModel):
    """Interpreter and GPU assignments for independent training/evaluation processes."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    train_python: str = sys.executable
    eval_python: str = sys.executable
    train_gpus: str | None = None
    eval_gpus: str | None = None
    overlap: bool = False

    @model_validator(mode="after")
    def validate_devices(self) -> Self:
        """Require explicit disjoint GPU assignments when stages overlap."""
        if self.overlap:
            if not self.train_gpus or not self.eval_gpus:
                raise ValueError("Overlapping stages require explicit train_gpus and eval_gpus")
            train = {value.strip() for value in self.train_gpus.split(",")}
            evaluation = {value.strip() for value in self.eval_gpus.split(",")}
            if "" in train | evaluation or train & evaluation:
                raise ValueError("Overlapping stages require disjoint nonempty GPU assignments")
        return self


class ExperimentSpec(BaseModel):
    """CG task semantics and execution settings around a native training template."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1] = 1
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    family: Literal["l4", "l4_2s", "l5"]
    instruction: str = "full"
    conditioning: Literal["static", "active_v1"] = "static"
    sets: tuple[str, ...] = ("full",)
    eval_set: str | None = None
    seeds: tuple[int, ...] = (11, 22, 33)
    run: RunConfig
    matrix: dict[str, list[Any]] = Field(default_factory=dict)
    output_root: str = "runs/corl2026_cg"
    finetune_from: str | None = None
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        """Reject ambiguous jobs and unsupported instruction layouts before writing files."""
        if not self.sets or not self.seeds:
            raise ValueError("sets and seeds must be nonempty")
        if len(set(self.sets)) != len(self.sets) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Duplicate task sets or training seeds")
        if any(not values for values in self.matrix.values()):
            raise ValueError("Matrix axes must be nonempty")
        vocab_sizes(self.family, self.instruction)
        resolve_profile(self.family, self.instruction, self.conditioning)
        return self


def set_nested(values: dict[str, Any], dotted: str, value: Any) -> None:
    """Apply an explicit existing dotted path, including numeric list indices."""
    parts = dotted.split(".")
    if any(not part for part in parts):
        raise ValueError(f"Unknown override path: {dotted}")
    current: Any = values
    try:
        for part in parts[:-1]:
            if isinstance(current, list) and not part.isdecimal():
                raise ValueError(part)
            current = current[int(part)] if isinstance(current, list) else current[part]
        if isinstance(current, list) and not parts[-1].isdecimal():
            raise ValueError(parts[-1])
        final: Any = int(parts[-1]) if isinstance(current, list) else parts[-1]
        if isinstance(current, dict) and final not in current:
            raise KeyError(final)
        current[final] = value
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"Unknown override path: {dotted}") from error


def load_experiment(path: Path, overrides: list[str] | None = None) -> ExperimentSpec:
    """Load one strict experiment JSON; all relative paths are invocation-relative."""
    values = ExperimentSpec.model_validate_json(path.read_text()).model_dump(mode="json")
    for override in overrides or []:
        key, separator, raw = override.partition("=")
        if not separator:
            raise ValueError(f"Expected PATH=VALUE, received {override!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        set_nested(values, key, value)
    return ExperimentSpec.model_validate(values)


def resolve_jobs(spec: ExperimentSpec) -> list[dict[str, Any]]:
    """Expand a finite matrix and return explicit, collision-resistant job manifests."""
    axes = sorted(spec.matrix)
    variants = itertools.product(*(spec.matrix[axis] for axis in axes))
    jobs = []
    identities: set[str] = set()
    for variant in variants:
        for set_name in spec.sets:
            task_set = build_task_set(
                spec.family, set_name, spec.instruction, eval_set=spec.eval_set
            )
            profile = resolve_profile(spec.family, spec.instruction, spec.conditioning)
            base = spec.run.model_dump(mode="json")
            for axis, value in zip(axes, variant):
                set_nested(base, axis, value)
            base["dataset"]["selection"]["include_tasks"] = list(task_set.train_slugs)
            base["policy"]["conditioning"] = {
                "type": "indexed",
                "vocab_sizes": [12]
                if profile
                else list(vocab_sizes(spec.family, spec.instruction)),
            }
            fields = instruction_fields(spec.family, spec.instruction)
            features = [f for f in base["features"]["features"] if f["key"] != "task_indices"]
            features.append(
                {
                    "key": "task_indices",
                    "shape": [1 if profile else len(fields)],
                    "paths": [profile.path]
                    if profile
                    else [f"cg/task/{field}" for field in fields],
                    "dtype": "int64",
                }
            )
            base["features"]["features"] = features
            # Only training seed and run destinations are excluded; design/data seeds remain.
            identity_config = copy.deepcopy(base)
            identity_config.pop("seed", None)
            identity_config["logging"].pop("out_dir", None)
            identity_config["logging"]["wandb"].pop("name", None)
            identity_config["logging"]["wandb"].pop("group", None)
            identity = {
                "family": spec.family,
                "instruction": spec.instruction,
                "set": set_name,
                "run": identity_config,
                "finetune": spec.finetune_from,
            }
            if profile is not None:
                identity["active_profile"] = profile.to_dict()
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", set_name)
            group = f"{spec.name}_{slug}_{digest}"
            for seed in spec.seeds:
                name = f"{group}_seed{seed}"
                if name in identities:
                    raise ValueError(f"Duplicate resolved experiment: {name}")
                identities.add(name)
                directory = (Path(spec.output_root) / name).resolve()
                config = copy.deepcopy(base)
                config["seed"] = seed
                config["logging"]["out_dir"] = str(directory / "train")
                config["logging"]["wandb"].update(name=name, group=group)
                config = RunConfig.model_validate(config).model_dump(mode="json")
                jobs.append(
                    {
                        "name": name,
                        "group": group,
                        "seed": seed,
                        "family": spec.family,
                        "instruction": spec.instruction,
                        "conditioning": spec.conditioning,
                        "active_profile": profile.to_dict() if profile else None,
                        "active_profile_digest": profile.digest if profile else None,
                        "set_name": set_name,
                        "eval_set": spec.eval_set,
                        "task_set": task_set.to_dict(),
                        "design_metrics": task_set.coverage(),
                        "run": config,
                        "finetune_from": spec.finetune_from,
                        "checkpoint": str(directory / "train/latest.pt"),
                        "directory": str(directory),
                        "evaluation": spec.evaluation.model_dump(mode="json"),
                        "execution": spec.execution.model_dump(mode="json"),
                    }
                )
    return jobs


def write_jobs(jobs: list[dict[str, Any]], *, overwrite: bool = False) -> list[Path]:
    """Save resolved inputs, refusing to overwrite an existing experiment by default."""
    paths: list[Path] = []
    directories: set[Path] = set()
    names: set[str] = set()
    # Check the complete batch first so a later collision cannot publish earlier jobs.
    for job in jobs:
        directory = Path(job["directory"]).resolve()
        if directory in directories or job["name"] in names:
            raise ValueError(f"Duplicate output job: {job['name']} at {directory}")
        directories.add(directory)
        names.add(job["name"])
        path = directory / "experiment.json"
        if directory.exists() and (
            not directory.is_dir() or (not overwrite and any(directory.iterdir()))
        ):
            raise FileExistsError(
                f"Experiment output exists: {directory}; use --overwrite explicitly"
            )
        paths.append(path)
    for job, path in zip(jobs, paths):
        directory = path.parent
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(job, indent=2) + "\n")
        (directory / "run.json").write_text(json.dumps(job["run"], indent=2) + "\n")
    return paths
