"""Evaluate canonical policy checkpoints and produce reproducible CG reports."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from robopolicy import FeatureSchema, load_policy, read_checkpoint
from robopolicy.training.logging import RunIdentity, log_evaluation

from corl2026_cg.instructions import ActiveInstructionProfile
from corl2026_cg.instructions.contracts import validate_feature_conditioning
from corl2026_cg.tasks import TaskSet, parse_task

from .backend import SimulatorBackend
from .config import EvaluationConfig
from .media import EpisodeVideos
from .rollout import run_batch


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate an explicit population; empty ID/OOD subsets remain absent."""
    if not episodes:
        return {"n_episodes": 0, "n_success": 0, "pc_success": None}
    count = len(episodes)
    return {
        "n_episodes": count,
        "n_success": sum(bool(episode["success"]) for episode in episodes),
        "pc_success": 100 * sum(bool(episode["success"]) for episode in episodes) / count,
        "pc_stage0_success": 100
        * sum(bool(episode["stage0_done"]) for episode in episodes)
        / count,
        "pc_stage1_success": 100
        * sum(bool(episode["stage1_done"]) for episode in episodes)
        / count,
        "pc_same_container_violation": 100
        * sum(bool(episode["same_container_violation"]) for episode in episodes)
        / count,
        "mean_return": sum(float(episode["return"]) for episode in episodes) / count,
        "mean_length": sum(int(episode["length"]) for episode in episodes) / count,
        "failures": dict(
            Counter(episode["failure_reason"] for episode in episodes if episode["failure_reason"])
        ),
    }


def validate_checkpoint_tasks(
    payload: dict[str, Any], task_set: TaskSet, conditioning: str = "static"
) -> ActiveInstructionProfile | None:
    """Reject mismatched instruction encodings and misleading ID/OOD membership."""
    schema = FeatureSchema.model_validate(payload["feature_schema"])
    task_feature = next(
        (feature for feature in schema.features if feature.key == "task_indices"), None
    )
    policy_conditioning = payload["policy_config"]["conditioning"]
    if task_feature is None or policy_conditioning.get("type") != "indexed":
        raise ValueError(
            "CG evaluation requires the checkpoint's explicit indexed task conditioning"
        )
    profile = validate_feature_conditioning(
        schema,
        tuple(policy_conditioning["vocab_sizes"]),
        task_set.family,
        task_set.instruction,
        conditioning,
    )
    expected = set(task_set.train_slugs)
    selection = payload["run_config"]["dataset"]["selection"]
    included, excluded = set(selection["include_tasks"]), set(selection["exclude_tasks"])
    actual = {identity[0] for identity in payload["selection"]}
    if expected & excluded or (included and included - excluded != expected) or actual != expected:
        raise ValueError(
            "Checkpoint training task selection differs from TaskSet; ID/OOD labels would be invalid"
        )
    return profile


def evaluate(
    checkpoint: Path,
    *,
    family: str,
    instruction: str,
    task_set: TaskSet,
    config: EvaluationConfig,
    out_dir: Path,
    conditioning: str = "static",
) -> dict[str, Any]:
    """Run all task trials through the public policy runtime and pinned simulator."""
    checkpoint, out_dir = Path(checkpoint), Path(out_dir)
    if any(
        (out_dir / name).exists()
        for name in ("eval_summary.json", "episodes.jsonl", "eval_config_used.json")
    ):
        raise FileExistsError(f"Evaluation results already exist in {out_dir}")
    if task_set.family != family or task_set.instruction != instruction:
        raise ValueError("Task set family/instruction does not match evaluation settings")
    payload = read_checkpoint(checkpoint)
    profile = validate_checkpoint_tasks(payload, task_set, conditioning)
    schema = FeatureSchema.model_validate(payload["feature_schema"])
    image_feature = next((feature for feature in schema.features if feature.key == "images"), None)
    cameras = (
        image_feature.cameras if image_feature else ("agentview",) if config.max_videos else ()
    )
    size = (image_feature.shape[-2], image_feature.shape[-1]) if image_feature else (256, 256)
    if config.max_attention_videos and image_feature is None:
        raise ValueError("Attention videos require an image policy with declared cameras")
    if config.n_execute > payload["policy_config"]["action_steps"]:
        raise ValueError("n_execute exceeds checkpoint action horizon")
    tasks = [parse_task(slug) for slug in task_set.eval_slugs]
    if not tasks or any(task.family != family for task in tasks):
        raise ValueError("Evaluation tasks must be nonempty and match the requested family")
    # Validate instruction layouts before allocating GPU or simulator resources.
    from .rollout import categorical_features

    for task in tasks:
        categorical_features(task, instruction, schema, profile)
    policy = load_policy(checkpoint, device=config.device, inference_config=config.inference)
    torch.manual_seed(config.seed)
    scheduled = []
    representatives = []
    for task in tasks:
        for trial in range(config.n_trials_per_task):
            concrete = task
            if instruction == "infer_cont0":
                allowed = [value for value in range(4) if value != task.container1]
                concrete = replace(task, container0=allowed[trial % len(allowed)])
            scheduled.append(concrete)
            representatives.append(task.slug)
    seed_sequences = np.random.SeedSequence(config.seed).spawn(len(scheduled))
    seeds = [int(sequence.generate_state(1)[0]) for sequence in seed_sequences]
    out_dir.mkdir(parents=True, exist_ok=True)
    design_metrics = task_set.coverage()
    resolved = {
        "family": family,
        "instruction": instruction,
        "conditioning": conditioning,
        "active_profile": profile.to_dict() if profile else None,
        "active_profile_digest": profile.digest if profile else None,
        "task_set": task_set.to_dict(),
        "design_metrics": design_metrics,
        "evaluation": config.model_dump(mode="json"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": payload["completed_updates"],
    }
    (out_dir / "eval_config_used.json").write_text(json.dumps(resolved, indent=2) + "\n")
    backend = SimulatorBackend(family, config, cameras, size)
    episodes: list[dict[str, Any]] = []
    try:
        for offset in range(0, len(scheduled), config.num_envs):
            batch_tasks = scheduled[offset : offset + config.num_envs]
            batch_seeds = seeds[offset : offset + config.num_envs]
            valid_count = len(batch_tasks)
            while len(batch_tasks) < config.num_envs:
                batch_tasks.append(batch_tasks[-1])
                batch_seeds.append(batch_seeds[-1])
            videos = EpisodeVideos(
                out_dir,
                policy,
                offset=offset,
                max_videos=config.max_videos,
                max_attention_videos=config.max_attention_videos,
                fps=config.fps,
                profile=profile,
            )
            try:
                records = run_batch(
                    backend,
                    policy,
                    schema,
                    batch_tasks,
                    batch_seeds,
                    instruction=instruction,
                    config=config,
                    valid_count=valid_count,
                    profile=profile,
                    on_frame=videos if config.max_videos or config.max_attention_videos else None,
                )
            finally:
                videos.close()
            for record in records:
                record["evaluation_task"] = representatives[len(episodes)]
                record["episode"] = len(episodes)
                record["split"] = "id" if task_set.is_id(record["task"]) else "ood"
                episodes.append(record)
            with (out_dir / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(f"[evaluate] {len(episodes)}/{len(scheduled)} episodes", flush=True)
    finally:
        backend.close()
    summary = {
        "schema_version": 1,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": payload["completed_updates"],
        "simulator": backend.provenance,
        "design_metrics": design_metrics,
        "aggregated": aggregate(episodes),
        "splits": {
            split: aggregate([record for record in episodes if record["split"] == split])
            for split in ("id", "ood")
        },
        "per_task": {
            task.slug: aggregate(
                [record for record in episodes if record["evaluation_task"] == task.slug]
            )
            for task in tasks
        },
    }
    output = out_dir / "eval_summary.json"
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    metrics = {
        f"eval/{key}": float(value)
        for key, value in summary["aggregated"].items()
        if isinstance(value, (int, float))
    }
    for split, values in summary["splits"].items():
        if values["pc_success"] is not None:
            metrics[f"eval/{split}/pc_success"] = values["pc_success"]
    logging = config.logging.model_copy(update={"out_dir": str(out_dir)})
    log_evaluation(
        logging,
        resolved,
        metrics,
        output,
        training_identity=RunIdentity.model_validate(payload["wandb"])
        if payload["wandb"]
        else None,
        checkpoint_step=payload["completed_updates"],
    )
    return summary
