"""Offline objective and sampled-action error on trained and unseen CG tasks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from robopolicy import (
    InferenceConfig,
    RunConfig,
    build_dataset,
    build_loader,
    load_policy,
    read_checkpoint,
)
from robopolicy.datasets.config import DatasetConfig, LoaderConfig
from robopolicy.datasets.index import read_manifest

from corl2026_cg.instructions import validate_dataset_profile
from corl2026_cg.instructions.contracts import policy_profile, validate_policy_conditioning
from corl2026_cg.tasks import (
    build_task_set,
    instruction_indices,
    parse_task,
)


def evaluate_losses(
    checkpoint: Path,
    *,
    family: str,
    instruction: str = "full",
    conditioning: str | None = None,
    dataset_root: str | None = None,
    samples: int = 500,
    batch_size: int = 50,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate a finite, seeded sample with checkpoint normalization.

    Both splits use the dataset's training membership. They compare seen and unseen
    task identities, not held-out validation demonstrations. Empty unseen sets are
    represented as not applicable, and short datasets report actual sample counts.
    """
    if samples < 1 or batch_size < 1:
        raise ValueError("samples and batch_size must be positive")
    payload = read_checkpoint(checkpoint)
    config = RunConfig.model_validate(payload["run_config"])
    policy_conditioning = config.policy.conditioning
    if policy_conditioning.type != "indexed":
        raise ValueError("CG task loss evaluation requires indexed checkpoint conditioning")
    paths = config.features["task_indices"].paths
    mode = conditioning or ("active_v1" if policy_profile(config) else "static")
    profile = validate_policy_conditioning(config, family, instruction, mode)
    trained = {str(identity[0]) for identity in payload["selection"]}
    configured = set(config.dataset.selection.include_tasks) | set(
        config.dataset.selection.exclude_tasks
    )
    for slug in trained | configured:
        if parse_task(slug).family != family:
            raise ValueError(f"Checkpoint task {slug!r} does not match family {family!r}")
    dataset_values = config.dataset.model_dump(mode="json")
    if dataset_root is not None:
        dataset_values["root"] = dataset_root
    dataset_values["sampling"]["seed"] = seed
    universe = set(build_task_set(family, "full", instruction).train_slugs)
    visible = {instruction_indices(parse_task(slug), instruction) for slug in trained}
    unseen = {
        slug
        for slug in universe
        if instruction_indices(parse_task(slug), instruction) not in visible
    }
    available = {
        entry.task_id for entry in read_manifest(Path(dataset_values["root"]).expanduser())
    }
    if trained - available:
        raise ValueError(f"Dataset is missing trained tasks: {sorted(trained - available)}")
    inference = load_policy(
        checkpoint, device=device, inference_config=InferenceConfig(compile="disabled")
    )
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint.resolve()),
        "algorithm": config.policy.algorithm.type,
        "checkpoint_step": payload["completed_updates"],
        "training_seed": config.seed,
        "family": family,
        "instruction": instruction,
        "conditioning": mode,
        "active_profile": profile.to_dict() if profile else None,
        "active_profile_digest": profile.digest if profile else None,
        "task_indices_paths": list(paths),
        "vocab_sizes": list(policy_conditioning.vocab_sizes),
        "training_task_slugs": sorted(trained),
        "metric_seed": seed,
        "sampling_seed": seed,
        "dataset_root": str(Path(dataset_values["root"]).expanduser().resolve()),
        "dataset_fingerprints": {},
        "requested_samples": samples,
        "batch_size": batch_size,
        "membership": "training demonstrations, partitioned by visible task identity",
        "task_universe": "full family, independently of the rollout evaluation set",
        "splits": {},
    }
    for name, tasks in (("trained_tasks", trained), ("unseen_tasks", unseen)):
        if not tasks:
            result["splits"][name] = {"status": "not_applicable", "num_samples": 0}
            continue
        missing = sorted(tasks - available)
        present = tasks & available
        if not present:
            result["splits"][name] = {
                "status": "unavailable",
                "num_samples": 0,
                "missing_task_slugs": missing,
            }
            continue
        values = {**dataset_values, "selection": dict(dataset_values["selection"])}
        values["selection"].update(include_tasks=sorted(present), exclude_tasks=[])
        if name == "unseen_tasks":
            values["selection"].update(
                max_demos=None, max_demos_per_task=None, balance_demos_across_tasks=False
            )
        dataset_config = DatasetConfig.model_validate(values)
        if profile:
            validate_dataset_profile(config.model_copy(update={"dataset": dataset_config}), profile)
        dataset = build_dataset(dataset_config, config.features, "train")
        result["dataset_fingerprints"][name] = dataset.fingerprint
        total = 0
        objective = normalized_error = raw_error = 0.0
        started = time.perf_counter()
        try:
            with torch.random.fork_rng(
                devices=[torch.device(device).index or 0]
                if torch.device(device).type == "cuda"
                else []
            ):
                torch.manual_seed(seed)
                with build_loader(dataset, LoaderConfig(batch_size=batch_size)) as loader:
                    with torch.no_grad():
                        for batch in loader:
                            remaining = samples - total
                            if remaining <= 0:
                                break
                            batch = {
                                key: value[:remaining] if isinstance(value, torch.Tensor) else value
                                for key, value in batch.items()
                            }
                            inputs = inference.packer(batch, device=device, require_actions=True)
                            assert inputs.actions is not None
                            count = int(inputs.state.shape[0])
                            objective += float(inference.policy.loss(inputs).item()) * count
                            prediction = inference.policy.predict(inputs)
                            normalized_error += (
                                float((prediction - inputs.actions).square().mean().item()) * count
                            )
                            raw_prediction = inference.packer.actions_to_raw(prediction)
                            target = batch["actions"].to(device)
                            raw_error += (
                                float((raw_prediction - target).square().mean().item()) * count
                            )
                            total += count
        finally:
            dataset.close()
        if not total:
            raise ValueError(f"No eligible samples for {name}")
        result["splits"][name] = {
            "status": "complete",
            "num_samples": total,
            "task_slugs": sorted(present),
            "requested_task_count": len(tasks),
            "available_task_count": len(present),
            "missing_task_slugs": missing,
            "objective_loss": objective / total,
            "action_mse_normalized": normalized_error / total,
            "action_mse_raw": raw_error / total,
            "elapsed_seconds": time.perf_counter() - started,
        }
    return result
