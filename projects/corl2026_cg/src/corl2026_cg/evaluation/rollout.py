"""Batched action-chunk execution with consecutive observation histories."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from robopolicy import FeatureSchema

from corl2026_cg.instructions import ActiveInstructionProfile, InstructionTracker
from corl2026_cg.tasks import CGTask, instruction_indices, validate_conditioning

from .config import EvaluationConfig
from .success import EpisodeProgress


def categorical_features(
    task: CGTask,
    instruction: str,
    schema: FeatureSchema,
    profile: ActiveInstructionProfile | None = None,
) -> dict[str, np.ndarray]:
    """Match explicitly materialized training task/entity index fields."""
    result = {}
    for feature in schema.features:
        if feature.key == "world_entities":
            raise ValueError(
                "world_entities is reserved for evaluation trackers, not policy inputs"
            )
        if feature.key == "task_indices":
            if profile is None:
                validate_conditioning(task.family, instruction, feature.paths)
                result[feature.key] = np.asarray(
                    instruction_indices(task, instruction), dtype=np.int64
                )
            else:
                profile.validate_conditioning(feature.paths)
                if feature.shape != (1,) or feature.window_size != 1 or feature.dtype != "int64":
                    raise ValueError("Active instructions require one current-step int64 token")
                # The tracker replaces this placeholder before the first model call.
                result[feature.key] = np.zeros(1, dtype=np.int64)
        elif feature.key == "entity_indices":
            indices = []
            for path in feature.paths:
                field = path.rsplit("/", 1)[-1]
                value = getattr(task, field, None)
                if value is None:
                    raise ValueError(
                        f"Cannot derive entity selection from {path!r} for {task.slug}"
                    )
                offset = 8 if field == "button" else 4 if field.startswith("container") else 0
                indices.append(int(value) + offset)
            result[feature.key] = np.asarray(indices, dtype=np.int64)
    for feature in schema.features:
        if feature.key in result and tuple(result[feature.key].shape) != feature.shape:
            raise ValueError(f"Instruction shape does not match checkpoint feature {feature.key}")
    return result


class ObservationHistory:
    """Keep every environment step, padding initial history with its first frame."""

    def __init__(
        self,
        schema: FeatureSchema,
        observations: list[dict[str, np.ndarray]],
        tasks: list[CGTask],
        instruction: str,
        profile: ActiveInstructionProfile | None = None,
    ) -> None:
        self.schema = schema
        self.profile = profile
        self.categories = [
            categorical_features(task, instruction, schema, profile) for task in tasks
        ]
        self.buffers: list[dict[str, deque[np.ndarray]]] = []
        for index, observation in enumerate(observations):
            if profile is not None:
                self.categories[index]["task_indices"] = observation["task_indices"].copy()
            buffers = {}
            for feature in schema.features:
                if feature.key in {"actions", "task_indices", "entity_indices"}:
                    continue
                if feature.future:
                    raise ValueError(f"Rollout observations must be causal: {feature.key}")
                value = observation[feature.key]
                buffers[feature.key] = deque(
                    (value.copy() for _ in range(feature.window_size)), maxlen=feature.window_size
                )
            self.buffers.append(buffers)

    def append(self, observations: list[dict[str, np.ndarray]], active: list[bool]) -> None:
        """Append each post-action observation once, including intermediate MPC steps."""
        for index, observation in enumerate(observations):
            if active[index]:
                if self.profile is not None:
                    self.categories[index]["task_indices"] = observation["task_indices"].copy()
                for key, history in self.buffers[index].items():
                    history.append(observation[key].copy())

    def batch(self) -> dict[str, torch.Tensor]:
        """Return fixed-size raw tensors in the checkpoint camera/feature order."""
        result = {}
        for feature in self.schema.features:
            if feature.key == "actions":
                continue
            values = [
                categories[feature.key][None]
                if feature.key in categories
                else np.stack(buffers[feature.key])
                for buffers, categories in zip(self.buffers, self.categories, strict=True)
            ]
            result[feature.key] = torch.from_numpy(np.stack(values))
        return result


def run_batch(
    backend: Any,
    policy: Any,
    schema: FeatureSchema,
    tasks: list[CGTask],
    reset_seeds: list[int],
    *,
    instruction: str,
    config: EvaluationConfig,
    valid_count: int | None = None,
    profile: ActiveInstructionProfile | None = None,
    on_frame: Callable[[int, dict[str, np.ndarray], dict[str, torch.Tensor] | None], None]
    | None = None,
) -> list[dict[str, Any]]:
    """Run a fixed batch, preserving a compiled policy's input signature.

    Completed/padded instances remain in model batches but receive no actions.
    Every active transition updates observation history, whether or not inference
    occurs. Stage or active-instruction changes discard only that environment's
    old action chunk and replan next step. Padded rows preserve the model shape.
    """
    count = len(tasks)
    valid_count = count if valid_count is None else valid_count
    observations = backend.reset(tasks, reset_seeds)
    trackers = [InstructionTracker(profile, task) for task in tasks] if profile else []
    instruction_ids = [
        tracker.update(observation["state"], observation["world_entities"])
        for tracker, observation in zip(trackers, observations)
    ]
    for observation, instruction_id in zip(observations, instruction_ids):
        observation["task_indices"] = np.asarray([instruction_id], dtype=np.int64)
    history = ObservationHistory(schema, observations, tasks, instruction, profile)
    progress = [EpisodeProgress() for _ in tasks]
    active = [index < valid_count for index in range(count)]
    queues: list[deque[np.ndarray]] = [deque() for _ in tasks]
    records: list[dict[str, Any]] = [
        {
            "task": task.slug,
            "reset_seed": int(getattr(backend, "reset_seeds", reset_seeds)[index]),
            "length": 0,
            "return": 0.0,
            "actions": [],
            "rewards": [],
        }
        for index, task in enumerate(tasks)
    ]
    if profile is not None:
        for record, instruction_id in zip(records, instruction_ids, strict=True):
            # One ID per observation: reset, then every executed action's successor.
            record["active_instruction_ids"] = [instruction_id]
            record["instruction_transitions"] = []
    for _ in range(config.horizon):
        if not any(active):
            break
        need_query = [index for index in range(count) if active[index] and not queues[index]]
        raw_batch = None
        if need_query:
            raw_batch = history.batch()
            predicted = policy.predict(raw_batch).detach().cpu().numpy()
            if predicted.ndim != 3 or predicted.shape[0] != count or predicted.shape[2] != 7:
                raise ValueError(f"CG requires predicted actions (B,T,7), got {predicted.shape}")
            if config.n_execute > predicted.shape[1]:
                raise ValueError("n_execute exceeds the checkpoint action horizon")
            if not np.isfinite(predicted).all():
                raise ValueError("Policy generated nonfinite actions")
            for index in need_query:
                queues[index].extend(predicted[index, : config.n_execute].copy())
        if on_frame is not None:
            for index in range(valid_count):
                if active[index]:
                    on_frame(index, observations[index], raw_batch)
        actions: NDArray[np.float32] = np.zeros((count, 7), dtype=np.float32)
        for index in range(count):
            if active[index]:
                actions[index] = queues[index].popleft()
        was_active = active.copy()
        observations, rewards, done = backend.step(actions, active)
        if profile is not None:
            for index, tracker in enumerate(trackers):
                if active[index]:
                    instruction_id = tracker.update(
                        observations[index]["state"], observations[index]["world_entities"]
                    )
                    records[index]["active_instruction_ids"].append(instruction_id)
                    if instruction_id != instruction_ids[index]:
                        records[index]["instruction_transitions"].append(
                            {
                                "step": records[index]["length"] + 1,
                                "from": instruction_ids[index],
                                "to": instruction_id,
                            }
                        )
                        instruction_ids[index] = instruction_id
                        queues[index].clear()
                observations[index]["task_indices"] = np.asarray(
                    [instruction_ids[index]], dtype=np.int64
                )
        history.append(observations, was_active)
        for index in range(valid_count):
            if not active[index]:
                continue
            record = records[index]
            record["length"] += 1
            record["return"] += float(rewards[index])
            record["actions"].append(actions[index].tolist())
            record["rewards"].append(float(rewards[index]))
            if progress[index].update(backend, index, tasks[index], instruction):
                queues[index].clear()
            if progress[index].success or done[index]:
                active[index] = False
                record["termination"] = "success" if progress[index].success else "environment"
    for index, record in enumerate(records[:valid_count]):
        record.update(asdict(progress[index]))
        record.setdefault("termination", "horizon")
        record["failure_reason"] = (
            None
            if record["success"]
            else "same_container"
            if record["same_container_violation"]
            else "stage1_incomplete"
            if record["stage0_done"]
            else "stage0_incomplete"
        )
    return records[:valid_count]
