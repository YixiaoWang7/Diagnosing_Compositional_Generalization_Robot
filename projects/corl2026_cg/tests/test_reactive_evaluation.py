"""Reactive conditioning follows every physical transition without goal leakage."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from robopolicy import Feature, FeatureSchema, RunConfig, build_policy
from robopolicy.checkpoints.serialization import capture_rng, save_checkpoint
from robopolicy.processing.normalization import Statistics
from robopolicy.training.logging import LoggingConfig, WandbConfig

from corl2026_cg.evaluation import runner
from corl2026_cg.evaluation.backend import SimulatorBackend
from corl2026_cg.evaluation.config import EvaluationConfig
from corl2026_cg.evaluation.media import EpisodeVideos
from corl2026_cg.evaluation.rollout import run_batch
from corl2026_cg.instructions import get_profile, label_trajectory
from corl2026_cg.tasks import CONTAINERS, OBJECTS, CGTask, TaskSet


def active_schema() -> FeatureSchema:
    """Declare one active token alongside a consecutive physical-state history."""
    return FeatureSchema(
        features=(
            Feature(key="state", shape=(9,), paths=("state",), history=2),
            Feature(
                key="task_indices",
                shape=(1,),
                paths=(get_profile("l5_v1").path,),
                dtype="int64",
            ),
        )
    )


def observation(grasped: bool = False) -> dict[str, np.ndarray]:
    """Build an unplaced object with optional physical end-effector contact."""
    state = np.array([0, 0, 1, 0, 0, 0, 1, 0.02, -0.02], dtype=np.float32)
    entities = np.zeros((12, 7), dtype=np.float32)
    entities[:, 0] = np.arange(12) + 1
    entities[:, 2], entities[:, 6] = 0.9, 1
    entities[0, :3] = state[:3] if grasped else [0.3, 0, 0.9]
    return {"state": state, "world_entities": entities}


class PhysicalSequence:
    """Expose controlled physical changes while recording all executed actions."""

    def __init__(self, *, regress: bool = False, latch_success_stage: bool = False) -> None:
        self.regress, self.latch_success_stage = regress, latch_success_stage
        self.actions: list[np.ndarray] = []
        self.observations: list[list[dict[str, np.ndarray]]] = []
        self.step_number = 0
        self.provenance = {"revision": "physical-sequence-test"}

    def reset(self, tasks: list[CGTask], seeds: list[int]) -> list[dict[str, np.ndarray]]:
        """Reset each instance with no previous instruction-tracker state."""
        self.step_number = 0
        self.actions.clear()
        self.observations = [[observation() for _ in tasks]]
        return deepcopy(self.observations[-1])

    def step(
        self, actions: np.ndarray, active: list[bool]
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray, list[bool]]:
        """Change only the first instance and leave all other instances ungrasped."""
        self.step_number += 1
        self.actions.append(actions.copy())
        grasped = not self.regress or self.step_number <= 2
        self.observations.append(
            [observation(grasped=grasped and index == 0) for index in range(len(active))]
        )
        return deepcopy(self.observations[-1]), np.zeros(len(active)), [False] * len(active)

    def native_success(self, index: int) -> bool:
        """Keep episodes active so action queues can be inspected."""
        return False

    def place_stage_success(self, index: int) -> bool:
        """Supply an independently latched task-success state when requested."""
        return self.latch_success_stage and self.step_number >= 1

    def close(self) -> None:
        """No resources are held by this deterministic physical sequence."""


class ChunkPolicy:
    """Encode query number and batch row into each predicted action chunk."""

    def __init__(self) -> None:
        self.batches: list[dict[str, torch.Tensor]] = []

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Record raw history and return actions of shape (B,4,7)."""
        self.batches.append({key: value.clone() for key, value in batch.items()})
        count = batch["state"].shape[0]
        actions = torch.zeros(count, 4, 7)
        actions[:, :, 0] = (10 * len(self.batches) + torch.arange(count))[:, None]
        return actions


def test_instruction_changes_replan_only_affected_queues_with_fixed_batch() -> None:
    backend, policy = PhysicalSequence(), ChunkPolicy()
    records = run_batch(
        backend,
        policy,
        active_schema(),
        [CGTask("l5", 0, 0, button=0)] * 3,
        [1, 2, 3],
        instruction="full",
        profile=get_profile("l5_v1"),
        valid_count=2,
        config=EvaluationConfig(device="cpu", num_envs=3, horizon=5, n_execute=4),
    )
    assert [batch["state"].shape[0] for batch in policy.batches] == [3, 3, 3]
    np.testing.assert_array_equal(np.stack(backend.actions)[:, 0, 0], [10, 10, 20, 20, 20])
    np.testing.assert_array_equal(np.stack(backend.actions)[:, 1, 0], [11, 11, 11, 11, 31])
    np.testing.assert_array_equal(np.stack(backend.actions)[:, 2], 0)
    assert [batch["task_indices"][:, 0, 0].tolist() for batch in policy.batches] == [
        [0, 0, 0],
        [4, 0, 0],
        [4, 0, 0],
    ]
    assert all(set(batch) == {"state", "task_indices"} for batch in policy.batches)
    assert records[0]["instruction_transitions"] == [{"step": 2, "from": 0, "to": 4}]
    assert records[1]["instruction_transitions"] == []
    assert records[0]["active_instruction_ids"] == [0, 0, 4, 4, 4, 4]


def test_online_labels_match_offline_and_regress_independently_of_success() -> None:
    backend, policy = PhysicalSequence(regress=True, latch_success_stage=True), ChunkPolicy()
    task, profile = CGTask("l5", 0, 0, button=0), get_profile("l5_v1")
    records = run_batch(
        backend,
        policy,
        active_schema(),
        [task],
        [1],
        instruction="full",
        profile=profile,
        config=EvaluationConfig(device="cpu", num_envs=1, horizon=5, n_execute=4),
    )
    trajectory = [observations[0] for observations in backend.observations]
    labels = label_trajectory(
        profile,
        task,
        np.stack([item["state"] for item in trajectory]),
        np.stack([item["world_entities"] for item in trajectory]),
    )
    assert records[0]["active_instruction_ids"] == labels[:, 0].tolist() == [0, 0, 4, 4, 0, 0]
    assert records[0]["stage0_done"] and not records[0]["success"]
    assert records[0]["instruction_transitions"][-1] == {"step": 4, "from": 4, "to": 0}
    # The same evaluator creates fresh trackers for each independent reset.
    repeated = run_batch(
        backend,
        policy,
        active_schema(),
        [task],
        [2],
        instruction="full",
        profile=profile,
        config=EvaluationConfig(device="cpu", num_envs=1, horizon=1, n_execute=4),
    )
    assert repeated[0]["active_instruction_ids"] == [0, 0]


@pytest.mark.parametrize("coordinates", ["world", "relative"])
def test_tracker_world_poses_do_not_depend_on_policy_entity_coordinates(coordinates: str) -> None:
    backend = SimulatorBackend.__new__(SimulatorBackend)
    backend.config = EvaluationConfig(entity_coordinates=coordinates)
    backend.family, backend.cameras = "l4", ()
    backend.envs = [SimpleNamespace()]
    raw = {
        "robot0_eef_pos": np.array([0.1, 0.2, 1]),
        "robot0_eef_quat": np.array([0, 0, 0, 1]),
        "robot0_gripper_qpos": np.zeros(2),
    }
    for index, name in enumerate(OBJECTS + CONTAINERS):
        raw[f"{name}_pos"] = np.array([index, index + 1, 0.9])
        raw[f"{name}_quat"] = np.array([0, 0, 0, 1])
    backend.observations = [raw]
    observed = backend.observe()[0]
    np.testing.assert_allclose(observed["world_entities"][0], [0, 1, 0.9, 0, 0, 0, 1])
    expected = [0, 1, 0.9] if coordinates == "world" else [-0.1, 0.8, -0.1]
    np.testing.assert_allclose(observed["entities"][0, :3], expected, atol=1e-7)


def checkpoint_payload() -> dict[str, Any]:
    """Build a minimal native payload with exact active-conditioning semantics."""
    task = CGTask("l5", 0, 0, button=0)
    return {
        "feature_schema": active_schema().model_dump(mode="json"),
        "policy_config": {
            "action_steps": 4,
            "conditioning": {"type": "indexed", "vocab_sizes": [12]},
        },
        "run_config": {
            "dataset": {"selection": {"include_tasks": [task.slug], "exclude_tasks": []}}
        },
        "selection": [(task.slug, 0, "fixture")],
        "completed_updates": 17,
        "wandb": None,
    }


@pytest.mark.parametrize(
    "change", ["path", "vocabulary", "shape", "history", "dtype", "normalization"]
)
def test_incompatible_active_checkpoint_fails_before_policy_load(
    change: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = checkpoint_payload()
    feature = payload["feature_schema"]["features"][1]
    if change == "path":
        feature["paths"] = [get_profile("l4_infer_cont0_v1").path]
    elif change == "vocabulary":
        payload["policy_config"]["conditioning"]["vocab_sizes"] = [13]
    elif change == "shape":
        feature["shape"] = [2]
    elif change == "history":
        feature["history"] = 1
    elif change == "dtype":
        feature["dtype"] = "float32"
    else:
        feature["normalization"] = "standard"
    monkeypatch.setattr(runner, "read_checkpoint", lambda _: payload)
    monkeypatch.setattr(runner, "load_policy", lambda *args, **kwargs: pytest.fail("loaded policy"))
    task = CGTask("l5", 0, 0, button=0)
    with pytest.raises(ValueError):
        runner.evaluate(
            tmp_path / "checkpoint.pt",
            family="l5",
            instruction="full",
            conditioning="active_v1",
            task_set=TaskSet("l5", "test", "full", (task.slug,), (task.slug,)),
            config=EvaluationConfig(device="cpu"),
            out_dir=tmp_path / "evaluation",
        )
    assert not (tmp_path / "evaluation").exists()


def test_active_evaluation_records_profile_and_full_composition_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = checkpoint_payload()
    backend = PhysicalSequence()
    monkeypatch.setattr(runner, "read_checkpoint", lambda _: payload)
    monkeypatch.setattr(runner, "load_policy", lambda *args, **kwargs: ChunkPolicy())
    monkeypatch.setattr(runner, "SimulatorBackend", lambda *args: backend)
    monkeypatch.setattr(runner, "log_evaluation", lambda *args, **kwargs: None)
    trained, unseen = CGTask("l5", 0, 0, button=0), CGTask("l5", 0, 0, button=1)
    output = tmp_path / "evaluation"
    result = runner.evaluate(
        tmp_path / "checkpoint.pt",
        family="l5",
        instruction="full",
        conditioning="active_v1",
        task_set=TaskSet("l5", "test", "full", (trained.slug,), (trained.slug, unseen.slug)),
        config=EvaluationConfig(
            device="cpu", num_envs=2, horizon=3, n_execute=4, n_trials_per_task=1
        ),
        out_dir=output,
    )
    resolved = json.loads((output / "eval_config_used.json").read_text())
    assert resolved["conditioning"] == "active_v1"
    assert resolved["active_profile"] == get_profile("l5_v1").to_dict()
    assert resolved["active_profile_digest"] == get_profile("l5_v1").digest
    assert resolved["design_metrics"] == result["design_metrics"]
    assert resolved["design_metrics"]["concrete_train_count"] == 1
    assert resolved["design_metrics"]["ood_with_unseen_pairs_slugs"] == [unseen.slug]
    assert result["checkpoint_step"] == 17
    assert result["splits"]["id"]["n_episodes"] == result["splits"]["ood"]["n_episodes"] == 1
    episodes = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert episodes[0]["active_instruction_ids"][0] == episodes[1]["active_instruction_ids"][0]
    assert [episode["split"] for episode in episodes] == ["id", "ood"]


def test_bounded_video_overlay_leaves_input_frame_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    videos = EpisodeVideos(
        tmp_path,
        None,
        offset=0,
        max_videos=1,
        max_attention_videos=0,
        fps=20,
        profile=get_profile("l5_v1"),
    )
    written = []
    monkeypatch.setattr(videos, "_write", lambda index, kind, frame: written.append(frame.copy()))
    observed = {"images": np.zeros((2, 3, 64, 64), np.uint8), "task_indices": np.array([4])}
    videos(0, observed, None)
    videos(1, observed, None)
    assert len(written) == 1 and np.any(written[0])
    assert not np.any(observed["images"])
    videos.close()


def active_checkpoint(path: Path, task: CGTask, profile_name: str) -> None:
    """Write a tiny untrained native image iMF checkpoint for simulator integration."""
    profile = get_profile(profile_name)
    config = RunConfig.model_validate(
        {
            "features": {
                "features": [
                    {"key": "state", "shape": [9], "paths": ["state"]},
                    {"key": "actions", "shape": [7], "paths": ["actions"], "future": 1},
                    {
                        "key": "images",
                        "shape": [3, 64, 64],
                        "source": "video",
                        "cameras": ["robot0_eye_in_hand", "agentview"],
                        "dtype": "uint8",
                    },
                    {
                        "key": "task_indices",
                        "shape": [1],
                        "paths": [profile.path],
                        "dtype": "int64",
                    },
                ]
            },
            "dataset": {"root": "unused-active-simulator-fixture"},
            "policy": {
                "state_dim": 9,
                "action_dim": 7,
                "action_steps": 2,
                "obs_steps": 1,
                "cameras": 2,
                "vision": {"type": "patch", "dim": 16, "patch_size": 16},
                "conditioning": {"type": "indexed", "vocab_sizes": [12]},
                "fusion": {"type": "cross", "dim": 16, "heads": 2, "layers": 1, "ff_dim": 32},
                "head": {
                    "type": "transformer",
                    "dim": 16,
                    "heads": 2,
                    "layers": 1,
                    "ff_dim": 32,
                    "time_dim": 8,
                },
                "algorithm": {"type": "improved_mean_flow"},
            },
            "logging": {"wandb": {"enabled": False}},
        }
    )
    policy = build_policy(config.policy)
    fingerprint = hashlib.sha256(b"untrained active simulator integration fixture").hexdigest()
    selection = ((task.slug, 0, "fixture"),)
    statistics = Statistics(
        feature_schema=config.features,
        dataset_fingerprint=fingerprint,
        selection=selection,
        features={},
    )
    save_checkpoint(
        path,
        {
            "format": "robopolicy.training",
            "schema_version": 1,
            "policy_config": config.policy.model_dump(mode="json"),
            "model": policy.state_dict(),
            "statistics": statistics.model_dump(mode="json"),
            "feature_schema": config.features.model_dump(mode="json"),
            "processing": config.processing.model_dump(mode="json"),
            "optimizer": torch.optim.AdamW(policy.parameters()).state_dict(),
            "scheduler": {},
            "scaler": {},
            "completed_updates": 0,
            "data_epoch": 0,
            "skipped_updates": 0,
            "rng": capture_rng("cpu"),
            "dataset_fingerprint": fingerprint,
            "selection": selection,
            "run_config": config.model_dump(mode="json"),
            "wandb": None,
            "segment": "untrained-active-simulator-integration-fixture",
        },
    )


@pytest.mark.simulator
@pytest.mark.parametrize("profile_name", ["l5_v1", "l4_infer_cont0_v1"])
def test_real_active_checkpoint_evaluation(profile_name: str, tmp_path: Path) -> None:
    simulator_root = os.environ.get("CG_SIMULATOR_ROOT")
    if not simulator_root:
        pytest.skip("Set CG_SIMULATOR_ROOT to explicitly opt into real simulator checks")
    profile = get_profile(profile_name)
    task = (
        CGTask("l5", 0, 0, button=0)
        if profile.family == "l5"
        else CGTask("l4_2s", 0, 0, object1=1, container1=1)
    )
    checkpoint = tmp_path / "untrained-active.pt"
    active_checkpoint(checkpoint, task, profile_name)
    output = tmp_path / "evaluation"
    result = runner.evaluate(
        checkpoint,
        family=profile.family,
        instruction=profile.instruction,
        conditioning="active_v1",
        task_set=TaskSet(
            profile.family, "integration", profile.instruction, (task.slug,), (task.slug,)
        ),
        config=EvaluationConfig(
            simulator_root=Path(simulator_root),
            device="cpu",
            num_envs=1,
            horizon=2,
            n_execute=1,
            n_trials_per_task=1,
            max_videos=1,
            max_attention_videos=1,
            logging=LoggingConfig(wandb=WandbConfig(enabled=False)),
        ),
        out_dir=output,
    )
    episode = json.loads((output / "episodes.jsonl").read_text())
    assert len(episode["active_instruction_ids"]) == episode["length"] + 1
    assert all(0 <= instruction < 12 for instruction in episode["active_instruction_ids"])
    assert result["aggregated"]["n_episodes"] == 1 and result["checkpoint_step"] == 0
    assert (output / "videos/episode-000000.mp4").stat().st_size > 0
    assert (output / "attention/episode-000000.mp4").stat().st_size > 0
