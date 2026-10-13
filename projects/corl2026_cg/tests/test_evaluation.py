"""Regression checks for task success and online observation/control semantics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from robopolicy import Feature, FeatureSchema

from corl2026_cg.evaluation.backend import SimulatorBackend, relative_pose
from corl2026_cg.evaluation.config import EvaluationConfig
from corl2026_cg.evaluation.rollout import ObservationHistory, categorical_features, run_batch
from corl2026_cg.evaluation.runner import aggregate
from corl2026_cg.evaluation.success import EpisodeProgress
from corl2026_cg.tasks import CGTask


class Probe:
    def __init__(self):
        self.placements = set()
        self.targets = []
        self.place = self.native = False

    def placement_success(self, index, object_id, container_id):
        return (object_id, container_id) in self.placements

    def native_success(self, index):
        return self.native

    def place_stage_success(self, index):
        return self.place

    def set_target(self, index, object_id, container_id):
        self.targets.append((object_id, container_id))


@pytest.mark.parametrize(
    "instruction,placement,success",
    [
        ("full", (0, 1), False),
        ("object_only", (0, 1), True),
        ("object_only", (1, 1), False),
        ("container_only", (2, 0), True),
        ("container_latents", (3, 0), True),
        ("container_only", (0, 1), False),
    ],
)
def test_visible_single_stage_instruction_controls_success(instruction, placement, success):
    env = Probe()
    env.placements.add(placement)
    progress = EpisodeProgress()
    progress.update(env, 0, CGTask("l4", 0, 0), instruction)
    assert progress.success is success


def test_two_stage_latches_first_stage_and_requires_separate_transition():
    env = Probe()
    task = CGTask("l4_2s", 0, 0, 1, 1)
    progress = EpisodeProgress()
    env.placements = {(0, 0), (1, 1)}
    assert progress.update(env, 0, task, "full")
    assert progress.stage0_done and not progress.success
    assert env.targets == [(1, 1)]
    env.placements = {(1, 1)}
    progress.update(env, 0, task, "full")
    assert progress.success


def test_open_container_excludes_first_destination():
    env = Probe()
    task = CGTask("l4_2s", 0, 0, 1, 1)
    progress = EpisodeProgress(stage0_done=True, stage0_container=0)
    env.placements = {(1, 0)}
    progress.update(env, 0, task, "open_cont")
    assert not progress.success
    env.placements = {(1, 3)}
    progress.update(env, 0, task, "open_cont")
    assert progress.success and progress.stage1_container == 3


def test_infer_cont0_violation_is_latched_and_hidden_destination_is_free():
    env = Probe()
    task = CGTask("l4_2s", 0, 0, 1, 1)
    valid = EpisodeProgress()
    env.placements = {(0, 3)}
    assert valid.update(env, 0, task, "infer_cont0")
    assert valid.stage0_container == 3
    invalid = EpisodeProgress()
    env.placements = {(0, 1)}
    invalid.update(env, 0, task, "infer_cont0")
    assert invalid.same_container_violation and not invalid.stage0_done
    env.placements = {(0, 2)}
    invalid.update(env, 0, task, "infer_cont0")
    env.placements = {(1, 1)}
    invalid.update(env, 0, task, "infer_cont0")
    assert invalid.stage1_done and not invalid.success


def test_l5_requires_physical_button_after_placement():
    env = Probe()
    task = CGTask("l5", 0, 0, button=2)
    progress = EpisodeProgress()
    env.place = True
    progress.update(env, 0, task, "full")
    assert progress.stage0_done and not progress.success
    env.native = True
    progress.update(env, 0, task, "full")
    assert progress.stage1_done and progress.success


class SteppingEnvironment(Probe):
    def reset(self, tasks, seeds):
        self.step_number = 0
        return [{"state": np.zeros(9, np.float32)} for _ in tasks]

    def step(self, actions, active):
        self.step_number += 1
        return (
            [{"state": np.full(9, self.step_number, np.float32)} for _ in active],
            np.ones(len(active)),
            [False] * len(active),
        )


class RecordingPolicy:
    def __init__(self):
        self.inputs = []

    def predict(self, batch):
        self.inputs.append(batch["state"].clone())
        return torch.zeros(batch["state"].shape[0], 2, 7)


def test_history_contains_consecutive_frames_between_action_chunks():
    schema = FeatureSchema(
        features=(Feature(key="state", shape=(9,), paths=("state",), history=2),)
    )
    policy = RecordingPolicy()
    records = run_batch(
        SteppingEnvironment(),
        policy,
        schema,
        [CGTask("l4", 0, 0)] * 2,
        [1, 2],
        instruction="full",
        valid_count=1,
        config=EvaluationConfig(device="cpu", num_envs=2, horizon=5, n_execute=2),
    )
    assert [batch.shape[0] for batch in policy.inputs] == [2, 2, 2]
    assert policy.inputs[0][0, :, 0].tolist() == [0, 0, 0]
    assert policy.inputs[1][0, :, 0].tolist() == [0, 1, 2]
    assert policy.inputs[2][0, :, 0].tolist() == [2, 3, 4]
    assert len(records) == 1 and records[0]["length"] == 5


def test_future_observation_windows_are_rejected():
    schema = FeatureSchema(features=(Feature(key="state", shape=(9,), paths=("state",), future=1),))
    with pytest.raises(ValueError, match="causal"):
        ObservationHistory(
            schema, [{"state": np.zeros(9, np.float32)}], [CGTask("l4", 0, 0)], "full"
        )


def test_entity_indices_match_prepared_semantic_paths():
    schema = FeatureSchema(
        features=(
            Feature(
                key="task_indices",
                shape=(3,),
                paths=("cg/task/object0", "cg/task/container0", "cg/task/button"),
                dtype="int64",
            ),
            Feature(
                key="entity_indices",
                shape=(3,),
                paths=("cg/entity/object0", "cg/entity/container0", "cg/entity/button"),
                dtype="int64",
            ),
        )
    )
    values = categorical_features(CGTask("l5", 2, 1, button=3), "full", schema)
    assert values["task_indices"].tolist() == [2, 1, 3]
    assert values["entity_indices"].tolist() == [2, 5, 11]


def test_backend_preserves_rgb_channels_and_camera_order():
    from corl2026_cg.tasks import CONTAINERS, OBJECTS

    backend = SimulatorBackend.__new__(SimulatorBackend)
    backend.config = EvaluationConfig(entity_coordinates="world")
    backend.family, backend.cameras = "l4", ("wrist", "front")
    backend.envs = [SimpleNamespace()]
    raw = {
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0, 0, 0, 1]),
        "robot0_gripper_qpos": np.zeros(2),
    }
    for name in OBJECTS + CONTAINERS:
        raw[f"{name}_pos"], raw[f"{name}_quat"] = np.zeros(3), np.array([0, 0, 0, 1])
    raw["wrist_image"] = np.array([[[255, 0, 0]], [[0, 255, 0]]], np.uint8)
    raw["front_image"] = np.full((2, 1, 3), 12, np.uint8)
    backend.observations = [raw]
    observed = backend.observe()[0]
    assert observed["images"][0, :, 0, 0].tolist() == [0, 255, 0]
    assert observed["images"][0, :, 1, 0].tolist() == [255, 0, 0]
    assert observed["images"][1, :, 0, 0].tolist() == [12, 12, 12]
    assert observed["entities"].shape == (8, 7)


def test_relative_pose_matches_exported_position_delta_and_xyzw():
    pose = np.array([3, 4, 5, 0, 0, 0, -1], np.float32)
    relative = relative_pose(pose, np.ones(3), np.array([0, 0, 0, 1]))
    np.testing.assert_array_equal(relative, [2, 3, 4, 0, 0, 0, 1])


def test_empty_ood_population_is_not_zero_success():
    assert aggregate([]) == {"n_episodes": 0, "n_success": 0, "pc_success": None}


def test_standalone_evaluation_writes_summary_and_absent_ood(tmp_path, monkeypatch):
    from robopolicy.training.logging import LoggingConfig, WandbConfig

    from corl2026_cg.evaluation import runner
    from corl2026_cg.tasks import TaskSet

    schema = FeatureSchema(
        features=(
            Feature(key="state", shape=(9,), paths=("state",)),
            Feature(
                key="task_indices",
                shape=(2,),
                dtype="int64",
                paths=("cg/task/object0", "cg/task/container0"),
            ),
        )
    )
    payload = {
        "feature_schema": schema.model_dump(),
        "policy_config": {
            "action_steps": 2,
            "conditioning": {"type": "indexed", "vocab_sizes": [4, 4]},
        },
        "run_config": {
            "dataset": {"selection": {"include_tasks": ["cross_into_bin"], "exclude_tasks": []}}
        },
        "selection": [("cross_into_bin", 0, "fixture")],
        "completed_updates": 10,
        "wandb": None,
    }
    backend = SteppingEnvironment()
    backend.provenance = {"revision": "test"}
    backend.close = lambda: None
    monkeypatch.setattr(runner, "read_checkpoint", lambda _: payload)
    monkeypatch.setattr(runner, "load_policy", lambda *args, **kwargs: RecordingPolicy())
    monkeypatch.setattr(runner, "SimulatorBackend", lambda *args: backend)
    logged = []
    monkeypatch.setattr(runner, "log_evaluation", lambda *args, **kwargs: logged.append(args))
    task_set = TaskSet("l4", "test", "full", ("cross_into_bin",), ("cross_into_bin",))
    config = EvaluationConfig(
        device="cpu",
        num_envs=2,
        horizon=2,
        n_execute=2,
        n_trials_per_task=1,
        logging=LoggingConfig(wandb=WandbConfig(enabled=False)),
    )
    result = runner.evaluate(
        tmp_path / "checkpoint.pt",
        family="l4",
        instruction="full",
        task_set=task_set,
        config=config,
        out_dir=tmp_path / "evaluation",
    )
    assert result["aggregated"]["n_episodes"] == 1
    assert result["splits"]["ood"]["pc_success"] is None
    assert (tmp_path / "evaluation" / "episodes.jsonl").exists()
    assert logged
    with pytest.raises(FileExistsError):
        runner.evaluate(
            tmp_path / "checkpoint.pt",
            family="l4",
            instruction="full",
            task_set=task_set,
            config=config,
            out_dir=tmp_path / "evaluation",
        )


def test_stage_change_discards_old_actions_and_replans():
    class StageEnvironment(SteppingEnvironment):
        def step(self, actions, active):
            observation, rewards, done = super().step(actions, active)
            self.placements = {(0, 0)} if self.step_number == 1 else {(1, 1)}
            return observation, rewards, done

    schema = FeatureSchema(features=(Feature(key="state", shape=(9,), paths=("state",)),))
    policy = RecordingPolicy()
    records = run_batch(
        StageEnvironment(),
        policy,
        schema,
        [CGTask("l4_2s", 0, 0, 1, 1)],
        [1],
        instruction="full",
        config=EvaluationConfig(device="cpu", num_envs=1, horizon=4, n_execute=2),
    )
    assert len(policy.inputs) == 2
    assert records[0]["success"] and records[0]["length"] == 2


@pytest.mark.parametrize(
    "change",
    [
        "same_width_instruction",
        "reordered_fields",
        "vocabulary",
        "training_tasks",
        "excluded_tasks",
        "actual_demos",
    ],
)
def test_evaluation_rejects_incompatible_checkpoint_contract(change):
    from corl2026_cg.evaluation.runner import validate_checkpoint_tasks
    from corl2026_cg.tasks import TaskSet

    payload = {
        "feature_schema": {
            "features": [
                {
                    "key": "task_indices",
                    "shape": [2],
                    "dtype": "int64",
                    "paths": ["cg/task/object0", "cg/task/container0"],
                }
            ]
        },
        "policy_config": {"conditioning": {"type": "indexed", "vocab_sizes": [4, 4]}},
        "run_config": {
            "dataset": {"selection": {"include_tasks": ["cross_into_bin"], "exclude_tasks": []}}
        },
        "selection": [("cross_into_bin", 0, "fixture")],
    }
    task_set = TaskSet("l4", "test", "full", ("cross_into_bin",), ("cross_into_bin",))
    if change == "same_width_instruction":
        task_set = TaskSet("l4", "test", "container_only", ("cross_into_bin",), ("cross_into_bin",))
        payload["feature_schema"]["features"][0].update(shape=[1], paths=["cg/task/object0"])
        payload["policy_config"]["conditioning"]["vocab_sizes"] = [4]
    elif change == "reordered_fields":
        payload["feature_schema"]["features"][0]["paths"].reverse()
    elif change == "vocabulary":
        payload["policy_config"]["conditioning"]["vocab_sizes"] = [4, 3]
    elif change == "training_tasks":
        payload["run_config"]["dataset"]["selection"]["include_tasks"] = ["cube_into_bin"]
    elif change == "excluded_tasks":
        payload["run_config"]["dataset"]["selection"]["exclude_tasks"] = ["cross_into_bin"]
    else:
        payload["selection"] = [("cube_into_bin", 0, "fixture")]
    with pytest.raises(ValueError):
        validate_checkpoint_tasks(payload, task_set)
