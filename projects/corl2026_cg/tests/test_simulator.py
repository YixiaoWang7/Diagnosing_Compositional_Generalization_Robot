"""Opt-in acceptance checks against the pinned real CG simulator and renderer.

Run with CG_SIMULATOR_ROOT=/path/to/robot_envs MUJOCO_GL=egl pytest -m simulator.
The tiny checkpoint is intentionally untrained; these tests check integration,
RGB observations, reset/step, native inference and artifacts, not policy quality.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from robopolicy import RunConfig, build_policy
from robopolicy.checkpoints.serialization import capture_rng, save_checkpoint
from robopolicy.processing.normalization import Statistics
from robopolicy.training.logging import LoggingConfig, WandbConfig

from corl2026_cg.evaluation import EvaluationConfig, evaluate
from corl2026_cg.evaluation.backend import SimulatorBackend
from corl2026_cg.tasks import CGTask, TaskSet, instruction_fields

pytestmark = pytest.mark.simulator


@pytest.fixture
def simulator_root() -> Path:
    value = os.environ.get("CG_SIMULATOR_ROOT")
    if not value:
        pytest.skip("Set CG_SIMULATOR_ROOT to explicitly opt into real simulator checks")
    return Path(value)


@pytest.mark.parametrize("family,entity_count", [("l4", 8), ("l5", 12)])
def test_real_simulator_rgb_reset_step(
    simulator_root: Path, family: str, entity_count: int
) -> None:
    config = EvaluationConfig(
        simulator_root=simulator_root, num_envs=1, horizon=3, n_execute=1, device="cpu"
    )
    backend = SimulatorBackend(family, config, ("robot0_eye_in_hand", "agentview"), (64, 64))
    try:
        task = CGTask(family, 0, 0, button=0 if family == "l5" else None)
        observation = backend.reset([task], [123])[0]
        assert observation["state"].shape == (9,)
        assert observation["entities"].shape == (entity_count, 7)
        assert observation["images"].shape == (2, 3, 64, 64)
        assert observation["images"].dtype == np.uint8
        for index, camera in enumerate(backend.cameras):
            expected = np.flipud(backend.observations[0][f"{camera}_image"]).transpose(2, 0, 1)
            np.testing.assert_array_equal(observation["images"][index], expected)
        observation, rewards, done = backend.step(np.zeros((1, 7), np.float32), [True])
        assert np.isfinite(observation[0]["state"]).all()
        assert rewards.shape == (1,) and done == [False]
        assert not backend.native_success(0)
        if family == "l4":
            original = backend.envs[0].task
            backend.placement_success(0, 2, 3)
            assert backend.envs[0].task == original
            backend.set_target(0, 1, 1)
            assert "cube" in backend.envs[0].task and "mug" in backend.envs[0].task
        else:
            assert not backend.place_stage_success(0)
    finally:
        backend.close()


def _checkpoint(path: Path, task: CGTask) -> None:
    """Save a native tiny image MeanFlow checkpoint without downloading weights."""
    fields = instruction_fields(task.family, "full")
    config = RunConfig.model_validate(
        {
            "features": {
                "features": [
                    {"key": "state", "shape": [9], "paths": ["state"], "history": 1},
                    {"key": "actions", "shape": [7], "paths": ["actions"], "future": 1},
                    {
                        "key": "images",
                        "shape": [3, 64, 64],
                        "source": "video",
                        "history": 1,
                        "cameras": ["robot0_eye_in_hand", "agentview"],
                        "dtype": "uint8",
                    },
                    {
                        "key": "task_indices",
                        "shape": [len(fields)],
                        "dtype": "int64",
                        "paths": [f"cg/task/{field}" for field in fields],
                    },
                ]
            },
            "dataset": {"root": "unused-offline-fixture"},
            "policy": {
                "state_dim": 9,
                "action_dim": 7,
                "action_steps": 2,
                "obs_steps": 2,
                "cameras": 2,
                "vision": {"type": "patch", "dim": 16, "patch_size": 16},
                "conditioning": {"type": "indexed", "vocab_sizes": [4] * len(fields)},
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
    optimizer = torch.optim.AdamW(policy.parameters())
    fingerprint = hashlib.sha256(b"explicit untrained simulator integration fixture").hexdigest()
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
            "optimizer": optimizer.state_dict(),
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
            "segment": "untrained-simulator-integration-fixture",
        },
    )


@pytest.mark.parametrize("family", ["l4", "l5"])
def test_real_checkpoint_evaluation(simulator_root: Path, tmp_path: Path, family: str) -> None:
    task = CGTask(family, 0, 0, button=0 if family == "l5" else None)
    checkpoint = tmp_path / "untrained.pt"
    _checkpoint(checkpoint, task)
    task_set = TaskSet(family, "integration", "full", (task.slug,), (task.slug,))
    config = EvaluationConfig(
        simulator_root=simulator_root,
        num_envs=1,
        horizon=2,
        n_execute=1,
        n_trials_per_task=1,
        device="cpu",
        max_videos=1,
        max_attention_videos=1,
        logging=LoggingConfig(wandb=WandbConfig(enabled=False)),
    )
    result = evaluate(
        checkpoint,
        family=family,
        instruction="full",
        task_set=task_set,
        config=config,
        out_dir=tmp_path / "evaluation",
    )
    assert result["aggregated"]["n_episodes"] == 1
    assert result["checkpoint_step"] == 0
    assert result["simulator"]["revision"] == config.simulator_revision
    assert result["splits"]["ood"]["pc_success"] is None
    assert (tmp_path / "evaluation/videos/episode-000000.mp4").stat().st_size > 0
    assert (tmp_path / "evaluation/attention/episode-000000.mp4").stat().st_size > 0
