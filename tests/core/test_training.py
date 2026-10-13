"""Training state, checkpoint recovery, and validation behavior."""

import json
import random
import shutil

import numpy as np
import pytest
import torch

from robopolicy.checkpoints.inference import InferenceConfig, load_policy
from robopolicy.checkpoints.serialization import (
    read_checkpoint,
    restore_rng,
    save_checkpoint,
)
from robopolicy.config.run import RunConfig
from robopolicy.contracts.schema import FeatureSchema
from robopolicy.datasets.factory import build_loader
from robopolicy.training.trainer import Trainer, validate
from tests.core.test_dataset import write_dataset


def run_config(tmp_path, device="cpu", amp=False, head="transformer"):
    """Generate real numeric data and configure a small complete run."""
    schema = write_dataset(tmp_path / "data", lengths=((8, 9, 10, 11, 12),))
    features = FeatureSchema(features=tuple(f for f in schema.features if f.key != "task_indices"))
    return RunConfig(
        features=features,
        dataset={"root": str(tmp_path / "data"), "selection": {"val_fraction": 0.3}},
        loader={"batch_size": 4},
        policy={
            "state_dim": 1,
            "action_dim": 1,
            "obs_steps": 3,
            "action_steps": 3,
            "fusion": {"dim": 16, "heads": 2, "layers": 1, "ff_dim": 32},
            "head": {
                "type": "transformer",
                "dim": 16,
                "heads": 2,
                "layers": 1,
                "ff_dim": 32,
                "time_dim": 8,
            },
        },
        training={
            "device": device,
            "amp": amp,
            "max_updates": 6,
            "checkpoint_every": 3,
            "validate_every": 3,
            "log_every": 1,
        },
        logging={"out_dir": str(tmp_path / "run"), "wandb": {"enabled": False}},
    )


@pytest.mark.parametrize("algorithm", ["flow_matching", "improved_mean_flow"])
def test_resume_all_state_and_prediction_parity(tmp_path, monkeypatch, algorithm):
    config = run_config(tmp_path)
    values = config.model_dump()
    values["policy"]["algorithm"] = {"type": algorithm}
    config = RunConfig.model_validate(values)
    trainer = Trainer(config)
    initial = {key: value.detach().clone() for key, value in trainer.policy.state_dict().items()}
    trainer.fit(until_step=3)
    assert any(
        not torch.equal(initial[key], value) for key, value in trainer.policy.state_dict().items()
    )
    path = tmp_path / "run/latest.pt"
    saved = read_checkpoint(path)
    assert saved["completed_updates"] == 3 and saved["optimizer"]["state"]
    with build_loader(trainer.dataset) as loader:
        batch = next(iter(loader))
    inputs = trainer.packer(batch)
    noise = torch.zeros(inputs.state.shape[0], 3, 1)
    trainer.policy.eval()
    expected = trainer.packer.actions_to_raw(trainer.policy.predict(inputs, noise))
    # No pretrained downloads or dataset access may occur during inference loading.
    monkeypatch.setattr("torch.hub.load", lambda *a, **k: pytest.fail("unexpected download"))
    shutil.move(tmp_path / "data", tmp_path / "relocated")
    inference = load_policy(path, device="cpu")
    actual = inference.predict({k: v for k, v in batch.items() if k != "actions"}, noise)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    overridden = load_policy(
        path, device="cpu", inference_config=InferenceConfig(num_inference_steps=2)
    )
    assert overridden.policy.algorithm.config.type == algorithm
    assert overridden.policy.algorithm.config.num_inference_steps == 2
    restore_rng(saved["rng"], "cpu")
    rng_expected = (random.random(), np.random.rand(), torch.rand(3))
    resumed = Trainer.resume(path, dataset_root=str(tmp_path / "relocated"))
    assert resumed.completed_updates == 3 and resumed.data_epoch == saved["data_epoch"] + 1
    assert resumed.scheduler.state_dict() == saved["scheduler"]
    assert resumed.optimizer.state_dict()["param_groups"] == saved["optimizer"]["param_groups"]
    assert random.random() == rng_expected[0] and np.random.rand() == rng_expected[1]
    torch.testing.assert_close(torch.rand(3), rng_expected[2], rtol=0, atol=0)
    for key, value in saved["optimizer"]["state"].items():
        for field, expected_value in value.items():
            torch.testing.assert_close(
                resumed.optimizer.state_dict()["state"][key][field], expected_value
            )
    resumed.fit()
    assert resumed.completed_updates == 6
    events = [
        json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()
    ]
    assert [row["train/step"] for row in events if row["event"] == "train"] == [1, 2, 3, 4, 5, 6]
    assert len({row["resume/segment"] for row in events}) == 2


def test_validation_partial_batches_and_mode_exception(tmp_path, monkeypatch):
    trainer = Trainer(run_config(tmp_path))
    with build_loader(trainer.dataset) as loader:
        raw = next(iter(loader))
    small = {
        key: (value[:1] if isinstance(value, torch.Tensor) else value) for key, value in raw.items()
    }

    class FakeLoss:
        training = True

        def train(self, mode=True):
            self.training = mode

        def eval(self):
            self.train(False)

        def loss(self, inputs):
            return torch.tensor(float(inputs.state.shape[0]))

    policy = FakeLoss()
    metrics = validate(policy, [raw, small], trainer.packer, "cpu")
    assert metrics["val/loss"] == (raw["state"].shape[0] ** 2 + 1) / (raw["state"].shape[0] + 1)
    assert policy.training

    def fail(_):
        raise RuntimeError("validation failed")

    policy.loss = fail
    policy.train(False)
    with pytest.raises(RuntimeError, match="validation failed"):
        validate(policy, [raw], trainer.packer, "cpu")
    assert not policy.training
    trainer.close()


def test_atomic_failure_and_incompatible_schema(tmp_path, monkeypatch):
    target = tmp_path / "checkpoint.pt"
    target.write_bytes(b"original")

    def fail(*args, **kwargs):
        raise OSError("disk error")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk error"):
        save_checkpoint(target, {"value": torch.zeros(2)})
    assert target.read_bytes() == b"original" and list(tmp_path.iterdir()) == [target]
    monkeypatch.undo()
    torch.save({"schema_version": 99}, target)
    with pytest.raises(ValueError, match="Incompatible"):
        read_checkpoint(target)


def test_dataset_change_rejected(tmp_path):
    import h5py

    trainer = Trainer(run_config(tmp_path))
    trainer.fit(until_step=3)
    episode = trainer.dataset.episodes[0]
    with h5py.File(episode.path, "r+") as handle:
        handle[f"data/{episode.episode_id}/state"][0] = 999
    with pytest.raises(ValueError, match="Dataset identity"):
        Trainer.resume(tmp_path / "run/latest.pt")


@pytest.mark.cuda
def test_cuda_amp_step_and_skip(tmp_path, monkeypatch):
    assert torch.cuda.is_available(), "Run CUDA acceptance tests on a CUDA runner"
    trainer = Trainer(run_config(tmp_path, "cuda:0", True))
    with build_loader(trainer.dataset) as loader:
        raw = next(iter(loader))
    _, updated = trainer.update(raw)
    # Initial overflow is permissible; the explicit inf injection below must skip.
    previous = trainer.completed_updates
    scheduler = trainer.scheduler.state_dict()
    normal_loss = trainer.policy.loss
    monkeypatch.setattr(trainer.policy, "loss", lambda inputs: normal_loss(inputs) * float("inf"))
    _, updated = trainer.update(raw)
    assert not updated and trainer.completed_updates == previous
    assert trainer.scheduler.state_dict() == scheduler and trainer.skipped_updates >= 1
    trainer.close()


def test_deterministic_toy_learning(tmp_path):
    """The production trainer learns a constant trajectory from a fixed toy batch."""
    config = run_config(tmp_path)
    values = config.model_dump(mode="json")
    values["training"].update(
        max_updates=160, learning_rate=0.005, schedule="constant", weight_decay=0
    )
    trainer = Trainer(RunConfig.model_validate(values))
    with build_loader(trainer.dataset) as loader:
        raw = next(iter(loader))
    raw["state"].zero_()
    raw["actions"].fill_(trainer.packer.normalizer.statistics.features["actions"].offset[0])
    inputs = trainer.packer(raw, require_actions=True)
    noise = torch.ones_like(inputs.actions)
    trainer.policy.eval()
    before = torch.mean((trainer.policy.predict(inputs, noise) - inputs.actions) ** 2).item()
    for _ in range(160):
        trainer.update(raw)
    trainer.policy.eval()
    after = torch.mean((trainer.policy.predict(inputs, noise) - inputs.actions) ** 2).item()
    assert after < before * 0.2, (before, after)
    trainer.close()


def test_atomic_rename_failure_preserves_previous(tmp_path, monkeypatch):
    target = tmp_path / "saved.pt"
    save_checkpoint(target, {"old": 1})
    original = target.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("rename error")

    monkeypatch.setattr("robopolicy.checkpoints.serialization.os.replace", fail)
    with pytest.raises(OSError, match="rename error"):
        save_checkpoint(target, {"new": 2})
    assert target.read_bytes() == original and len(list(tmp_path.iterdir())) == 1
