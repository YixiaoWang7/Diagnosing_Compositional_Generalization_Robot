"""Real checkpoint contracts and reusable cross-runtime inference acceptance."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

from robopolicy import InferenceConfig, Trainer, dataset_identity, load_policy, read_checkpoint
from tests.core.test_dataset import write_dataset
from tests.core.test_training import run_config


@pytest.fixture
def checkpoint(tmp_path):
    trainer = Trainer(run_config(tmp_path))
    trainer.fit(until_step=1)
    return tmp_path / "run/latest.pt"


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("schema_version", 99, "schema"),
        ("completed_updates", True, "integer"),
        ("data_epoch", -1, "integer"),
        ("model", {"bad": 1}, "state dict"),
        ("policy_config.action_steps", 9, "conflicts"),
        ("feature_schema.features.0.normalization", "none", "conflicts"),
        ("processing.image_size", [4, 4], "conflicts"),
        ("run_config.policy.state_dim", 9, "shape"),
        ("statistics.dataset_fingerprint", "a" * 64, "fingerprint"),
        ("statistics.selection", [], "selection"),
        ("statistics.features.state.scale", [0], "positive"),
        ("statistics.features.state.offset", [float("nan")], "finite"),
        ("statistics.features.state.shape", [2], "shape"),
        ("statistics.features.state.offset", [], "length"),
    ],
)
def test_checkpoint_metadata_rejected(checkpoint, field, value, match):
    payload = read_checkpoint(checkpoint)
    target = payload
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    target[parts[-1]] = value
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match=match):
        read_checkpoint(checkpoint)


def test_reader_needs_no_dataset_or_model(checkpoint, monkeypatch):
    payload = read_checkpoint(checkpoint)
    shutil.move(checkpoint.parent.parent / "data", checkpoint.parent.parent / "relocated")

    def forbidden(*args, **kwargs):
        pytest.fail("checkpoint reader constructed a model or opened data")

    monkeypatch.setattr("robopolicy.checkpoints.inference.build_policy", forbidden)
    monkeypatch.setattr("robopolicy.datasets.index.read_manifest", forbidden)
    assert read_checkpoint(checkpoint)["dataset_fingerprint"] == payload["dataset_fingerprint"]
    del payload["statistics"]
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="missing fields"):
        read_checkpoint(checkpoint)


def test_identity_matches_training_without_decoder(tmp_path, monkeypatch):
    schema = write_dataset(tmp_path / "data", video=True)
    from robopolicy import DatasetConfig, build_dataset

    config = DatasetConfig(root=str(tmp_path / "data"))
    dataset = build_dataset(config, schema)
    expected = (dataset.fingerprint, dataset.selection)

    def forbidden(*args, **kwargs):
        pytest.fail("identity initialized execution or decoded video")

    monkeypatch.setattr("robopolicy.datasets.factory.PolicyDataset", forbidden)
    monkeypatch.setattr("robopolicy.datasets.video.require_decoder", forbidden)
    monkeypatch.setattr("robopolicy.datasets.video.DecoderHandle", forbidden)
    identity = dataset_identity(config, schema)
    assert (identity.fingerprint, identity.selection) == expected
    shutil.copytree(tmp_path / "data", tmp_path / "disk/data")
    relocated = config.model_copy(update={"root": str(tmp_path / "disk/data")})
    assert dataset_identity(relocated, schema) == identity


@pytest.mark.parametrize("version", ["2.7.0+cu128", "2.8.0+cpu"])
def test_older_runtime_eager_and_step_override(checkpoint, monkeypatch, version):
    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(torch, "compile", lambda *a, **k: pytest.fail("unexpected compilation"))
    runtime = load_policy(
        checkpoint, device="cpu", inference_config=InferenceConfig(num_inference_steps=3)
    )
    assert not runtime.compiled
    assert runtime.policy.algorithm.config.num_inference_steps == 3
    with pytest.raises(ValueError, match="requires PyTorch 2.9"):
        load_policy(checkpoint, device="cpu", inference_config=InferenceConfig(compile="enabled"))


def write_portability_fixtures(root: Path) -> None:
    """Train tiny state, entity, patch-image, and DINO models for another runtime."""
    import h5py
    import numpy as np

    from robopolicy import RunConfig, build_loader

    root.mkdir(parents=True, exist_ok=False)
    for modality in ("state", "entities", "images", "dinov2"):
        destination = root / modality
        config = run_config(destination).model_dump(mode="json")
        config["dataset"]["video"] = {"decoder": "opencv"}
        if modality in ("images", "dinov2"):
            schema = write_dataset(destination / "data", lengths=((8, 9, 10),), video=True)
            config["features"] = {
                "features": [
                    f.model_dump(mode="json") for f in schema.features if f.key != "task_indices"
                ]
            }
            config["policy"].update(
                cameras=2,
                vision={
                    "type": "dinov2" if modality == "dinov2" else "patch",
                    "patch_size": 4,
                    "dim": 16,
                    "depth": 1,
                    "heads": 2,
                    "image_size": 16,
                },
            )
        if modality == "entities":
            with h5py.File(destination / "data/task/0.h5", "r+") as file:
                for group in file["data"].values():
                    count = len(group["state"])
                    group["entities"] = np.ones((count, 2, 3), np.float32)
                    group["entity_indices"] = np.tile([0, 1], (count, 1)).astype(np.int64)
            config["features"]["features"].extend(
                [
                    {"key": "entities", "shape": [2, 3], "paths": ["entities"], "history": 2},
                    {
                        "key": "entity_indices",
                        "shape": [2],
                        "paths": ["entity_indices"],
                        "dtype": "int64",
                    },
                ]
            )
            config["policy"].update(entity_dim=3, selected_entities=2)
        trainer = Trainer(RunConfig.model_validate(config))
        with build_loader(trainer.dataset, trainer.config.loader) as loader:
            raw = next(iter(loader))
        raw = {key: value for key, value in raw.items() if key not in ("actions", "metadata")}
        trainer.fit(until_step=1)
        runtime = load_policy(destination / "run/latest.pt", device="cpu")
        noise = torch.ones(len(raw["state"]), 3, 1)
        torch.save(
            {"raw": raw, "noise": noise, "actions": runtime.predict(raw, noise)},
            destination / "prediction.pt",
        )
        shutil.rmtree(destination / "data")


def check_portability_fixtures(root: Path) -> None:
    """Load checkpoints with absent training data and compare fixed-noise predictions."""
    for modality in ("state", "entities", "images", "dinov2"):
        destination = root / modality
        sample = torch.load(destination / "prediction.pt", weights_only=True)
        runtime = load_policy(destination / "run/latest.pt", device="cpu")
        assert not runtime.compiled
        actual = runtime.predict(sample["raw"], sample["noise"])
        torch.testing.assert_close(actual, sample["actions"], rtol=1e-4, atol=1e-5)
    print(
        json.dumps({"python": sys.version.split()[0], "torch": torch.__version__, "modalities": 4})
    )


if __name__ == "__main__":
    operation, directory = sys.argv[1:]
    {"write": write_portability_fixtures, "check": check_portability_fixtures}[operation](
        Path(directory)
    )
