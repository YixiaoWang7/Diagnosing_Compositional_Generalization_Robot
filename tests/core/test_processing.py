"""Selected-demo normalization and shared, non-mutating input packing."""

import numpy as np
import pytest
import torch

from robopolicy.contracts.schema import FeatureSchema
from robopolicy.datasets.config import DatasetConfig, SelectionConfig
from robopolicy.datasets.factory import build_dataset, build_loader
from robopolicy.datasets.statistics import fit_statistics
from robopolicy.datasets.storage import iter_numeric_chunks
from robopolicy.processing.packing import InputPacker, ProcessingConfig
from tests.core.test_dataset import write_dataset
from tests.core.test_inference import inference_fixture


def test_training_only_incremental_roundtrip(tmp_path):
    schema = write_dataset(tmp_path / "data", lengths=((8, 9, 10, 11, 12),))
    config = DatasetConfig(
        root=str(tmp_path / "data"), selection=SelectionConfig(val_fraction=0.3, max_demos=2)
    )
    dataset = build_dataset(config, schema)
    validation = build_dataset(config, schema, "val")
    assert len(dataset.selection) == 2 and set(dataset.selection).isdisjoint(validation.selection)
    statistics = fit_statistics(dataset, tmp_path / "stats.json", chunk_size=2)
    values = np.concatenate(
        [chunk["state"] for chunk in iter_numeric_chunks(dataset.episodes, schema)]
    )
    np.testing.assert_allclose(statistics.features["state"].offset, values.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(statistics.features["state"].scale, values.std(axis=0), rtol=1e-6)
    assert statistics.features["state"].count == sum(episode.length for episode in dataset.episodes)
    assert "task_indices" not in statistics.features
    with pytest.raises(ValueError, match="training selection"):
        fit_statistics(validation)
    with build_loader(dataset) as loader:
        raw = next(iter(loader))
        copied = raw["state"].clone()
        packer = InputPacker(schema, statistics)
        inputs = packer(raw, require_actions=True)
        torch.testing.assert_close(packer.actions_to_raw(inputs.actions), raw["actions"])
        torch.testing.assert_close(raw["state"], copied)
        assert inputs.task_indices.dtype == torch.int64 and inputs.task_indices.ndim == 2
        observation = packer({key: value for key, value in raw.items() if key != "actions"})
        torch.testing.assert_close(inputs.state, observation.state)
        assert observation.actions is None


def test_cache_and_constant_dimensions(tmp_path):
    import h5py

    schema = write_dataset(tmp_path / "data")
    config = DatasetConfig(
        root=str(tmp_path / "data"), selection=SelectionConfig(val_fraction=0, max_demos=1)
    )
    cache = tmp_path / "stats.json"
    initial = fit_statistics(build_dataset(config, schema), cache, 2)
    assert fit_statistics(build_dataset(config, schema), cache, 3) == initial
    config2 = DatasetConfig.model_validate(
        {**config.model_dump(), "selection": {"val_fraction": 0, "max_demos": 2}}
    )
    assert fit_statistics(build_dataset(config2, schema), cache).selection != initial.selection
    episode = build_dataset(config, schema).episodes[0]
    with h5py.File(episode.path, "r+") as handle:
        handle[f"data/{episode.episode_id}/state"][:] = 42
    changed = fit_statistics(build_dataset(config, schema), cache)
    assert changed.dataset_fingerprint != initial.dataset_fingerprint
    assert changed.features["state"].offset == (42,) and changed.features["state"].scale == (1,)
    altered = FeatureSchema.model_validate(
        {
            "features": [
                {
                    **f.model_dump(),
                    "normalization": "minmax" if f.key == "state" else f.normalization,
                }
                for f in schema.features
            ]
        }
    )
    assert fit_statistics(build_dataset(config, altered), cache).feature_schema == altered


def test_signed_difference_preserves_sign_current_frame_and_caller_storage():
    runtime, raw, _ = inference_fixture(modality="images")
    raw["images"][:, 0] = torch.tensor([255, 0, 127], dtype=torch.uint8).view(1, 1, 3, 1, 1)
    raw["images"][:, 1] = torch.tensor([0, 255, 127], dtype=torch.uint8).view(1, 1, 3, 1, 1)
    raw["images"][:, :, 1] = raw["images"][:, :, 0].flip(1)
    original = raw["images"].clone()
    baseline = runtime.packer(raw).images
    config = ProcessingConfig.model_validate(
        {
            **runtime.packer.config.model_dump(),
            "image_representation": "current_and_signed_difference",
        }
    )
    packer = InputPacker(runtime.packer.schema, runtime.packer.normalizer.statistics, config)
    packed = packer(raw)
    mean = torch.tensor(config.image_mean).view(1, 1, 3, 1, 1)
    std = torch.tensor(config.image_std).view(1, 1, 3, 1, 1)
    pixels = packed.images[:, 0] * std + mean
    expected = torch.tensor([[0, 1, 0.5], [1, 0, 0.5]]).view(1, 2, 3, 1, 1)
    torch.testing.assert_close(pixels, expected.expand_as(pixels))
    torch.testing.assert_close(packed.images[:, 1], baseline[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(packer(raw, training=True).images, packed.images, rtol=0, atol=0)
    assert torch.equal(raw["images"], original)
    raw["images"][:, 0] = raw["images"][:, 1]
    neutral = packer(raw).images[:, 0] * std + mean
    torch.testing.assert_close(neutral, torch.full_like(neutral, 0.5))
    with pytest.raises(ValueError, match="exactly two"):
        packer.images(raw["images"][:, :1], training=False)


@pytest.mark.parametrize("window", [None, (0, 0), (2, 0), (0, 1)])
def test_signed_difference_rejects_missing_or_noncausal_pair(window):
    runtime, _, _ = inference_fixture(modality="images")
    values = runtime.packer.schema.model_dump()
    for feature in values["features"]:
        if feature["key"] == "images" and window is not None:
            feature["history"], feature["future"] = window
    if window is None:
        values["features"] = tuple(f for f in values["features"] if f["key"] != "images")
    schema = FeatureSchema.model_validate(values)
    config = ProcessingConfig(image_representation="current_and_signed_difference")
    with pytest.raises(ValueError, match="exactly two"):
        config.validate_schema(schema)
