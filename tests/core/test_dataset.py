"""Behavioral dataset checks using generated numeric data and real encoded video."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from robopolicy.contracts.schema import Feature, FeatureSchema
from robopolicy.datasets.config import (
    DatasetConfig,
    LoaderConfig,
    SamplingConfig,
    SelectionConfig,
    VideoConfig,
)
from robopolicy.datasets.factory import build_dataset, build_loader
from robopolicy.datasets.planning import feature_indices, partition_bounds


def write_dataset(
    root: Path,
    lengths: tuple[tuple[int, ...], ...] = ((7, 4), (8,), (2,)),
    video: bool = False,
    mismatch: bool = False,
) -> FeatureSchema:
    """Write tiny aligned shards; numeric values and RGB colors encode identity."""
    import h5py

    task = root / "task"
    task.mkdir(parents=True, exist_ok=True)
    shards = []
    for shard_id, episode_lengths in enumerate(lengths):
        demos = []
        with h5py.File(task / f"{shard_id}.h5", "w") as handle:
            for episode_id, length in enumerate(episode_lengths):
                name = f"demo_{episode_id}"
                demos.append({"demo_key": name, "length": length})
                group = handle.create_group(f"data/{name}")
                values = (shard_id * 50 + episode_id * 20 + np.arange(length)).astype(np.float32)[
                    :, None
                ]
                group["state"] = values
                group["actions"] = np.concatenate([values, values[-1:]]) if mismatch else values
                group["indices"] = np.zeros((length, 2), dtype=np.int64)
        videos = {}
        if video:
            import cv2

            for camera_id, camera in enumerate(("left", "right")):
                path = task / f"{shard_id}-{camera}.avi"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 10, (16, 16))
                assert writer.isOpened()
                segments = []
                offset = 0
                for episode_id, length in enumerate(episode_lengths):
                    segments.append(
                        {
                            "demo_key": f"demo_{episode_id}",
                            "frame_start": offset,
                            "frame_count": length,
                        }
                    )
                    for step in range(length):
                        rgb = np.empty((16, 16, 3), dtype=np.uint8)
                        rgb[:] = [shard_id * 50 + episode_id * 20 + step, camera_id * 80, 200]
                        writer.write(np.ascontiguousarray(rgb[..., ::-1]))
                    offset += length
                writer.release()
                sidecar = f"{shard_id}-{camera}.json"
                (task / sidecar).write_text(json.dumps({"frames": offset, "segments": segments}))
                videos[camera] = {"mp4": path.name, "sidecar": sidecar}
        shards.append(
            {"shard_id": shard_id, "hdf5": f"{shard_id}.h5", "demos": demos, "videos": videos}
        )
    (task / "manifest.json").write_text(json.dumps({"shards": shards}))
    (root / "dataset_manifest.json").write_text(
        json.dumps({"tasks": [{"task_slug": "opaque-id", "manifest": "task/manifest.json"}]})
    )
    features = [
        Feature(key="state", shape=(1,), paths=("state",), history=2, normalization="standard"),
        Feature(key="actions", shape=(1,), paths=("actions",), future=2, normalization="minmax"),
        Feature(key="task_indices", shape=(2,), paths=("indices",), dtype="int64"),
    ]
    if video:
        features.append(
            Feature(
                key="images",
                shape=(3, 16, 16),
                source="video",
                cameras=("right", "left"),
                dtype="uint8",
                history=2,
            )
        )
    return FeatureSchema(features=tuple(features))


@pytest.mark.parametrize("total", [0, 1, 2, 7, 11])
@pytest.mark.parametrize("count", [1, 2, 5, 20])
def test_partitions_are_disjoint_exhaustive(total: int, count: int) -> None:
    parts = [list(range(*partition_bounds(total, part, count))) for part in range(count)]
    assert [index for part in parts for index in part] == list(range(total))
    assert max(map(len, parts)) - min(map(len, parts)) <= 1


@pytest.mark.parametrize(
    ("step", "history", "future", "expected"),
    [
        (0, 2, 0, [0, 0, 0]),
        (1, 2, 0, [0, 0, 1]),
        (2, 4, 1, [0, 0, 0, 1, 2, 3]),
        (2, 2, 0, [0, 1, 2]),
        (3, 2, 2, [1, 2, 3, 4, 5]),
        (0, 0, 3, [0, 1, 2, 3]),
        (4, 0, 0, [4]),
    ],
)
def test_history_contract(step: int, history: int, future: int, expected: list[int]) -> None:
    feature = Feature(key="state", shape=(1,), paths=("state",), history=history, future=future)
    indices = feature_indices(step, feature)
    np.testing.assert_array_equal(indices, expected)
    assert indices.dtype == np.int64
    assert indices.shape == (feature.window_size,)
    assert indices[history] == step
    assert np.all(np.diff(indices) >= 0)


@pytest.mark.parametrize("decoder", ["opencv", "torchcodec"])
@pytest.mark.parametrize("strategy", ["window_major", "global_pass"])
@pytest.mark.parametrize("partitions", [1, 3, 20])
def test_all_decoder_sampler_combinations(
    tmp_path: Path, decoder: str, strategy: str, partitions: int
) -> None:
    schema = write_dataset(tmp_path, video=True)
    config = DatasetConfig(
        root=str(tmp_path),
        video=VideoConfig(decoder=decoder, max_handles=1),
        selection=SelectionConfig(val_fraction=0),
        sampling=SamplingConfig(strategy=strategy, k_passes=partitions, block_size=3),
    )
    dataset = build_dataset(config, schema)
    samples = list(dataset)
    identities = [
        (
            sample["metadata"]["shard_id"],
            sample["metadata"]["episode_id"],
            sample["metadata"]["step"],
        )
        for sample in samples
    ]
    assert len(identities) == len(set(identities)) == len(dataset) == 13
    assert dataset.resources["live_handles"] == 0
    assert dataset.resources["peak_handles"] == 1
    for sample in samples:
        assert sample["images"].shape == (3, 2, 3, 16, 16)
        assert sample["images"].dtype == torch.uint8
        assert sample["task_indices"].dtype == torch.int64
        assert torch.equal(sample["images"][:, 0, 0, 0, 0].float(), sample["state"][:, 0])
        assert torch.equal(sample["images"][:, 1, 0, 0, 0].float(), sample["state"][:, 0])
        assert torch.all(sample["images"][:, 0, 1] == 80)
        assert torch.all(sample["images"][:, 1, 1] == 0)
        assert torch.all(sample["images"][:, :, 2] == 200)
        assert sample["actions"][-1, 0] == sample["state"][-1, 0] + 2
    assert list(dataset)[0]["metadata"] == samples[0]["metadata"]


@pytest.mark.parametrize("workers", [0, 2, 5])
@pytest.mark.parametrize("strategy", ["window_major", "global_pass"])
def test_worker_epochs_and_early_cleanup(tmp_path: Path, workers: int, strategy: str) -> None:
    schema = write_dataset(tmp_path)
    dataset = build_dataset(
        DatasetConfig(
            root=str(tmp_path),
            selection=SelectionConfig(val_fraction=0),
            sampling=SamplingConfig(strategy=strategy, k_passes=4, block_size=2),
        ),
        schema,
    )
    with build_loader(dataset, LoaderConfig(batch_size=4, num_workers=workers)) as loader:
        orders = []
        for epoch in (0, 1, 0):
            dataset.set_epoch(epoch)
            values = [value for batch in loader for value in batch["state"][:, -1, 0].tolist()]
            assert len(values) == len(set(values)) == len(dataset)
            orders.append(values)
        assert orders[0] == orders[2]
        next(iter(loader))
    assert dataset.resources["live_handles"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        iter(loader)


def test_alignment_failure_and_state_only_decoder_independence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = write_dataset(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="Unaligned modalities"):
        build_dataset(DatasetConfig(root=str(tmp_path)), schema)
    write_dataset(tmp_path, mismatch=False)

    def unavailable(*args: object) -> None:
        raise AssertionError("State-only loading must not instantiate or require video")

    monkeypatch.setattr("robopolicy.datasets.factory.require_decoder", unavailable)
    dataset = build_dataset(
        DatasetConfig(root=str(tmp_path), selection=SelectionConfig(val_fraction=0)), schema
    )
    assert len(list(dataset)) == 13


def test_decode_exception_releases_handles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    schema = write_dataset(tmp_path, video=True)
    from robopolicy.datasets.video import DecoderHandle

    def fail(*args: object) -> None:
        raise RuntimeError("injected decode error")

    monkeypatch.setattr(DecoderHandle, "read", fail)
    dataset = build_dataset(
        DatasetConfig(root=str(tmp_path), selection=SelectionConfig(val_fraction=0)), schema
    )
    with pytest.raises(RuntimeError, match="injected decode"):
        list(dataset)
    assert dataset.resources["live_handles"] == 0
