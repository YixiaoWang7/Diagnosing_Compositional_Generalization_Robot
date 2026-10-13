"""Dataset preparation preserves sources and feeds native RoboPolicy readers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
from robopolicy import (
    DatasetConfig,
    Feature,
    FeatureSchema,
    LoaderConfig,
    build_dataset,
    build_loader,
    dataset_identity,
)

from corl2026_cg.data import prepare_dataset

SLUG = "cross_into_bin__cube_into_mug"


def make_source(root: Path, slug: str = SLUG, entities: int = 8) -> Path:
    """Create aligned numeric episodes and opaque video bytes with sidecars."""
    task = root / slug
    task.mkdir(parents=True)
    demos = []
    segments = []
    offset = 0
    with h5py.File(task / "shard.h5", "w") as handle:
        for index, length in enumerate((4, 6)):
            key = f"episode-{index}"
            group = handle.create_group(f"data/{key}")
            group["actions"] = np.zeros((length, 7), dtype=np.float32)
            group["obs/state"] = np.ones((length, 9), dtype=np.float32)
            poses = np.arange(length * (entities * 7 + 3), dtype=np.float32)
            group["obs/object_rel_eef"] = poses.reshape(length, entities * 7 + 3)
            demos.append({"demo_key": key, "length": length})
            segments.append({"demo_key": key, "frame_start": offset, "frame_count": length})
            offset += length
    (task / "front.mp4").write_bytes(b"immutable-video-placeholder")
    (task / "front.json").write_text(json.dumps({"frames": offset, "segments": segments}))
    (task / "manifest.json").write_text(
        json.dumps(
            {
                "task": slug,
                "shards": [
                    {
                        "shard_id": 0,
                        "hdf5": "shard.h5",
                        "demos": demos,
                        "videos": {"front": {"mp4": "front.mp4", "sidecar": "front.json"}},
                    }
                ],
            }
        )
    )
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "tasks": [{"task_slug": slug, "manifest": f"{slug}/manifest.json"}],
            }
        )
    )
    return root


def snapshot(root: Path) -> dict[str, str]:
    """Hash source files so tests detect accidental in-place edits."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_prepare_feeds_native_partial_query_and_entity_batches(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source")
    (source / SLUG / "shard.h5").chmod(0o444)
    before = snapshot(source)
    destination = prepare_dataset(
        source,
        tmp_path / "prepared",
        entity_path="obs/object_rel_eef",
        entity_count=8,
    )
    assert snapshot(source) == before
    assert (source / SLUG / "shard.h5").stat().st_ino != (
        destination / SLUG / "shard.h5"
    ).stat().st_ino
    assert (source / SLUG / "front.mp4").stat().st_ino == (
        destination / SLUG / "front.mp4"
    ).stat().st_ino
    schema = FeatureSchema(
        features=(
            Feature(key="state", shape=(9,), paths=("obs/state",)),
            Feature(key="actions", shape=(7,), paths=("actions",)),
            Feature(
                key="task_indices",
                shape=(3,),
                dtype="int64",
                paths=(
                    "cg/task/object0",
                    "cg/task/object1",
                    "cg/task/container1",
                ),
            ),
            Feature(key="entities", shape=(8, 7), paths=("cg/entities",)),
            Feature(
                key="entity_indices",
                shape=(3,),
                dtype="int64",
                paths=(
                    "cg/entity/object0",
                    "cg/entity/object1",
                    "cg/entity/container1",
                ),
            ),
        )
    )
    config = DatasetConfig(root=str(destination), selection={"val_fraction": 0})
    identity = dataset_identity(config, schema)
    assert len(identity.selection) == 2
    with build_loader(build_dataset(config, schema), LoaderConfig(batch_size=3)) as loader:
        batch = next(iter(loader))
    assert batch["task_indices"].shape == (3, 1, 3)
    assert batch["task_indices"].tolist() == [[[0, 1, 1]]] * 3
    assert batch["entity_indices"].tolist() == [[[0, 1, 5]]] * 3
    assert batch["entities"].shape == (3, 1, 8, 7)
    provenance = json.loads((destination / "preparation.json").read_text())
    assert provenance["status"] == "complete"
    assert provenance["entity_path"] == "obs/object_rel_eef"
    assert provenance["task_values"][SLUG]["constant"] == 0
    assert {item["path"]: item["sha256"] for item in provenance["source_files"]} == before
    moved = tmp_path / "relocated"
    destination.rename(moved)
    assert dataset_identity(config.model_copy(update={"root": str(moved)}), schema) == identity


def test_l5_entity_ids_and_video_copy_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slug = "milk_into_mug_no_handle_press_yellow"
    source = make_source(tmp_path / "source", slug, entities=12)

    def unavailable_link(source: Path, destination: Path) -> None:
        """Simulate a cross-device export."""
        raise OSError("cross-device")

    monkeypatch.setattr(os, "link", unavailable_link)
    destination = prepare_dataset(
        source,
        tmp_path / "prepared",
        entity_path="obs/object_rel_eef",
        entity_count=12,
    )
    with h5py.File(destination / slug / "shard.h5") as handle:
        group = handle["data/episode-0"]
        assert group["cg/task/button"][:, 0].tolist() == [3] * 4
        assert group["cg/entity/button"][:, 0].tolist() == [11] * 4
        assert group["cg/entity/container0"][:, 0].tolist() == [7] * 4
    assert (destination / slug / "front.mp4").read_bytes() == (
        source / slug / "front.mp4"
    ).read_bytes()
    assert (destination / slug / "front.mp4").stat().st_ino != (
        source / slug / "front.mp4"
    ).stat().st_ino


@pytest.mark.parametrize("failure", ["numeric", "video", "external_link", "entity_count"])
def test_failed_export_is_not_published_and_preserves_source(tmp_path: Path, failure: str) -> None:
    source = make_source(tmp_path / "source")
    if failure in {"numeric", "external_link"}:
        with h5py.File(source / SLUG / "shard.h5", "a") as handle:
            group = handle["data/episode-0"]
            if failure == "numeric":
                group["misaligned"] = np.zeros((3, 1))
            else:
                group["external"] = h5py.ExternalLink("outside.h5", "/data")
    if failure == "video":
        path = source / SLUG / "front.json"
        sidecar = json.loads(path.read_text())
        sidecar["segments"][0]["frame_count"] = 3
        path.write_text(json.dumps(sidecar))
    before = snapshot(source)
    destination = tmp_path / "prepared"
    kwargs = (
        {"entity_path": "obs/object_rel_eef", "entity_count": 4}
        if failure == "entity_count"
        else {}
    )
    with pytest.raises(ValueError):
        prepare_dataset(source, destination, **kwargs)
    assert not destination.exists()
    assert not list(tmp_path.glob(".prepared*"))
    assert snapshot(source) == before


def test_rejects_path_escape_existing_destinations_and_interrupted_lock(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source")
    destination = tmp_path / "prepared"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        prepare_dataset(source, destination)
    destination.rmdir()
    lock = tmp_path / ".prepared.prepare.lock"
    lock.write_text("interrupted export")
    with pytest.raises(FileExistsError):
        prepare_dataset(source, destination)
    assert lock.read_text() == "interrupted export"
    lock.unlink()
    with pytest.raises(ValueError, match="disjoint"):
        prepare_dataset(source, source / "prepared")
    path = source / SLUG / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["shards"][0]["hdf5"] = "../dataset_manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="outside"):
        prepare_dataset(source, destination)


def test_state_only_export_needs_no_entity_or_video_dependencies(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source", "cross_into_bin")
    manifest_path = source / "cross_into_bin/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0].pop("videos")
    manifest_path.write_text(json.dumps(manifest))
    destination = prepare_dataset(source, tmp_path / "prepared")
    with h5py.File(destination / "cross_into_bin/shard.h5") as handle:
        assert "cg/entities" not in handle["data/episode-0"]
        assert handle["data/episode-0/cg/task/constant"][:, 0].tolist() == [0] * 4
