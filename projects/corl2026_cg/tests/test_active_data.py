"""Immutable active-label exports, prepared-input validation, and visual inspection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from corl2026_cg.data import inspect_instructions, prepare_dataset
from corl2026_cg.data.active import source_layout, validate_label_dataset
from corl2026_cg.instructions import get_profile, validate_dataset_profile

SLUG = "cross_into_bin_press_red"


def source(root: Path, *, video: bool = False) -> Path:
    """Create a pick/place demo with canonical simulator physical observations."""
    directory = root / SLUG
    directory.mkdir(parents=True)
    length = 6
    entities = np.zeros((length, 12, 7), dtype=np.float32)
    entities[:, :, 0] = np.arange(12) * 0.25
    entities[:, :, 2], entities[:, :, 6] = 0.8, 1
    state = np.repeat(
        np.array([[0, 0, 1.5, 0, 0, 0, 1, 0.04, -0.04]], dtype=np.float32), length, axis=0
    )
    state[1:3, :3], state[1:3, 7:] = entities[0, 0, :3], [0.02, -0.02]
    entities[3:, 0] = entities[3:, 4]
    with h5py.File(directory / "shard.h5", "w") as handle:
        group = handle.create_group("data/demo")
        group["actions"] = np.zeros((length, 7), dtype=np.float32)
        group["obs/robot0_eef_pos"] = state[:, :3]
        group["obs/robot0_eef_quat"] = state[:, 3:7]
        group["obs/robot0_gripper_qpos"] = state[:, 7:]
        group["obs/object"] = entities.reshape(length, -1)
        group["obs/object_rel_eef"] = entities.reshape(length, -1)
        group["timestamps"] = np.arange(length) / 20
    shard = {"shard_id": 0, "hdf5": "shard.h5", "demos": [{"demo_key": "demo", "length": length}]}
    if video:
        import imageio.v2 as imageio

        with imageio.get_writer(directory / "front.mp4", fps=20, macro_block_size=1) as writer:
            for step in range(length):
                writer.append_data(np.full((64, 64, 3), step * 30, dtype=np.uint8))
        (directory / "front.json").write_text(
            json.dumps(
                {
                    "frames": length,
                    "fps": 20,
                    "segments": [{"demo_key": "demo", "frame_start": 0, "frame_count": length}],
                }
            )
        )
        shard["videos"] = {"front": {"mp4": "front.mp4", "sidecar": "front.json"}}
    (directory / "manifest.json").write_text(json.dumps({"shards": [shard]}))
    (root / "dataset_manifest.json").write_text(
        json.dumps({"tasks": [{"task_slug": SLUG, "manifest": f"{SLUG}/manifest.json"}]})
    )
    return root


def hashes(root: Path) -> dict[str, str]:
    """Fingerprint original files to detect unexpected writes."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_prepare_active_labels_from_raw_and_prepared_preserves_every_source(tmp_path: Path) -> None:
    raw = source(tmp_path / "raw")
    before = hashes(raw)
    prepared = prepare_dataset(
        raw, tmp_path / "prepared", entity_path="obs/object_rel_eef", entity_count=12
    )
    prepared_before = hashes(prepared)
    active = prepare_dataset(prepared, tmp_path / "active", active_profile="l5_v1")
    profile = get_profile("l5_v1")
    with h5py.File(active / SLUG / "shard.h5") as handle:
        group = handle["data/demo"]
        assert group[profile.path][:, 0].tolist() == [0, 0, 4, 4, 8, 8]
        validate_label_dataset(group, profile, 6, source_layout(group, 6))
        assert group["cg/entities"].shape == (6, 12, 7)
    repeated = prepare_dataset(active, tmp_path / "repeated", active_profile="l5_v1")
    assert repeated.is_dir() and hashes(raw) == before and hashes(prepared) == prepared_before
    provenance = json.loads((active / "preparation.json").read_text())
    assert provenance["active_profile"]["digest"] == profile.digest


@pytest.mark.parametrize("corruption", ["task", "profile", "labels", "world", "layout"])
def test_corrupt_prepared_inputs_are_rejected_atomically(tmp_path: Path, corruption: str) -> None:
    raw = source(tmp_path / "raw")
    prepared = prepare_dataset(raw, tmp_path / "prepared", active_profile="l5_v1")
    profile = get_profile("l5_v1")
    with h5py.File(prepared / SLUG / "shard.h5", "a") as handle:
        group = handle["data/demo"]
        if corruption == "task":
            group["cg/task/object0"][0, 0] = 1
        elif corruption == "profile":
            group[profile.path].attrs["profile_digest"] = "different"
        elif corruption == "labels":
            group[profile.path][0, 0] = 2
        elif corruption == "world":
            del group["obs/object"]
        else:
            group[profile.path].attrs["source_layout"] = "{}"
    before = hashes(prepared)
    with pytest.raises(ValueError):
        prepare_dataset(prepared, tmp_path / "failed", active_profile="l5_v1")
    assert not (tmp_path / "failed").exists() and hashes(prepared) == before
    assert not list(tmp_path.glob(".failed*"))


def test_inspection_writes_bounded_video_and_reports_real_phase_changes(tmp_path: Path) -> None:
    raw = source(tmp_path / "raw", video=True)
    before = hashes(raw)
    summary = inspect_instructions(raw, tmp_path / "qa", samples=4)
    sample = summary["samples"][0]
    assert [item["token"] for item in sample["transitions"]] == [0, 4, 8]
    assert sample["phase_counts"] == [2, 2, 2]
    assert (tmp_path / "qa" / sample["video"]).stat().st_size > 0
    assert hashes(raw) == before


def test_selected_label_guard_matches_native_demo_budget_and_integer_batches(
    tmp_path: Path,
) -> None:
    from robopolicy import (
        DatasetConfig,
        Feature,
        FeatureSchema,
        LoaderConfig,
        build_dataset,
        build_loader,
    )
    from robopolicy.datasets.index import read_manifest
    from robopolicy.datasets.selection import select_entries

    raw = source(tmp_path / "raw")
    with h5py.File(raw / SLUG / "shard.h5", "a") as handle:
        handle.copy("data/demo", "data/demo2")
    manifest_path = raw / SLUG / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0]["demos"].append({"demo_key": "demo2", "length": 6})
    manifest_path.write_text(json.dumps(manifest))
    prepared = prepare_dataset(raw, tmp_path / "prepared", active_profile="l5_v1")
    profile = get_profile("l5_v1")
    config = DatasetConfig(root=str(prepared), selection={"val_fraction": 0, "max_demos": 1})
    entries = read_manifest(prepared)
    chosen = select_entries(entries, config.selection, "train")[0].episode_id
    excluded = next(entry.episode_id for entry in entries if entry.episode_id != chosen)
    with h5py.File(prepared / SLUG / "shard.h5", "a") as handle:
        handle[f"data/{excluded}/{profile.path}"].attrs["profile_digest"] = "bad"
    validate_dataset_profile(SimpleNamespace(dataset=config), profile)
    schema = FeatureSchema(
        features=(
            Feature(key="task_indices", shape=(1,), paths=(profile.path,), dtype="int64"),
            Feature(key="actions", shape=(7,), paths=("actions",)),
        )
    )
    with build_loader(build_dataset(config, schema), LoaderConfig(batch_size=6)) as loader:
        batch = next(iter(loader))
    assert sorted(batch["task_indices"][:, 0, 0].tolist()) == [0, 0, 4, 4, 8, 8]
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_dataset_profile(
            SimpleNamespace(
                dataset=DatasetConfig(root=str(prepared), selection={"val_fraction": 0})
            ),
            profile,
        )


def test_instruction_video_matches_training_indices_despite_conflicting_clocks(
    tmp_path: Path,
) -> None:
    import cv2

    raw = source(tmp_path / "raw", video=True)
    with h5py.File(raw / SLUG / "shard.h5", "a") as handle:
        group = handle["data/demo"]
        group["timestamps"][:] = np.arange(6) / 50
        group.attrs["timestamps_source"] = "assumed"
        group["modalities/images/front/timestamps"] = np.arange(6) / 20
    sidecar_path = raw / SLUG / "front.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["segments"][0]["timestamps_source"] = "assumed_video_fps"
    sidecar_path.write_text(json.dumps(sidecar))
    summary = inspect_instructions(raw, tmp_path / "qa", samples=1)
    sample = summary["samples"][0]
    alignment = sample["alignment"]
    assert alignment["mode"] == "step_index" and alignment["warnings"]
    assert alignment["cameras"]["front"]["max_timestamp_difference_seconds"] == pytest.approx(0.15)
    capture = cv2.VideoCapture(str(tmp_path / "qa" / sample["video"]))
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(20)
    try:
        for step in range(6):
            ok, frame = capture.read()
            assert ok
            # The bottom rows are untouched by text; every source frame must appear once.
            assert float(frame[-8:].mean()) == pytest.approx(step * 30, abs=5)
    finally:
        capture.release()


def test_instruction_inspection_rejects_unaligned_frame_counts_atomically(tmp_path: Path) -> None:
    raw = source(tmp_path / "raw", video=True)
    sidecar_path = raw / SLUG / "front.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["segments"][0]["frame_count"] = 5
    sidecar_path.write_text(json.dumps(sidecar))
    before = hashes(raw)
    with pytest.raises(ValueError, match="one video frame per numeric step"):
        inspect_instructions(raw, tmp_path / "qa", samples=1)
    assert not (tmp_path / "qa").exists() and hashes(raw) == before
    assert not list(tmp_path.glob(".qa.incomplete-*"))
