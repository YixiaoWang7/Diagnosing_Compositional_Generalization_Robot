"""HDF5 bridge for immutable reactive labels; numerical tracking stays storage-free."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

import h5py
import numpy as np
from numpy.typing import NDArray

from corl2026_cg.instructions.profiles import ActiveInstructionProfile
from corl2026_cg.instructions.tracker import InstructionTracker
from corl2026_cg.tasks import CGTask, parse_task

if TYPE_CHECKING:
    from robopolicy import RunConfig

_STATE_PARTS = (
    ("obs/robot0_eef_pos", 3),
    ("obs/robot0_eef_quat", 4),
    ("obs/robot0_gripper_qpos", 2),
)


def source_layout(group: h5py.Group, length: int) -> dict[str, Any]:
    """Validate canonical physical inputs without loading a complete trajectory."""
    parts: tuple[tuple[str, int], ...]
    if all(path in group for path, _ in _STATE_PARTS):
        parts = _STATE_PARTS
    elif "obs/state" in group:
        parts = (("obs/state", 9),)
    else:
        raise ValueError(f"Missing canonical robot state for active labels: {group.name}")
    for path, width in parts:
        node = group[path]
        if not isinstance(node, h5py.Dataset) or node.shape != (length, width):
            raise ValueError(f"Invalid active-label state shape: {group.name}/{path}")
    if "obs/object" not in group:
        raise ValueError(f"Active labels require world poses at {group.name}/obs/object")
    poses = group["obs/object"]
    if (
        not isinstance(poses, h5py.Dataset)
        or poses.ndim != 2
        or poses.shape[0] != length
        or poses.shape[1] < 56
        or poses.shape[1] % 7
    ):
        raise ValueError(f"World pose source must be (T,N*7), N>=8: {group.name}/obs/object")
    return {"state_paths": [path for path, _ in parts], "world_entities_path": "obs/object"}


def read_observations(
    group: h5py.Group, layout: dict[str, Any], start: int, stop: int
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Read bounded (T_block,9) state and (T_block,N,7) world-pose blocks."""
    states = np.concatenate(
        [np.asarray(group[path][start:stop]) for path in layout["state_paths"]], axis=1
    )
    flat_poses = np.asarray(group[layout["world_entities_path"]][start:stop])
    return states, flat_poses.reshape(stop - start, -1, 7)


def validate_label_dataset(
    group: h5py.Group, profile: ActiveInstructionProfile, length: int, layout: dict[str, Any]
) -> None:
    """Validate integer labels, frozen semantics, and recorded physical source layout."""
    if profile.path not in group:
        raise ValueError(f"Missing active labels {group.name}/{profile.path}")
    labels = group[profile.path]
    if (
        not isinstance(labels, h5py.Dataset)
        or labels.shape != (length, 1)
        or labels.dtype != np.dtype("int64")
    ):
        raise ValueError(f"Active labels must have shape (T,1) and int64 dtype: {labels.name}")
    if labels.attrs.get("profile_digest") != profile.digest:
        raise ValueError(f"Active instruction profile digest mismatch: {labels.name}")
    if labels.attrs.get("profile_json") != json.dumps(profile.to_dict(), sort_keys=True):
        raise ValueError(f"Active instruction profile definition mismatch: {labels.name}")
    if labels.attrs.get("source_layout") != json.dumps(layout, sort_keys=True):
        raise ValueError(f"Active instruction source layout mismatch: {labels.name}")
    for start in range(0, length, 4096):
        block = labels[start : start + 4096]
        if np.any((block < 0) | (block >= 12)):
            raise ValueError(f"Active instruction token outside [0,12): {labels.name}")


def add_active_labels(
    group: h5py.Group, task: CGTask, profile: ActiveInstructionProfile, length: int
) -> None:
    """Compute labels once in bounded blocks, or verify an existing prepared profile."""
    tracker = InstructionTracker(profile, task)
    layout = source_layout(group, length)
    existing = profile.path in group
    if existing:
        validate_label_dataset(group, profile, length, layout)
        labels = group[profile.path]
    else:
        labels = group.create_dataset(profile.path, (length, 1), dtype="int64")
        labels.attrs["profile_digest"] = profile.digest
        labels.attrs["profile_json"] = json.dumps(profile.to_dict(), sort_keys=True)
        labels.attrs["source_layout"] = json.dumps(layout, sort_keys=True)
    for start in range(0, length, 4096):
        stop = min(start + 4096, length)
        states, poses = read_observations(group, layout, start, stop)
        block = np.asarray(
            [
                tracker.update(state, entities)
                for state, entities in zip(states, poses, strict=True)
            ],
            dtype=np.int64,
        ).reshape(-1, 1)
        if existing:
            if not np.array_equal(labels[start:stop], block):
                raise ValueError(
                    f"Existing active labels disagree with physical observations: {labels.name}"
                )
        else:
            labels[start:stop] = block


def validate_dataset_profile(
    config: RunConfig,
    profile: ActiveInstructionProfile,
    split: Literal["train", "val", "all"] = "train",
) -> None:
    """Check only selected demonstrations against their checkpoint-visible profile.

    The native selection helper keeps validation aligned with demo split/budget
    semantics. Labels are scanned in bounded blocks without decoding video.
    """
    from pathlib import Path

    from robopolicy.datasets.index import ManifestEntry, read_manifest
    from robopolicy.datasets.selection import select_entries

    root = Path(config.dataset.root).expanduser().resolve()
    entries = select_entries(read_manifest(root), config.dataset.selection, split)
    grouped: dict[Path, list[ManifestEntry]] = {}
    for entry in entries:
        if parse_task(entry.task_id).family != profile.family:
            raise ValueError(
                f"Profile {profile.name} does not support selected task {entry.task_id}"
            )
        grouped.setdefault(entry.path, []).append(entry)
    for path, selected in grouped.items():
        with h5py.File(path, "r") as handle:
            for entry in selected:
                group = handle[f"data/{entry.episode_id}"]
                length = int(group["actions"].shape[0])
                layout = source_layout(group, length)
                validate_label_dataset(group, profile, length, layout)
