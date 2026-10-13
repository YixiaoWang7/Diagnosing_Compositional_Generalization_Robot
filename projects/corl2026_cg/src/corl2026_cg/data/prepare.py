"""Materialize CG conditioning once, outside the policy library's data path."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from corl2026_cg.data.active import add_active_labels
from corl2026_cg.instructions.profiles import ActiveInstructionProfile, get_profile
from corl2026_cg.tasks import CONTAINERS, OBJECTS, parse_task, semantic_values


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object with a useful error for a malformed manifest."""
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _source_path(root: Path, relative: str) -> Path:
    """Resolve a regular source inside the declaring manifest directory."""
    path = Path(relative)
    if path.is_absolute():
        raise ValueError(f"Source paths must be relative: {relative}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Missing source or path outside manifest directory: {relative}")
    return resolved


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write deterministic readable metadata inside an unpublished export."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _hash_file(path: Path) -> str:
    """Hash a source file with bounded memory for export provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _check_hdf5_storage(group: h5py.Group, visited: set[int]) -> None:
    """Reject external storage and links that escape content fingerprinting."""
    identity = hash(group.id)
    if identity in visited:
        return
    visited.add(identity)
    for name in group:
        if not isinstance(group.get(name, getlink=True), h5py.HardLink):
            raise ValueError(f"HDF5 links must be local hard links: {group.name}/{name}")
        node = group[name]
        if isinstance(node, h5py.Group):
            _check_hdf5_storage(node, visited)
        elif node.is_virtual or node.external:
            raise ValueError(f"External HDF5 storage is unsupported: {node.name}")


def _episode_length(group: h5py.Group, declared: Any) -> int:
    """Validate aligned temporal numeric arrays and return the action length."""
    if "actions" not in group or not isinstance(group["actions"], h5py.Dataset):
        raise ValueError(f"Missing actions in {group.name}")
    actions = group["actions"]
    if actions.ndim != 2 or actions.shape[0] < 1:
        raise ValueError(f"Actions require nonempty (T, D_action): {actions.name}")
    length = int(actions.shape[0])
    if declared is not None and int(declared) != length:
        raise ValueError(f"Unaligned manifest length for {group.name}: {declared} != {length}")

    def check(name: str, node: h5py.Group | h5py.Dataset) -> None:
        """Check stored numeric time axes without loading the arrays."""
        if isinstance(node, h5py.Dataset) and node.ndim and np.issubdtype(node.dtype, np.number):
            if node.shape[0] != length:
                raise ValueError(f"Unaligned numeric source {group.name}/{name}")

    group.visititems(check)
    return length


def _constant(group: h5py.Group, path: str, length: int, value: int) -> None:
    """Store an implicit constant int64 array with shape (T, 1)."""
    if path in group:
        stored = group[path]
        if stored.shape != (length, 1) or stored.dtype != np.dtype("int64"):
            raise ValueError(f"Invalid prepared categorical feature: {stored.name}")
        for start in range(0, length, 4096):
            if not np.all(stored[start : start + 4096] == value):
                raise ValueError(f"Prepared categorical values disagree with task: {stored.name}")
        return
    group.create_dataset(path, shape=(length, 1), dtype="int64", fillvalue=value)


def _add_conditioning(
    group: h5py.Group,
    values: dict[str, int],
    length: int,
    entity_path: str | None,
    entity_count: int | None,
    allow_prepared: bool = False,
) -> None:
    """Write semantic IDs and optional explicit (T, N_entity, 7) poses."""
    if "cg" in group:
        if not allow_prepared:
            raise ValueError(f"Reserved CG features already exist: {group.name}/cg")
        if set(group["cg"]) - {"task", "entities", "entity", "active_instruction"}:
            raise ValueError(f"Unknown prepared CG fields: {group.name}/cg")
        if "cg/task" not in group or set(group["cg/task"]) != set(values):
            raise ValueError(f"Prepared CG task fields disagree with task: {group.name}/cg/task")
    for field, value in values.items():
        _constant(group, f"cg/task/{field}", length, value)
    if "cg/entities" in group:
        existing = group["cg/entities"]
        if existing.ndim != 3 or existing.shape[0] != length or existing.shape[2] != 7:
            raise ValueError(f"Invalid prepared entity shape: {existing.name}")
        if entity_count is not None and existing.shape[1] != entity_count:
            raise ValueError(
                f"Prepared entity count disagrees with requested count: {existing.name}"
            )
        for start in range(0, length, 4096):
            if not np.isfinite(existing[start : start + 4096]).all():
                raise ValueError(f"Non-finite prepared entities: {existing.name}")
        entity_count = int(existing.shape[1])
    if entity_count is None:
        return
    if entity_path is None and "cg/entities" not in group:
        return
    if entity_path is not None:
        _prepare_entities(group, entity_path, entity_count, length)
    for field, value in values.items():
        if field == "constant":
            continue
        if field.startswith("object"):
            offset = 0
        elif field.startswith("container"):
            offset = len(OBJECTS)
        else:
            offset = len(OBJECTS) + len(CONTAINERS)
        index = value + offset
        if index >= entity_count:
            raise ValueError(
                f"Entity index {index} for {field} exceeds entity_count={entity_count}"
            )
        _constant(group, f"cg/entity/{field}", length, index)


def _prepare_entities(group: h5py.Group, entity_path: str, entity_count: int, length: int) -> None:
    """Copy explicit poses or verify existing poses against their declared source."""
    if entity_path not in group or not isinstance(group[entity_path], h5py.Dataset):
        raise ValueError(f"Missing entity source {group.name}/{entity_path}")
    source = group[entity_path]
    if source.ndim != 2 or source.shape[0] != length or source.shape[1] < entity_count * 7:
        raise ValueError(f"Entity source must have shape (T, >= N_entity*7): {source.name}")
    existed = "cg/entities" in group
    entities = (
        group["cg/entities"]
        if existed
        else group.create_dataset(
            "cg/entities",
            shape=(length, entity_count, 7),
            dtype="float32",
            chunks=(min(length, 4096), entity_count, 7),
            compression="gzip",
        )
    )
    for start in range(0, length, 4096):
        end = min(length, start + 4096)
        poses = np.asarray(source[start:end, : entity_count * 7], dtype=np.float32)
        if not np.isfinite(poses).all():
            raise ValueError(f"Non-finite entity poses: {source.name}")
        shaped = poses.reshape(end - start, entity_count, 7)
        # Shape: (T_chunk, N_entity, 7), preserving the source coordinate frame.
        if existed:
            if not np.array_equal(entities[start:end], shaped):
                raise ValueError(f"Prepared entities disagree with source: {entities.name}")
        else:
            entities[start:end] = shaped


def _copy_shared(source: Path, destination: Path) -> None:
    """Reuse immutable videos without copying bytes when a hard link is possible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _prepare_task(
    source: Path,
    staging: Path,
    task: dict[str, Any],
    source_files: set[Path],
    entity_path: str | None,
    entity_count: int | None,
    active_profile: ActiveInstructionProfile | None,
) -> tuple[dict[str, str], dict[str, int]]:
    """Prepare one task's numeric shards, sidecars, and shared video files."""
    slug = str(task["task_slug"])
    values = dict(semantic_values(parse_task(slug)))
    manifest_path = _source_path(source, str(task["manifest"]))
    source_files.add(manifest_path)
    manifest = _read_json(manifest_path)
    task_root = manifest_path.parent
    output_root = staging / slug
    output_root.mkdir()
    seen_shards: set[int] = set()
    seen_sources: set[Path] = set()
    copied_media: set[Path] = set()
    episode_count = 0
    for shard in manifest.get("shards", []):
        shard_id = int(shard["shard_id"])
        if shard_id in seen_shards:
            raise ValueError(f"Duplicate shard {slug}/{shard_id}")
        seen_shards.add(shard_id)
        numeric = _source_path(task_root, str(shard["hdf5"]))
        if numeric in seen_sources:
            raise ValueError(f"A numeric source is referenced by multiple shards: {numeric}")
        seen_sources.add(numeric)
        source_files.add(numeric)
        relative = numeric.relative_to(task_root)
        copied = output_root / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(numeric, "r") as handle:
            _check_hdf5_storage(handle, set())
        shutil.copy2(numeric, copied)
        copied.chmod(copied.stat().st_mode | stat.S_IWUSR)
        shard["hdf5"] = relative.as_posix()
        lengths: dict[str, int] = {}
        with h5py.File(copied, "r+") as handle:
            for demo in shard.get("demos", []):
                key = str(demo["demo_key"])
                if not key or key in lengths or f"data/{key}" not in handle:
                    raise ValueError(f"Missing or duplicate episode {slug}/{shard_id}/{key}")
                group = handle[f"data/{key}"]
                length = _episode_length(group, demo.get("length"))
                _add_conditioning(
                    group, values, length, entity_path, entity_count, active_profile is not None
                )
                if active_profile is not None:
                    add_active_labels(group, parse_task(slug), active_profile, length)
                lengths[key] = length
                demo["length"] = length
                episode_count += 1
        for camera, video in shard.get("videos", {}).items():
            sidecar_path = _source_path(task_root, str(video["sidecar"]))
            sidecar = _read_json(sidecar_path)
            total = int(sidecar["frames"])
            segments: dict[str, tuple[int, int]] = {}
            for segment in sidecar.get("segments", []):
                key = str(segment["demo_key"])
                start, count = int(segment["frame_start"]), int(segment["frame_count"])
                if key in segments or min(start, count, total) < 0 or start + count > total:
                    raise ValueError(f"Invalid or duplicate video segment: {sidecar_path}")
                segments[key] = start, count
            for key, length in lengths.items():
                if key not in segments or segments[key][1] != length:
                    raise ValueError(f"Unaligned video {slug}/{shard_id}/{key} camera={camera}")
            for field in ("mp4", "sidecar"):
                media = _source_path(task_root, str(video[field]))
                source_files.add(media)
                relative = media.relative_to(task_root)
                destination = output_root / relative
                if destination not in copied_media:
                    if destination.exists():
                        raise ValueError(f"Conflicting source paths: {destination}")
                    if field == "mp4":
                        _copy_shared(media, destination)
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(media, destination)
                    copied_media.add(destination)
                video[field] = relative.as_posix()
    if not episode_count:
        raise ValueError(f"No episodes in task {slug}")
    if (output_root / "manifest.json").exists():
        raise ValueError(f"Source path conflicts with output manifest: {slug}")
    _write_json(output_root / "manifest.json", manifest)
    return {"task_slug": slug, "manifest": f"{slug}/manifest.json"}, values


def prepare_dataset(
    source: Path,
    destination: Path,
    *,
    entity_path: str | None = None,
    entity_count: int | None = None,
    active_profile: str | None = None,
) -> Path:
    """Export an aligned dataset with explicit task and entity conditioning.

    Args:
        source: Existing small-files dataset root. Original files are never edited.
        destination: New dataset root, published atomically after full preparation.
        entity_path: Optional flat HDF5 pose source relative to each episode.
        entity_count: Number of leading seven-dimensional poses to retain. Pose
            coordinates are preserved; inference must use the same convention.
        active_profile: Optional immutable reactive-instruction profile name.
            Prepared inputs are verified before adding labels to a fresh export.

    Returns:
        Absolute prepared dataset root containing dataset_manifest.json and
        preparation.json. Videos are hard-linked when possible and must remain
        immutable; HDF5 and metadata files are independently copied.

    Raises:
        ValueError: Source modalities are unaligned or paths/conditioning invalid.
        FileExistsError: Destination exists or an interrupted export lock remains.
    """
    source = Path(source).expanduser().resolve(strict=True)
    destination = Path(destination).expanduser().resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination must be disjoint dataset directories")
    if (entity_path is None) != (entity_count is None) or (
        entity_count is not None and entity_count < 1
    ):
        raise ValueError("entity_path and positive entity_count must be supplied together")
    if entity_path is not None and (entity_path.startswith("/") or ".." in Path(entity_path).parts):
        raise ValueError("entity_path must be relative to each episode")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    source_manifest = source / "dataset_manifest.json"
    manifest = _read_json(source_manifest)
    profile = get_profile(active_profile) if active_profile is not None else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / f".{destination.name}.prepare.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"source={source}\ndestination={destination}\n")
    staging: Path | None = None
    try:
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.incomplete-", dir=destination.parent)
        )
        source_files = {source_manifest}
        tasks: list[dict[str, str]] = []
        task_values: dict[str, dict[str, int]] = {}
        for task in manifest.get("tasks", []):
            slug = str(task["task_slug"])
            if slug in task_values:
                raise ValueError(f"Duplicate task ID: {slug}")
            prepared, values = _prepare_task(
                source, staging, task, source_files, entity_path, entity_count, profile
            )
            tasks.append(prepared)
            task_values[slug] = values
        if not tasks:
            raise ValueError("No tasks in source dataset manifest")
        manifest["tasks"] = tasks
        _write_json(staging / "dataset_manifest.json", manifest)
        _write_json(
            staging / "preparation.json",
            {
                "version": 1,
                "exporter": "corl2026_cg",
                "status": "complete",
                "source_root": str(source),
                "entity_path": entity_path,
                "entity_count": entity_count,
                "active_profile": (
                    {"definition": profile.to_dict(), "digest": profile.digest}
                    if profile is not None
                    else None
                ),
                "task_values": task_values,
                "source_files": [
                    {
                        "path": path.relative_to(source).as_posix(),
                        "sha256": _hash_file(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in sorted(source_files)
                ],
            },
        )
        if destination.exists():
            raise FileExistsError(f"Destination appeared during preparation: {destination}")
        staging.rename(destination)
        return destination
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        lock.unlink()
