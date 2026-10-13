"""Read and validate small-files manifests without decoding any video."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robopolicy.contracts.schema import FeatureSchema


@dataclass(frozen=True)
class VideoSource:
    """Validated video segment for one camera and episode."""

    path: Path
    start: int
    count: int
    total_frames: int


@dataclass(frozen=True)
class Episode:
    """An episode's storage locations, aligned length, and opaque identity."""

    task_id: str
    shard_id: int
    episode_id: str
    path: Path
    length: int
    videos: dict[str, VideoSource]
    task_name: str

    @property
    def identity(self) -> tuple[str, int, str]:
        """Return a location-independent key used by selection and recovery."""
        return self.task_id, self.shard_id, self.episode_id


@dataclass(frozen=True)
class ManifestEntry:
    """Unresolved episode metadata prior to selection and storage validation."""

    task_id: str
    shard_id: int
    episode_id: str
    path: Path
    length: int | None
    videos: dict[str, dict[str, str]]
    task_root: Path
    manifest_path: Path
    task_name: str

    @property
    def identity(self) -> tuple[str, int, str]:
        """Return the opaque identity, independent of absolute storage paths."""
        return self.task_id, self.shard_id, self.episode_id


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, reporting its path on invalid structure."""
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def source_path(root: Path, relative: str) -> Path:
    """Resolve an existing source file within its manifest's root."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Source path escapes dataset root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset source: {path}")
    return path


def read_manifest(root: Path) -> tuple[ManifestEntry, ...]:
    """Discover episodes and shards without reading arrays or opening decoders.

    Args:
        root: Directory containing ``dataset_manifest.json``.

    Returns:
        Entries in deterministic manifest order, with unique episode identities.
    """
    dataset = read_json(root / "dataset_manifest.json")
    entries: list[ManifestEntry] = []
    seen: set[tuple[str, int, str]] = set()
    task_ids: set[str] = set()
    for task in dataset.get("tasks", []):
        task_id = str(task["task_slug"])
        if not task_id or task_id in task_ids:
            raise ValueError(f"Duplicate or empty task identifier: {task_id!r}")
        task_ids.add(task_id)
        manifest_path = source_path(root, task["manifest"])
        manifest = read_json(manifest_path)
        shard_ids: set[int] = set()
        for shard in manifest.get("shards", []):
            shard_id = int(shard["shard_id"])
            if shard_id in shard_ids:
                raise ValueError(f"Duplicate shard {task_id}/{shard_id}")
            shard_ids.add(shard_id)
            path = source_path(manifest_path.parent, shard["hdf5"])
            for demo in shard.get("demos", []):
                entry = ManifestEntry(
                    task_id,
                    shard_id,
                    str(demo["demo_key"]),
                    path,
                    int(demo["length"]) if "length" in demo else None,
                    shard.get("videos", {}),
                    manifest_path.parent,
                    manifest_path,
                    str(demo.get("task", manifest.get("task", task_id))),
                )
                if not entry.episode_id or entry.identity in seen:
                    raise ValueError(f"Duplicate or empty episode: {entry.identity}")
                seen.add(entry.identity)
                entries.append(entry)
    if not entries:
        raise ValueError(f"No episodes in {root / 'dataset_manifest.json'}")
    return tuple(entries)


def validate_entries(
    entries: tuple[ManifestEntry, ...], schema: FeatureSchema
) -> tuple[Episode, ...]:
    """Validate selected numeric shapes and aligned video sidecar lengths.

    Args:
        entries: Already resolved selection; excluded demos are never loaded.
        schema: Required numeric per-step and RGB ``(3, H, W)`` shapes.

    Returns:
        Validated episodes, including short episodes with no eligible windows.
    """
    try:
        import h5py
    except ImportError as error:
        raise ImportError("HDF5 reading requires robopolicy[dataset]") from error

    episodes: list[Episode] = []
    sidecars: dict[Path, dict[str, Any]] = {}
    # Open each shard once during validation, even when it has many episodes.
    grouped: dict[Path, list[ManifestEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.path, []).append(entry)
    for path, group in grouped.items():
        with h5py.File(path, "r") as handle:
            for entry in group:
                episode_path = f"data/{entry.episode_id}"
                if episode_path not in handle:
                    raise ValueError(f"Missing episode {episode_path} in {path}")
                demo = handle[episode_path]
                lengths: dict[str, int] = {}
                for feature in schema.features:
                    if feature.source != "hdf5":
                        continue
                    shapes: list[tuple[int, ...]] = []
                    for key in feature.paths:
                        if key not in demo:
                            raise ValueError(f"Missing source {path}:{episode_path}/{key}")
                        shape = tuple(demo[key].shape)
                        if len(shape) < 2:
                            raise ValueError(
                                f"Feature {feature.key} must include a time and feature axis"
                            )
                        shapes.append(shape[1:])
                        lengths[f"{feature.key}:{key}"] = shape[0]
                    if any(shape[:-1] != shapes[0][:-1] for shape in shapes):
                        raise ValueError(f"Incompatible concatenation shapes for {feature.key}")
                    actual = (*shapes[0][:-1], sum(shape[-1] for shape in shapes))
                    if actual != feature.shape:
                        raise ValueError(
                            f"Shape mismatch for {feature.key}: {actual} != {feature.shape}"
                        )
                if entry.length is not None:
                    lengths["manifest"] = entry.length
                videos: dict[str, VideoSource] = {}
                for camera in schema.cameras:
                    if camera not in entry.videos:
                        raise ValueError(f"Missing camera {camera!r} for {entry.identity}")
                    source = entry.videos[camera]
                    video_path = source_path(entry.task_root, source["mp4"])
                    sidecar_path = source_path(entry.task_root, source["sidecar"])
                    if sidecar_path not in sidecars:
                        sidecars[sidecar_path] = read_json(sidecar_path)
                    sidecar = sidecars[sidecar_path]
                    segments = sidecar.get("segments", [])
                    matches = [seg for seg in segments if seg["demo_key"] == entry.episode_id]
                    if len(matches) != 1:
                        raise ValueError(
                            f"Expected one segment for {entry.identity} camera={camera}"
                        )
                    segment = matches[0]
                    if "frame_count" not in segment or "frames" not in sidecar:
                        raise ValueError(
                            f"Sidecar requires frame_count and total frames: {sidecar_path}"
                        )
                    start, count, total = (
                        int(segment["frame_start"]),
                        int(segment["frame_count"]),
                        int(sidecar["frames"]),
                    )
                    if min(start, count, total) < 0 or start + count > total:
                        raise ValueError(f"Invalid video segment bounds in {sidecar_path}")
                    videos[camera] = VideoSource(video_path, start, count, total)
                    lengths[f"video:{camera}"] = count
                if not lengths or len(set(lengths.values())) != 1:
                    raise ValueError(
                        f"Unaligned modalities for {entry.identity}: {lengths}; prepare aligned data first"
                    )
                length = next(iter(lengths.values()))
                if length < 0:
                    raise ValueError(f"Negative episode length for {entry.identity}")
                episodes.append(Episode(*entry.identity, path, length, videos, entry.task_name))
    return tuple(episodes)


def fingerprint(
    root: Path, entries: tuple[ManifestEntry, ...], episodes: tuple[Episode, ...]
) -> str:
    """Hash selected source contents and identities independently of location.

    Reads files incrementally; no dataset arrays or video frames are retained.
    The caller caches this result for the lifetime of its resolved dataset.
    """
    files = {path for entry in entries for path in (entry.path, entry.manifest_path)}
    files.add(root / "dataset_manifest.json")
    requested_cameras = {camera for episode in episodes for camera in episode.videos}
    for entry in entries:
        for camera in requested_cameras:
            source = entry.videos[camera]
            for key in ("mp4", "sidecar"):
                files.add(source_path(entry.task_root, source[key]))
    digest = hashlib.sha256()
    digest.update(json.dumps([episode.identity for episode in episodes]).encode())
    for path in sorted(files, key=lambda value: value.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()
