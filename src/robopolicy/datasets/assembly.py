"""Shared frame-request collection and episode-local sample assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from robopolicy.contracts.batch import SampleMetadata
from robopolicy.contracts.schema import FeatureSchema
from robopolicy.datasets.planning import SampleRef, feature_indices
from robopolicy.datasets.storage import NumericArrays
from robopolicy.datasets.video import DecoderPool

FrameBlock = dict[tuple[Path, int], Tensor]


def decode_block(
    block: tuple[SampleRef, ...], schema: FeatureSchema, pool: DecoderPool
) -> FrameBlock:
    """Decode each requested frame once for a single bounded sample block."""
    requests: dict[Path, set[int]] = {}
    lengths: dict[Path, int] = {}
    for reference in block:
        for feature in schema.features:
            for camera in feature.cameras:
                video = reference.episode.videos[camera]
                indices = feature_indices(reference.step, feature) + video.start
                requests.setdefault(video.path, set()).update(int(index) for index in indices)
                lengths[video.path] = video.total_frames
    frames: FrameBlock = {}
    for path, requested in requests.items():
        ordered = sorted(requested)
        decoded = pool.read(path, ordered, lengths[path])  # (N_requested, 3, H, W)
        frames.update({(path, index): frame for index, frame in zip(ordered, decoded, strict=True)})
    return frames


def assemble_sample(
    reference: SampleRef,
    schema: FeatureSchema,
    arrays: NumericArrays,
    frames: FrameBlock,
) -> dict[str, Any]:
    """Assemble owned numeric windows and RGB images without normalization.

    Returns numeric ``(T_window, *shape)`` and image
    ``(T_window, N_camera, 3, H, W)`` tensors plus opaque metadata.
    """
    sample: dict[str, Any] = {}
    episode = reference.episode
    for feature in schema.features:
        indices = feature_indices(reference.step, feature)
        if len(indices) and indices[-1] >= episode.length:
            raise ValueError(f"Future window crosses episode boundary: {episode.identity}")
        if feature.source == "hdf5":
            sample[feature.key] = torch.from_numpy(arrays[feature.key][indices].copy())
        else:
            cameras: list[Tensor] = []
            for camera in feature.cameras:
                video = episode.videos[camera]
                camera_frames = torch.stack(
                    [frames[(video.path, int(index) + video.start)] for index in indices]
                )
                # Shape: (T_window, 3, H, W).
                if tuple(camera_frames.shape[1:]) != feature.shape:
                    raise ValueError(
                        f"Image shape mismatch for camera {camera}: {tuple(camera_frames.shape[1:])} != {feature.shape}"
                    )
                cameras.append(camera_frames)
            sample[feature.key] = torch.stack(cameras, dim=1)
            # Shape: (T_window, N_camera, 3, H, W), preserving camera declaration order.
    sample["metadata"] = SampleMetadata(
        task_id=episode.task_id,
        task_name=episode.task_name,
        shard_id=episode.shard_id,
        episode_id=episode.episode_id,
        step=reference.step,
    )
    return sample
