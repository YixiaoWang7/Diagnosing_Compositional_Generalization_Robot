"""Incremental numeric storage readers shared by loading and statistics fitting."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robopolicy.contracts.schema import Feature, FeatureSchema
from robopolicy.datasets.index import Episode

NumericArrays = dict[str, NDArray[Any]]


def read_feature(demo: Any, feature: Feature, rows: slice) -> NDArray[Any]:
    """Read requested ``(T, *shape)`` rows without modifying cached arrays."""
    parts = [np.asarray(demo[path][rows]) for path in feature.paths]
    if feature.dtype == "int64" and any(
        not np.issubdtype(part.dtype, np.integer) for part in parts
    ):
        raise ValueError(f"Categorical source {feature.key} must have an integer storage dtype")
    values = np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]
    values = values.astype(feature.dtype, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in feature {feature.key}")
    return values


def load_window(
    episodes: tuple[Episode, ...], schema: FeatureSchema
) -> dict[tuple[str, int, str], NumericArrays]:
    """Load selected arrays for one shard window and close every HDF5 handle."""
    import h5py

    arrays: dict[tuple[str, int, str], NumericArrays] = {}
    grouped: dict[Path, list[Episode]] = {}
    for episode in episodes:
        grouped.setdefault(episode.path, []).append(episode)
    for path, group in grouped.items():
        with h5py.File(path, "r") as handle:
            for episode in group:
                demo = handle[f"data/{episode.episode_id}"]
                arrays[episode.identity] = {
                    feature.key: read_feature(demo, feature, slice(None))
                    for feature in schema.features
                    if feature.source == "hdf5"
                }
    return arrays


def iter_numeric_chunks(
    episodes: tuple[Episode, ...], schema: FeatureSchema, chunk_size: int = 4096
) -> Iterator[NumericArrays]:
    """Yield bounded chunks from exactly the resolved demos for statistics.

    Args:
        episodes: Selected training episodes, each visited once.
        schema: Features with explicit per-feature normalization settings.
        chunk_size: Maximum time rows retained per episode and feature.

    Yields:
        Numeric arrays shaped ``(T_chunk, *feature.shape)``.
    """
    import h5py

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    for episode in episodes:
        with h5py.File(episode.path, "r") as handle:
            demo = handle[f"data/{episode.episode_id}"]
            for start in range(0, episode.length, chunk_size):
                yield {
                    feature.key: read_feature(demo, feature, slice(start, start + chunk_size))
                    for feature in schema.features
                    if feature.normalization != "none"
                }
