"""Incrementally fit normalization on the dataset's exact resolved selection."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robopolicy.datasets.factory import PolicyDataset
from robopolicy.datasets.storage import iter_numeric_chunks
from robopolicy.processing.normalization import FeatureStatistics, Normalizer, Statistics


class Moments:
    """Merge chunk moments in float64 without concatenating a dataset in memory."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Allocate per-feature ``shape`` accumulators independent of dataset size."""
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)
        self.minimum = np.full(shape, np.inf)
        self.maximum = np.full(shape, -np.inf)

    def update(self, values: NDArray[Any]) -> None:
        """Merge a ``(T_chunk, *feature.shape)`` numeric block using parallel variance."""
        if len(values) == 0:
            return
        block = values.astype(np.float64)
        count = len(block)
        mean = block.mean(axis=0)  # (*feature.shape)
        difference = mean - self.mean
        total = self.count + count
        self.m2 += ((block - mean) ** 2).sum(axis=0) + difference**2 * (self.count * count / total)
        self.mean += difference * count / total
        self.minimum = np.minimum(self.minimum, block.min(axis=0))
        self.maximum = np.maximum(self.maximum, block.max(axis=0))
        self.count = total


def fit_statistics(
    dataset: PolicyDataset, cache: Path | None = None, chunk_size: int = 4096
) -> Statistics:
    """Fit or reuse statistics for the exact training selection and source contents.

    Args:
        dataset: Resolved training dataset; validation fitting is rejected.
        cache: Optional JSON cache outside the dataset, invalidated by schema,
            selected IDs, or content fingerprint changes.
        chunk_size: Maximum temporal rows per numeric read.

    Returns:
        Fixed inference-compatible normalization values with full provenance.
    """
    if dataset.split != "train":
        raise ValueError("Normalization must be fitted on the training selection")
    identity = dataset.fingerprint
    if cache is not None and cache.exists():
        try:
            saved = Statistics.model_validate_json(cache.read_text())
            if (
                saved.feature_schema == dataset.schema
                and saved.selection == dataset.selection
                and saved.dataset_fingerprint == identity
            ):
                Normalizer(saved)
                return saved
        except (ValueError, KeyError):
            pass
    moments = {
        feature.key: Moments(feature.shape)
        for feature in dataset.schema.features
        if feature.normalization != "none"
    }
    for chunk in iter_numeric_chunks(dataset.episodes, dataset.schema, chunk_size):
        for key, chunk_values in chunk.items():
            moments[key].update(chunk_values)
    features: dict[str, FeatureStatistics] = {}
    for key, values in moments.items():
        if values.count == 0:
            raise ValueError(f"No training values available for normalization: {key}")
        if dataset.schema[key].normalization == "standard":
            offset = values.mean
            scale = np.sqrt(np.maximum(values.m2 / values.count, 0))
        else:
            offset = (values.minimum + values.maximum) / 2
            scale = (values.maximum - values.minimum) / 2
        scale = np.where(scale > 1e-12, scale, 1.0)
        features[key] = FeatureStatistics(
            shape=dataset.schema[key].shape,
            count=values.count,
            offset=tuple(offset.flat),
            scale=tuple(scale.flat),
        )
    statistics = Statistics(
        feature_schema=dataset.schema,
        dataset_fingerprint=identity,
        selection=dataset.selection,
        features=features,
    )
    Normalizer(statistics)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=cache.parent, prefix=f".{cache.name}.")
        try:
            with os.fdopen(descriptor, "w") as handle:
                json.dump(statistics.model_dump(mode="json"), handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, cache)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return statistics
