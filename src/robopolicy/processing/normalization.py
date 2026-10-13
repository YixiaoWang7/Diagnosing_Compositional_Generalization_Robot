"""Tensor-only normalization from supplied statistics; no dataset imports."""

from __future__ import annotations

from math import isfinite, prod
from typing import Literal

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn
from typing_extensions import Self

from robopolicy.contracts.schema import FeatureSchema, StrictConfig


class FeatureStatistics(StrictConfig):
    """Flattened per-dimension affine statistics with a declared feature shape."""

    shape: tuple[int, ...]
    count: int = Field(ge=1)
    offset: tuple[float, ...]
    scale: tuple[float, ...]


class Statistics(StrictConfig):
    """Versioned normalization provenance and selected-training-demo statistics."""

    version: Literal[1] = 1
    feature_schema: FeatureSchema
    dataset_fingerprint: str
    selection: tuple[tuple[str, int, str], ...]
    features: dict[str, FeatureStatistics]

    @model_validator(mode="after")
    def validate_statistics(self) -> Self:
        """Check normalization semantics without constructing tensor buffers."""
        required = {
            feature.key
            for feature in self.feature_schema.features
            if feature.normalization != "none"
        }
        if required != set(self.features):
            raise ValueError("Normalization feature keys do not match the saved feature schema")
        for key, values in self.features.items():
            expected = self.feature_schema[key].shape
            if values.shape != expected:
                raise ValueError(f"Statistics shape mismatch for {key}")
            if len(values.offset) != prod(expected) or len(values.scale) != prod(expected):
                raise ValueError(f"Statistics length mismatch for {key}")
            if not all(isfinite(value) for value in values.offset) or not all(
                isfinite(value) and value > 0 for value in values.scale
            ):
                raise ValueError(f"Statistics for {key} must be finite with positive scales")
        return self


class Normalizer(nn.Module):
    """Apply reversible transforms using movable, nonpersistent statistics buffers.

    Attributes:
        statistics: Serializable statistics; buffers are rebuilt from these values.
    """

    def __init__(self, statistics: Statistics) -> None:
        """Validate supplied statistics before using them in training or inference."""
        super().__init__()
        self.statistics = statistics
        self._buffer_names: dict[str, tuple[str, str]] = {}
        for key, values in statistics.features.items():
            number = len(self._buffer_names)
            names = (f"offset_{number}", f"scale_{number}")
            self._buffer_names[key] = names
            # Preserve supplied precision until conversion to the input dtype.
            self.register_buffer(
                names[0],
                torch.tensor(values.offset, dtype=torch.float64).reshape(values.shape),
                persistent=False,
            )  # (*feature.shape)
            self.register_buffer(
                names[1],
                torch.tensor(values.scale, dtype=torch.float64).reshape(values.shape),
                persistent=False,
            )  # (*feature.shape)

    def transform(self, key: str, tensor: Tensor, inverse: bool = False) -> Tensor:
        """Normalize or invert ``(..., *feature.shape)`` into a fresh tensor."""
        names = self._buffer_names.get(key)
        if names is None:
            return tensor.clone()
        offset = self.get_buffer(names[0]).to(tensor)  # (*feature.shape)
        scale = self.get_buffer(names[1]).to(tensor)  # (*feature.shape)
        return tensor * scale + offset if inverse else (tensor - offset) / scale
