"""Serializable feature declarations shared by storage and tensor processing."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class StrictConfig(BaseModel):
    """Immutable component settings that reject unknown configuration keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Feature(StrictConfig):
    """Declare one per-step feature and its inclusive temporal window.

    Attributes:
        key: Canonical batch key, independent of task metadata.
        shape: Per-step numeric shape, or image shape ``(3, H, W)``.
        source: Storage reader to use.
        paths: HDF5 paths relative to an episode, concatenated on the last axis.
        cameras: Camera identifiers in the desired output order.
        dtype: Output tensor dtype; categorical features must be int64.
        normalization: Explicit numeric normalization method.
        history: Number of previous steps, padded with the first episode step.
        future: Number of future steps, which must exist in the same episode.
    """

    key: str = Field(min_length=1)
    shape: tuple[int, ...]
    source: Literal["hdf5", "video"] = "hdf5"
    paths: tuple[str, ...] = ()
    cameras: tuple[str, ...] = ()
    dtype: Literal["float32", "int64", "uint8"] = "float32"
    normalization: Literal["none", "standard", "minmax"] = "none"
    history: int = Field(default=0, ge=0)
    future: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_feature(self) -> Self:
        """Reject ambiguous sources, invalid shapes, and normalized categories."""
        if not self.shape or any(size < 1 for size in self.shape):
            raise ValueError("Feature shape must have positive dimensions")
        if self.source == "video":
            if len(self.shape) != 3 or self.shape[0] != 3:
                raise ValueError("Video shape must be RGB (3, H, W)")
            if not self.cameras or self.paths or self.dtype != "uint8":
                raise ValueError("Video requires cameras, dtype=uint8, and no HDF5 paths")
            if len(set(self.cameras)) != len(self.cameras):
                raise ValueError("Camera identifiers must be unique")
        elif not self.paths or self.cameras:
            raise ValueError("Numeric features require paths and no cameras")
        if self.dtype != "float32" and self.normalization != "none":
            raise ValueError("Categorical and uint8 features cannot be normalized")
        if self.key in {"task_indices", "entity_indices"}:
            if self.dtype != "int64" or self.history or self.future:
                raise ValueError("Selection indices require int64 and a current-step window")
        return self

    @property
    def window_size(self) -> int:
        """Return history plus current plus future steps."""
        return self.history + 1 + self.future


class FeatureSchema(StrictConfig):
    """Ordered feature declarations; the schema is saved with every checkpoint."""

    features: tuple[Feature, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_keys(self) -> Self:
        """Reject duplicate feature names and the reserved metadata key."""
        keys = [feature.key for feature in self.features]
        if len(keys) != len(set(keys)) or "metadata" in keys:
            raise ValueError("Feature keys must be unique and cannot be 'metadata'")
        return self

    def __getitem__(self, key: str) -> Feature:
        """Return the declaration for a canonical key, or raise KeyError."""
        for feature in self.features:
            if feature.key == key:
                return feature
        raise KeyError(key)

    @property
    def max_future(self) -> int:
        """Return the minimum future extent required for an eligible sample."""
        return max(feature.future for feature in self.features)

    @property
    def cameras(self) -> tuple[str, ...]:
        """Return unique requested cameras in declaration order."""
        return tuple(dict.fromkeys(cam for f in self.features for cam in f.cameras))
