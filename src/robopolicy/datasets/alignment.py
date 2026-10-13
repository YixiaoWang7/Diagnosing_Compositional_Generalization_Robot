"""Explicit preparation utilities; runtime readers require already aligned data."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = ["align_timestamps"]


def align_timestamps(
    values: NDArray[Any],
    source_times: NDArray[np.float64],
    target_times: NDArray[np.float64],
    *,
    method: Literal["nearest", "linear"] = "nearest",
) -> NDArray[Any]:
    """Resample one modality onto supplied timestamps during dataset preparation.

    Args:
        values: Source numeric values, shape ``(T_source, *feature_shape)``.
        source_times: Finite, strictly increasing times, shape ``(T_source,)``.
        target_times: Finite, strictly increasing times, shape ``(T_target,)``.
        method: Nearest preserves integer categories and chooses the earlier
            sample on a tie. Linear accepts floating values and returns float64.

    Returns:
        Owned values of shape ``(T_target, *feature_shape)``. Neither timestamps
        nor caller-owned values are modified.

    Raises:
        ValueError: Times are unordered, lengths disagree, an extrapolation is
            requested, or categorical values would be interpolated.
    """
    source = np.asarray(source_times, dtype=np.float64)
    target = np.asarray(target_times, dtype=np.float64)
    for name, times in (("source", source), ("target", target)):
        if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"{name} timestamps must be finite and strictly increasing")
    if not len(source) or values.ndim < 1 or len(values) != len(source):
        raise ValueError("Source timestamps and modality length must match and be nonempty")
    if not np.isfinite(values).all():
        raise ValueError("Source values must be finite")
    if len(target) and (target[0] < source[0] or target[-1] > source[-1]):
        raise ValueError("Target timestamps require extrapolation outside the source interval")
    right = np.searchsorted(source, target).clip(0, len(source) - 1)
    left = (right - 1).clip(0)
    if method == "nearest":
        indices = np.where(target - source[left] <= source[right] - target, left, right)
        selected: NDArray[Any] = values[indices].copy()  # (T_target, *feature_shape)
        return selected
    if method != "linear" or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("Use nearest for categorical values; linear requires floating values")
    width = source[right] - source[left]
    weight = np.divide(target - source[left], width, out=np.zeros_like(target), where=width > 0)
    weight = weight.reshape((-1,) + (1,) * (values.ndim - 1))  # (T_target, 1, ...)
    result: NDArray[Any] = values[left] * (1 - weight) + values[right] * weight
    return result  # (T_target, *feature_shape)
