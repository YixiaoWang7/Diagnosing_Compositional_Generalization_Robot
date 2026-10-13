"""Timestamp preparation never silently aligns runtime samples."""

import numpy as np
import pytest

from robopolicy.datasets.alignment import align_timestamps


def test_preparation_interpolation_and_category_selection():
    source = np.array([0.0, 1.0, 2.0])
    target = np.array([0.0, 0.5, 1.5, 2.0])
    values = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
    expected = np.array([[0.0, 2.0], [1.0, 3.0], [3.0, 5.0], [4.0, 6.0]])
    np.testing.assert_allclose(align_timestamps(values, source, target, method="linear"), expected)
    categories = np.array([5, 9, 3], dtype=np.int64)
    aligned = align_timestamps(categories, source, target)
    np.testing.assert_array_equal(aligned, [5, 5, 9, 3])
    assert aligned.dtype == np.int64 and not np.shares_memory(aligned, categories)
    with pytest.raises(ValueError, match="categorical"):
        align_timestamps(categories, source, target, method="linear")
    assert align_timestamps(values, source, np.array([])).shape == (0, 2)


@pytest.mark.parametrize(
    "source,target",
    [
        ([0.0, 0.0], [0.0]),
        ([1.0, 0.0], [0.0]),
        ([0.0, 1.0], [-1.0]),
        ([0.0, 1.0], [2.0]),
        ([0.0, 1.0], [float("nan")]),
    ],
)
def test_preparation_rejects_ambiguous_or_outside_times(source, target):
    with pytest.raises(ValueError):
        align_timestamps(np.array([0.0, 1.0]), np.array(source), np.array(target))
