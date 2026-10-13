"""Deterministic balanced budgets over opaque demonstration identities."""

from collections import Counter
from pathlib import Path

import pytest

from robopolicy.datasets.config import SelectionConfig
from robopolicy.datasets.index import ManifestEntry
from robopolicy.datasets.selection import select_entries, stable_seed


def entries(counts: tuple[int, ...]) -> tuple[ManifestEntry, ...]:
    """Build manifest entries without any storage dependencies."""
    return tuple(
        ManifestEntry(
            chr(65 + task),
            0,
            f"demo_{demo}",
            Path("data.h5"),
            2,
            {},
            Path("."),
            Path("manifest.json"),
            "opaque",
        )
        for task, count in enumerate(counts)
        for demo in range(count)
    )


@pytest.mark.parametrize(
    ("budget", "expected"), [(8, (2, 2, 2, 2)), (7, (2, 2, 2, 1)), (2, (1, 1, 0, 0))]
)
def test_balanced_total_budget(budget: int, expected: tuple[int, ...]) -> None:
    source = entries((10, 10, 10, 10))
    config = SelectionConfig(val_fraction=0, max_demos=budget, balance_demos_across_tasks=True)
    selected = select_entries(source, config, "train")
    counts = Counter(entry.task_id for entry in selected)
    assert tuple(counts[task] for task in "ABCD") == expected
    assert selected == tuple(entry for entry in source if entry in selected)


def test_balanced_caps_preserve_unfilled_quotas_and_subset_seed() -> None:
    source = entries((1, 20, 20))
    config = SelectionConfig(
        val_fraction=0,
        max_demos=12,
        max_demos_per_task=3,
        balance_demos_across_tasks=True,
    )
    selected = select_entries(source, config, "train")
    alternate = select_entries(source, config.model_copy(update={"subset_seed": 17}), "train")
    assert Counter(entry.task_id for entry in selected) == {"A": 1, "B": 3, "C": 3}
    assert Counter(entry.task_id for entry in alternate) == Counter(
        entry.task_id for entry in selected
    )
    assert selected != alternate
    assert not select_entries(
        source, config.model_copy(update={"exclude_tasks": ("A", "B", "C")}), "all"
    )


def test_default_global_ranking_and_validation_are_unchanged() -> None:
    source = entries((100, 100, 100, 100))
    config = SelectionConfig(val_fraction=0.2, max_demos=7)
    baseline = select_entries(source, config.model_copy(update={"max_demos": None}), "train")
    ranked = sorted(baseline, key=lambda entry: (stable_seed(0, *entry.identity), entry.identity))[
        :7
    ]
    assert select_entries(source, config, "train") == tuple(
        entry for entry in source if entry in ranked
    )
    balanced = config.model_copy(update={"max_demos": 3, "balance_demos_across_tasks": True})
    assert select_entries(source, config, "val") == select_entries(source, balanced, "val")
    assert not select_entries(source, balanced.model_copy(update={"val_fraction": 0}), "val")
    train_ids = {entry.identity for entry in select_entries(source, balanced, "train")}
    val_ids = {entry.identity for entry in select_entries(source, balanced, "val")}
    assert not train_ids & val_ids


def test_balancing_without_total_limit_keeps_per_task_selection() -> None:
    source = entries((5, 5))
    config = SelectionConfig(val_fraction=0, max_demos_per_task=2)
    balanced = config.model_copy(update={"balance_demos_across_tasks": True})
    assert select_entries(source, config, "all") == select_entries(source, balanced, "all")
