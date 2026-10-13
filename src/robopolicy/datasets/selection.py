"""Resolve task identifiers, split membership, and demo limits exactly once."""

from __future__ import annotations

import hashlib
from typing import Literal

from robopolicy.datasets.config import SelectionConfig
from robopolicy.datasets.index import ManifestEntry


def stable_seed(*parts: object) -> int:
    """Return a stable seed independent of Python's process hash randomization."""
    encoded = "|".join(str(part) for part in parts).encode()
    return int(hashlib.md5(encoded, usedforsecurity=False).hexdigest()[:8], 16)


def select_entries(
    entries: tuple[ManifestEntry, ...],
    config: SelectionConfig,
    split: Literal["train", "val", "all"],
) -> tuple[ManifestEntry, ...]:
    """Resolve an immutable selection using opaque identifiers.

    Training limits apply after splitting. Validation membership remains fixed
    when training demo limits change. A zero validation fraction yields no
    validation demos, rather than reusing the training set.
    """
    available = {entry.task_id for entry in entries}
    missing = set(config.include_tasks) - available
    if missing:
        raise ValueError(f"Unknown task IDs: {sorted(missing)}; available: {sorted(available)}")
    selected = [
        entry
        for entry in entries
        if (
            (not config.include_tasks or entry.task_id in config.include_tasks)
            and entry.task_id not in config.exclude_tasks
        )
    ]
    if split != "all":
        selected = [
            entry
            for entry in selected
            if (
                (stable_seed(config.split_seed, *entry.identity) / 2**32 < config.val_fraction)
                == (split == "val")
            )
        ]
    if split != "val":
        ranked = sorted(
            selected,
            key=lambda entry: (stable_seed(config.subset_seed, *entry.identity), entry.identity),
        )
        task_counts: dict[str, int] = {}
        limited: list[ManifestEntry] = []
        for entry in ranked:
            count = task_counts.get(entry.task_id, 0)
            if config.max_demos_per_task is not None and count >= config.max_demos_per_task:
                continue
            limited.append(entry)
            task_counts[entry.task_id] = count + 1
        if config.max_demos is not None:
            if config.balance_demos_across_tasks and limited:
                tasks = sorted(task_counts)
                base, remainder = divmod(config.max_demos, len(tasks))
                quotas = {task: base + (index < remainder) for index, task in enumerate(tasks)}
                balanced: list[ManifestEntry] = []
                for entry in limited:
                    if quotas[entry.task_id]:
                        balanced.append(entry)
                        quotas[entry.task_id] -= 1
                limited = balanced
            else:
                limited = limited[: config.max_demos]
        identities = {entry.identity for entry in limited}
        selected = [entry for entry in selected if entry.identity in identities]
    return tuple(selected)
