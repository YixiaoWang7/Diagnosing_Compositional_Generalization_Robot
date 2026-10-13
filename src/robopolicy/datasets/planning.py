"""Pure finite-epoch planning; no array storage or video decoder imports."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from robopolicy.contracts.schema import Feature, FeatureSchema
from robopolicy.datasets.config import SamplingConfig
from robopolicy.datasets.index import Episode
from robopolicy.datasets.selection import stable_seed


@dataclass(frozen=True)
class SampleRef:
    """Reference to an eligible anchor step within one episode."""

    episode: Episode
    step: int


@dataclass(frozen=True)
class WindowPlan:
    """Episodes loaded together and partitions traversed before releasing them."""

    shards: tuple[tuple[Episode, ...], ...]
    partitions: tuple[int, ...]

    @property
    def episodes(self) -> tuple[Episode, ...]:
        """Return all selected episodes needed by this window."""
        return tuple(episode for shard in self.shards for episode in shard)


def partition_bounds(total: int, partition: int, count: int) -> tuple[int, int]:
    """Split a sequence into disjoint, exhaustive partitions, including empties."""
    if total < 0 or count < 1 or not 0 <= partition < count:
        raise ValueError("Invalid total, partition index, or partition count")
    base, extra = divmod(total, count)
    start = partition * base + min(partition, extra)
    return start, start + base + int(partition < extra)


def feature_indices(step: int, feature: Feature) -> NDArray[np.int64]:
    """Return episode-local ``(T_window,)`` indices with first-step history padding."""
    if step < 0:
        raise ValueError("Anchor step must be nonnegative")
    return np.maximum(
        0, np.arange(step - feature.history, step + feature.future + 1, dtype=np.int64)
    )


def plan_windows(
    episodes: tuple[Episode, ...],
    config: SamplingConfig,
    epoch: int,
    worker_id: int,
    worker_count: int,
    validation: bool,
) -> Iterator[WindowPlan]:
    """Assign whole shards to workers, then choose window or partition traversal.

    Global-pass ordering is local to each worker and introduces no inter-worker
    barrier. Worker counts affect ordering but never duplicate shard ownership.
    """
    if epoch < 0 or worker_count < 1 or not 0 <= worker_id < worker_count:
        raise ValueError("Invalid epoch or worker assignment")
    grouped: dict[tuple[str, int], list[Episode]] = {}
    for episode in episodes:
        grouped.setdefault((episode.task_id, episode.shard_id), []).append(episode)
    shards = [tuple(grouped[key]) for key in sorted(grouped)]
    if not validation:
        order = np.random.default_rng(stable_seed(config.seed, epoch, "shards")).permutation(
            len(shards)
        )
        shards = [shards[int(index)] for index in order]
    assigned = shards[worker_id::worker_count]
    windows = [
        tuple(assigned[start : start + config.window_shards])
        for start in range(0, len(assigned), config.window_shards)
    ]
    partitions = tuple(range(1 if validation else config.k_passes))
    if config.strategy == "global_pass" and not validation:
        for partition in partitions:
            for window in windows:
                yield WindowPlan(window, (partition,))
    else:
        for window in windows:
            yield WindowPlan(window, partitions)


def plan_blocks(
    window: WindowPlan,
    schema: FeatureSchema,
    config: SamplingConfig,
    epoch: int,
    validation: bool,
) -> Iterator[tuple[SampleRef, ...]]:
    """Partition the same per-shard permutation for either traversal strategy.

    Blocks sort requests for video locality. Every sample remains in its
    original disjoint partition. Future indices are bounded by construction.
    """
    for partition in window.partitions:
        for shard in window.shards:
            references = [
                SampleRef(episode, step)
                for episode in shard
                for step in range(max(0, episode.length - schema.max_future))
            ]
            order = np.arange(len(references))
            if not validation:
                first = shard[0]
                np.random.default_rng(
                    stable_seed(config.seed, epoch, first.task_id, first.shard_id)
                ).shuffle(order)
            start, end = partition_bounds(
                len(order), partition, 1 if validation else config.k_passes
            )
            for offset in range(start, end, config.block_size):
                block = [
                    references[int(index)]
                    for index in order[offset : min(offset + config.block_size, end)]
                ]
                if not validation:
                    block.sort(key=lambda ref: (ref.episode.identity, ref.step))
                yield tuple(block)
