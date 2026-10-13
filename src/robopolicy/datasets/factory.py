"""Public factories for finite datasets and explicitly owned DataLoader workers."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Generator, Iterator
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, cast

from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from robopolicy.contracts.schema import FeatureSchema
from robopolicy.datasets.config import DatasetConfig, LoaderConfig
from robopolicy.datasets.execution import execute_epoch
from robopolicy.datasets.index import fingerprint, read_manifest, validate_entries
from robopolicy.datasets.selection import select_entries
from robopolicy.datasets.video import require_decoder


class PolicyDataset(IterableDataset[dict[str, Any]]):
    """Resolved selection with finite, reproducible, worker-aware epoch iterators.

    Attributes:
        config: Independent storage, selection, and execution settings.
        schema: Validated feature declarations used by readers and processing.
        episodes: Immutable selected episode metadata.
        split: Selection role; validation ordering never changes across epochs.
    """

    def __init__(
        self, config: DatasetConfig, schema: FeatureSchema, split: Literal["train", "val", "all"]
    ) -> None:
        """Resolve selection and validate metadata without decoding video frames."""
        super().__init__()
        self.config, self.schema, self.split = config, schema, split
        self.root = Path(config.root).expanduser().resolve()
        self.entries = select_entries(read_manifest(self.root), config.selection, split)
        self.episodes = validate_entries(self.entries, schema)
        if schema.cameras:
            require_decoder(config.video)
        context = mp.get_context("spawn")
        self._epoch = context.Value("q", 0)
        self._counters = context.Array("q", [0, 0, 0])
        self._active: Generator[dict[str, Any], None, None] | None = None

    @cached_property
    def fingerprint(self) -> str:
        """Return a content-based identity, unchanged by dataset relocation."""
        return fingerprint(self.root, self.entries, self.episodes)

    @property
    def selection(self) -> tuple[tuple[str, int, str], ...]:
        """Return resolved episode IDs for normalization and recovery auditing."""
        return tuple(episode.identity for episode in self.episodes)

    @property
    def epoch(self) -> int:
        """Return the trainer-owned epoch shared with persistent workers."""
        return int(self._epoch.value)

    def set_epoch(self, epoch: int) -> None:
        """Set the next iterator's logical epoch; call between iterations only."""
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self._epoch.value = epoch

    def __len__(self) -> int:
        """Return exact eligible-sample coverage, including final valid windows."""
        return sum(max(0, episode.length - self.schema.max_future) for episode in self.episodes)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Start one finite worker epoch and close any abandoned local iterator."""
        self.close()
        worker = get_worker_info()
        worker_id, worker_count = (0, 1) if worker is None else (worker.id, worker.num_workers)
        self._active = execute_epoch(
            self.episodes,
            self.schema,
            self.config,
            self.epoch,
            worker_id,
            worker_count,
            self.split == "val",
            self._counters,
        )
        return self._active

    @property
    def resources(self) -> dict[str, int]:
        """Return aggregate live handles, peak handles, and block decode requests."""
        with self._counters.get_lock():
            return dict(
                zip(
                    ("live_handles", "peak_handles", "decode_requests"),
                    [self._counters[index] for index in range(3)],
                    strict=True,
                )
            )

    def close(self) -> None:
        """Close the process-local iterator; DataLoader owns other processes."""
        if self._active is not None:
            self._active.close()
            self._active = None


class PolicyLoader:
    """Own DataLoader workers; use a context manager or call close explicitly.

    PyTorch currently lacks a public worker-shutdown method. That dependency is
    contained here and covered by multiprocessing lifecycle tests.
    """

    def __init__(self, dataset: PolicyDataset, config: LoaderConfig) -> None:
        """Configure complete coverage with drop_last=False for every worker."""
        self.dataset, self.config = dataset, config
        kwargs: dict[str, Any] = {}
        if config.num_workers:
            kwargs.update(
                prefetch_factor=config.prefetch_factor,
                persistent_workers=config.persistent_workers,
                multiprocessing_context=config.multiprocessing_context,
            )
        self.loader: DataLoader[dict[str, Any]] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,
            **kwargs,
        )
        self._iterator: Any = None
        self._closed = False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Create a finite epoch iterator, propagating the dataset's current epoch."""
        if self._closed:
            raise RuntimeError("Loader is closed; build a new loader")
        self._iterator = iter(self.loader)
        return cast(Iterator[dict[str, Any]], self._iterator)

    def close(self) -> None:
        """Stop workers and release process-local readers after early or full exit."""
        if self._iterator is not None:
            shutdown = getattr(self._iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
            self._iterator = None
        self.dataset.close()
        self._closed = True
        # Worker termination also releases native handles. Clear live telemetry
        # because Python generator finalizers do not run after process shutdown.
        with self.dataset._counters.get_lock():
            self.dataset._counters[0] = 0

    def __enter__(self) -> PolicyLoader:
        """Return this resource owner for a with statement."""
        return self

    def __exit__(self, *exception: object) -> None:
        """Always release workers, including when consumption raises."""
        self.close()


def build_dataset(
    config: DatasetConfig, schema: FeatureSchema, split: Literal["train", "val", "all"] = "train"
) -> PolicyDataset:
    """Resolve and validate a dataset using decoder and sampler independently."""
    return PolicyDataset(config, schema, split)


def build_loader(dataset: PolicyDataset, config: LoaderConfig | None = None) -> PolicyLoader:
    """Build a finite loader with explicit shutdown and exhaustive sample coverage."""
    return PolicyLoader(dataset, config or LoaderConfig())
