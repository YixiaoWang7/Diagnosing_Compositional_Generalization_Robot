"""One resource lifetime for window prefetch, block prefetch, and native decoders."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

from robopolicy.contracts.schema import FeatureSchema
from robopolicy.datasets.assembly import FrameBlock, assemble_sample, decode_block
from robopolicy.datasets.config import DatasetConfig
from robopolicy.datasets.index import Episode
from robopolicy.datasets.planning import SampleRef, WindowPlan, plan_blocks, plan_windows
from robopolicy.datasets.storage import NumericArrays, load_window
from robopolicy.datasets.video import DecoderPool

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def prefetch_map(
    function: Callable[[InputT], OutputT],
    inputs: Iterable[InputT],
    enabled: bool,
) -> Generator[OutputT, None, None]:
    """Map with at most one pending result and join resources on every exit path."""
    if not enabled:
        for value in inputs:
            yield function(value)
        return
    iterator = iter(inputs)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robopolicy-prefetch")
    pending: Future[OutputT] | None = None
    try:
        try:
            pending = executor.submit(function, next(iterator))
        except StopIteration:
            return
        while pending is not None:
            result = pending.result()
            try:
                pending = executor.submit(function, next(iterator))
            except StopIteration:
                pending = None
            yield result
            del result
    finally:
        if pending is not None:
            pending.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def execute_epoch(
    episodes: tuple[Episode, ...],
    schema: FeatureSchema,
    config: DatasetConfig,
    epoch: int,
    worker_id: int,
    worker_count: int,
    validation: bool,
    counters: Any,
) -> Generator[dict[str, Any], None, None]:
    """Execute a finite worker shard plan and close threads before decoder handles."""
    pool = DecoderPool(config.video, counters)

    def load(plan: WindowPlan) -> tuple[WindowPlan, dict[tuple[str, int, str], NumericArrays]]:
        """Read one current or prefetched shard window into owned numeric arrays."""
        return plan, load_window(plan.episodes, schema)

    def decode(block: tuple[SampleRef, ...]) -> tuple[tuple[SampleRef, ...], FrameBlock]:
        """Read block-scoped frames on the sole decoder thread."""
        return block, decode_block(block, schema, pool)

    plans = plan_windows(episodes, config.sampling, epoch, worker_id, worker_count, validation)
    windows = prefetch_map(load, plans, config.sampling.prefetch_window)
    try:
        for plan, arrays in windows:
            blocks = prefetch_map(
                decode,
                plan_blocks(plan, schema, config.sampling, epoch, validation),
                config.sampling.prefetch_block and bool(schema.cameras),
            )
            try:
                for block, frames in blocks:
                    for reference in block:
                        yield assemble_sample(
                            reference, schema, arrays[reference.episode.identity], frames
                        )
            finally:
                blocks.close()
    finally:
        windows.close()
        pool.close()
