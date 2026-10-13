"""Independent dataset selection, sampling, decoding, and execution settings."""

from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from robopolicy.contracts.schema import StrictConfig


class VideoConfig(StrictConfig):
    """Bound native decoder handles per worker; decoded frames are block-scoped.

    Attributes:
        decoder: Explicit backend; there is no automatic fallback.
        max_handles: Maximum simultaneously open decoders in each worker.
        threads: FFmpeg threads per TorchCodec decoder. OpenCV manages its own
            codec threads; use max_handles to bound its retained native buffers.
    """

    decoder: Literal["opencv", "torchcodec"] = "torchcodec"
    max_handles: int = Field(default=8, ge=1)
    threads: int = Field(default=1, ge=1)


class SamplingConfig(StrictConfig):
    """Finite epoch planning settings independent of the video implementation."""

    strategy: Literal["window_major", "global_pass"] = "global_pass"
    k_passes: int = Field(default=1, ge=1)
    seed: int = 42
    window_shards: int = Field(default=2, ge=1)
    block_size: int = Field(default=32, ge=1)
    prefetch_window: bool = True
    prefetch_block: bool = True


class SelectionConfig(StrictConfig):
    """Resolve opaque task IDs, splits, and training demo limits once.

    Balanced total budgets assign equal quotas in sorted task order; the first
    tasks receive remainder slots. Unfilled quotas are not redistributed.
    """

    include_tasks: tuple[str, ...] = ()
    exclude_tasks: tuple[str, ...] = ()
    val_fraction: float = Field(default=0.05, ge=0, lt=1)
    split_seed: int = 0
    subset_seed: int = 0
    max_demos: int | None = Field(default=None, ge=1)
    max_demos_per_task: int | None = Field(default=None, ge=1)
    balance_demos_across_tasks: bool = False

    @model_validator(mode="after")
    def validate_filters(self) -> Self:
        """Reject contradictory task filters."""
        if set(self.include_tasks) & set(self.exclude_tasks):
            raise ValueError("Include and exclude task filters overlap")
        return self


class DatasetConfig(StrictConfig):
    """Compose dataset settings without any simulator or model assumptions."""

    root: str
    video: VideoConfig = Field(default_factory=VideoConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)


class LoaderConfig(StrictConfig):
    """Single-process or multiprocessing loading with explicit resource lifetime."""

    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=0, ge=0)
    prefetch_factor: int = Field(default=2, ge=1)
    persistent_workers: bool = True
    pin_memory: bool = False
    multiprocessing_context: Literal["spawn", "forkserver"] = "spawn"
