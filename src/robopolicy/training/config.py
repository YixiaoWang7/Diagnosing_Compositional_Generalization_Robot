"""Single-device optimizer and training-loop settings."""

from typing import Literal

from pydantic import Field

from robopolicy.contracts.schema import StrictConfig


class TrainConfig(StrictConfig):
    """Configure completed optimizer updates rather than attempted AMP batches."""

    device: str = "cpu"
    max_updates: int = Field(default=1000, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    amp: bool = False
    grad_clip_norm: float = Field(default=1.0, gt=0)
    schedule: Literal["constant", "cosine"] = "cosine"
    warmup_updates: int = Field(default=0, ge=0)
    min_lr_ratio: float = Field(default=0.01, ge=0, le=1)
    log_every: int = Field(default=10, ge=1)
    validate_every: int = Field(default=100, ge=1)
    validation_batches: int | None = Field(default=None, ge=1)
    checkpoint_every: int = Field(default=100, ge=1)
    max_consecutive_skips: int = Field(default=100, ge=1)
