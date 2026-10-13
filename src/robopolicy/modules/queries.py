"""Independent categorical query vocabularies and task-free learned queries."""

from __future__ import annotations

from typing import Annotated, Literal

import torch
from pydantic import Field, field_validator
from torch import Tensor, nn

from robopolicy.contracts.schema import StrictConfig


class IndexedConditioning(StrictConfig):
    """One independent embedding table per explicit application-defined slot."""

    type: Literal["indexed"] = "indexed"
    vocab_sizes: tuple[int, ...] = Field(min_length=1)

    @field_validator("vocab_sizes")
    @classmethod
    def validate_sizes(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        """Reject empty vocabularies instead of deriving task-specific offsets."""
        if any(value < 1 for value in values):
            raise ValueError("Query vocabulary sizes must be positive")
        return values


class LearnedConditioning(StrictConfig):
    """Task-free learned query tokens requiring no categorical IDs."""

    type: Literal["learned"] = "learned"
    num_tokens: int = Field(default=1, ge=1)


ConditioningConfig = Annotated[
    IndexedConditioning | LearnedConditioning, Field(discriminator="type")
]


class QueryTokens(nn.Module):
    """Centralize query construction, validation, and camera/time expansion."""

    def __init__(self, config: ConditioningConfig, dim: int) -> None:
        """Construct independent slot embeddings or a learned ``(K, D)`` tensor."""
        super().__init__()
        self.config = config
        self.count = (
            len(config.vocab_sizes)
            if isinstance(config, IndexedConditioning)
            else config.num_tokens
        )
        self.embeddings = (
            nn.ModuleList(nn.Embedding(size, dim) for size in config.vocab_sizes)
            if isinstance(config, IndexedConditioning)
            else nn.ModuleList()
        )
        self.tokens = (
            nn.Parameter(torch.randn(self.count, dim) * 0.02)
            if isinstance(config, LearnedConditioning)
            else None
        )

    def forward(self, batch_size: int, indices: Tensor | None = None) -> Tensor:
        """Return queries ``(B, K, D)`` after validating integer IDs and slot count."""
        self.validate(batch_size, indices)
        return self.embed(batch_size, indices)

    def validate(self, batch_size: int, indices: Tensor | None) -> None:
        """Validate ``(B, K)`` IDs outside compiled neural computations."""
        if isinstance(self.config, IndexedConditioning):
            if indices is None or indices.shape != (batch_size, self.count):
                raise ValueError(
                    f"Indexed conditioning requires task_indices shaped (B, {self.count})"
                )
            if indices.dtype not in {torch.int64, torch.int32}:
                raise ValueError("Task indices must be integer tensors")
            for slot, size in enumerate(self.config.vocab_sizes):
                if bool(((indices[:, slot] < 0) | (indices[:, slot] >= size)).any()):
                    raise ValueError(f"Task indices for slot {slot} must be in [0, {size})")
        elif indices is not None:
            raise ValueError("Learned conditioning does not accept task_indices")

    def embed(self, batch_size: int, indices: Tensor | None) -> Tensor:
        """Embed previously validated ``(B, K)`` IDs into ``(B, K, D)`` queries."""
        if isinstance(self.config, IndexedConditioning):
            assert indices is not None
            return torch.stack(
                [embedding(indices[:, slot]) for slot, embedding in enumerate(self.embeddings)],
                dim=1,
            )
            # Shape: (B, K, D); each slot has its own parameters.
        assert self.tokens is not None
        return self.tokens.unsqueeze(0).expand(batch_size, -1, -1)  # (B, K, D)

    @staticmethod
    def expand_observations(queries: Tensor, steps: int, cameras: int) -> Tensor:
        """Expand ``(B, K, D)`` across time/cameras into ``(B*T*N, K, D)``."""
        batch, slots, dim = queries.shape
        return (
            queries[:, None, None]
            .expand(batch, steps, cameras, slots, dim)
            .reshape(batch * steps * cameras, slots, dim)
        )
