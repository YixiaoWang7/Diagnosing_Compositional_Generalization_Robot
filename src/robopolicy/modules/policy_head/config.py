"""Action-head architecture settings, composed by PolicyConfig."""

from typing import Annotated, Literal

from pydantic import Field

from robopolicy.contracts.schema import StrictConfig


class ConvHeadConfig(StrictConfig):
    """FiLM-conditioned convolutional U-Net settings."""

    type: Literal["conv"] = "conv"
    down_dims: tuple[int, ...] = (64, 128, 256)
    time_dim: int = Field(default=64, ge=4, multiple_of=2)
    kernel_size: int = Field(default=5, ge=1)
    groups: int = Field(default=8, ge=1)


class TransformerHeadConfig(StrictConfig):
    """Adaptive LayerNorm Transformer action decoder settings."""

    type: Literal["transformer"] = "transformer"
    dim: int = Field(default=128, ge=8)
    layers: int = Field(default=4, ge=1)
    heads: int = Field(default=4, ge=1)
    ff_dim: int = Field(default=256, ge=1)
    time_dim: int = Field(default=64, ge=4, multiple_of=2)
    dropout: float = Field(default=0, ge=0, lt=1)
    attention_dropout: float = Field(default=0, ge=0, lt=1)
    state_conditioning: Literal["mean", "current"] = "mean"


HeadConfig = Annotated[ConvHeadConfig | TransformerHeadConfig, Field(discriminator="type")]
