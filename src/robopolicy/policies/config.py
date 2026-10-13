"""One architecture configuration for training and inference construction."""

from __future__ import annotations

from pydantic import Field, model_validator
from typing_extensions import Self

from robopolicy.algorithms import AlgorithmConfig
from robopolicy.algorithms.flow_matching import FlowMatchingConfig
from robopolicy.algorithms.improved_mean_flow import ImprovedMeanFlowConfig
from robopolicy.contracts.schema import StrictConfig
from robopolicy.modules.observation import FusionConfig
from robopolicy.modules.policy_head.config import ConvHeadConfig, HeadConfig, TransformerHeadConfig
from robopolicy.modules.queries import ConditioningConfig, LearnedConditioning
from robopolicy.modules.vision import VisionConfig


class PolicyConfig(StrictConfig):
    """Compose component-owned defaults and validate cross-component dimensions."""

    state_dim: int = Field(ge=1)
    action_dim: int = Field(ge=1)
    action_steps: int = Field(ge=1)
    obs_steps: int = Field(default=1, ge=1)
    cameras: int = Field(default=0, ge=0)
    entity_dim: int = Field(default=0, ge=0)
    selected_entities: int = Field(default=0, ge=0)
    conditioning: ConditioningConfig = Field(default_factory=LearnedConditioning)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    vision: VisionConfig | None = None
    head: HeadConfig = Field(default_factory=TransformerHeadConfig)
    algorithm: AlgorithmConfig = Field(default_factory=FlowMatchingConfig)

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        """Reject incompatible dimensions before allocating neural parameters."""
        if bool(self.cameras) != (self.vision is not None):
            raise ValueError("cameras and vision must be configured together")
        if bool(self.entity_dim) != bool(self.selected_entities):
            raise ValueError("entity_dim and selected_entities must be configured together")
        if self.fusion.dim % self.fusion.heads:
            raise ValueError("Fusion dimension must be divisible by head count")
        if self.fusion.type == "cross" and self.fusion.mask == "query_strict":
            raise ValueError(
                "query_strict is a self-attention fusion mask; cross fusion uses query"
            )
        if isinstance(self.head, ConvHeadConfig):
            if isinstance(self.algorithm, ImprovedMeanFlowConfig):
                raise ValueError("Improved MeanFlow requires a Transformer action head")
            if len(self.head.down_dims) < 2 or any(
                dim < 8 or dim % self.head.groups or dim % 8 for dim in self.head.down_dims
            ):
                raise ValueError(
                    "Conv down_dims require at least two widths divisible by groups and 8"
                )
            if self.head.kernel_size % 2 != 1:
                raise ValueError("Convolution kernel_size must be odd")
            if self.action_steps % (2 ** (len(self.head.down_dims) - 1)):
                raise ValueError(
                    "Conv action_steps must be divisible by the temporal downsampling factor"
                )
        elif self.head.dim % self.head.heads:
            raise ValueError("Transformer dimension must be divisible by head count")
        if (
            self.vision is not None
            and self.vision.type == "dinov2"
            and self.vision.dim % self.vision.heads
        ):
            raise ValueError("DINOv2 dimension must be divisible by head count")
        return self
