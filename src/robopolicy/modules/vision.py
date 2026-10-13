"""Reconstructible vision encoders; pretrained loading is explicitly opt-in."""

from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import Field
from torch import Tensor, nn

from robopolicy.contracts.schema import StrictConfig


class VisionConfig(StrictConfig):
    """Vision architecture stored with weights for offline inference construction."""

    type: Literal["patch", "dinov2"] = "patch"
    patch_size: int = Field(default=8, ge=1)
    dim: int = Field(default=64, ge=8)
    depth: int = Field(default=12, ge=1)
    heads: int = Field(default=6, ge=1)
    image_size: int = Field(default=224, ge=1)
    freeze: bool = False
    pretrained: str | None = None


class VisionEncoder(nn.Module):
    """Encode preprocessed RGB ``(B, 3, H, W)`` into patch tokens ``(B, L, D)``."""

    def __init__(self, config: VisionConfig, initialize_pretrained: bool) -> None:
        """Construct saved architecture without downloads unless explicitly enabled."""
        super().__init__()
        self.config = config
        self.backbone: nn.Module
        if config.type == "patch":
            if config.pretrained is not None:
                raise ValueError("pretrained is supported only for the dinov2 encoder")
            self.backbone = nn.Conv2d(
                3, config.dim, kernel_size=config.patch_size, stride=config.patch_size
            )
        else:
            try:
                from transformers import Dinov2Config, Dinov2Model
            except ImportError as error:
                raise ImportError("DINOv2 requires robopolicy[vision]") from error
            config_factory: Any = Dinov2Config  # HuggingFace configuration lacks typed signatures.
            architecture = config_factory(
                hidden_size=config.dim,
                num_hidden_layers=config.depth,
                num_attention_heads=config.heads,
                image_size=config.image_size,
                patch_size=config.patch_size,
            )
            if initialize_pretrained and config.pretrained:
                self.backbone = Dinov2Model.from_pretrained(config.pretrained, config=architecture)
            else:
                self.backbone = Dinov2Model(architecture)
        if config.freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> VisionEncoder:
        """Keep a frozen backbone in evaluation mode while training its consumers."""
        super().train(mode)
        if self.config.freeze:
            self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        """Return patch tokens ``(B, L, D)`` for floating RGB images ``(B, 3, H, W)``."""
        if self.config.type == "patch":
            encoded = cast(Tensor, self.backbone(images))
            return encoded.flatten(2).transpose(1, 2)  # (B, H_patch*W_patch, D)
        result: Any = self.backbone(pixel_values=images)
        return cast(Tensor, result.last_hidden_state[:, 1:])  # (B, L_patch, D)
