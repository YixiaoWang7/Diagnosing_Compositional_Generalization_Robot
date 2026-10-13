"""Canonical non-mutating input packing for training, validation, and inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch
import torch.nn.functional as functional
from pydantic import model_validator
from torch import Tensor, nn
from typing_extensions import Self

from robopolicy.contracts.batch import PolicyInput
from robopolicy.contracts.schema import FeatureSchema, StrictConfig
from robopolicy.processing.normalization import Normalizer, Statistics


class ProcessingConfig(StrictConfig):
    """Tensor preprocessing settings; statistics fitting belongs to dataset readers."""

    image_size: tuple[int, int] | None = None
    crop_size: tuple[int, int] | None = None
    random_crop: bool = False
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    image_representation: Literal["rgb", "current_and_signed_difference"] = "rgb"

    @model_validator(mode="after")
    def validate_sizes(self) -> Self:
        """Reject invalid sizes and zero image scaling factors."""
        for size in (self.image_size, self.crop_size):
            if size is not None and min(size) < 1:
                raise ValueError("Image and crop sizes must be positive")
        if min(self.image_std) <= 0:
            raise ValueError("Image standard deviations must be positive")
        return self

    def validate_schema(self, schema: FeatureSchema) -> None:
        """Require one previous and one current image for signed differences."""
        if self.image_representation == "current_and_signed_difference":
            images = next((feature for feature in schema.features if feature.key == "images"), None)
            if images is None or images.source != "video" or images.history != 1 or images.future:
                raise ValueError("Signed differences require exactly two causal image observations")


class InputPacker(nn.Module):
    """Convert raw batches to the common PolicyInput using supplied fixed statistics."""

    def __init__(
        self, schema: FeatureSchema, statistics: Statistics, config: ProcessingConfig | None = None
    ) -> None:
        """Bind the saved schema and statistics without consulting any dataset."""
        super().__init__()
        if schema != statistics.feature_schema:
            raise ValueError("Input schema and normalization schema differ")
        self.schema = schema
        self.normalizer = Normalizer(statistics)
        self.config = config or ProcessingConfig()
        self.config.validate_schema(schema)
        self._image_mean: Tensor
        self._image_std: Tensor
        self.register_buffer(
            "_image_mean",
            torch.tensor(self.config.image_mean).reshape(1, 3, 1, 1),
            persistent=False,
        )  # (1, 3, 1, 1)
        self.register_buffer(
            "_image_std",
            torch.tensor(self.config.image_std).reshape(1, 3, 1, 1),
            persistent=False,
        )  # (1, 3, 1, 1)

    def images(self, images: Tensor, training: bool) -> Tensor:
        """Preprocess uint8 ``(B, T, N, 3, H, W)`` without changing caller storage.

        Difference mode replaces the previous image with a signed, gray-centered
        difference at unit gain. Geometry is shared and normalization happens last.
        """
        if images.dtype != torch.uint8 or images.ndim != 6:
            raise ValueError("Raw images must be RGB uint8 (B, T_obs, N_camera, 3, H, W)")
        batch, steps, cameras = images.shape[:3]
        if self.config.image_representation == "current_and_signed_difference" and steps != 2:
            raise ValueError("Signed differences require exactly two causal image observations")
        pixels = images.flatten(0, 2).to(torch.float32) / 255.0
        # Shape: (B*T_obs*N_camera, 3, H, W), now owned floating pixels.
        if self.config.crop_size is not None:
            height, width = pixels.shape[-2:]
            crop_h, crop_w = self.config.crop_size
            if crop_h > height or crop_w > width:
                raise ValueError("Crop size exceeds supplied image dimensions")
            if training and self.config.random_crop:
                top = int(torch.randint(height - crop_h + 1, ()).item())
                left = int(torch.randint(width - crop_w + 1, ()).item())
            else:
                top, left = (height - crop_h) // 2, (width - crop_w) // 2
            pixels = pixels[..., top : top + crop_h, left : left + crop_w]
            # Shape: (B*T_obs*N_camera, 3, crop_h, crop_w).
        if self.config.image_size is not None:
            pixels = functional.interpolate(
                pixels,
                size=self.config.image_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            # Shape: (B*T_obs*N_camera, 3, resized_h, resized_w).
        if self.config.image_representation == "current_and_signed_difference":
            frames = pixels.reshape(batch, steps, cameras, *pixels.shape[1:])
            # Shape: (B, 2, N_camera, 3, output_h, output_w).
            difference = 0.5 + 0.5 * (frames[:, 1] - frames[:, 0])
            pixels = torch.stack((difference, frames[:, 1]), dim=1).flatten(0, 2)
            # Shape: (B*2*N_camera, 3, output_h, output_w), ordered [difference, current].
        mean = self._image_mean.to(pixels)
        std = self._image_std.to(pixels)
        pixels = (pixels - mean) / std
        return pixels.reshape(batch, steps, cameras, *pixels.shape[1:])
        # Shape: (B, T_obs, N_camera, 3, output_h, output_w).

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        device: str = "cpu",
        training: bool = False,
        require_actions: bool = False,
    ) -> PolicyInput:
        """Pack supplied observations, adding targets only when they are present.

        Args:
            batch: Raw tensors in declared temporal shapes and optional metadata.
            device: Destination for every tensor, preserving categorical integers.
            training: Enable configured random cropping.
            require_actions: Fail clearly if training targets are absent.

        Returns:
            A new PolicyInput; no caller-owned tensor or array is modified.
        """
        values = self.validate(batch, require_actions=require_actions)
        moved = {key: value.to(device, non_blocking=True) for key, value in values.items()}
        return self.pack_tensors(moved, training=training)

    def validate(
        self, batch: Mapping[str, Any], *, require_actions: bool = False
    ) -> dict[str, Tensor]:
        """Validate raw feature shapes/dtypes and return tensor references.

        Args:
            batch: Raw tensors with batch and declared temporal dimensions.
            require_actions: Require action targets for training.

        Returns:
            Declared tensors in schema order, without copies or device transfers.
        """
        values: dict[str, Tensor] = {}
        for feature in self.schema.features:
            key = feature.key
            if key not in batch:
                if key == "actions" and not require_actions:
                    continue
                raise ValueError(f"Missing required input feature: {key}")
            value = batch[key]
            if not isinstance(value, Tensor):
                raise ValueError(f"Input feature {key} must be a Tensor")
            expected = (
                (feature.window_size, len(feature.cameras), *feature.shape)
                if feature.source == "video"
                else (feature.window_size, *feature.shape)
            )
            if tuple(value.shape[1:]) != expected:
                raise ValueError(
                    f"Input shape mismatch for {key}: {tuple(value.shape[1:])} != {expected}"
                )
            if value.dtype != getattr(torch, feature.dtype):
                raise ValueError(
                    f"Input dtype mismatch for {key}: expected {feature.dtype}, got {value.dtype}"
                )
            if key == "images" and self.config.crop_size is not None:
                if any(crop > size for crop, size in zip(self.config.crop_size, value.shape[-2:])):
                    raise ValueError("Crop size exceeds supplied image dimensions")
            values[key] = value
        if "state" not in values or (require_actions and "actions" not in values):
            raise ValueError("Packing requires state and, for training, actions")
        allowed = {"state", "actions", "images", "entities", "task_indices", "entity_indices"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(
                f"Feature keys need a project adapter before packing: {sorted(unknown)}"
            )
        batch_size = values["state"].shape[0]
        if batch_size < 1 or any(value.shape[0] != batch_size for value in values.values()):
            raise ValueError("Input features must share a nonempty batch dimension")
        return values

    @staticmethod
    def canonical_inputs(values: Mapping[str, Tensor]) -> PolicyInput:
        """Wrap tensors and remove the current-step axis from categorical inputs.

        Args:
            values: Features with shapes validated against their schema.

        Returns:
            Policy inputs; categorical ``(B, 1, K)`` tensors become ``(B, K)``.
        """
        return PolicyInput(
            **{
                key: tensor[:, 0] if key in {"task_indices", "entity_indices"} else tensor
                for key, tensor in values.items()
            }
        )

    def pack_tensors(self, values: Mapping[str, Tensor], *, training: bool = False) -> PolicyInput:
        """Preprocess validated tensors already placed on the destination device.

        Args:
            values: Raw tensors in their declared temporal shapes.
            training: Enable configured random cropping for training only.

        Returns:
            Canonical inputs, including normalized floating images and features.
        """
        transformed = {
            key: self.images(tensor, training)
            if key == "images"
            else self.normalizer.transform(key, tensor)
            for key, tensor in values.items()
        }
        return self.canonical_inputs(transformed)

    def actions_to_raw(self, actions: Tensor) -> Tensor:
        """Invert action normalization for predictions ``(B, T_action, D_action)``."""
        return self.normalizer.transform("actions", actions, inverse=True)
