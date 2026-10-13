"""Shared state, entity, and image encoding for every action-head variant."""

from __future__ import annotations

from typing import Literal, cast

import torch
from pydantic import Field
from torch import Tensor, nn

from robopolicy.contracts.batch import PolicyInput
from robopolicy.contracts.schema import StrictConfig
from robopolicy.modules.fusion.attention_projector import (
    CrossAttentionPooling,
    SelfAttentionPooling,
)
from robopolicy.modules.queries import ConditioningConfig, QueryTokens
from robopolicy.modules.vision import VisionConfig, VisionEncoder


class FusionConfig(StrictConfig):
    """Attention settings owned by observation fusion rather than run configuration."""

    type: Literal["cross", "self"] = "cross"
    dim: int = Field(default=64, ge=8)
    heads: int = Field(default=4, ge=1)
    layers: int = Field(default=2, ge=1)
    ff_dim: int = Field(default=128, ge=1)
    dropout: float = Field(default=0, ge=0, lt=1)
    mask: Literal["none", "query", "query_strict"] = "none"
    temporal_encoding: Literal["none", "learned"] = "none"


class ObservationEncoder(nn.Module):
    """Compose explicit state/entities and camera queries using one encoding path."""

    def __init__(
        self,
        state_dim: int,
        obs_steps: int,
        cameras: int,
        entity_dim: int,
        selected_entities: int,
        conditioning: ConditioningConfig,
        fusion: FusionConfig,
        vision: VisionConfig | None,
        initialize_pretrained: bool,
    ) -> None:
        """Configure token count once for both convolutional and Transformer heads."""
        super().__init__()
        self.state_dim = state_dim
        self.obs_steps, self.cameras = obs_steps, cameras
        self.entity_dim, self.selected_entities = entity_dim, selected_entities
        self.queries = QueryTokens(conditioning, fusion.dim)
        self.state_projection = nn.Linear(state_dim, fusion.dim)
        self.temporal_embedding = (
            nn.Parameter(torch.randn(1, obs_steps, 1, fusion.dim) * 0.02)
            if fusion.temporal_encoding == "learned"
            else None
        )
        self.entity_projection = nn.Linear(entity_dim, fusion.dim) if entity_dim else None
        self.vision = VisionEncoder(vision, initialize_pretrained) if vision is not None else None
        self.fusion: nn.Module | None = None
        if vision is not None:
            pooling = CrossAttentionPooling if fusion.type == "cross" else SelfAttentionPooling
            self.fusion = pooling(
                input_shape=(vision.dim, 1, 1),
                embed_dim=fusion.dim,
                out_dim=fusion.dim,
                num_layers=fusion.layers,
                num_heads=fusion.heads,
                ff_dim=fusion.ff_dim,
                dropout=fusion.dropout,
                mask_type=fusion.mask,
            )
        self.token_count = (
            obs_steps * (1 + selected_entities + cameras * self.queries.count)
            if cameras
            else obs_steps * (1 + selected_entities) + self.queries.count
        )
        self.dim = fusion.dim

    def forward(
        self, inputs: PolicyInput, return_attention: bool = False
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Encode observations and optionally return maps from that exact computation.

        Args:
            inputs: Canonical state/images/entities/task indices.
            return_attention: Request attention weights without a separate encoder pass.

        Returns:
            Context ``(B, L_context, D)`` and optional attention maps with leading
            dimensions ``(B, T_obs, N_camera, K_query)``.
        """
        self.validate(inputs)
        return self.encode_tensors(inputs, return_attention)

    def validate(self, inputs: PolicyInput) -> None:
        """Check observation shapes and IDs without executing neural operations.

        Args:
            inputs: Canonical observations. Images may be raw uint8 while checking
                an inference batch before preprocessing and device transfer.
        """
        if inputs.state.ndim != 3 or inputs.state.shape[1:] != (self.obs_steps, self.state_dim):
            raise ValueError("State must match configured (B, T_obs, D_state)")
        if not inputs.state.is_floating_point():
            raise ValueError("State must be floating point")
        batch, steps, _ = inputs.state.shape
        if batch < 1:
            raise ValueError("Observations require a nonempty batch")
        self.queries.validate(batch, inputs.task_indices)
        if self.entity_projection is not None:
            entities, indices = inputs.entities, inputs.entity_indices
            if entities is None or indices is None:
                raise ValueError("Entity policies require entities and explicit entity_indices")
            if (
                entities.ndim != 4
                or entities.shape[:2] != (batch, steps)
                or entities.shape[-1] != self.entity_dim
            ):
                raise ValueError("Entities must have shape (B, T_obs, N_entity, D_entity)")
            if indices.dtype != torch.int64 or indices.shape != (batch, self.selected_entities):
                raise ValueError("entity_indices must be int64 (B, N_selected)")
            if bool(((indices < 0) | (indices >= entities.shape[2])).any()):
                raise ValueError("Entity selection index out of range")
        elif inputs.entities is not None or inputs.entity_indices is not None:
            raise ValueError("This architecture does not declare entity inputs")
        if self.vision is not None:
            if (
                inputs.images is None
                or inputs.images.ndim != 6
                or inputs.images.shape[:3] != (batch, steps, self.cameras)
                or inputs.images.shape[3] != 3
            ):
                raise ValueError("Images must have shape (B, T_obs, N_camera, 3, H, W)")
        elif inputs.images is not None:
            raise ValueError("This architecture does not declare images")

    def encode_tensors(
        self, inputs: PolicyInput, return_attention: bool = False
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Encode validated observations using the shared neural computation.

        Args:
            inputs: Validated observations with floating images and state.
            return_attention: Include maps from the same encoder computation.

        Returns:
            Context ``(B, L_context, D)`` and optional maps with leading
            dimensions ``(B, T_obs, N_camera, K_query)``.
        """
        batch, steps, _ = inputs.state.shape
        queries = self.queries.embed(batch, inputs.task_indices)  # (B, K, D)
        state = cast(Tensor, self.state_projection(inputs.state))
        if self.temporal_embedding is not None:
            state = state + self.temporal_embedding[:, :, 0]
        tokens = [state]  # (B, T_obs, D)
        if self.entity_projection is not None:
            entities, indices = inputs.entities, inputs.entity_indices
            assert entities is not None and indices is not None
            selection = indices[:, None, :, None].expand(
                batch, steps, self.selected_entities, self.entity_dim
            )  # (B, T_obs, N_selected, D_entity)
            selected = entities.gather(2, selection)  # (B, T_obs, N_selected, D_entity)
            encoded = cast(Tensor, self.entity_projection(selected))
            if self.temporal_embedding is not None:
                encoded = encoded + self.temporal_embedding
            tokens.append(encoded.flatten(1, 2))  # (B, T_obs*N_selected, D)
        attention: dict[str, Tensor] = {}
        if self.vision is not None:
            assert inputs.images is not None
            images = inputs.images.flatten(0, 2)  # (B*T_obs*N_camera, 3, H, W)
            image_tokens = self.vision(images)  # (B*T_obs*N_camera, L_patch, D_vision)
            expanded = self.queries.expand_observations(queries, steps, self.cameras)
            assert self.fusion is not None
            result = self.fusion(
                image_tokens, query_tokens=expanded, return_intermediate=return_attention
            )
            if return_attention:
                image_context, maps = result
                for key, values in maps.items():
                    attention[key] = values.reshape(batch, steps, self.cameras, *values.shape[1:])
                    # Shape: (B, T_obs, N_camera, K_query, D_or_L_patch).
            else:
                image_context = result
            if self.temporal_embedding is not None:
                image_context = image_context.reshape(batch, steps, self.cameras, -1, self.dim)
                image_context = image_context + self.temporal_embedding[:, :, None]
            tokens.append(image_context.reshape(batch, -1, self.dim))
            # Shape: (B, T_obs*N_camera*K_query, D).
        else:
            tokens.append(queries)
        return torch.cat(tokens, dim=1), attention  # (B, L_context, D)
