#!/usr/bin/env python

"""Policy Head Module with Multi-Modal Conditioning.

Moved from `flow_models.models.policy_head` as part of the method-agnostic refactor.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, cast

import torch
import torch.nn as nn
from torch import Tensor


class _ResidualSettings(TypedDict):
    """Typed settings for time/context-conditioned residual blocks."""

    cond_dim: int
    kernel_size: int
    n_groups: int
    use_film_scale_modulation: bool


class _MultiResidualSettings(TypedDict):
    """Typed settings for multi-modal residual blocks."""

    modality_dims: dict[str, int]
    kernel_size: int
    n_groups: int
    use_film_scale: bool
    film_fusion: Literal["concat", "gated", "add"]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ModalityConfig:
    """Configuration for a single conditioning modality."""

    name: str
    dim: int
    n_steps: int = 1
    enabled: bool = True

    @property
    def total_dim(self) -> int:
        """Return the feature width after flattening the observation-step axis.

        Returns:
            The computed int result.
        """
        return self.dim * self.n_steps


@dataclass
class PolicyHeadConfig:
    """Configuration for the FiLM-conditioned policy head."""

    action_dim: int
    horizon: int
    modalities: list[ModalityConfig] = field(default_factory=list)
    time_embed_dim: int = 128
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    use_film_scale_modulation: bool = True
    film_fusion: Literal["concat", "gated", "add"] = "gated"

    def __post_init__(self) -> None:
        """Resolve explicit modality declarations into typed component settings."""
        self.modalities = [
            ModalityConfig(**m) if isinstance(m, dict) else m for m in self.modalities
        ]

    @property
    def enabled_modalities(self) -> list[ModalityConfig]:
        """Return only modalities explicitly enabled in this head.

        Returns:
            The computed list[ModalityConfig] result.
        """
        return [m for m in self.modalities if m.enabled]

    @property
    def modality_names(self) -> list[str]:
        """Return the identifiers of enabled conditioning modalities.

        Returns:
            The computed list[str] result.
        """
        return [m.name for m in self.enabled_modalities]

    def get_modality(self, name: str) -> ModalityConfig | None:
        """Find an explicit conditioning declaration by name.

        Args:
            name: Name setting for this component.

        Returns:
            The computed ModalityConfig | None result.
        """
        for m in self.modalities:
            if m.name == name:
                return m
        return None


# =============================================================================
# Time Embedding
# =============================================================================


class SinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings for time conditioning."""

    def __init__(self, dim: int) -> None:
        """Configure SinusoidalPosEmb and allocate its owned state.

        Args:
            dim: Dim setting for this component.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute sinusoidal time embeddings.

        Args:
            x (Tensor): Shape [B]. Scalar timestep values.

        Returns:
            Tensor: Shape [B, dim]. Concatenated sin/cos embedding.
        """
        device = x.device
        half_dim = self.dim // 2
        frequency_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -frequency_scale)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# =============================================================================
# FiLM Modules
# =============================================================================


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer for a single modality."""

    def __init__(self, cond_dim: int, out_channels: int, use_scale: bool = True) -> None:
        """Configure FiLMLayer and allocate its owned state.

        Args:
            cond_dim: Cond dim setting for this component.
            out_channels: Out channels setting for this component.
            use_scale: Use scale setting for this component.
        """
        super().__init__()
        self.out_channels = out_channels
        self.use_scale = use_scale
        film_dim = out_channels * 2 if use_scale else out_channels
        self.film_generator = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, film_dim))

    def forward(self, cond: Tensor) -> tuple[Tensor | None, Tensor]:
        """
        Generate FiLM parameters from a conditioning vector.

        Args:
            cond (Tensor): Shape [B, D_cond]. Conditioning vector.

        Returns:
            tuple:
                - scale (Tensor | None): Shape [B, C, 1] if enabled, else None.
                - bias (Tensor): Shape [B, C, 1].
        """
        film_params = self.film_generator(cond).unsqueeze(-1)
        if self.use_scale:
            scale = film_params[:, : self.out_channels]
            bias = film_params[:, self.out_channels :]
            return scale, bias
        return None, film_params


class MultiModalFiLM(nn.Module):
    """Multi-modal FiLM conditioning module with multiple fusion strategies."""

    def __init__(
        self,
        modality_dims: dict[str, int],
        out_channels: int,
        use_scale: bool = True,
        fusion: Literal["concat", "gated", "add"] = "concat",
    ) -> None:
        """Configure MultiModalFiLM and allocate its owned state.

        Args:
            modality_dims: Modality dims setting for this component.
            out_channels: Out channels setting for this component.
            use_scale: Use scale setting for this component.
            fusion: Fusion setting for this component.
        """
        super().__init__()
        self.modality_dims = dict(modality_dims)
        self.modality_names = list(modality_dims.keys())
        self.out_channels = out_channels
        self.use_scale = use_scale
        self.fusion = fusion

        self.unified_film: FiLMLayer | None
        if fusion == "concat":
            total_dim = sum(modality_dims.values())
            self.unified_film = FiLMLayer(total_dim, out_channels, use_scale)
            self.film_layers = None
            self.gate_networks = None
        elif fusion == "gated":
            self.film_layers = nn.ModuleDict(
                {
                    name: FiLMLayer(dim, out_channels, use_scale)
                    for name, dim in modality_dims.items()
                }
            )
            self.gate_networks = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.Linear(dim, dim // 2),
                        nn.ReLU(),
                        nn.Linear(dim // 2, 1),
                        nn.Sigmoid(),
                    )
                    for name, dim in modality_dims.items()
                }
            )
            self.unified_film = None
        elif fusion == "add":
            self.film_layers = nn.ModuleDict(
                {
                    name: FiLMLayer(dim, out_channels, use_scale)
                    for name, dim in modality_dims.items()
                }
            )
            self.unified_film = None
            self.gate_networks = None
        else:
            raise ValueError(f"Unknown fusion: {fusion}. Choose from: 'concat', 'gated', 'add'")

    def forward(self, features: Tensor, conditions: dict[str, Tensor]) -> Tensor:
        """
        Apply multi-modal FiLM modulation to features.

        Args:
            features (Tensor): Shape [B, C, L]. Features to modulate.
            conditions (dict[str, Tensor]): Each value is shape [B, D_mod] for the modality.

        Returns:
            Tensor: Shape [B, C, L]. Modulated features (same shape as input).
        """
        if self.fusion == "concat":
            return self._forward_concat(features, conditions)
        if self.fusion == "gated":
            return self._forward_gated(features, conditions)
        if self.fusion == "add":
            return self._forward_add(features, conditions)
        raise ValueError(f"Unknown fusion: {self.fusion}")

    def _forward_concat(self, features: Tensor, conditions: dict[str, Tensor]) -> Tensor:
        """Apply FiLM from concatenated modality vectors.

        Args:
            features: Feature tensor shaped (B, C, T).
            conditions: Named conditioning vectors, each shaped (B, D_modality).

        Returns:
            The computed Tensor result.
        """
        B = features.shape[0]
        cond_list: list[Tensor] = []
        any_provided = False

        for name in self.modality_names:
            expected_dim = self.modality_dims[name]
            if name in conditions and conditions[name] is not None:
                cond = conditions[name]
                if cond.shape[0] != B or cond.shape[-1] != expected_dim:
                    raise ValueError(
                        f"Condition '{name}' has shape {tuple(cond.shape)} but expected (B={B}, {expected_dim})."
                    )
                cond_list.append(cond)
                any_provided = True
            else:
                cond_list.append(
                    torch.zeros((B, expected_dim), device=features.device, dtype=features.dtype)
                )

        if not any_provided:
            return features

        unified_cond = torch.cat(cond_list, dim=-1)
        assert self.unified_film is not None
        scale, bias = self.unified_film(unified_cond)
        if self.use_scale and scale is not None:
            return cast(Tensor, scale * features + bias)
        return cast(Tensor, features + bias)

    def _forward_gated(self, features: Tensor, conditions: dict[str, Tensor]) -> Tensor:
        """Combine modality-specific FiLM outputs with learned gates.

        Args:
            features: Feature tensor shaped (B, C, T).
            conditions: Named conditioning vectors, each shaped (B, D_modality).

        Returns:
            The computed Tensor result.
        """
        modulated_outputs = []
        gate_values = []

        for name in self.modality_names:
            if name not in conditions:
                continue
            cond = conditions[name]
            scale, bias = self.film_layers[name](cond)  # type: ignore[index]
            modulated = (
                (scale * features + bias)
                if (self.use_scale and scale is not None)
                else (features + bias)
            )
            gate = self.gate_networks[name](cond).unsqueeze(-1)  # type: ignore[index]
            modulated_outputs.append(gate * modulated)
            gate_values.append(gate)

        if not modulated_outputs:
            return features

        output = sum(modulated_outputs)
        total_gate = sum(gate_values)
        return cast(Tensor, output / (total_gate + 1e-8))

    def _forward_add(self, features: Tensor, conditions: dict[str, Tensor]) -> Tensor:
        """Add independent modality FiLM contributions.

        Args:
            features: Feature tensor shaped (B, C, T).
            conditions: Named conditioning vectors, each shaped (B, D_modality).

        Returns:
            The computed Tensor result.
        """
        output = features
        for name in self.modality_names:
            if name not in conditions:
                continue
            cond = conditions[name]
            scale, bias = self.film_layers[name](cond)  # type: ignore[index]
            modulated = (
                (scale * features + bias)
                if (self.use_scale and scale is not None)
                else (features + bias)
            )
            output = output + (modulated - features)
        return output


# =============================================================================
# Convolutional Blocks
# =============================================================================


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(
        self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8
    ) -> None:
        """Configure Conv1dBlock and allocate its owned state.

        Args:
            inp_channels: Inp channels setting for this component.
            out_channels: Out channels setting for this component.
            kernel_size: Kernel size setting for this component.
            n_groups: N groups setting for this component.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Convolutional block forward.

        Args:
            x (Tensor): Shape [B, C_in, L].

        Returns:
            Tensor: Shape [B, C_out, L].
        """
        return cast(Tensor, self.block(x))


class MultiModalResidualBlock1d(nn.Module):
    """Residual block with multi-modal FiLM conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modality_dims: dict[str, int],
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film_scale: bool = True,
        film_fusion: Literal["concat", "gated", "add"] = "gated",
    ) -> None:
        """Configure MultiModalResidualBlock1d and allocate its owned state.

        Args:
            in_channels: In channels setting for this component.
            out_channels: Out channels setting for this component.
            modality_dims: Modality dims setting for this component.
            kernel_size: Kernel size setting for this component.
            n_groups: N groups setting for this component.
            use_film_scale: Use film scale setting for this component.
            film_fusion: Film fusion setting for this component.
        """
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        self.film = MultiModalFiLM(
            modality_dims=modality_dims,
            out_channels=out_channels,
            use_scale=use_film_scale,
            fusion=film_fusion,
        )
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, conditions: dict[str, Tensor]) -> Tensor:
        """
        Residual block forward with multi-modal FiLM conditioning.

        Args:
            x (Tensor): Shape [B, C_in, L].
            conditions (dict[str, Tensor]): Per-modality conditioning vectors, each shape [B, D_mod].

        Returns:
            Tensor: Shape [B, C_out, L].
        """
        out = self.conv1(x)
        out = self.film(out, conditions)
        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return cast(Tensor, out)


class ConditionalResidualBlock1d(nn.Module):
    """Legacy single-modality conditional residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film_scale_modulation: bool = True,
    ) -> None:
        """Configure ConditionalResidualBlock1d and allocate its owned state.

        Args:
            in_channels: In channels setting for this component.
            out_channels: Out channels setting for this component.
            cond_dim: Cond dim setting for this component.
            kernel_size: Kernel size setting for this component.
            n_groups: N groups setting for this component.
            use_film_scale_modulation: Use film scale modulation setting for this component.
        """
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Legacy residual block forward with single conditioning vector.

        Args:
            x (Tensor): Shape [B, C_in, L].
            cond (Tensor): Shape [B, D_cond]. Single conditioning vector.

        Returns:
            Tensor: Shape [B, C_out, L].
        """
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            out = out + cond_embed
        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return cast(Tensor, out)


class MultiModalPolicyHead(nn.Module):
    """Multi-modal FiLM-conditioned U-Net for action prediction."""

    def __init__(self, config: PolicyHeadConfig) -> None:
        """Configure MultiModalPolicyHead and allocate its owned state.

        Args:
            config: Validated settings owned by this component.
        """
        super().__init__()
        self.config = config

        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(config.time_embed_dim),
            nn.Linear(config.time_embed_dim, config.time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.time_embed_dim * 4, config.time_embed_dim),
        )

        self.modality_dims: dict[str, int] = {"time": config.time_embed_dim}
        for mod in config.enabled_modalities:
            self.modality_dims[mod.name] = mod.total_dim

        in_out = [(config.action_dim, config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        common_kwargs: _MultiResidualSettings = {
            "modality_dims": self.modality_dims,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale": config.use_film_scale_modulation,
            "film_fusion": config.film_fusion,
        }

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        MultiModalResidualBlock1d(dim_in, dim_out, **common_kwargs),
                        MultiModalResidualBlock1d(dim_out, dim_out, **common_kwargs),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.mid_modules = nn.ModuleList(
            [
                MultiModalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_kwargs
                ),
                MultiModalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_kwargs
                ),
            ]
        )

        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        MultiModalResidualBlock1d(dim_in * 2, dim_out, **common_kwargs),
                        MultiModalResidualBlock1d(dim_out, dim_out, **common_kwargs),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_dim, 1),
        )

    def forward(
        self, x: Tensor, timestep: Tensor, conditions: dict[str, Tensor] | None = None
    ) -> Tensor:
        """
        Multi-modal policy head forward.

        Args:
            x (Tensor): Shape [B, T, A]. Input trajectory-like tensor (e.g., noisy actions).
            timestep (Tensor): Shape [B]. Diffusion/flow timestep.
            conditions (dict[str, Tensor] | None): Optional modality vectors. Each value shape [B, D_mod].

        Returns:
            Tensor: Shape [B, T, A]. Predicted output over the horizon (same shape as x).

        Details:
            1. Internally uses channel-first conv: x -> [B, A, T].
            2. Builds a conditioning dict containing time embedding + provided modalities.
            3. U-Net style down/mid/up with FiLM-modulated residual blocks.
        """
        x = x.transpose(1, 2)
        time_embed = self.time_encoder(timestep)
        cond_dict = {"time": time_embed}
        if conditions is not None:
            cond_dict.update(conditions)

        encoder_skip_features = []
        for stage in self.down_modules:
            resnet, resnet2, downsample = cast(nn.ModuleList, stage)
            x = resnet(x, cond_dict)
            x = resnet2(x, cond_dict)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, cond_dict)

        for stage in self.up_modules:
            resnet, resnet2, upsample = cast(nn.ModuleList, stage)
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, cond_dict)
            x = resnet2(x, cond_dict)
            x = upsample(x)

        x = self.final_conv(x)
        return x.transpose(1, 2)


class FiLMConvPolicyHead(nn.Module):
    """FiLM-conditioned Convolutional U-Net for action prediction."""

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        global_cond_dim: int,
        n_obs_steps: int = 1,
        time_embed_dim: int = 128,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        use_film_scale_modulation: bool = True,
    ) -> None:
        """Configure FiLMConvPolicyHead and allocate its owned state.

        Args:
            action_dim: Action dim setting for this component.
            horizon: Horizon setting for this component.
            global_cond_dim: Global cond dim setting for this component.
            n_obs_steps: N obs steps setting for this component.
            time_embed_dim: Time embed dim setting for this component.
            down_dims: Down dims setting for this component.
            kernel_size: Kernel size setting for this component.
            n_groups: N groups setting for this component.
            use_film_scale_modulation: Use film scale modulation setting for this component.
        """
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.global_cond_dim = global_cond_dim
        self.n_obs_steps = n_obs_steps

        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        cond_dim = time_embed_dim + global_cond_dim * n_obs_steps
        in_out = [(action_dim, down_dims[0])] + list(
            zip(down_dims[:-1], down_dims[1:], strict=True)
        )
        common_res_block_kwargs: _ResidualSettings = {
            "cond_dim": cond_dim,
            "kernel_size": kernel_size,
            "n_groups": n_groups,
            "use_film_scale_modulation": use_film_scale_modulation,
        }

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        ConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1d(down_dims[-1], down_dims[-1], **common_res_block_kwargs),
                ConditionalResidualBlock1d(down_dims[-1], down_dims[-1], **common_res_block_kwargs),
            ]
        )

        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        ConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size=kernel_size),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(self, x: Tensor, timestep: Tensor, global_cond: Tensor) -> Tensor:
        """
        Convolutional policy head forward (single `global_cond`).

        Args:
            x (Tensor): Shape [B, T, A]. Input trajectory-like tensor.
            timestep (Tensor): Shape [B]. Diffusion/flow timestep.
            global_cond (Tensor): Shape [B, D] or [B, n_obs_steps, D]. Global conditioning features.

        Returns:
            Tensor: Shape [B, T, A]. Predicted output over the horizon (same shape as x).

        Details:
            1. Flattens `global_cond` over observation steps if needed.
            2. Concatenates time embedding + global conditioning into a single vector.
            3. U-Net style conv with conditional residual blocks.
        """
        # Explicitly materialize NCL strides: contiguous() is a no-op when the
        # action dimension is one, leaving an ambiguous layout for cuDNN/Inductor.
        x = x.transpose(1, 2)  # (B, A, T)
        if x.shape[1] == 1:
            x = x.clone(memory_format=torch.contiguous_format)
        timestep_embed = self.time_encoder(timestep)
        global_feature = (
            torch.cat([timestep_embed, global_cond.flatten(start_dim=1)], dim=-1)
            if global_cond is not None
            else timestep_embed
        )

        encoder_skip_features = []
        for stage in self.down_modules:
            resnet, resnet2, downsample = cast(nn.ModuleList, stage)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for stage in self.up_modules:
            resnet, resnet2, upsample = cast(nn.ModuleList, stage)
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.transpose(1, 2)


def create_policy_head(
    action_dim: int,
    horizon: int,
    modalities: list[ModalityConfig | dict[str, Any]] | None = None,
    global_cond_dim: int | None = None,
    n_obs_steps: int = 1,
    **kwargs: Any,
) -> MultiModalPolicyHead | FiLMConvPolicyHead:
    """
    Policy head factory.

    Args:
        action_dim (int): A (last dim of trajectory tensors).
        horizon (int): T (sequence length).
        modalities (list[ModalityConfig | dict[str, Any]] | None): If provided, constructs `MultiModalPolicyHead`.
        global_cond_dim (int | None): If provided (and modalities is None), constructs `FiLMConvPolicyHead`.
        n_obs_steps (int): Observation steps used when global_cond is [B, n_obs_steps, D].
        **kwargs: Passed through to the head constructor/config.

    Returns:
        MultiModalPolicyHead | FiLMConvPolicyHead: Instantiated policy head.
    """
    if modalities is not None:
        config = PolicyHeadConfig(
            action_dim=action_dim,
            horizon=horizon,
            modalities=[
                ModalityConfig(**item) if isinstance(item, dict) else item for item in modalities
            ],
            **kwargs,
        )
        return MultiModalPolicyHead(config)
    if global_cond_dim is not None:
        return FiLMConvPolicyHead(
            action_dim=action_dim,
            horizon=horizon,
            global_cond_dim=global_cond_dim,
            n_obs_steps=n_obs_steps,
            **kwargs,
        )
    raise ValueError("Either 'modalities' or 'global_cond_dim' must be provided")
