#!/usr/bin/env python
"""Transformer-based action decoder for flow/diffusion matching (DiT style).

Implements an Adaptive Layer Norm (AdaLN) transformer decoder that:
- Conditions on timestep + robot state via AdaLN (efficient for scalar/low-dim inputs)
- Cross-attends to a VLM token sequence for visual + language context
- Applies self-attention over the action horizon for global temporal reasoning

This is the action decoder counterpart to FiLMConvPolicyHead, designed for VLM-based
policies where the full VLM token sequence is available (no pre-compression needed).

Reference architecture: DiT (Peebles & Xie, 2023), π0 (Black et al., 2024).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn as nn
from torch import Tensor

from robopolicy.modules.transformer.transformer import (
    CrossAttention,
    FeedForward,
    KVCache,
    SelfAttention,
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class SinusoidalPosEmb(nn.Module):
    """1-D sinusoidal positional / timestep embedding."""

    def __init__(self, dim: int) -> None:
        """Configure SinusoidalPosEmb and allocate its owned state.

        Args:
            dim: Dim setting for this component.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """x: (B,) float timestep values → (B, dim)."""
        device = x.device
        half_dim = self.dim // 2
        frequency_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -frequency_scale)
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ---------------------------------------------------------------------------
# Adaptive Layer Norm
# ---------------------------------------------------------------------------


class AdaLN(nn.Module):
    """Adaptive Layer Norm: modulate normalized activations with (scale, shift) from cond.

    Replaces standard LayerNorm(x) with:
        x_norm = LayerNorm(x)
        scale, shift = Linear(cond).chunk(2)
        out = x_norm * (1 + scale) + shift

    The Linear is zero-initialized so the block starts as an identity residual.
    """

    def __init__(self, dim: int, cond_dim: int) -> None:
        """Configure AdaLN and allocate its owned state.

        Args:
            dim: Dim setting for this component.
            cond_dim: Cond dim setting for this component.
        """
        super().__init__()
        # elementwise_affine=False: no learned γ/β — AdaLN provides those from cond
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * dim, bias=True),
        )
        # Zero-init: block starts as identity (scale=0, shift=0)
        nn.init.zeros_(cast(nn.Linear, self.modulation[-1]).weight)
        nn.init.zeros_(cast(nn.Linear, self.modulation[-1]).bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        x: (B, T, D)
        cond: (B, cond_dim)
        returns: (B, T, D) adaptively normalized
        """
        shift, scale = self.modulation(cond).chunk(2, dim=-1)  # each (B, D)
        return cast(Tensor, self.norm(x) * (1.0 + scale[:, None, :]) + shift[:, None, :])


# ---------------------------------------------------------------------------
# AdaLN Transformer Block
# ---------------------------------------------------------------------------


class AdaLNTransformerBlock(nn.Module):
    """Transformer block with AdaLN conditioning and cross-attention.

    Structure (pre-norm with AdaLN):
        x = x + drop(self_attn(AdaLN(x, cond)))
        x = x + drop(cross_attn(AdaLN(x, cond), context))
        x = x + drop(ffn(AdaLN(x, cond)))

    Args:
        dim: hidden dimension of action tokens (D_dec)
        num_heads: number of attention heads
        ff_dim: feedforward hidden dimension (typically 4 * dim)
        cond_dim: conditioning vector dimension (timestep + robot state)
        context_dim: VLM token hidden dimension (D_vlm)
        dropout: residual dropout
        attention_dropout: attention weight dropout
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_dim: int,
        cond_dim: int,
        context_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        """Configure AdaLNTransformerBlock and allocate its owned state.

        Args:
            dim: Dim setting for this component.
            num_heads: Num heads setting for this component.
            ff_dim: Ff dim setting for this component.
            cond_dim: Cond dim setting for this component.
            context_dim: Context dim setting for this component.
            dropout: Dropout setting for this component.
            attention_dropout: Attention dropout setting for this component.
        """
        super().__init__()

        # Self-attention branch
        self.ada_norm1 = AdaLN(dim, cond_dim)
        self.self_attn = SelfAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.drop1 = nn.Dropout(dropout)

        # Cross-attention branch (action tokens attend to VLM context)
        self.ada_norm2 = AdaLN(dim, cond_dim)
        self.cross_attn = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.drop2 = nn.Dropout(dropout)

        # FFN branch
        self.ada_norm3 = AdaLN(dim, cond_dim)
        self.ff = FeedForward(dim=dim, hidden_dim=ff_dim, dropout=dropout, activation="gelu")
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        context: Tensor,
        *,
        context_key_padding_mask: Tensor | None = None,
        cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> Tensor:
        """
        Args:
            x: (B, T_action, D) action tokens
            cond: (B, cond_dim) AdaLN conditioning (timestep + robot state)
            context: (B, L_vlm, D_vlm) VLM token sequence
            context_key_padding_mask: (B, L_vlm) bool, True = padding to ignore
            cache: optional KVCache for cross-attention (caches projected VLM K/V)
            layer_idx: layer index for KVCache keying
        Returns:
            (B, T_action, D)
        """
        # Self-attention
        x_sa = self.ada_norm1(x, cond)
        sa_out, _ = self.self_attn(x_sa, layer_idx=layer_idx, cache_name="self")
        x = x + self.drop1(sa_out)

        # Cross-attention: action tokens attend to VLM context
        x_ca = self.ada_norm2(x, cond)
        ca_out, _ = self.cross_attn(
            x_ca,
            context=context,
            context_key_padding_mask=context_key_padding_mask,
            cache=cache,
            layer_idx=layer_idx,
            cache_name="cross",
            static_kv=True,  # VLM context is constant during denoising iterations
        )
        x = x + self.drop2(ca_out)

        # FFN
        x_ff = self.ada_norm3(x, cond)
        x = x + self.drop3(self.ff(x_ff))

        return x


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TransformerActionDecoderConfig:
    """Configuration for TransformerActionDecoder."""

    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    ff_dim: int = 2048
    time_embed_dim: int = 256
    dropout: float = 0.1
    attention_dropout: float = 0.0
    interval_conditioning: bool = False
    state_conditioning: Literal["mean", "current"] = "mean"


# ---------------------------------------------------------------------------
# Full Decoder
# ---------------------------------------------------------------------------


class TransformerActionDecoder(nn.Module):
    """Transformer-based action decoder for flow/diffusion matching.

    Replaces FiLMConvPolicyHead when a VLM token sequence is available as context.
    The decoder cross-attends to VLM tokens directly — no prior compression needed.

    Pipeline:
        1. Embed noisy actions: Linear(action_dim → d_model) + learned positional embed
        2. Build AdaLN cond: SinPosEmb(t) → MLP,  robot_state → Linear,  concat
        3. N × AdaLNTransformerBlock: self-attn → cross-attn (VLM) → FFN, each with AdaLN
        4. LayerNorm + Linear(d_model → action_dim) [zero-initialized]

    Args:
        action_dim: dimensionality of each action step
        horizon: number of action steps (T_action)
        context_dim: VLM encoder output dimension (D_vlm)
        robot_state_dim: robot proprioception dimension
        config: TransformerActionDecoderConfig (architecture hyperparameters)
    """

    def __init__(
        self,
        *,
        action_dim: int,
        horizon: int,
        context_dim: int,
        robot_state_dim: int,
        config: TransformerActionDecoderConfig | None = None,
    ) -> None:
        """Configure TransformerActionDecoder and allocate its owned state.

        Args:
            action_dim: Action dim setting for this component.
            horizon: Horizon setting for this component.
            context_dim: Context dim setting for this component.
            robot_state_dim: Robot state dim setting for this component.
            config: Validated settings owned by this component.
        """
        super().__init__()
        if config is None:
            config = TransformerActionDecoderConfig()
        self.config = config
        self.action_dim = action_dim
        self.horizon = horizon
        self.d_model = config.d_model

        # --- Action token embedding ---
        self.action_proj = nn.Linear(action_dim, config.d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, horizon, config.d_model) * 0.02)

        # --- Conditioning: sinusoidal timestep + robot state ---
        self.time_emb = SinusoidalPosEmb(config.time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embed_dim, config.time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(config.time_embed_dim * 2, config.time_embed_dim),
        )
        self.state_proj = nn.Linear(robot_state_dim, config.time_embed_dim)
        # cond_dim = time_embed_dim (from t) + time_embed_dim (from state)
        cond_dim = config.time_embed_dim * 2

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList(
            [
                AdaLNTransformerBlock(
                    dim=config.d_model,
                    num_heads=config.n_heads,
                    ff_dim=config.ff_dim,
                    cond_dim=cond_dim,
                    context_dim=context_dim,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

        # --- Output projection ---
        self.out_norm = nn.LayerNorm(config.d_model)
        self.out_proj = nn.Linear(config.d_model, action_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Allocate after the original layers to preserve their seeded initialization.
        self.interval_mlp: nn.Sequential | None = None
        if config.interval_conditioning:
            self.interval_mlp = nn.Sequential(
                nn.Linear(config.time_embed_dim, config.time_embed_dim * 2),
                nn.SiLU(),
                nn.Linear(config.time_embed_dim * 2, config.time_embed_dim),
            )
            nn.init.zeros_(cast(nn.Linear, self.interval_mlp[-1]).weight)
            nn.init.zeros_(cast(nn.Linear, self.interval_mlp[-1]).bias)

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        context: Tensor,
        robot_state: Tensor,
        *,
        time_delta: Tensor | None = None,
        context_key_padding_mask: Tensor | None = None,
        cache: KVCache | None = None,
    ) -> Tensor:
        """
        Args:
            x: (B, horizon, action_dim) noisy action sequence
            timestep: (B,) diffusion timestep in [0, 1]
            context: (B, L_vlm, D_vlm) VLM token sequence (frozen encoder output)
            robot_state: (B, n_obs_steps, Dr) robot proprioception
                         Averaged over n_obs_steps before projection.
            time_delta: (B,) interval length, required only by the iMF head.
            context_key_padding_mask: (B, L_vlm) bool, True = padding position
            cache: optional KVCache to cache cross-attention K/V projections
        Returns:
            (B, horizon, action_dim) predicted velocity field
        """
        # Action token embedding
        tokens = self.action_proj(x) + self.pos_embed  # (B, horizon, d_model)

        # Build AdaLN conditioning vector
        t_emb = self.time_mlp(self.time_emb(timestep))  # (B, time_embed_dim)
        if self.interval_mlp is not None:
            if time_delta is None:
                raise ValueError("Interval-conditioned head requires time_delta")
            t_emb = t_emb + self.interval_mlp(self.time_emb(time_delta))
        elif time_delta is not None:
            raise ValueError("This head does not support interval conditioning")
        state = (
            robot_state[:, -1]
            if self.config.state_conditioning == "current"
            else robot_state.mean(dim=1)
        )
        state_emb = self.state_proj(state)  # (B, time_embed_dim)
        cond = torch.cat([t_emb, state_emb], dim=-1)  # (B, cond_dim)

        # Transformer blocks
        for i, block in enumerate(self.blocks):
            tokens = block(
                tokens,
                cond=cond,
                context=context,
                context_key_padding_mask=context_key_padding_mask,
                cache=cache,
                layer_idx=i,
            )

        return cast(Tensor, self.out_proj(self.out_norm(tokens)))  # (B, horizon, action_dim)
