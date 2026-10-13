#!/usr/bin/env python

"""Attention-based fusion/pooling modules.

These modules fuse information from token sequences or spatial feature maps using
*externally provided* query tokens and a small Transformer stack.

Query token generation/conditioning (learnable queries, language prompts, etc.) lives in
`robopolicy.modules.task`.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from robopolicy.modules.transformer.transformer import TransformerStack


def _require_query_tokens(
    *,
    batch_size: int,
    device: torch.device,
    query_tokens: Tensor | None,
) -> Tensor:
    """Return query tokens of shape (B, Tq, D) on `device`."""
    if query_tokens is None:
        raise ValueError(
            "query_tokens is required. Provide it explicitly, e.g. from `robopolicy.modules.task` "
            "(learnable queries, language prompts, etc.)."
        )

    q = query_tokens.to(device=device)
    if q.ndim == 2:
        q = q.unsqueeze(0).expand(batch_size, q.shape[0], q.shape[1])
    if q.ndim != 3 or q.shape[0] != batch_size:
        raise ValueError(
            f"query_tokens must have shape (Tq, D) or (B, Tq, D), got {tuple(q.shape)}"
        )
    return q


class SelfAttentionPooling(nn.Module):
    """Self-attention based pooling for image features.

    Similar to TokenizerSelfAttentionGlobalConditioner from flow_state but adapted for image features.
    Applies self-attention over the concatenation of (query tokens, spatial tokens).
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        embed_dim: int | None = None,
        out_dim: int | None = None,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        *,
        mask_type: str = "none",
    ) -> None:
        """Configure SelfAttentionPooling and allocate its owned state.

        Args:
            input_shape: Per-sample feature map shape (C, H, W).
            embed_dim: Embed dim setting for this component.
            out_dim: Out dim setting for this component.
            num_layers: Num layers setting for this component.
            num_heads: Num heads setting for this component.
            ff_dim: Ff dim setting for this component.
            dropout: Dropout setting for this component.
            mask_type: Mask type setting for this component.
        """
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape
        self.spatial_size = self._in_h * self._in_w

        self.embed_dim = embed_dim if embed_dim is not None else self._in_c
        self.out_dim = out_dim if out_dim is not None else self.embed_dim
        self.mask_type = str(mask_type)

        # Input projection
        self.input_proj = nn.Linear(self._in_c, self.embed_dim)
        self.input_norm = nn.LayerNorm(self.embed_dim)

        # Self-attention encoder
        self.encoder = TransformerStack(
            num_layers=num_layers,
            dim=self.embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_dim,
            dropout=dropout,
            attention_dropout=dropout,
            activation="gelu",
            use_cross_attention=False,
        )

        # Output normalization and projection
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, self.out_dim)

    def get_mask(
        self,
        device: torch.device,
        *,
        num_queries: int,
        num_tokens: int,
    ) -> Tensor | None:
        """Self-attention mask for the concatenated sequence [queries, tokens].

        Notes:
        - "query": blocks query-to-query attention (off-diagonal) only. This prevents direct
          query-query mixing within a single layer, but with multiple layers queries can still
          communicate indirectly via token updates (tokens can read queries in layer ℓ, then
          queries read tokens in layer ℓ+1).
        - "query_strict": additionally blocks token->query attention, preventing tokens from
          reading queries. This makes queries conditionally independent given the (fixed) token
          stream even across multiple layers.
        """
        if self.mask_type == "none":
            return None
        if self.mask_type not in {"query", "query_strict"}:
            raise ValueError(
                f"Unknown mask_type={self.mask_type!r} (expected 'none', 'query', 'query_strict')."
            )

        t = int(num_queries) + int(num_tokens)
        if num_queries <= 0:
            raise ValueError(f"num_queries must be > 0, got {num_queries}")
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be >= 0, got {num_tokens}")

        attn_mask = torch.zeros((t, t), dtype=torch.bool, device=device)

        # Block attention between different query positions.
        qq = torch.ones((num_queries, num_queries), dtype=torch.bool, device=device)
        qq.fill_diagonal_(False)
        attn_mask[:num_queries, :num_queries] = qq

        if self.mask_type == "query_strict" and num_tokens > 0:
            # Block tokens from attending to queries (prevents query info from entering token stream).
            attn_mask[num_queries:, :num_queries] = True

        return attn_mask

    def forward(
        self,
        features: Tensor,
        *,
        query_tokens: Tensor,
        return_intermediate: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """
        Args:
            features: (B, C, H, W) spatial feature map, or (B, Tk, C) token sequence
            query_tokens: query tokens, shape (Tq, D) or (B, Tq, D)
        Returns:
            (B, Tq, out_dim) pooled features for each query token
        """
        # Accept either feature maps (B, C, H, W) or token sequences (B, Tk, C).
        if features.ndim == 4:
            B, C, H, W = features.shape
            assert C == self._in_c and H == self._in_h and W == self._in_w
            tokens = features.flatten(2).transpose(1, 2)  # (B, N, C)
        elif features.ndim == 3:
            B, Tk, C = features.shape
            assert C == self._in_c
            tokens = features
        else:
            raise ValueError(f"Expected features to have ndim=3 or 4, got {features.ndim}.")

        tokens = self.input_norm(self.input_proj(tokens))  # (B, Tk, embed_dim)

        q_tokens = _require_query_tokens(
            batch_size=B,
            device=features.device,
            query_tokens=query_tokens,
        )  # (B, Tq, D)

        # Concatenate query tokens with spatial tokens
        all_tokens = torch.cat([q_tokens, tokens], dim=1)  # (B, Tq+Tk, embed_dim)

        # Apply self-attention
        Tq = q_tokens.shape[1]
        Tk = tokens.shape[1]
        attn_mask = self.get_mask(features.device, num_queries=Tq, num_tokens=Tk)
        encoded_tokens, attention_layers = self.encoder(
            all_tokens, self_attn_mask=attn_mask, return_attention=return_intermediate
        )

        # Extract query token outputs
        token_embed = self.output_norm(encoded_tokens[:, :Tq, :])  # (B, Tq, embed_dim)
        output_tokens = self.output_proj(token_embed)  # (B, Tq, out_dim)

        if not return_intermediate:
            return cast(Tensor, output_tokens)

        intermediate: dict[str, Tensor] = {"query_token_embed": token_embed}
        if attention_layers:
            weights = [
                layer["self"].attn_weights[:, :, :Tq, Tq:]
                for layer in attention_layers.values()
                if "self" in layer
            ]
            if weights:
                intermediate["query_attn"] = torch.stack(weights).mean(dim=(0, 2))
                # Shape: (B, T_query, T_spatial), averaging layers and heads.
        return cast(Tensor, output_tokens), intermediate


class CrossAttentionPooling(nn.Module):
    """Cross-attention based pooling for image features.

    Similar to TokenizerCrossAttentionGlobalConditioner from flow_state but adapted for image features.
    Uses externally provided query tokens that cross-attend to spatial feature tokens.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        embed_dim: int | None = None,
        out_dim: int | None = None,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        *,
        mask_type: str = "none",
    ) -> None:
        """Configure CrossAttentionPooling and allocate its owned state.

        Args:
            input_shape: Per-sample feature map shape (C, H, W).
            embed_dim: Embed dim setting for this component.
            out_dim: Out dim setting for this component.
            num_layers: Num layers setting for this component.
            num_heads: Num heads setting for this component.
            ff_dim: Ff dim setting for this component.
            dropout: Dropout setting for this component.
            mask_type: Mask type setting for this component.
        """
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape
        self.spatial_size = self._in_h * self._in_w

        self.embed_dim = embed_dim if embed_dim is not None else self._in_c
        self.out_dim = out_dim if out_dim is not None else self.embed_dim

        # Input projection for spatial tokens (keys/values)
        self.input_proj = nn.Linear(self._in_c, self.embed_dim)
        self.input_norm = nn.LayerNorm(self.embed_dim)

        # Cross-attention decoder
        self.decoder = TransformerStack(
            num_layers=num_layers,
            dim=self.embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_dim,
            dropout=dropout,
            attention_dropout=dropout,
            activation="gelu",
            use_cross_attention=True,
            context_dim=self.embed_dim,
        )

        # Output normalization and projection
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, self.out_dim)
        self.mask_type = str(mask_type)

    def get_mask(self, device: torch.device, *, num_queries: int) -> Tensor | None:
        """Query-side self-attention mask (no memory mask for image tokens)."""
        if self.mask_type == "none":
            return None
        elif self.mask_type == "query":
            # Block attention between different query positions (each query attends only to itself).
            query_mask = torch.ones(num_queries, num_queries, dtype=torch.bool, device=device)
            query_mask.fill_diagonal_(False)
            return query_mask
        else:
            raise ValueError(f"Unknown mask_type={self.mask_type!r} (expected 'none', 'query').")

    def forward(
        self,
        features: Tensor,
        *,
        query_tokens: Tensor,
        return_intermediate: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """
        Args:
            features: (B, C, H, W) spatial feature map, or (B, Tk, C) token sequence
            query_tokens: query tokens, shape (Tq, D) or (B, Tq, D)
        Returns:
            (B, Tq, out_dim) pooled features for each query token
        """
        # Accept either feature maps (B, C, H, W) or token sequences (B, Tk, C).
        if features.ndim == 4:
            B, C, H, W = features.shape
            assert C == self._in_c and H == self._in_h and W == self._in_w
            tokens = features.flatten(2).transpose(1, 2)  # (B, N, C)
        elif features.ndim == 3:
            B, Tk, C = features.shape
            assert C == self._in_c
            tokens = features
        else:
            raise ValueError(f"Expected features to have ndim=3 or 4, got {features.ndim}.")

        spatial_tokens = self.input_norm(self.input_proj(tokens))  # (B, Tk, embed_dim)

        q_tokens = _require_query_tokens(
            batch_size=B,
            device=features.device,
            query_tokens=query_tokens,
        )  # (B, Tq, D)

        query_mask = self.get_mask(features.device, num_queries=q_tokens.shape[1])
        # print(query_mask)

        # Apply cross-attention: queries attend to spatial tokens (no memory mask available for image tokens).
        decoded_tokens, attn_by_layer = self.decoder(
            q_tokens,
            context=spatial_tokens,
            self_attn_mask=query_mask,
            cross_attn_mask=None,
            return_attention=return_intermediate,
        )  # (B, Tq, embed_dim)

        # Normalize and project
        token_embed = self.output_norm(decoded_tokens)  # (B, Tq, embed_dim)
        output_tokens = self.output_proj(token_embed)  # (B, Tq, out_dim)

        if not return_intermediate:
            return cast(Tensor, output_tokens)

        # Extract per-query attention distributions over spatial tokens for overlap loss/metrics.
        query_attn: Tensor | None = None

        if attn_by_layer is not None:
            cross_by_layer = []
            for _, scores_dict in attn_by_layer.items():
                if "cross" not in scores_dict:
                    continue
                cross_by_layer.append(scores_dict["cross"].attn_weights)  # (B, H, Tq, Tk)
            if len(cross_by_layer) > 0:
                attn_all = torch.stack(cross_by_layer, dim=0).mean(dim=0)  # (B, H, Tq, Tk)
                attn_mean = attn_all.mean(dim=1)  # (B, Tq, Tk)
                query_attn = attn_mean

        intermediate: dict[str, Tensor] = {}
        if query_attn is not None:
            intermediate["query_attn"] = query_attn  # (B, Tq, Tk)
        # Also expose the query embeddings (pre-projection).
        intermediate["query_token_embed"] = token_embed  # (B, Tq, D)
        return cast(Tensor, output_tokens), intermediate
