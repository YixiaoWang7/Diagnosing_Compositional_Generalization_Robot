from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor


def _as_bld(x: Tensor, *, batch_size: int | None = None) -> Tensor:
    """Normalize inputs to (B, L, D). Accepts (L, D) or (B, L, D)."""
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(
            f"Expected embeddings to have shape (L,D) or (B,L,D), got {tuple(x.shape)}"
        )
    if batch_size is not None and x.shape[0] != batch_size:
        raise ValueError(f"Batch mismatch: expected B={batch_size}, got {x.shape[0]}")
    return x


def _pad_or_truncate_seq(x: Tensor, *, length: int) -> Tensor:
    """Pad/truncate sequence length to `length` along dim=1."""
    if x.ndim != 3:
        raise ValueError(f"Expected x to be (B,L,D), got {tuple(x.shape)}")
    B, L, D = x.shape
    Tq = int(length)
    if Tq <= 0:
        raise ValueError("length must be >= 1")
    if L == Tq:
        return x
    if L > Tq:
        return x[:, :Tq, :]
    pad = x.new_zeros((B, Tq - L, D))
    return torch.cat([x, pad], dim=1)


@dataclass(frozen=True)
class HFTextEncoderConfig:
    """Optional HuggingFace text encoder config.

    This is *optional* to avoid forcing a transformers dependency for the project.
    """

    model_name: str
    max_length: int = 32
    freeze: bool = True


class LanguageQueryTokens(nn.Module):
    """Convert task language prompts into query tokens for attention-based fusion.

    Output is always (B, Tq, D) where:
    - B is batch size
    - Tq is `num_query_tokens`
    - D is `dim`

    You can use it in two ways:
    - **Provide precomputed embeddings**: pass `embeddings=(B,L,E)` (or `(L,E)`) to `forward`.
      The module will project to `dim` and pad/truncate to `num_query_tokens`.
    - **Provide raw prompt strings**: pass `prompts=[str, ...]` to `forward` *if* you constructed
      this module with `hf=HFTextEncoderConfig(...)` and `transformers` is installed.
    """

    def __init__(
        self,
        *,
        num_query_tokens: int,
        dim: int,
        embed_dim_in: int | None = None,
        hf: HFTextEncoderConfig | None = None,
    ) -> None:
        """Configure LanguageQueryTokens and allocate its owned state.

        Args:
            num_query_tokens: Num query tokens setting for this component.
            dim: Dim setting for this component.
            embed_dim_in: Embed dim in setting for this component.
            hf: Hf setting for this component.
        """
        super().__init__()
        self.num_query_tokens = int(num_query_tokens)
        self.dim = int(dim)
        if self.num_query_tokens <= 0:
            raise ValueError("num_query_tokens must be >= 1")
        if self.dim <= 0:
            raise ValueError("dim must be >= 1")

        self._use_hf = hf is not None
        self._hf_cfg = hf

        # If using HF, infer encoder width at runtime after loading; else embed_dim_in required.
        if not self._use_hf and embed_dim_in is None:
            raise ValueError("embed_dim_in is required when hf is not provided")

        self._embed_dim_in = None if embed_dim_in is None else int(embed_dim_in)
        if self._embed_dim_in is not None and self._embed_dim_in <= 0:
            raise ValueError("embed_dim_in must be >= 1 when provided")

        self._tokenizer: Any = None
        self._text_model: Any = None

        # Projection is created lazily if HF is used and embed width isn't known yet.
        self.proj: nn.Linear | None
        if self._embed_dim_in is not None:
            self.proj = nn.Linear(self._embed_dim_in, self.dim)
        else:
            self.proj = None

        if self._use_hf:
            self._init_hf()

    def _init_hf(self) -> None:
        """Load the explicitly requested optional text encoder and tokenizer."""
        assert self._hf_cfg is not None
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "LanguageQueryTokens was configured with `hf=...` but `transformers` "
                "is not available. Install it or pass precomputed embeddings instead."
            ) from e

        tokenizer_factory: Any = AutoTokenizer
        tok = tokenizer_factory.from_pretrained(self._hf_cfg.model_name)
        model = AutoModel.from_pretrained(self._hf_cfg.model_name)
        if self._hf_cfg.freeze:
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

        self._tokenizer = tok
        self._text_model = model

        # Create projection now that we know hidden size.
        hidden = int(getattr(model.config, "hidden_size", None) or getattr(model.config, "d_model"))
        self._embed_dim_in = hidden
        self.proj = nn.Linear(hidden, self.dim)

    @torch.no_grad()
    def encode_prompts(self, prompts: list[str]) -> Tensor:
        """Encode prompts with HuggingFace model to (B, L, E)."""
        if not self._use_hf:
            raise RuntimeError("encode_prompts requires hf=... configuration")
        assert self._tokenizer is not None and self._text_model is not None
        assert self._hf_cfg is not None

        batch = self._tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=int(self._hf_cfg.max_length),
            return_tensors="pt",
        )
        # Move tokenizer outputs to model device.
        device = next(self._text_model.parameters()).device
        batch = {k: v.to(device=device) for k, v in batch.items()}
        out = self._text_model(**batch)
        # Most transformer encoders expose last_hidden_state.
        if not hasattr(out, "last_hidden_state"):
            raise RuntimeError("HuggingFace model output missing `last_hidden_state`")
        return cast(Tensor, out.last_hidden_state)  # (B, L, E)

    def forward(
        self,
        *,
        batch_size: int | None = None,
        prompts: list[str] | None = None,
        embeddings: Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return query tokens (B, Tq, D).

        Provide exactly one of:
        - prompts: list[str] of length B (requires hf=... + transformers)
        - embeddings: (B, L, E) or (L, E)
        """
        if (prompts is None) == (embeddings is None):
            raise ValueError("Provide exactly one of `prompts` or `embeddings`")

        if prompts is not None:
            if batch_size is not None and batch_size != len(prompts):
                raise ValueError(f"batch_size={batch_size} but len(prompts)={len(prompts)}")
            x = self.encode_prompts(prompts)  # (B, L, E) on model device
        else:
            assert embeddings is not None
            x = _as_bld(embeddings, batch_size=batch_size)

        # Decide output device/dtype.
        if device is None:
            device = x.device
        if dtype is None:
            dtype = x.dtype
        x = x.to(device=device, dtype=dtype)

        if self.proj is None:
            raise RuntimeError("Projection layer was not initialized")
        x = self.proj(x)  # (B, L, D)
        x = _pad_or_truncate_seq(x, length=self.num_query_tokens)  # (B, Tq, D)
        return x
