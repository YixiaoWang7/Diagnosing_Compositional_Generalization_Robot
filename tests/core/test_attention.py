"""Masked attention preserves blocking in fast and introspection paths."""

import pytest
import torch

from robopolicy.modules.transformer.transformer import scaled_dot_product_attention


@pytest.mark.parametrize("return_attention", [False, True])
@pytest.mark.parametrize("padding", [False, True])
def test_all_masked_attention_is_zero(return_attention, padding):
    queries = torch.randn(2, 2, 3, 4, requires_grad=True)
    keys = torch.randn(2, 2, 5, 4, requires_grad=True)
    values = torch.randn(2, 2, 5, 4, requires_grad=True)
    kwargs = (
        {"key_padding_mask": torch.ones(2, 5, dtype=torch.bool)}
        if padding
        else {"attn_mask": torch.ones(3, 5, dtype=torch.bool)}
    )
    output, scores = scaled_dot_product_attention(
        queries, keys, values, return_attention=return_attention, **kwargs
    )
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    assert torch.isfinite(queries.grad).all()
    if scores is not None:
        assert scores.attn_weights.count_nonzero() == 0
