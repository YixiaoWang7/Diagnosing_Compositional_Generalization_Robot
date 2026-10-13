"""iMF identities, gradients, interval sampling and production-head integration."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError
from torch import Tensor, nn

from robopolicy.algorithms import build_algorithm
from robopolicy.algorithms.improved_mean_flow import (
    ImprovedMeanFlowAlgorithm,
    ImprovedMeanFlowConfig,
)
from robopolicy.contracts.batch import PolicyInput
from robopolicy.policies.config import PolicyConfig
from robopolicy.policies.flow import build_policy
from tests.core.test_policy import policy_config


class AffineField(nn.Module):
    """Analytic field whose derivative exposes both interval and action tangents."""

    def __init__(self) -> None:
        """Initialize a scalar trainable coefficient."""
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(
        self, actions: Tensor, timestep: Tensor, global_cond: Tensor, *, time_delta: Tensor
    ) -> Tensor:
        """Return affine velocities (B,T,D), with scalar times/intervals (B,)."""
        return (
            self.scale * actions
            + timestep[:, None, None]
            + 2 * time_delta[:, None, None]
            + global_cond
        )


@pytest.mark.parametrize("boundary", [False, True])
def test_imf_analytic_jvp_and_stop_gradient(monkeypatch, boundary):
    algorithm = ImprovedMeanFlowAlgorithm()
    field = AffineField()
    actions = torch.tensor([[[0.3], [-0.2]], [[0.1], [0.6]]])
    noise = torch.tensor([[[0.5], [0.4]], [[-0.3], [0.8]]])
    context = torch.zeros_like(actions, requires_grad=True)
    timestep = torch.tensor([0.7, 0.6])
    endpoint = timestep if boundary else torch.tensor([0.2, 0.1])
    monkeypatch.setattr(algorithm, "sample_times", lambda _: (timestep, endpoint))
    monkeypatch.setattr(torch, "randn_like", lambda _: noise)
    time, interval = timestep[:, None, None], (timestep - endpoint)[:, None, None]
    samples = (1 - time) * actions + time * noise
    velocity = 0.2 * samples + time
    derivative = 0.2 * velocity + 3  # dz/dt=predicted v; d(t+2*(t-r))/dt=3.
    prediction = 0.2 * samples + time + 2 * interval
    residual = prediction + interval * derivative - (noise - actions)
    errors = residual.square().flatten(1).sum(1)
    weights = errors + 0.01
    loss = algorithm.compute_loss(field, actions, context)
    torch.testing.assert_close(loss, (errors / weights).mean())
    loss.backward()
    expected_grad = (2 * residual * samples).flatten(1).sum(1) / weights
    torch.testing.assert_close(field.scale.grad, expected_grad.mean())
    assert torch.isfinite(context.grad).all() and context.grad.abs().sum() > 0
    with torch.no_grad():
        torch.testing.assert_close(algorithm.compute_loss(field, actions, context), loss)


def test_imf_paired_times_and_configuration():
    torch.manual_seed(2)
    algorithm = ImprovedMeanFlowAlgorithm()
    timestep, endpoint = algorithm.sample_times(torch.empty(64, 16, 1))
    assert ((timestep > 0) & (timestep < 1) & (endpoint <= timestep)).all()
    assert torch.equal(timestep[:32], endpoint[:32])
    assert (timestep[32:] > endpoint[32:]).all()
    assert isinstance(build_algorithm(algorithm.config), ImprovedMeanFlowAlgorithm)
    for invalid in (
        {"data_proportion": 1.1},
        {"norm_eps": 0},
        {"norm_p": -1},
        {"num_inference_steps": 0},
        {"logit_normal_scale": 0},
        {"unknown": 1},
    ):
        with pytest.raises(ValidationError):
            ImprovedMeanFlowConfig(type="improved_mean_flow", **invalid)
    with pytest.raises(ValidationError):
        policy_config("conv", algorithm={"type": "improved_mean_flow"})
    with pytest.raises(ValidationError):
        policy_config(algorithm={"norm_p": 0.5})  # iMF needs an explicit tag.


@pytest.mark.parametrize("steps", [1, 2])
def test_imf_sampling_uses_interval_jumps(steps):
    class ConstantField(nn.Module):
        def __init__(self):
            super().__init__()
            self.times = []

        def forward(self, actions, timestep, global_cond, *, time_delta):
            self.times.append((timestep.clone(), time_delta.clone()))
            return torch.full_like(actions, 2)

    algorithm = ImprovedMeanFlowAlgorithm(
        ImprovedMeanFlowConfig(type="improved_mean_flow", num_inference_steps=steps)
    )
    field = ConstantField()
    noise = torch.ones(3, 4, 1)
    actual = algorithm.generate_samples(field, noise, torch.zeros(3, 2, 4))
    torch.testing.assert_close(actual, -noise)
    assert len(field.times) == steps and torch.equal(noise, torch.ones_like(noise))
    for index, (time, interval) in enumerate(field.times):
        torch.testing.assert_close(time, torch.full((3,), 1 - index / steps))
        torch.testing.assert_close(interval, torch.full((3,), 1 / steps))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_imf_image_policy_gradients_and_dropout(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.fail("CUDA acceptance requires a visible GPU")
    config = policy_config(images=True, algorithm={"type": "improved_mean_flow"})
    config = PolicyConfig.model_validate(
        {**config.model_dump(), "head": {**config.head.model_dump(), "dropout": 0.1}}
    )
    policy = build_policy(config).to(device).train()
    # The output projection starts at zero: prime it to test all upstream gradients.
    nn.init.normal_(policy.field.network.out_proj.weight, std=0.02)
    for block in policy.field.network.blocks:
        for name in ("ada_norm1", "ada_norm2", "ada_norm3"):
            nn.init.normal_(getattr(block, name).modulation[-1].weight, std=0.02)
    inputs = PolicyInput(
        state=torch.randn(3, 2, 2, device=device),
        images=torch.randn(3, 2, 2, 3, 16, 16, device=device),
        actions=torch.randn(3, 4, 1, device=device),
    )
    native_jvp = torch.func.jvp

    def checked_jvp(function, primals, tangents):
        prediction, derivative = native_jvp(function, primals, tangents)
        boundary = primals[1] == primals[2]
        torch.testing.assert_close(
            prediction[boundary].detach(), tangents[0][boundary], rtol=1e-4, atol=2e-6
        )
        return prediction, derivative

    monkeypatch.setattr(torch.func, "jvp", checked_jvp)
    with torch.autocast(device, dtype=torch.float16 if device == "cuda" else torch.bfloat16):
        loss = policy.loss(inputs)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    for module in (
        policy.encoder.vision,
        policy.encoder.queries,
        policy.field.network.interval_mlp,
    ):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
    policy.eval()
    with torch.no_grad():
        assert torch.isfinite(policy.loss(inputs))
    assert torch.isfinite(policy.predict(inputs)).all()


def test_imf_preserves_shared_initialization_and_default_fm_weights():
    config = policy_config(images=True)
    torch.manual_seed(2)
    fm = build_policy(config)
    torch.manual_seed(2)
    imf = build_policy(
        config.model_copy(update={"algorithm": ImprovedMeanFlowConfig(type="improved_mean_flow")})
    )
    common = fm.state_dict()
    for name, weight in common.items():
        torch.testing.assert_close(imf.state_dict()[name], weight, rtol=0, atol=0)
    assert set(imf.state_dict()) - set(common) == {
        f"field.network.interval_mlp.{layer}.{parameter}"
        for layer in (0, 2)
        for parameter in ("weight", "bias")
    }
