"""Shared-head improved MeanFlow for conditional action sequences.

Implements Algorithm 1 of https://arxiv.org/abs/2512.02012. Time runs from
clean actions (0) to noise (1), opposite to this package's FM convention.
Reference defaults: https://github.com/Lyy-iiis/imeanflow/blob/main/imf.py.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from pydantic import Field
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from robopolicy.algorithms.flow_matching import LogitNormalScheduler
from robopolicy.contracts.schema import StrictConfig


class ImprovedMeanFlowConfig(StrictConfig):
    """Configure paired times, boundary training, adaptive loss and sampling.

    Attributes:
        type: Explicit selection; untagged configurations continue to select FM.
        num_inference_steps: Number of average-velocity jumps from noise to actions.
        logit_normal_loc: Mean of Gaussian logits used to sample both times.
        logit_normal_scale: Standard deviation of the Gaussian logits.
        data_proportion: Fraction of each batch trained at the boundary r=t.
        norm_p: Exponent of the detached per-example squared-error weight.
        norm_eps: Positive stabilizer added before raising that weight to norm_p.
    """

    type: Literal["improved_mean_flow"]
    num_inference_steps: int = Field(default=1, ge=1)
    logit_normal_loc: float = -0.4
    logit_normal_scale: float = Field(default=1.0, gt=0)
    data_proportion: float = Field(default=0.5, ge=0, le=1)
    norm_p: float = Field(default=1.0, ge=0, le=1)
    norm_eps: float = Field(default=0.01, gt=0)


class ImprovedMeanFlowAlgorithm:
    """Train and sample average velocities using the existing conditional field.

    Attributes:
        config: Immutable algorithm settings.
        scheduler: Shared logit-normal time sampler.
    """

    def __init__(self, config: ImprovedMeanFlowConfig | None = None) -> None:
        """Bind settings without creating networks or modifying their parameters."""
        self.config = config or ImprovedMeanFlowConfig(type="improved_mean_flow")
        self.scheduler = LogitNormalScheduler(
            self.config.logit_normal_loc, self.config.logit_normal_scale
        )

    def sample_times(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Sample ordered times t>=r, each (B,), for actions (B, T_action, D_action)."""
        first = self.scheduler.sample(actions.shape[0], actions.device, actions.dtype)
        second = self.scheduler.sample(actions.shape[0], actions.device, actions.dtype)
        timestep, endpoint = torch.maximum(first, second), torch.minimum(first, second)
        boundary = torch.arange(actions.shape[0], device=actions.device) < int(
            actions.shape[0] * self.config.data_proportion
        )
        return timestep, torch.where(boundary, timestep, endpoint)

    def compute_loss(self, vector_field_net: nn.Module, x_1: Tensor, global_cond: Tensor) -> Tensor:
        """Return scalar iMF velocity loss with a detached directional derivative.

        Args:
            vector_field_net: Field accepting actions, timestep, global_cond and time_delta.
            x_1: Clean normalized actions (B, T_action, D_action), matching the FM API.
            global_cond: Encoded observations (B, L_context, D_context).

        Returns:
            Scalar differentiable adaptive loss. The ordinary prediction retains
            gradients into context; neither the tangent nor JVP is differentiated.
        """
        # Forward AD is unsupported by fused attention. Scope math attention and
        # FP32 to this loss; the encoder can still use AMP, and inference uses SDPA.
        with torch.autocast(x_1.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
            actions, context = x_1.float(), global_cond.float()
            timestep, endpoint = self.sample_times(actions)
            noise = torch.randn_like(actions)
            time = timestep[:, None, None]  # (B, 1, 1), broadcast over action chunks.
            noisy_actions = (1 - time) * actions + time * noise

            def average_velocity(samples: Tensor, current: Tensor, end: Tensor) -> Tensor:
                """Map samples (B,T,D) and times (B,) to interval-conditioned velocities."""
                return cast(
                    Tensor,
                    vector_field_net(
                        samples, current, global_cond=context, time_delta=current - end
                    ),
                )

            devices = [actions.device.index] if actions.device.type == "cuda" else []
            # Reuse the same dropout draw for the boundary prediction and JVP.
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                velocity = average_velocity(noisy_actions, timestep, timestep)
            prediction, derivative = torch.func.jvp(
                average_velocity,
                (noisy_actions, timestep, endpoint),
                (velocity, torch.ones_like(timestep), torch.zeros_like(endpoint)),
            )
            interval = (timestep - endpoint)[:, None, None]  # (B, 1, 1).
            predicted_velocity = prediction + interval * derivative.detach()
            error = (predicted_velocity - (noise - actions)).square().flatten(1).sum(1)  # (B,).
            weight = (error.detach() + self.config.norm_eps).pow(self.config.norm_p)
            return cast(Tensor, (error / weight).mean())

    def generate_samples(
        self, vector_field_net: nn.Module, initial_noise: Tensor, global_cond: Tensor
    ) -> Tensor:
        """Jump from noise to clean actions using one network evaluation per interval.

        Args:
            vector_field_net: Field supporting interval conditioning.
            initial_noise: Initial action noise (B, T_action, D_action).
            global_cond: Encoded observations (B, L_context, D_context).

        Returns:
            Generated actions (B, T_action, D_action), accumulated in the noise dtype.
        """
        samples = initial_noise
        times = torch.linspace(
            1,
            0,
            self.config.num_inference_steps + 1,
            device=samples.device,
            dtype=samples.dtype,
        )
        for index in range(self.config.num_inference_steps):
            timestep = times[index].expand(samples.shape[0])  # (B,).
            interval = (times[index] - times[index + 1]).expand_as(timestep)
            velocity = vector_field_net(
                samples, timestep, global_cond=global_cond, time_delta=interval
            )
            samples = samples - interval[:, None, None] * velocity.to(samples.dtype)
        return samples
