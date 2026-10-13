#!/usr/bin/env python

"""Flow Matching Algorithm Implementation.

Moved from `flow_models.flow.flow_matching` as part of the method-agnostic refactor.

This module provides a modular, extensible flow matching framework integrating
state-of-the-art techniques from recent research in generative modeling and
robot learning.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

# =============================================================================
# Flow Paths - Define how we interpolate between noise and data
# =============================================================================


class FlowPath(ABC):
    """Abstract base class for flow interpolation paths.

    A flow path defines:
    1. How to interpolate between x_0 (noise) and x_1 (data): x_t = path(x_0, x_1, t)
    2. The target velocity field: v_target = d(x_t)/dt
    """

    @abstractmethod
    def interpolate(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Compute x_t along the path."""
        pass

    @abstractmethod
    def velocity(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Compute target velocity v_t = dx_t/dt."""
        pass

    def expand_t(self, t: Tensor, target_shape: tuple[int, ...]) -> Tensor:
        """Expand time tensor to broadcast with target shape."""
        n_dims = len(target_shape) - 1  # Exclude batch dimension
        return t.view(-1, *([1] * n_dims))


class LinearPath(FlowPath):
    """Linear/Rectified Flow path."""

    def interpolate(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Interpolate source and target samples at batched times.

        Args:
            x_0: Source/noise tensor shaped (B, T_action, D_action).
            x_1: Target action tensor shaped (B, T_action, D_action).
            t: Training times shaped (B,).

        Returns:
            Interpolated samples shaped (B, T_action, D_action).
        """
        t_expanded = self.expand_t(t, x_0.shape)
        return (1 - t_expanded) * x_0 + t_expanded * x_1

    def velocity(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Compute target flow velocities at batched times.

        Args:
            x_0: Source/noise tensor shaped (B, T_action, D_action).
            x_1: Target action tensor shaped (B, T_action, D_action).
            t: Training times shaped (B,).

        Returns:
            Target velocities shaped (B, T_action, D_action).
        """
        return x_1 - x_0


class GaussianPath(FlowPath):
    """Gaussian/Conditional Flow Matching path with added noise."""

    def __init__(
        self,
        sigma_min: float = 1e-4,
        sigma_schedule: Literal["constant", "linear", "sqrt"] = "constant",
    ) -> None:
        """Configure GaussianPath and allocate its owned state.

        Args:
            sigma_min: Sigma min setting for this component.
            sigma_schedule: Sigma schedule setting for this component.
        """
        self.sigma_min = sigma_min
        self.sigma_schedule = sigma_schedule

    def get_sigma(self, t: Tensor) -> Tensor:
        """Evaluate the configured Gaussian path noise scale.

        Args:
            t: Training times shaped (B,).

        Returns:
            Noise scales shaped (B,).
        """
        if self.sigma_schedule == "constant":
            return torch.full_like(t, self.sigma_min)
        elif self.sigma_schedule == "linear":
            return self.sigma_min * (1 - t)
        elif self.sigma_schedule == "sqrt":
            return self.sigma_min * torch.sqrt(1 - t + 1e-8)
        else:
            raise ValueError(f"Unknown sigma_schedule: {self.sigma_schedule}")

    def interpolate(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Interpolate source and target samples at batched times.

        Args:
            x_0: Source/noise tensor shaped (B, T_action, D_action).
            x_1: Target action tensor shaped (B, T_action, D_action).
            t: Training times shaped (B,).

        Returns:
            Interpolated samples shaped (B, T_action, D_action).
        """
        t_expanded = self.expand_t(t, x_0.shape)
        sigma_t = self.expand_t(self.get_sigma(t), x_0.shape)
        epsilon = torch.randn_like(x_0)
        return (1 - t_expanded) * x_0 + t_expanded * x_1 + sigma_t * epsilon

    def velocity(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Compute target flow velocities at batched times.

        Args:
            x_0: Source/noise tensor shaped (B, T_action, D_action).
            x_1: Target action tensor shaped (B, T_action, D_action).
            t: Training times shaped (B,).

        Returns:
            Target velocities shaped (B, T_action, D_action).
        """
        return x_1 - x_0


class OptimalTransportPath(FlowPath):
    """Optimal Transport Conditional Flow Matching (OT-CFM) path."""

    def __init__(self, reg: float = 0.05, num_iters: int = 50) -> None:
        """Configure OptimalTransportPath and allocate its owned state.

        Args:
            reg: Reg setting for this component.
            num_iters: Num iters setting for this component.
        """
        self.reg = reg
        self.num_iters = num_iters

    def compute_ot_plan(self, x_0: Tensor, x_1: Tensor) -> Tensor:
        """Compute an entropy-regularized transport coupling between sample batches.

        Args:
            x_0: Source/noise tensor shaped (B, T_action, D_action).
            x_1: Target action tensor shaped (B, T_action, D_action).

        Returns:
            Coupling weights shaped (B, B).
        """
        batch_size = x_0.shape[0]
        x_0_flat = x_0.reshape(batch_size, -1)
        x_1_flat = x_1.reshape(batch_size, -1)
        cost = torch.cdist(x_0_flat, x_1_flat, p=2).pow(2)
        K = torch.exp(-cost / self.reg)
        u = torch.ones(batch_size, device=x_0.device, dtype=x_0.dtype)
        v = torch.ones(batch_size, device=x_0.device, dtype=x_0.dtype)
        for _ in range(self.num_iters):
            u = 1.0 / (K @ v + 1e-8)
            v = 1.0 / (K.T @ u + 1e-8)
        plan = u.unsqueeze(1) * K * v.unsqueeze(0)
        perm = plan.argmax(dim=1)
        return perm

    def interpolate(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Interpolate source and target samples at batched times.

        Args:
            x_0: Source/noise tensor shaped (B, T_action, D_action).
            x_1: Target action tensor shaped (B, T_action, D_action).
            t: Training times shaped (B,).

        Returns:
            Interpolated samples shaped (B, T_action, D_action).
        """
        perm = self.compute_ot_plan(x_0, x_1)
        x_1_matched = x_1[perm]
        t_expanded = self.expand_t(t, x_0.shape)
        return (1 - t_expanded) * x_0 + t_expanded * x_1_matched

    def velocity(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Compute target flow velocities at batched times.

        Args:
            x_0: Source/noise tensor shaped (B, T_action, D_action).
            x_1: Target action tensor shaped (B, T_action, D_action).
            t: Training times shaped (B,).

        Returns:
            Target velocities shaped (B, T_action, D_action).
        """
        perm = self.compute_ot_plan(x_0, x_1)
        x_1_matched = x_1[perm]
        return x_1_matched - x_0


class VPPath(FlowPath):
    """Variance-Preserving path connecting flow matching to diffusion."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        """Configure VPPath and allocate its owned state.

        Args:
            beta_min: Beta min setting for this component.
            beta_max: Beta max setting for this component.
        """
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_alpha_sigma(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate variance-preserving interpolation coefficients.

        Args:
            t: Training times shaped (B,).

        Returns:
            Alpha and sigma tensors, each shaped (B,).
        """
        log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        alpha_t = torch.exp(log_mean_coeff)
        sigma_t = torch.sqrt(1 - alpha_t**2 + 1e-8)
        return alpha_t, sigma_t

    def interpolate(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Interpolate source and target samples at batched times.

        Args:
            x_0: Source/noise tensor shaped (B, T_action, D_action).
            x_1: Target action tensor shaped (B, T_action, D_action).
            t: Training times shaped (B,).

        Returns:
            Interpolated samples shaped (B, T_action, D_action).
        """
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        alpha_t = self.expand_t(alpha_t, x_0.shape)
        sigma_t = self.expand_t(sigma_t, x_0.shape)
        return alpha_t * x_1 + sigma_t * x_0

    def velocity(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Compute target flow velocities at batched times.

        Args:
            x_0: Source/noise tensor shaped (B, T_action, D_action).
            x_1: Target action tensor shaped (B, T_action, D_action).
            t: Training times shaped (B,).

        Returns:
            Target velocities shaped (B, T_action, D_action).
        """
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        alpha_t = self.expand_t(alpha_t, x_0.shape)
        sigma_t = self.expand_t(sigma_t, x_0.shape)
        d_alpha = -0.5 * (self.beta_max - self.beta_min) * t - 0.5 * self.beta_min
        d_alpha = self.expand_t(d_alpha * alpha_t.squeeze(), x_0.shape)
        d_sigma = -alpha_t * d_alpha / (sigma_t + 1e-8)
        return d_alpha * x_1 + d_sigma * x_0


# =============================================================================
# Time Schedulers - Define how we sample training times
# =============================================================================


class TimeScheduler(ABC):
    """Interface for sampling training times in the flow interval."""

    @abstractmethod
    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Sample a vector of training times on the supplied device and dtype.

        Args:
            batch_size: Number of independent samples B.
            device: Destination inferred or selected by the caller.
            dtype: Floating dtype used by the supplied tensors.

        Returns:
            Training times shaped (B,).
        """
        pass


class UniformScheduler(TimeScheduler):
    """Sample training times uniformly, optionally excluding interval endpoints."""

    def __init__(self, eps: float = 0.0) -> None:
        """Configure UniformScheduler and allocate its owned state.

        Args:
            eps: Eps setting for this component.
        """
        self.eps = eps

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Sample a vector of training times on the supplied device and dtype.

        Args:
            batch_size: Number of independent samples B.
            device: Destination inferred or selected by the caller.
            dtype: Floating dtype used by the supplied tensors.

        Returns:
            Training times shaped (B,).
        """
        t = torch.rand(batch_size, device=device, dtype=dtype)
        return t * (1 - 2 * self.eps) + self.eps


class LogitNormalScheduler(TimeScheduler):
    """Sample logistic-normal training times from normal random variables."""

    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        """Configure LogitNormalScheduler and allocate its owned state.

        Args:
            loc: Loc setting for this component.
            scale: Scale setting for this component.
        """
        self.loc = loc
        self.scale = scale

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Sample a vector of training times on the supplied device and dtype.

        Args:
            batch_size: Number of independent samples B.
            device: Destination inferred or selected by the caller.
            dtype: Floating dtype used by the supplied tensors.

        Returns:
            Training times shaped (B,).
        """
        u = torch.randn(batch_size, device=device, dtype=dtype)
        u = self.loc + self.scale * u
        return torch.sigmoid(u)


class CosineScheduler(TimeScheduler):
    """Bias sampled training times with a cosine schedule."""

    def __init__(self, s: float = 0.008) -> None:
        """Configure CosineScheduler and allocate its owned state.

        Args:
            s: S setting for this component.
        """
        self.s = s

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Sample a vector of training times on the supplied device and dtype.

        Args:
            batch_size: Number of independent samples B.
            device: Destination inferred or selected by the caller.
            dtype: Floating dtype used by the supplied tensors.

        Returns:
            Training times shaped (B,).
        """
        u = torch.rand(batch_size, device=device, dtype=dtype)
        t = torch.acos(1 - 2 * u) / torch.pi
        return t.clamp(self.s, 1 - self.s)


class BetaScheduler(TimeScheduler):
    """Beta-distribution time scheduler biased toward t≈0 (noisy actions).

    Parameterised by the user's formulation:
      u = (t_max - t) / t_max,   u ~ Beta(alpha, beta)
      t = t_max * (1 - u)

    Beta(alpha=1.5, beta=1) density: f(u) ∝ u^0.5, mode at u=1 → t=0.
    CDF: F(u) = u^alpha, so P(t ≤ 0.6 | t_max=0.999) = 1 - 0.4^1.5 ≈ 75%.
    In other words ~75% of training samples land in t ∈ [0, 0.6],
    and only ~25% in t ∈ [0.6, 0.999] — fast drop-off past the midpoint.

    Raising alpha (e.g. 2.0, 3.0) steepens the decay and concentrates
    even more mass near t=0.
    """

    def __init__(self, alpha: float = 1.5, beta: float = 1.0, t_max: float = 1.0) -> None:
        """Configure BetaScheduler and allocate its owned state.

        Args:
            alpha: Alpha setting for this component.
            beta: Beta setting for this component.
            t_max: T max setting for this component.
        """
        self.alpha = alpha
        self.beta = beta
        self.t_max = t_max

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Sample a vector of training times on the supplied device and dtype.

        Args:
            batch_size: Number of independent samples B.
            device: Destination inferred or selected by the caller.
            dtype: Floating dtype used by the supplied tensors.

        Returns:
            Training times shaped (B,).
        """
        a = torch.tensor(self.alpha, device=device, dtype=dtype)
        b = torch.tensor(self.beta, device=device, dtype=dtype)
        u = torch.distributions.Beta(a, b).sample((batch_size,))  # u = (t_max-t)/t_max ∈ [0,1]
        return (self.t_max * (1.0 - u)).to(dtype)  # t = t_max*(1-u) ∈ [0, t_max]


class DenseJumpScheduler(TimeScheduler):
    """Concentrate training times around the inference solver grid."""

    def __init__(self, num_inference_steps: int = 10, density_scale: float = 2.0) -> None:
        """Configure DenseJumpScheduler and allocate its owned state.

        Args:
            num_inference_steps: Num inference steps setting for this component.
            density_scale: Density scale setting for this component.
        """
        self.num_inference_steps = num_inference_steps
        self.density_scale = density_scale

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Sample a vector of training times on the supplied device and dtype.

        Args:
            batch_size: Number of independent samples B.
            device: Destination inferred or selected by the caller.
            dtype: Floating dtype used by the supplied tensors.

        Returns:
            Training times shaped (B,).
        """
        dt = 1.0 / self.num_inference_steps
        segment = torch.randint(0, self.num_inference_steps, (batch_size,), device=device)
        u = torch.rand(batch_size, device=device, dtype=dtype)
        u = torch.pow(u, 1.0 / self.density_scale)
        if torch.rand(1).item() > 0.5:
            u = 1 - u
        t = (segment.float() + u) * dt
        return t.clamp(0, 1).to(dtype)


# =============================================================================
# ODE Solvers - Define how we integrate the ODE during inference
# =============================================================================


class ODESolver(ABC):
    """Integrate a batched vector field over a configurable time interval."""

    def __init__(self, num_steps: int = 10) -> None:
        """Configure ODESolver and allocate its owned state.

        Args:
            num_steps: Num steps setting for this component.
        """
        self.num_steps = num_steps

    @abstractmethod
    def solve(
        self,
        vector_field: Callable[[Tensor, Tensor], Tensor],
        x_0: Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> Tensor:
        """Integrate the vector field from initial samples to the requested endpoint.

        Args:
            vector_field: Callable mapping (samples, times) to same-shaped velocities.
            x_0: Source/noise tensor shaped (B, T_action, D_action).
            t_start: T start setting for this component.
            t_end: T end setting for this component.

        Returns:
            Final samples shaped (B, T_action, D_action).
        """
        pass

    def get_time_steps(
        self, t_start: float, t_end: float, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """Create the integration grid, including both requested endpoints.

        Args:
            t_start: T start setting for this component.
            t_end: T end setting for this component.
            device: Destination inferred or selected by the caller.
            dtype: Floating dtype used by the supplied tensors.

        Returns:
            An integration grid shaped (num_steps + 1,).
        """
        return torch.linspace(t_start, t_end, self.num_steps + 1, device=device, dtype=dtype)


class EulerSolver(ODESolver):
    """First-order explicit Euler integration of batched trajectories."""

    def solve(
        self,
        vector_field: Callable[[Tensor, Tensor], Tensor],
        x_0: Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> Tensor:
        """Integrate the vector field from initial samples to the requested endpoint.

        Args:
            vector_field: Callable mapping (samples, times) to same-shaped velocities.
            x_0: Source/noise tensor shaped (B, T_action, D_action).
            t_start: T start setting for this component.
            t_end: T end setting for this component.

        Returns:
            Final samples shaped (B, T_action, D_action).
        """
        x = x_0
        timesteps = self.get_time_steps(t_start, t_end, x_0.device, x_0.dtype)
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]
            t_batch = t.expand(x.shape[0])  # (B,), without a GPU-to-host scalar read.
            v = vector_field(x, t_batch)
            x = x + dt * v
        return x


class HeunSolver(ODESolver):
    """Second-order predictor-corrector integration of batched trajectories."""

    def solve(
        self,
        vector_field: Callable[[Tensor, Tensor], Tensor],
        x_0: Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> Tensor:
        """Integrate the vector field from initial samples to the requested endpoint.

        Args:
            vector_field: Callable mapping (samples, times) to same-shaped velocities.
            x_0: Source/noise tensor shaped (B, T_action, D_action).
            t_start: T start setting for this component.
            t_end: T end setting for this component.

        Returns:
            Final samples shaped (B, T_action, D_action).
        """
        x = x_0
        timesteps = self.get_time_steps(t_start, t_end, x_0.device, x_0.dtype)
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            t_batch = t.expand(x.shape[0])  # (B,), without a GPU-to-host scalar read.
            t_next_batch = t_next.expand(x.shape[0])  # (B,), without a GPU-to-host scalar read.
            v1 = vector_field(x, t_batch)
            x_pred = x + dt * v1
            v2 = vector_field(x_pred, t_next_batch)
            x = x + dt * 0.5 * (v1 + v2)
        return x


class MidpointSolver(ODESolver):
    """Second-order explicit midpoint integration of batched trajectories."""

    def solve(
        self,
        vector_field: Callable[[Tensor, Tensor], Tensor],
        x_0: Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> Tensor:
        """Integrate the vector field from initial samples to the requested endpoint.

        Args:
            vector_field: Callable mapping (samples, times) to same-shaped velocities.
            x_0: Source/noise tensor shaped (B, T_action, D_action).
            t_start: T start setting for this component.
            t_end: T end setting for this component.

        Returns:
            Final samples shaped (B, T_action, D_action).
        """
        x = x_0
        timesteps = self.get_time_steps(t_start, t_end, x_0.device, x_0.dtype)
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            t_mid = t + 0.5 * dt
            t_batch = t.expand(x.shape[0])  # (B,), without a GPU-to-host scalar read.
            t_mid_batch = t_mid.expand(x.shape[0])  # (B,), without a GPU-to-host scalar read.
            v1 = vector_field(x, t_batch)
            x_mid = x + 0.5 * dt * v1
            v_mid = vector_field(x_mid, t_mid_batch)
            x = x + dt * v_mid
        return x


class RK4Solver(ODESolver):
    """Fourth-order Runge-Kutta integration of batched trajectories."""

    def solve(
        self,
        vector_field: Callable[[Tensor, Tensor], Tensor],
        x_0: Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> Tensor:
        """Integrate the vector field from initial samples to the requested endpoint.

        Args:
            vector_field: Callable mapping (samples, times) to same-shaped velocities.
            x_0: Source/noise tensor shaped (B, T_action, D_action).
            t_start: T start setting for this component.
            t_end: T end setting for this component.

        Returns:
            Final samples shaped (B, T_action, D_action).
        """
        x = x_0
        timesteps = self.get_time_steps(t_start, t_end, x_0.device, x_0.dtype)
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            t_batch = t.expand(x.shape[0])  # (B,), without a GPU-to-host scalar read.
            t_mid_batch = (t + 0.5 * dt).expand(x.shape[0])  # (B,)
            t_next_batch = t_next.expand(x.shape[0])  # (B,), without a GPU-to-host scalar read.
            k1 = vector_field(x, t_batch)
            k2 = vector_field(x + 0.5 * dt * k1, t_mid_batch)
            k3 = vector_field(x + 0.5 * dt * k2, t_mid_batch)
            k4 = vector_field(x + dt * k3, t_next_batch)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x


PATH_REGISTRY: dict[str, type[FlowPath]] = {
    "linear": LinearPath,
    "gaussian": GaussianPath,
    "ot": OptimalTransportPath,
    "vp": VPPath,
}
SCHEDULER_REGISTRY: dict[str, type[TimeScheduler]] = {
    "uniform": UniformScheduler,
    "logit_normal": LogitNormalScheduler,
    "cosine": CosineScheduler,
    "dense_jump": DenseJumpScheduler,
    "beta": BetaScheduler,
}
SOLVER_REGISTRY: dict[str, type[ODESolver]] = {
    "euler": EulerSolver,
    "heun": HeunSolver,
    "midpoint": MidpointSolver,
    "rk4": RK4Solver,
}


def create_path(path_type: str, **kwargs: Any) -> FlowPath:
    """Construct one of the supported flow paths from explicit settings.

    Args:
        path_type: Path type setting for this component.
        kwargs: Kwargs setting for this component.

    Returns:
        The computed FlowPath result.
    """
    name = path_type.lower()
    if name not in PATH_REGISTRY:
        raise ValueError(
            f"Unknown path type: '{path_type}'. Available: {list(PATH_REGISTRY.keys())}"
        )
    return PATH_REGISTRY[name](**kwargs)


def create_scheduler(scheduler_type: str, **kwargs: Any) -> TimeScheduler:
    """Construct one of the supported training-time distributions.

    Args:
        scheduler_type: Scheduler type setting for this component.
        kwargs: Kwargs setting for this component.

    Returns:
        The computed TimeScheduler result.
    """
    name = scheduler_type.lower()
    if name not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler type: '{scheduler_type}'. Available: {list(SCHEDULER_REGISTRY.keys())}"
        )
    return SCHEDULER_REGISTRY[name](**kwargs)


def create_solver(solver_type: str, num_steps: int = 10, **kwargs: Any) -> ODESolver:
    """Construct one of the supported numerical integration methods.

    Args:
        solver_type: Solver type setting for this component.
        num_steps: Num steps setting for this component.
        kwargs: Kwargs setting for this component.

    Returns:
        The computed ODESolver result.
    """
    name = solver_type.lower()
    if name not in SOLVER_REGISTRY:
        raise ValueError(
            f"Unknown solver type: '{solver_type}'. Available: {list(SOLVER_REGISTRY.keys())}"
        )
    return SOLVER_REGISTRY[name](num_steps=num_steps, **kwargs)


class FlowMatchingConfig(BaseModel):
    """Immutable flow path, time distribution, and solver settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["flow_matching"] = "flow_matching"
    path_type: Literal["linear", "gaussian", "ot", "vp"] = "linear"
    scheduler_type: Literal["uniform", "logit_normal", "cosine", "dense_jump", "beta"] = "uniform"
    solver_type: Literal["euler", "heun", "midpoint", "rk4"] = "euler"
    num_inference_steps: int = Field(default=10, ge=1)
    guidance_scale: float = 1.0
    velocity_clamp: float | None = Field(default=None, gt=0)
    sigma_min: float = Field(default=1e-4, ge=0)
    sigma_schedule: Literal["constant", "linear", "sqrt"] = "constant"
    ot_reg: float = Field(default=0.05, gt=0)
    time_eps: float = Field(default=0.0, ge=0, lt=0.5)
    # Upper bound on sampled t. E.g. 0.999 keeps t in [time_eps, 0.999].
    t_max: float = Field(default=1.0, gt=0, le=1)
    logit_normal_loc: float = 0.0
    logit_normal_scale: float = Field(default=1.0, gt=0)
    dense_jump_density: float = Field(default=2.0, gt=0)
    # beta: s ~ Beta(alpha, beta), t = t_max * (1 - s) → high density near t=0
    beta_alpha: float = Field(default=1.5, gt=0)
    beta_beta: float = Field(default=1.0, gt=0)


class FlowMatchingAlgorithm:
    """Share training velocity loss and numerical action generation."""

    def __init__(self, config: FlowMatchingConfig | None = None) -> None:
        """Configure FlowMatchingAlgorithm and allocate its owned state.

        Args:
            config: Validated settings owned by this component.
        """
        if config is None:
            config = FlowMatchingConfig()
        self.config = config

        path_kwargs = {}
        if config.path_type == "gaussian":
            path_kwargs = {"sigma_min": config.sigma_min, "sigma_schedule": config.sigma_schedule}
        elif config.path_type == "ot":
            path_kwargs = {"reg": config.ot_reg}
        self.path = create_path(config.path_type, **path_kwargs)

        scheduler_kwargs = {}
        if config.scheduler_type == "uniform":
            scheduler_kwargs = {"eps": config.time_eps}
        elif config.scheduler_type == "logit_normal":
            scheduler_kwargs = {"loc": config.logit_normal_loc, "scale": config.logit_normal_scale}
        elif config.scheduler_type == "dense_jump":
            scheduler_kwargs = {
                "num_inference_steps": config.num_inference_steps,
                "density_scale": config.dense_jump_density,
            }
        elif config.scheduler_type == "beta":
            scheduler_kwargs = {
                "alpha": config.beta_alpha,
                "beta": config.beta_beta,
                "t_max": config.t_max,
            }
        self.scheduler = create_scheduler(config.scheduler_type, **scheduler_kwargs)
        self.solver = create_solver(config.solver_type, num_steps=config.num_inference_steps)

    def compute_loss(
        self,
        vector_field_net: nn.Module,
        x_1: Tensor,
        global_cond: Tensor,
        action_is_pad: Tensor | None = None,
        do_mask_loss_for_padding: bool = False,
    ) -> Tensor:
        """Compute scalar mean-squared error against the target flow velocity.

        Args:
            vector_field_net: Module mapping actions, times, and conditioning to velocities.
            x_1: Target action tensor shaped (B, T_action, D_action).
            global_cond: Conditioning tensor shaped (B, D) or (B, L_context, D).
            action_is_pad: Optional boolean padding mask shaped (B, T_action).
            do_mask_loss_for_padding: Do mask loss for padding setting for this component.

        Returns:
            A scalar differentiable loss tensor.
        """
        batch_size = x_1.shape[0]
        x_0 = torch.randn_like(x_1)
        t = self.scheduler.sample(batch_size, x_1.device, x_1.dtype)
        if self.config.t_max < 1.0:
            t = t.clamp(max=self.config.t_max)
        x_t = self.path.interpolate(x_0, x_1, t)
        target_v = self.path.velocity(x_0, x_1, t)
        predicted_v = vector_field_net(x_t, t, global_cond=global_cond)
        if self.config.velocity_clamp is not None:
            predicted_v = torch.clamp(
                predicted_v, -self.config.velocity_clamp, self.config.velocity_clamp
            )
        loss = F.mse_loss(predicted_v, target_v, reduction="none")
        if do_mask_loss_for_padding:
            if action_is_pad is None:
                raise ValueError(
                    "You need to provide 'action_is_pad' when do_mask_loss_for_padding=True"
                )
            in_episode_bound = ~action_is_pad
            loss = loss * in_episode_bound.unsqueeze(-1)
        return loss.mean()

    def generate_samples(
        self,
        vector_field_net: nn.Module,
        initial_noise: Tensor,
        global_cond: Tensor,
        guidance_scale: float | None = None,
        uncond_global_cond: Tensor | None = None,
    ) -> Tensor:
        """Generate action trajectories by integrating the learned velocity field.

        Args:
            vector_field_net: Module mapping actions, times, and conditioning to velocities.
            initial_noise: Initial samples shaped (B, T_action, D_action).
            global_cond: Conditioning tensor shaped (B, D) or (B, L_context, D).
            guidance_scale: Guidance scale setting for this component.
            uncond_global_cond: Optional unconditional tensor with the same shape as global_cond.

        Returns:
            Generated samples shaped (B, T_action, D_action).
        """
        guidance_scale = self.config.guidance_scale if guidance_scale is None else guidance_scale

        def vector_field(x: Tensor, t: Tensor) -> Tensor:
            """Evaluate conditional velocity with optional classifier-free guidance.

            Args:
                x: Input tensor shaped (B, T, D).
                t: Training times shaped (B,).

            Returns:
                Velocities shaped (B, T_action, D_action).
            """
            v = vector_field_net(x, t, global_cond=global_cond)
            if guidance_scale != 1.0 and uncond_global_cond is not None:
                v_uncond = vector_field_net(x, t, global_cond=uncond_global_cond)
                v = v_uncond + guidance_scale * (v - v_uncond)
            if self.config.velocity_clamp is not None:
                v = torch.clamp(v, -self.config.velocity_clamp, self.config.velocity_clamp)
            return cast(Tensor, v)

        return self.solver.solve(vector_field, initial_noise, t_start=0.0, t_end=1.0)


def create_flow_matching(**kwargs: Any) -> FlowMatchingAlgorithm:
    """Construct flow matching from validated component settings.

    Args:
        kwargs: Kwargs setting for this component.

    Returns:
        The computed FlowMatchingAlgorithm result.
    """
    return FlowMatchingAlgorithm(config=FlowMatchingConfig(**kwargs))
