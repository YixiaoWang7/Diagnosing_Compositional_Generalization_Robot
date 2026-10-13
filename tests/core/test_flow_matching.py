#!/usr/bin/env python
"""Tests for flow matching algorithms."""

import pytest
import torch
import torch.nn as nn

from robopolicy.algorithms.flow_matching import (
    # Registries
    PATH_REGISTRY,
    SCHEDULER_REGISTRY,
    SOLVER_REGISTRY,
    CosineScheduler,
    DenseJumpScheduler,
    EulerSolver,
    FlowMatchingAlgorithm,
    # Main classes
    FlowMatchingConfig,
    # Paths
    FlowPath,
    GaussianPath,
    HeunSolver,
    LinearPath,
    LogitNormalScheduler,
    ODESolver,
    OptimalTransportPath,
    RK4Solver,
    # Schedulers
    TimeScheduler,
    UniformScheduler,
    VPPath,
    create_flow_matching,
    # Factories
    create_path,
    create_scheduler,
    create_solver,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def batch_size():
    return 8


@pytest.fixture
def horizon():
    return 16


@pytest.fixture
def action_dim():
    return 7


@pytest.fixture
def cond_dim():
    return 256


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_x0(batch_size, horizon, action_dim, device):
    """Sample noise tensor."""
    return torch.randn(batch_size, horizon, action_dim, device=device)


@pytest.fixture
def sample_x1(batch_size, horizon, action_dim, device):
    """Sample data tensor."""
    return torch.randn(batch_size, horizon, action_dim, device=device)


@pytest.fixture
def sample_t(batch_size, device):
    """Sample time values."""
    return torch.rand(batch_size, device=device)


class SimpleVectorField(nn.Module):
    """Simple vector field network for testing."""

    def __init__(self, action_dim: int, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim + cond_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.action_dim = action_dim

    def forward(self, x, t, global_cond):
        # x: (B, H, A), t: (B,), global_cond: (B, cond_dim)
        B, H, A = x.shape

        # Flatten and concat
        x_flat = x.reshape(B, -1)[:, : self.action_dim]  # Take first action
        t_expanded = t.unsqueeze(-1)
        cond_flat = global_cond.reshape(B, -1)

        inp = torch.cat([x_flat, t_expanded, cond_flat], dim=-1)
        out = self.net(inp)

        # Expand to full horizon
        return out.unsqueeze(1).expand(-1, H, -1)


@pytest.fixture
def vector_field_net(action_dim, cond_dim, device):
    """Simple vector field network for testing."""
    return SimpleVectorField(action_dim, cond_dim).to(device)


# =============================================================================
# Test Flow Paths
# =============================================================================


class TestLinearPath:
    """Tests for LinearPath (Rectified Flow)."""

    def test_interpolate_t0(self, sample_x0, sample_x1):
        """At t=0, should return x_0."""
        path = LinearPath()
        t = torch.zeros(sample_x0.shape[0], device=sample_x0.device)
        x_t = path.interpolate(sample_x0, sample_x1, t)
        assert torch.allclose(x_t, sample_x0, atol=1e-6)

    def test_interpolate_t1(self, sample_x0, sample_x1):
        """At t=1, should return x_1."""
        path = LinearPath()
        t = torch.ones(sample_x0.shape[0], device=sample_x0.device)
        x_t = path.interpolate(sample_x0, sample_x1, t)
        assert torch.allclose(x_t, sample_x1, atol=1e-6)

    def test_interpolate_midpoint(self, sample_x0, sample_x1):
        """At t=0.5, should be midpoint."""
        path = LinearPath()
        t = torch.full((sample_x0.shape[0],), 0.5, device=sample_x0.device)
        x_t = path.interpolate(sample_x0, sample_x1, t)
        expected = 0.5 * sample_x0 + 0.5 * sample_x1
        assert torch.allclose(x_t, expected, atol=1e-6)

    def test_velocity(self, sample_x0, sample_x1, sample_t):
        """Velocity should be x_1 - x_0 (constant)."""
        path = LinearPath()
        v = path.velocity(sample_x0, sample_x1, sample_t)
        expected = sample_x1 - sample_x0
        assert torch.allclose(v, expected, atol=1e-6)


class TestGaussianPath:
    """Tests for GaussianPath (CFM)."""

    def test_interpolate_shape(self, sample_x0, sample_x1, sample_t):
        """Test output shape."""
        path = GaussianPath(sigma_min=1e-4)
        x_t = path.interpolate(sample_x0, sample_x1, sample_t)
        assert x_t.shape == sample_x0.shape

    def test_interpolate_with_noise(self, sample_x0, sample_x1, sample_t):
        """Test that noise is added."""
        path = GaussianPath(sigma_min=0.1)  # Larger sigma to be visible

        # Run twice - should get different results due to noise
        x_t1 = path.interpolate(sample_x0, sample_x1, sample_t)
        x_t2 = path.interpolate(sample_x0, sample_x1, sample_t)

        assert not torch.allclose(x_t1, x_t2)

    def test_sigma_schedules(self, sample_x0, sample_x1, sample_t):
        """Test different sigma schedules."""
        for schedule in ["constant", "linear", "sqrt"]:
            path = GaussianPath(sigma_min=1e-4, sigma_schedule=schedule)
            x_t = path.interpolate(sample_x0, sample_x1, sample_t)
            assert x_t.shape == sample_x0.shape


class TestOptimalTransportPath:
    """Tests for OT-CFM path."""

    def test_interpolate_shape(self, sample_x0, sample_x1, sample_t):
        """Test output shape."""
        path = OptimalTransportPath(reg=0.1, num_iters=10)
        x_t = path.interpolate(sample_x0, sample_x1, sample_t)
        assert x_t.shape == sample_x0.shape

    def test_ot_plan_permutation(self, sample_x0, sample_x1):
        """Test that OT plan returns valid permutation."""
        path = OptimalTransportPath(reg=0.1, num_iters=10)
        perm = path.compute_ot_plan(sample_x0, sample_x1)

        assert perm.shape == (sample_x0.shape[0],)
        assert perm.min() >= 0
        assert perm.max() < sample_x0.shape[0]


class TestVPPath:
    """Tests for Variance-Preserving path."""

    def test_interpolate_shape(self, sample_x0, sample_x1, sample_t):
        """Test output shape."""
        path = VPPath()
        x_t = path.interpolate(sample_x0, sample_x1, sample_t)
        assert x_t.shape == sample_x0.shape

    def test_alpha_sigma(self):
        """Test alpha and sigma values."""
        path = VPPath(beta_min=0.1, beta_max=20.0)

        t = torch.tensor([0.0, 0.5, 1.0])
        alpha, sigma = path.get_alpha_sigma(t)

        # At t=0, alpha should be ~1, sigma ~0
        assert alpha[0] > 0.9
        # At t=1, alpha should be smaller
        assert alpha[2] < alpha[0]


# =============================================================================
# Test Time Schedulers
# =============================================================================


class TestUniformScheduler:
    """Tests for uniform time scheduler."""

    def test_sample_range(self, batch_size, device):
        """Test samples are in [0, 1]."""
        scheduler = UniformScheduler()
        t = scheduler.sample(batch_size, device, torch.float32)

        assert t.shape == (batch_size,)
        assert t.min() >= 0
        assert t.max() <= 1

    def test_eps_margin(self, batch_size, device):
        """Test epsilon margin."""
        eps = 0.1
        scheduler = UniformScheduler(eps=eps)

        # Sample many times to test bounds
        for _ in range(10):
            t = scheduler.sample(batch_size, device, torch.float32)
            assert t.min() >= eps - 1e-6
            assert t.max() <= 1 - eps + 1e-6


class TestLogitNormalScheduler:
    """Tests for logit-normal scheduler."""

    def test_sample_range(self, batch_size, device):
        """Test samples are in (0, 1)."""
        scheduler = LogitNormalScheduler()
        t = scheduler.sample(batch_size, device, torch.float32)

        assert t.shape == (batch_size,)
        assert t.min() > 0
        assert t.max() < 1

    def test_loc_scale(self, batch_size, device):
        """Test different loc/scale parameters."""
        for loc, scale in [(0.0, 1.0), (-1.0, 0.5), (1.0, 2.0)]:
            scheduler = LogitNormalScheduler(loc=loc, scale=scale)
            t = scheduler.sample(batch_size, device, torch.float32)
            assert t.shape == (batch_size,)


class TestCosineScheduler:
    """Tests for cosine scheduler."""

    def test_sample_range(self, batch_size, device):
        """Test samples are in (0, 1)."""
        scheduler = CosineScheduler()
        t = scheduler.sample(batch_size, device, torch.float32)

        assert t.shape == (batch_size,)
        assert t.min() > 0
        assert t.max() < 1


class TestDenseJumpScheduler:
    """Tests for dense-jump scheduler."""

    def test_sample_range(self, batch_size, device):
        """Test samples are in [0, 1]."""
        scheduler = DenseJumpScheduler(num_inference_steps=10)
        t = scheduler.sample(batch_size, device, torch.float32)

        assert t.shape == (batch_size,)
        assert t.min() >= 0
        assert t.max() <= 1


# =============================================================================
# Test ODE Solvers
# =============================================================================


class TestEulerSolver:
    """Tests for Euler solver."""

    def test_solve_identity(self, sample_x0):
        """Test with zero velocity (identity map)."""
        solver = EulerSolver(num_steps=10)

        def zero_velocity(x, t):
            return torch.zeros_like(x)

        x_final = solver.solve(zero_velocity, sample_x0)
        assert torch.allclose(x_final, sample_x0, atol=1e-6)

    def test_solve_constant_velocity(self, sample_x0):
        """Test with constant velocity."""
        solver = EulerSolver(num_steps=100)
        v = torch.ones_like(sample_x0)

        def const_velocity(x, t):
            return v

        x_final = solver.solve(const_velocity, sample_x0)
        # After integrating velocity=1 from t=0 to t=1, should add 1
        expected = sample_x0 + v
        assert torch.allclose(x_final, expected, atol=0.1)


class TestHeunSolver:
    """Tests for Heun solver."""

    def test_solve_shape(self, sample_x0):
        """Test output shape."""
        solver = HeunSolver(num_steps=10)

        def velocity(x, t):
            return torch.zeros_like(x)

        x_final = solver.solve(velocity, sample_x0)
        assert x_final.shape == sample_x0.shape


class TestRK4Solver:
    """Tests for RK4 solver."""

    def test_higher_accuracy(self, sample_x0):
        """Test that RK4 is more accurate than Euler with same steps."""

        # Linear ODE: dx/dt = x, solution: x(t) = x_0 * exp(t)
        def velocity(x, t):
            return x

        euler = EulerSolver(num_steps=10)
        rk4 = RK4Solver(num_steps=10)

        x_euler = euler.solve(velocity, sample_x0)
        x_rk4 = rk4.solve(velocity, sample_x0)

        # Expected: x_0 * e
        expected = sample_x0 * torch.exp(torch.ones(1, device=sample_x0.device))

        euler_error = (x_euler - expected).abs().mean()
        rk4_error = (x_rk4 - expected).abs().mean()

        # RK4 should have smaller error
        assert rk4_error < euler_error


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_path(self):
        """Test creating paths."""
        for name in PATH_REGISTRY.keys():
            path = create_path(name)
            assert isinstance(path, FlowPath)

    def test_create_scheduler(self):
        """Test creating schedulers."""
        for name in SCHEDULER_REGISTRY.keys():
            scheduler = create_scheduler(name)
            assert isinstance(scheduler, TimeScheduler)

    def test_create_solver(self):
        """Test creating solvers."""
        for name in SOLVER_REGISTRY.keys():
            solver = create_solver(name, num_steps=5)
            assert isinstance(solver, ODESolver)

    def test_invalid_names(self):
        """Test error on invalid names."""
        with pytest.raises(ValueError):
            create_path("invalid")
        with pytest.raises(ValueError):
            create_scheduler("invalid")
        with pytest.raises(ValueError):
            create_solver("invalid")


# =============================================================================
# Test FlowMatchingAlgorithm
# =============================================================================


class TestFlowMatchingAlgorithm:
    """Tests for main FlowMatchingAlgorithm class."""

    def test_config_creation(self):
        """Test creating algorithm with config."""
        config = FlowMatchingConfig(
            path_type="linear",
            scheduler_type="uniform",
            solver_type="euler",
            num_inference_steps=10,
        )
        algo = FlowMatchingAlgorithm(config=config)

        assert algo.config.path_type == "linear"
        assert isinstance(algo.path, LinearPath)

    def test_kwargs_creation(self):
        """Test creating algorithm with kwargs."""
        algo = FlowMatchingAlgorithm(
            FlowMatchingConfig(
                path_type="gaussian",
                scheduler_type="logit_normal",
                solver_type="heun",
                num_inference_steps=20,
            )
        )

        assert algo.config.path_type == "gaussian"
        assert isinstance(algo.path, GaussianPath)

    def test_compute_loss(self, vector_field_net, sample_x1, cond_dim, batch_size, device):
        """Test loss computation."""
        algo = FlowMatchingAlgorithm(FlowMatchingConfig(path_type="linear"))

        global_cond = torch.randn(batch_size, cond_dim, device=device)

        loss = algo.compute_loss(
            vector_field_net=vector_field_net,
            x_1=sample_x1,
            global_cond=global_cond,
        )

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_generate_samples(self, vector_field_net, sample_x0, cond_dim, batch_size, device):
        """Test sample generation."""
        algo = FlowMatchingAlgorithm(
            FlowMatchingConfig(
                path_type="linear",
                solver_type="euler",
                num_inference_steps=5,
            )
        )

        global_cond = torch.randn(batch_size, cond_dim, device=device)

        samples = algo.generate_samples(
            vector_field_net=vector_field_net,
            initial_noise=sample_x0,
            global_cond=global_cond,
        )

        assert samples.shape == sample_x0.shape

    def test_different_solvers(self, vector_field_net, sample_x0, cond_dim, batch_size, device):
        """Test with different solvers."""
        global_cond = torch.randn(batch_size, cond_dim, device=device)

        for solver_type in ["euler", "heun", "midpoint", "rk4"]:
            algo = FlowMatchingAlgorithm(
                FlowMatchingConfig(
                    path_type="linear",
                    solver_type=solver_type,
                    num_inference_steps=3,
                )
            )

            samples = algo.generate_samples(
                vector_field_net=vector_field_net,
                initial_noise=sample_x0,
                global_cond=global_cond,
            )

            assert samples.shape == sample_x0.shape


class TestCreateFlowMatching:
    """Tests for convenience factory."""

    def test_create(self):
        """Test factory function."""
        algo = create_flow_matching(
            path_type="linear",
            scheduler_type="uniform",
            solver_type="euler",
            num_inference_steps=10,
        )

        assert isinstance(algo, FlowMatchingAlgorithm)
        assert algo.config.path_type == "linear"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_training_loop(self, vector_field_net, sample_x1, cond_dim, batch_size, device):
        """Test a training step."""
        algo = FlowMatchingAlgorithm(FlowMatchingConfig(path_type="linear"))
        optimizer = torch.optim.Adam(vector_field_net.parameters(), lr=1e-3)

        global_cond = torch.randn(batch_size, cond_dim, device=device)

        # Forward
        loss = algo.compute_loss(
            vector_field_net=vector_field_net,
            x_1=sample_x1,
            global_cond=global_cond,
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0

    def test_inference_pipeline(self, vector_field_net, sample_x0, cond_dim, batch_size, device):
        """Test inference pipeline."""
        algo = FlowMatchingAlgorithm(
            FlowMatchingConfig(
                path_type="linear",
                solver_type="heun",
                num_inference_steps=10,
            )
        )

        vector_field_net.eval()
        global_cond = torch.randn(batch_size, cond_dim, device=device)

        with torch.no_grad():
            samples = algo.generate_samples(
                vector_field_net=vector_field_net,
                initial_noise=sample_x0,
                global_cond=global_cond,
            )

        assert samples.shape == sample_x0.shape
        assert not torch.isnan(samples).any()


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.parametrize(
    "settings",
    [
        {"num_inference_steps": 0},
        {"time_eps": 0.6},
        {"t_max": 1.1},
        {"beta_alpha": 0},
        {"ot_reg": -1},
        {"velocity_clamp": -1},
        {"solver_type": "missing"},
        {"path_type": "missing"},
    ],
)
def test_invalid_flow_configuration(settings):
    """Reject invalid numerical settings before allocating or training a policy."""
    with pytest.raises(ValueError):
        FlowMatchingConfig(**settings)
