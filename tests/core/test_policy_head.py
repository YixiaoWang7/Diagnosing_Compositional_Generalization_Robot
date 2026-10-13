#!/usr/bin/env python
"""Tests for policy head module with multi-modal conditioning."""

import pytest
import torch

from robopolicy.modules.policy_head.unet import (
    FiLMConvPolicyHead,
    FiLMLayer,
    ModalityConfig,
    MultiModalFiLM,
    MultiModalPolicyHead,
    MultiModalResidualBlock1d,
    PolicyHeadConfig,
    SinusoidalPosEmb,
    create_policy_head,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def action_dim():
    return 7


@pytest.fixture
def horizon():
    return 16


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_modalities():
    """Sample modality configurations for testing."""
    return [
        ModalityConfig(name="image", dim=512, n_steps=2),
        ModalityConfig(name="force", dim=6, n_steps=1),
        ModalityConfig(name="proprioception", dim=9, n_steps=2),
    ]


@pytest.fixture
def sample_config(action_dim, horizon, sample_modalities):
    """Sample policy head config."""
    return PolicyHeadConfig(
        action_dim=action_dim,
        horizon=horizon,
        modalities=sample_modalities,
        down_dims=(64, 128, 256),  # Smaller for faster tests
    )


# =============================================================================
# Test Configuration
# =============================================================================


class TestModalityConfig:
    """Tests for ModalityConfig dataclass."""

    def test_basic_creation(self):
        """Test basic modality config creation."""
        config = ModalityConfig(name="image", dim=512)
        assert config.name == "image"
        assert config.dim == 512
        assert config.n_steps == 1
        assert config.enabled is True
        assert config.total_dim == 512

    def test_multi_step(self):
        """Test modality with multiple timesteps."""
        config = ModalityConfig(name="image", dim=256, n_steps=3)
        assert config.total_dim == 256 * 3

    def test_disabled(self):
        """Test disabled modality."""
        config = ModalityConfig(name="force", dim=6, enabled=False)
        assert config.enabled is False


class TestPolicyHeadConfig:
    """Tests for PolicyHeadConfig dataclass."""

    def test_basic_creation(self, action_dim, horizon, sample_modalities):
        """Test basic config creation."""
        config = PolicyHeadConfig(
            action_dim=action_dim,
            horizon=horizon,
            modalities=sample_modalities,
        )
        assert config.action_dim == action_dim
        assert config.horizon == horizon
        assert len(config.modalities) == 3

    def test_enabled_modalities(self, action_dim, horizon):
        """Test filtering enabled modalities."""
        modalities = [
            ModalityConfig(name="image", dim=512, enabled=True),
            ModalityConfig(name="force", dim=6, enabled=False),
            ModalityConfig(name="audio", dim=128, enabled=True),
        ]
        config = PolicyHeadConfig(
            action_dim=action_dim,
            horizon=horizon,
            modalities=modalities,
        )
        enabled = config.enabled_modalities
        assert len(enabled) == 2
        assert enabled[0].name == "image"
        assert enabled[1].name == "audio"

    def test_modality_names(self, sample_config):
        """Test getting modality names."""
        names = sample_config.modality_names
        assert names == ["image", "force", "proprioception"]

    def test_get_modality(self, sample_config):
        """Test getting modality by name."""
        image_mod = sample_config.get_modality("image")
        assert image_mod is not None
        assert image_mod.dim == 512

        unknown = sample_config.get_modality("unknown")
        assert unknown is None

    def test_dict_conversion(self, action_dim, horizon):
        """Test creating config with dict modalities."""
        config = PolicyHeadConfig(
            action_dim=action_dim,
            horizon=horizon,
            modalities=[
                {"name": "image", "dim": 512},
                {"name": "force", "dim": 6},
            ],
        )
        assert isinstance(config.modalities[0], ModalityConfig)
        assert config.modalities[0].name == "image"


# =============================================================================
# Test Components
# =============================================================================


class TestSinusoidalPosEmb:
    """Tests for sinusoidal positional embedding."""

    def test_output_shape(self, batch_size, device):
        """Test embedding output shape."""
        dim = 128
        emb = SinusoidalPosEmb(dim).to(device)
        t = torch.rand(batch_size, device=device)
        output = emb(t)
        assert output.shape == (batch_size, dim)

    def test_different_times(self, device):
        """Test that different times give different embeddings."""
        emb = SinusoidalPosEmb(64).to(device)
        t1 = torch.tensor([0.0], device=device)
        t2 = torch.tensor([1.0], device=device)

        out1 = emb(t1)
        out2 = emb(t2)

        assert not torch.allclose(out1, out2)


class TestFiLMLayer:
    """Tests for FiLM layer."""

    def test_with_scale(self, batch_size, device):
        """Test FiLM layer with scale modulation."""
        cond_dim = 64
        out_channels = 128

        film = FiLMLayer(cond_dim, out_channels, use_scale=True).to(device)
        cond = torch.randn(batch_size, cond_dim, device=device)

        scale, bias = film(cond)

        assert scale is not None
        assert scale.shape == (batch_size, out_channels, 1)
        assert bias.shape == (batch_size, out_channels, 1)

    def test_without_scale(self, batch_size, device):
        """Test FiLM layer without scale modulation."""
        cond_dim = 64
        out_channels = 128

        film = FiLMLayer(cond_dim, out_channels, use_scale=False).to(device)
        cond = torch.randn(batch_size, cond_dim, device=device)

        scale, bias = film(cond)

        assert scale is None
        assert bias.shape == (batch_size, out_channels, 1)


class TestMultiModalFiLM:
    """Tests for multi-modal FiLM module."""

    def test_single_modality(self, batch_size, device):
        """Test with single modality."""
        modality_dims = {"image": 256}
        out_channels = 64

        film = MultiModalFiLM(modality_dims, out_channels).to(device)
        features = torch.randn(batch_size, out_channels, 16, device=device)
        conditions = {"image": torch.randn(batch_size, 256, device=device)}

        output = film(features, conditions)

        assert output.shape == features.shape

    def test_multiple_modalities(self, batch_size, device):
        """Test with multiple modalities."""
        modality_dims = {
            "image": 256,
            "force": 32,
            "audio": 128,
        }
        out_channels = 64

        film = MultiModalFiLM(modality_dims, out_channels, fusion="add").to(device)
        features = torch.randn(batch_size, out_channels, 16, device=device)
        conditions = {
            "image": torch.randn(batch_size, 256, device=device),
            "force": torch.randn(batch_size, 32, device=device),
            "audio": torch.randn(batch_size, 128, device=device),
        }

        output = film(features, conditions)

        assert output.shape == features.shape

    def test_partial_conditions(self, batch_size, device):
        """Test with missing modalities in conditions."""
        modality_dims = {"image": 256, "force": 32}
        out_channels = 64

        film = MultiModalFiLM(modality_dims, out_channels).to(device)
        features = torch.randn(batch_size, out_channels, 16, device=device)

        # Only provide image condition
        conditions = {"image": torch.randn(batch_size, 256, device=device)}

        output = film(features, conditions)
        assert output.shape == features.shape

    def test_fusion_modes(self, batch_size, device):
        """Test different fusion modes."""
        modality_dims = {"a": 64, "b": 64}
        out_channels = 32

        for fusion in ["add", "concat"]:
            film = MultiModalFiLM(modality_dims, out_channels, fusion=fusion).to(device)
            features = torch.randn(batch_size, out_channels, 8, device=device)
            conditions = {
                "a": torch.randn(batch_size, 64, device=device),
                "b": torch.randn(batch_size, 64, device=device),
            }

            output = film(features, conditions)
            assert output.shape == features.shape


class TestMultiModalResidualBlock1d:
    """Tests for multi-modal residual block."""

    def test_basic_forward(self, batch_size, device):
        """Test basic forward pass."""
        in_channels = 32
        out_channels = 64
        seq_len = 16
        modality_dims = {"time": 128, "image": 256}

        block = MultiModalResidualBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            modality_dims=modality_dims,
        ).to(device)

        x = torch.randn(batch_size, in_channels, seq_len, device=device)
        conditions = {
            "time": torch.randn(batch_size, 128, device=device),
            "image": torch.randn(batch_size, 256, device=device),
        }

        output = block(x, conditions)

        assert output.shape == (batch_size, out_channels, seq_len)

    def test_same_channels(self, batch_size, device):
        """Test with same input/output channels (identity residual)."""
        channels = 64
        seq_len = 16
        modality_dims = {"time": 128}

        block = MultiModalResidualBlock1d(
            in_channels=channels,
            out_channels=channels,
            modality_dims=modality_dims,
        ).to(device)

        x = torch.randn(batch_size, channels, seq_len, device=device)
        conditions = {"time": torch.randn(batch_size, 128, device=device)}

        output = block(x, conditions)

        assert output.shape == x.shape


# =============================================================================
# Test Policy Heads
# =============================================================================


class TestMultiModalPolicyHead:
    """Tests for multi-modal policy head."""

    def test_basic_forward(self, sample_config, batch_size, device):
        """Test basic forward pass."""
        head = MultiModalPolicyHead(sample_config).to(device)

        x = torch.randn(batch_size, sample_config.horizon, sample_config.action_dim, device=device)
        timestep = torch.rand(batch_size, device=device)
        conditions = {
            "image": torch.randn(batch_size, 512 * 2, device=device),
            "force": torch.randn(batch_size, 6, device=device),
            "proprioception": torch.randn(batch_size, 9 * 2, device=device),
        }

        output = head(x, timestep, conditions)

        assert output.shape == x.shape

    def test_time_only_conditioning(self, sample_config, batch_size, device):
        """Test with only time conditioning (no other modalities)."""
        head = MultiModalPolicyHead(sample_config).to(device)

        x = torch.randn(batch_size, sample_config.horizon, sample_config.action_dim, device=device)
        timestep = torch.rand(batch_size, device=device)

        # No conditions dict
        output = head(x, timestep, conditions=None)

        assert output.shape == x.shape

    def test_partial_conditions(self, sample_config, batch_size, device):
        """Test with partial modality conditions."""
        head = MultiModalPolicyHead(sample_config).to(device)

        x = torch.randn(batch_size, sample_config.horizon, sample_config.action_dim, device=device)
        timestep = torch.rand(batch_size, device=device)

        # Only provide image condition
        conditions = {
            "image": torch.randn(batch_size, 512 * 2, device=device),
        }

        output = head(x, timestep, conditions)

        assert output.shape == x.shape

    def test_gradient_flow(self, sample_config, batch_size, device):
        """Test that gradients flow through the network."""
        head = MultiModalPolicyHead(sample_config).to(device)

        x = torch.randn(
            batch_size,
            sample_config.horizon,
            sample_config.action_dim,
            device=device,
            requires_grad=True,
        )
        timestep = torch.rand(batch_size, device=device)
        conditions = {
            "image": torch.randn(batch_size, 512 * 2, device=device, requires_grad=True),
        }

        output = head(x, timestep, conditions)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert conditions["image"].grad is not None


class TestFiLMConvPolicyHead:
    """Tests for legacy FiLM policy head."""

    def test_basic_forward(self, batch_size, action_dim, horizon, device):
        """Test basic forward pass."""
        global_cond_dim = 256
        n_obs_steps = 2

        head = FiLMConvPolicyHead(
            action_dim=action_dim,
            horizon=horizon,
            global_cond_dim=global_cond_dim,
            n_obs_steps=n_obs_steps,
            down_dims=(64, 128, 256),
        ).to(device)

        x = torch.randn(batch_size, horizon, action_dim, device=device)
        timestep = torch.rand(batch_size, device=device)
        global_cond = torch.randn(batch_size, n_obs_steps, global_cond_dim, device=device)

        output = head(x, timestep, global_cond)

        assert output.shape == x.shape


class TestCreatePolicyHead:
    """Tests for factory function."""

    def test_create_multimodal(self, action_dim, horizon, sample_modalities):
        """Test creating multi-modal head."""
        head = create_policy_head(
            action_dim=action_dim,
            horizon=horizon,
            modalities=sample_modalities,
            down_dims=(64, 128),
        )

        assert isinstance(head, MultiModalPolicyHead)

    def test_create_legacy(self, action_dim, horizon):
        """Test creating legacy head."""
        head = create_policy_head(
            action_dim=action_dim,
            horizon=horizon,
            global_cond_dim=256,
            n_obs_steps=2,
            down_dims=(64, 128),
        )

        assert isinstance(head, FiLMConvPolicyHead)

    def test_error_on_missing_args(self, action_dim, horizon):
        """Test error when neither modalities nor global_cond_dim provided."""
        with pytest.raises(ValueError):
            create_policy_head(action_dim=action_dim, horizon=horizon)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_training_step(self, sample_config, batch_size, device):
        """Simulate a training step."""
        head = MultiModalPolicyHead(sample_config).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-4)

        # Forward
        x = torch.randn(batch_size, sample_config.horizon, sample_config.action_dim, device=device)
        timestep = torch.rand(batch_size, device=device)
        target = torch.randn_like(x)
        conditions = {
            "image": torch.randn(batch_size, 512 * 2, device=device),
            "force": torch.randn(batch_size, 6, device=device),
            "proprioception": torch.randn(batch_size, 9 * 2, device=device),
        }

        output = head(x, timestep, conditions)
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0

    def test_inference(self, sample_config, device):
        """Test inference mode."""
        head = MultiModalPolicyHead(sample_config).to(device)
        head.eval()

        with torch.no_grad():
            x = torch.randn(1, sample_config.horizon, sample_config.action_dim, device=device)
            timestep = torch.tensor([0.5], device=device)
            conditions = {
                "image": torch.randn(1, 512 * 2, device=device),
            }

            output = head(x, timestep, conditions)

        assert output.shape == x.shape


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
