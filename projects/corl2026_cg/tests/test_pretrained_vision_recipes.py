"""Pretrained positional grids remain separate from the policy's image crops."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from robopolicy.modules.vision import VisionConfig, VisionEncoder
from transformers import Dinov2Config, Dinov2Model

from corl2026_cg.experiments.config import load_experiment

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.json")), ids=lambda path: path.stem)
def test_pretrained_dino_recipes_preserve_grid_and_training_crop(path: Path) -> None:
    """Facebook DINOv2-small weights require 37x37 saved positional patches."""
    run = load_experiment(path).run
    vision = run.policy.vision
    if vision is None or vision.type != "dinov2" or vision.pretrained != "facebook/dinov2-small":
        return
    assert (vision.dim, vision.depth, vision.heads, vision.patch_size) == (384, 12, 6, 14)
    assert vision.image_size == 518
    assert run.processing.crop_size == (224, 224)


def test_local_pretrained_grid_loads_smaller_images_and_reconstructs_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise real HF loading and interpolation without downloads or large weights."""
    source = Dinov2Model(
        Dinov2Config(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=28,
            patch_size=4,
        )
    ).eval()
    pretrained = tmp_path / "local-pretrained"
    source.save_pretrained(pretrained)
    config = VisionConfig(
        type="dinov2",
        dim=16,
        depth=1,
        heads=2,
        image_size=28,
        patch_size=4,
        freeze=True,
        pretrained=str(pretrained),
    )
    encoder = VisionEncoder(config, initialize_pretrained=True)
    images = torch.linspace(0, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    with torch.inference_mode():
        expected = source(pixel_values=images).last_hidden_state[:, 1:]
        actual = encoder(images)
    assert tuple(encoder.state_dict()["backbone.embeddings.position_embeddings"].shape) == (
        1,
        50,
        16,
    )
    assert actual.shape == (1, 16, 16)
    torch.testing.assert_close(actual, expected)
    assert all(not parameter.requires_grad for parameter in encoder.parameters())

    checkpoint = tmp_path / "encoder.pt"
    torch.save(
        {"config": config.model_dump(mode="json"), "model": encoder.state_dict()}, checkpoint
    )
    shutil.rmtree(pretrained)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("offline reconstruction must not consult pretrained files or the network")

    monkeypatch.setattr(Dinov2Model, "from_pretrained", forbidden)
    payload = torch.load(checkpoint, weights_only=True)
    reconstructed = VisionEncoder(
        VisionConfig.model_validate(payload["config"]), initialize_pretrained=False
    )
    reconstructed.load_state_dict(payload["model"], strict=True)
    with torch.inference_mode():
        restored = reconstructed(images)
    torch.testing.assert_close(restored, actual)
