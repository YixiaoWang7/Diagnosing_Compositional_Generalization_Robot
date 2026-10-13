"""Compiled inference preserves prediction, validation, and output ownership."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import Tensor

from robopolicy import InferenceConfig, InferencePolicy, InputPacker
from robopolicy.checkpoints.inference import inference_precision
from robopolicy.contracts.schema import Feature, FeatureSchema
from robopolicy.policies.flow import build_policy
from robopolicy.processing.normalization import FeatureStatistics, Statistics
from robopolicy.processing.packing import ProcessingConfig
from tests.core.test_policy import policy_config


def inference_fixture(
    head: str = "transformer",
    modality: str = "state",
    solver: str = "euler",
    device: str = "cpu",
    compiled: bool = False,
    precision: str = "fp32",
) -> tuple[InferencePolicy, dict[str, Tensor], Tensor]:
    """Build a small production policy with nonidentity statistics and raw inputs."""
    torch.manual_seed(19)
    images = modality in {"images", "dinov2", "difference"}
    indexed = modality == "indexed"
    entities = modality == "entities"
    config = policy_config(
        head,
        images,
        entity_dim=3 if entities else 0,
        selected_entities=2 if entities else 0,
        conditioning={"type": "indexed", "vocab_sizes": [2, 3, 4]}
        if indexed
        else {"type": "learned", "num_tokens": 2},
        algorithm={"solver_type": solver, "num_inference_steps": 2, "velocity_clamp": 2.0},
    )
    if modality == "dinov2":
        config = type(config).model_validate(
            {
                **config.model_dump(),
                "vision": {
                    "type": "dinov2",
                    "dim": 16,
                    "heads": 2,
                    "depth": 1,
                    "image_size": 16,
                    "patch_size": 4,
                },
            }
        )
    features = [
        Feature(key="state", shape=(2,), paths=("state",), history=1, normalization="standard"),
        Feature(key="actions", shape=(1,), paths=("actions",), future=3, normalization="standard"),
    ]
    raw = {"state": torch.randn(1, 2, 2)}
    if images:
        features.append(
            Feature(
                key="images",
                shape=(3, 20, 20),
                source="video",
                cameras=("left", "right"),
                dtype="uint8",
                history=1,
            )
        )
        raw["images"] = torch.randint(256, (1, 2, 2, 3, 20, 20), dtype=torch.uint8)
    if indexed:
        features.append(Feature(key="task_indices", shape=(3,), paths=("task",), dtype="int64"))
        raw["task_indices"] = torch.tensor([[[1, 2, 3]]])
    if entities:
        features.extend(
            [
                Feature(key="entities", shape=(4, 3), paths=("entities",), history=1),
                Feature(key="entity_indices", shape=(2,), paths=("indices",), dtype="int64"),
            ]
        )
        raw["entities"] = torch.randn(1, 2, 4, 3)
        raw["entity_indices"] = torch.tensor([[[3, 0]]])
    schema = FeatureSchema(features=tuple(features))
    stats = Statistics(
        feature_schema=schema,
        dataset_fingerprint="fixture",
        selection=(),
        features={
            "state": FeatureStatistics(shape=(2,), count=5, offset=(0.3, -0.2), scale=(0.7, 1.2)),
            "actions": FeatureStatistics(shape=(1,), count=5, offset=(1.3,), scale=(0.8,)),
        },
    )
    packer = InputPacker(
        schema,
        stats,
        ProcessingConfig(
            image_size=(16, 16),
            crop_size=(18, 18),
            image_representation="current_and_signed_difference"
            if modality == "difference"
            else "rgb",
        ),
    )
    settings = InferenceConfig.model_validate(
        {
            "compile": "enabled" if compiled else "disabled",
            "mode": "reduce-overhead",
            "precision": precision,
        }
    )
    policy = build_policy(config)
    if head == "transformer":
        # Production initializes this projection to zero. Give the fixture a
        # nonzero field so parity and changed-input tests exercise the network.
        torch.nn.init.normal_(policy.field.network.out_proj.weight, std=0.05)
    return (
        InferencePolicy(policy, packer, device, settings),
        raw,
        torch.randn(1, 4, 1),
    )


@pytest.mark.parametrize("head", ["conv", "transformer"])
@pytest.mark.parametrize("solver", ["euler", "heun", "midpoint", "rk4"])
@pytest.mark.parametrize("modality", ["state", "images", "indexed", "entities", "difference"])
def test_fullgraph_matches_public_prediction(head: str, solver: str, modality: str) -> None:
    runtime, raw, noise = inference_fixture(head, modality, solver)
    original = {key: value.clone() for key, value in raw.items()}
    with torch.inference_mode():
        inputs = runtime.packer(raw)
        expected = runtime.packer.actions_to_raw(runtime.policy.predict(inputs, noise))
        # Capture the whole numerical path. Separate CUDA tests use real Inductor.
        captured = torch.compile(
            runtime._prediction, backend="eager", fullgraph=True, dynamic=False
        )
        actual = captured(raw, noise)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(runtime.predict(raw, noise), expected, rtol=0, atol=0)
    for key, value in raw.items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    assert runtime.packer.state_dict() == {}  # Statistics stay in their versioned schema.
    assert not any("_orig_mod" in key for key in runtime.policy.state_dict())
    torch._dynamo.reset()


@pytest.mark.parametrize("modality", ["indexed", "entities"])
def test_inference_validates_changing_ids(modality: str) -> None:
    runtime, raw, noise = inference_fixture(modality=modality)
    runtime.predict(raw, noise)
    key = "task_indices" if modality == "indexed" else "entity_indices"
    with pytest.raises(ValueError, match="slot|range"):
        runtime.predict({**raw, key: torch.full_like(raw[key], 99)}, noise)
    with pytest.raises(ValueError, match="dtype"):
        runtime.predict({**raw, key: raw[key].float()}, noise)
    with pytest.raises(ValueError, match="noise"):
        runtime.predict(raw, noise.double())


def test_warmup_preserves_rng_and_cpu_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, raw, _ = inference_fixture()
    monkeypatch.setattr(torch, "compile", lambda *a, **k: pytest.fail("CPU auto compiled"))
    runtime = InferencePolicy(runtime.policy, runtime.packer, "cpu")
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    runtime.warmup(raw)
    assert before[0] == random.getstate()
    assert before[1][0] == np.random.get_state()[0]
    np.testing.assert_array_equal(before[1][1], np.random.get_state()[1])
    assert before[1][2:] == np.random.get_state()[2:]
    assert torch.equal(before[2], torch.get_rng_state())
    assert not runtime.compiled
    with pytest.raises(ValueError, match="positive"):
        runtime.warmup(raw, iterations=0)
    with pytest.raises(ValueError, match="CUDA"):
        InferencePolicy(runtime.policy, runtime.packer, "cpu", InferenceConfig(compile="enabled"))
    with pytest.raises(ValueError, match="BF16"):
        InferencePolicy(runtime.policy, runtime.packer, "cpu", InferenceConfig(precision="bf16"))
    with pytest.raises(ValueError):
        InferenceConfig.model_validate({"version": 2})


@pytest.mark.cuda
@pytest.mark.parametrize("head", ["conv", "transformer"])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("modality", ["images", "difference"])
def test_inductor_prediction_lifecycle(head: str, precision: str, modality: str) -> None:
    from torch._dynamo.utils import counters

    assert torch.cuda.is_available(), "CUDA acceptance requires a GPU"
    runtime, raw, noise = inference_fixture(
        head, modality, device="cuda:0", compiled=True, precision=precision
    )
    before_weights = {key: value.clone() for key, value in runtime.policy.state_dict().items()}
    cuda_rng = torch.cuda.get_rng_state("cuda:0")
    runtime.warmup(raw)
    graphs = counters["stats"]["unique_graphs"]
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state("cuda:0"))
    with torch.inference_mode():
        values = {key: value.to(runtime.device) for key, value in raw.items()}
        with inference_precision(runtime.config):
            expected = runtime._prediction(values, noise.to(runtime.device))
        actual = runtime.predict(raw, noise)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-4 if precision == "fp32" else 2e-2,
            atol=1e-5 if precision == "fp32" else 1e-2,
        )
        retained = actual.clone()
        changed = {**raw, "state": raw["state"] + 1.0, "images": 255 - raw["images"]}
        second = runtime.predict(changed, noise + 0.25)
        assert not torch.equal(second, actual)
        torch.testing.assert_close(actual, retained, rtol=0, atol=0)
        gpu_result = runtime.predict(values, noise)
        torch.testing.assert_close(gpu_result, retained, rtol=0, atol=0)
        # Caller strides and source devices cannot cause runtime specialization.
        strided = {
            key: value.transpose(-1, -2).contiguous().transpose(-1, -2)
            for key, value in raw.items()
        }
        torch.testing.assert_close(runtime.predict(strided, noise), retained, rtol=0, atol=0)
        assert actual.dtype == torch.float32 and not actual.requires_grad
    with pytest.raises(ValueError, match="signature"):
        runtime.predict(
            {key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in raw.items()}
        )
    for key, weight in runtime.policy.state_dict().items():
        torch.testing.assert_close(weight, before_weights[key], rtol=0, atol=0)
    assert counters["stats"]["unique_graphs"] == graphs
    torch._dynamo.reset()


@pytest.mark.cuda
@pytest.mark.parametrize("modality", ["indexed", "entities"])
def test_compiled_selection_changes_without_recompilation(modality: str) -> None:
    from torch._dynamo.utils import counters

    runtime, raw, noise = inference_fixture(modality=modality, device="cuda:0", compiled=True)
    runtime.warmup(raw)
    graphs = counters["stats"]["unique_graphs"]
    first = runtime.predict(raw, noise)
    key = "task_indices" if modality == "indexed" else "entity_indices"
    changed = {**raw, key: torch.zeros_like(raw[key])}
    second = runtime.predict(changed, noise)
    assert not torch.equal(first, second)
    assert counters["stats"]["unique_graphs"] == graphs
    with pytest.raises(ValueError, match="slot|range"):
        runtime.predict({**raw, key: torch.full_like(raw[key], 99)}, noise)
    torch._dynamo.reset()


@pytest.mark.cuda
@pytest.mark.vision
def test_dinov2_real_inductor() -> None:
    assert torch.cuda.is_available(), "DINOv2 Inductor acceptance requires CUDA"
    runtime, raw, noise = inference_fixture(modality="dinov2", device="cuda:0", compiled=True)
    runtime.warmup(raw)
    with torch.inference_mode():
        packed = runtime.packer(raw, device=runtime.device)
        with inference_precision(runtime.config):
            expected = runtime.packer.actions_to_raw(
                runtime.policy.predict(packed, noise.to(runtime.device))
            )
        torch.testing.assert_close(runtime.predict(raw, noise), expected, rtol=1e-4, atol=1e-5)
    torch._dynamo.reset()


@pytest.mark.cuda
def test_compiler_failure_is_explicit_and_preserves_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, raw, _ = inference_fixture(device="cuda:0", compiled=True)
    before = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
    )
    backends = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)

    def broken(*args: object) -> Tensor:
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
        random.random()
        np.random.rand()
        torch.rand(1)
        torch.rand(1, device=runtime.device)
        raise RuntimeError("deliberate compiler failure")

    monkeypatch.setattr(runtime, "_execute", broken)
    with pytest.raises(RuntimeError, match="no eager fallback") as error:
        runtime.warmup(raw)
    assert "deliberate compiler failure" in str(error.value.__cause__)
    assert before[0] == random.getstate()
    np.testing.assert_array_equal(before[1][1], np.random.get_state()[1])
    assert torch.equal(before[2], torch.get_rng_state())
    assert torch.equal(before[3], torch.cuda.get_rng_state())
    assert backends == (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    assert runtime._signature is None
