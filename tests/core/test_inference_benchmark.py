"""Benchmark orchestration fails clearly and closes its online evaluation run."""

import subprocess
from pathlib import Path

import pytest
import torch

from robopolicy.training import inference_benchmark as benchmark
from robopolicy.training.logging import RunIdentity
from tests.core.test_inference import inference_fixture


def test_latency_summary_includes_arithmetic_mean() -> None:
    summary = benchmark._percentiles([1.0, 2.0, 9.0])
    assert summary["mean_ms"] == 4.0 and summary["p50_ms"] == 2.0


@pytest.mark.cuda
@pytest.mark.parametrize("algorithm", ["flow_matching", "improved_mean_flow"])
def test_component_profile_preserves_prediction_and_removes_hooks(algorithm: str) -> None:
    if not torch.cuda.is_available():
        pytest.fail("Component profiling acceptance requires CUDA")
    runtime, raw, noise = inference_fixture(modality="images", device="cuda")
    if algorithm == "improved_mean_flow":
        from robopolicy import build_policy

        config = runtime.policy.config
        values = config.model_dump()
        values["algorithm"] = {"type": algorithm, "num_inference_steps": 2}
        runtime = benchmark.InferencePolicy(
            build_policy(type(config).model_validate(values)),
            runtime.packer,
            device="cuda",
            inference_config=benchmark.InferenceConfig(compile="disabled"),
        )
    raw = {key: value.to(runtime.device) for key, value in raw.items()}
    noise = noise.to(runtime.device)
    expected = runtime.predict(raw, noise)
    profile = benchmark.profile_components(runtime, raw, noise, 3)
    torch.testing.assert_close(runtime.predict(raw, noise), expected)
    assert profile["parameters"]["total"]["total"] == sum(
        p.numel() for p in runtime.policy.parameters()
    )
    assert profile["timings"]["decoder"]["mean_ms"] > 0
    assert profile["timings"]["qformer"]["mean_ms"] > 0
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks
        for module in runtime.policy.modules()
    )


def test_baseline_validation_precedes_online_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = benchmark.InferenceBenchmarkConfig(
        checkpoints={"model": "saved.pt"},
        baselines={"model": str(tmp_path / "missing.json")},
        logging={"out_dir": str(tmp_path), "wandb": {"enabled": True}},
    )
    monkeypatch.setattr(
        benchmark, "_start_online_run", lambda *a, **k: pytest.fail("Premature online run")
    )
    with pytest.raises(FileNotFoundError):
        benchmark.run_benchmark_suite(config)


def test_worker_failure_finishes_run_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finished: list[int] = []

    class Run:
        def finish(self, *, exit_code: int) -> None:
            finished.append(exit_code)

    identity = RunIdentity(entity="yixiao_wang", project="robopolicy-pilot", id="fixture")
    monkeypatch.setattr(benchmark, "_start_online_run", lambda *a, **k: (None, Run(), identity))
    monkeypatch.setattr(
        benchmark, "_upload_file", lambda *a, **k: pytest.fail("Uploaded failed results")
    )

    def fail_worker(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "measurement worker")

    monkeypatch.setattr(benchmark.subprocess, "run", fail_worker)
    config = benchmark.InferenceBenchmarkConfig(
        checkpoints={"model": "saved.pt"},
        logging={"out_dir": str(tmp_path), "wandb": {"enabled": True}},
    )
    with pytest.raises(subprocess.CalledProcessError):
        benchmark.run_benchmark_suite(config)
    assert finished == [1]
