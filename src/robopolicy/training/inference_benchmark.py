"""Reproducible inference measurements, with a separately logged evaluation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import Field
from torch import Tensor

from robopolicy.checkpoints.inference import (
    InferenceConfig,
    InferencePolicy,
    inference_precision,
    load_policy,
)
from robopolicy.contracts.schema import StrictConfig
from robopolicy.training.logging import LoggingConfig, WandbConfig, _start_online_run, _upload_file


class InferenceBenchmarkConfig(StrictConfig):
    """Fixed-shape GPU measurement settings and explicit evaluation logging.

    Attributes:
        checkpoints: Named checkpoint paths, used only for inference reconstruction.
        device: One CUDA device, isolated from other measured jobs.
        seed: Generated fixture seed; no dataset is required.
        iterations: Predictions in each measured repetition and timing path.
        repetitions: Independent steady-state measurement repetitions.
        logging: Local artifacts and optional online evaluation tracking.
    """

    version: Literal[1] = 1
    checkpoints: dict[str, str] = Field(min_length=1)
    baselines: dict[str, str] = Field(default_factory=dict)
    device: str = "cuda:0"
    seed: int = 2026
    iterations: int = Field(default=1000, ge=1)
    repetitions: int = Field(default=3, ge=1)
    logging: LoggingConfig = Field(
        default_factory=lambda: LoggingConfig(
            out_dir="runs/inference-benchmark",
            wandb=WandbConfig(project="robopolicy-pilot", name="compiled-inference"),
        )
    )


def generated_inputs(runtime: InferencePolicy, seed: int) -> tuple[dict[str, Tensor], Tensor]:
    """Create reproducible CPU observations and noise in checkpoint-declared shapes.

    Numeric values span the saved normalization scales. Images are generated RGB
    pixels; these fixtures measure execution and numerical agreement, not robot
    success or task accuracy.

    Args:
        runtime: Reconstructed model and feature declarations.
        seed: Local generator seed, independent of global RNG state.

    Returns:
        Raw observations with B=1 and noise ``(1, T_action, D_action)``.
    """
    generator = torch.Generator().manual_seed(seed)
    raw: dict[str, Tensor] = {}
    for feature in runtime.packer.schema.features:
        if feature.key == "actions":
            continue
        shape = (
            1,
            feature.window_size,
            *((len(feature.cameras),) if feature.source == "video" else ()),
            *feature.shape,
        )
        if feature.dtype == "uint8":
            value = torch.randint(256, shape, dtype=torch.uint8, generator=generator)
        elif feature.dtype == "int64":
            value = torch.zeros(shape, dtype=torch.int64)
        else:
            value = torch.randn(shape, generator=generator)
            stats = runtime.packer.normalizer.statistics.features.get(feature.key)
            if stats is not None:
                value = value * value.new_tensor(stats.scale).reshape(
                    stats.shape
                ) + value.new_tensor(stats.offset).reshape(stats.shape)
                # Shape unchanged: (B, T_feature, *feature.shape).
        raw[feature.key] = value
    config = runtime.policy.config
    return raw, torch.randn(1, config.action_steps, config.action_dim, generator=generator)


def _percentiles(values: list[float]) -> dict[str, float]:
    """Summarize measured milliseconds without rounding away comparison precision."""
    return {
        "mean_ms": float(np.mean(values)),
        **dict(zip(("p50_ms", "p95_ms", "p99_ms"), np.percentile(values, [50, 95, 99]).tolist())),
    }


@torch.inference_mode()
def profile_components(
    runtime: InferencePolicy, raw: dict[str, Tensor], noise: Tensor, iterations: int
) -> dict[str, Any]:
    """Time eager components separately from the uninstrumented API benchmark.

    Args:
        runtime: Eager FP32 CUDA inference runtime.
        raw: Validated device-resident observations with batch size one.
        noise: Device-resident initial actions ``(1, T_action, D_action)``.
        iterations: Number of instrumented predictions.

    Returns:
        CUDA timings per prediction and parameter counts. Observation children
        overlap observation_total; decoder children overlap solver_total. Repeated
        decoder calls are summed within each prediction, not treated as samples.
    """
    if iterations < 1:
        raise ValueError("Component profiling requires positive iterations")
    if runtime.compiled or runtime.config.precision != "fp32":
        raise ValueError("Component profiling requires eager FP32 inference")
    encoder = runtime.policy.encoder
    head = runtime.policy.field.network
    modules = {
        "vision": encoder.vision,
        "qformer": encoder.fusion,
        "state_projection": encoder.state_projection,
        "decoder": head,
        "decoder_state_conditioning": getattr(head, "state_proj", None),
    }
    events: dict[str, list[tuple[Any, Any]]] = {name: [] for name in modules}
    handles = []
    event_factory: Any = torch.cuda.Event

    def before(name: str, *_args: Any) -> None:
        """Record a start event without synchronizing or changing model inputs."""
        start, end = event_factory(enable_timing=True), event_factory(enable_timing=True)
        start.record()
        events[name].append((start, end))

    def after(name: str, *_args: Any) -> None:
        """Close the matching module event, including each solver invocation."""
        events[name][-1][1].record()

    from functools import partial

    for name, module in modules.items():
        if module is not None:
            handles.append(module.register_forward_pre_hook(partial(before, name)))
            handles.append(module.register_forward_hook(partial(after, name)))
    stage_names = ("preprocessing", "observation_total", "solver_total", "denormalization")
    samples: dict[str, list[float]] = {name: [] for name in (*modules, *stage_names)}
    try:
        expected = runtime.predict(raw, noise)
        for _ in range(iterations):
            for pairs in events.values():
                pairs.clear()
            boundaries = [event_factory(enable_timing=True) for _ in range(5)]
            boundaries[0].record()
            inputs = runtime.packer.pack_tensors(raw)
            boundaries[1].record()
            context, _ = encoder.encode_tensors(inputs)
            boundaries[2].record()
            actions = runtime.policy.algorithm.generate_samples(
                runtime.policy.field, noise, context
            )
            boundaries[3].record()
            actual = runtime.packer.actions_to_raw(actions)
            boundaries[4].record()
            boundaries[-1].synchronize()
            for name, pairs in events.items():
                if pairs:
                    samples[name].append(sum(start.elapsed_time(end) for start, end in pairs))
            for index, name in enumerate(stage_names):
                samples[name].append(boundaries[index].elapsed_time(boundaries[index + 1]))
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    finally:
        for handle in handles:
            handle.remove()
    groups = {
        "vision": encoder.vision,
        "qformer": encoder.fusion,
        "queries": encoder.queries,
        "state_projection": encoder.state_projection,
        "decoder": head,
        "total": runtime.policy,
    }
    return {
        "iterations": iterations,
        "timings": {name: _percentiles(values) for name, values in samples.items() if values},
        "parameters": {
            name: {
                "total": sum(parameter.numel() for parameter in module.parameters()),
                "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad),
            }
            for name, module in groups.items()
            if module is not None
        },
        "note": "Nested timings overlap; decoder timings sum all solver calls per prediction.",
    }


def benchmark_checkpoint(
    checkpoint: str,
    settings: InferenceConfig,
    *,
    device: str = "cuda:0",
    seed: int = 2026,
    iterations: int = 1000,
    repetitions: int = 3,
    component_profile: bool = False,
) -> dict[str, Any]:
    """Measure one runtime in an isolated process, including transfers and ownership.

    Args:
        checkpoint: Saved model path.
        settings: Eager/compiled execution and FP32/BF16 selection.
        device: CUDA device used for all events and computation.
        seed: Generated observation/noise seed.
        iterations: Predictions per repetition and measurement path.
        repetitions: Number of timing repetitions.
        component_profile: Add a separate eager FP32 component measurement pass.

    Returns:
        Serializable configuration, latency, memory, numerical, and graph evidence.
    """
    from torch._dynamo.utils import counters

    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("Inference benchmark acceptance requires CUDA")
    if min(iterations, repetitions) < 1:
        raise ValueError("Benchmark iteration and repetition counts must be positive")
    if component_profile and (settings.compile != "disabled" or settings.precision != "fp32"):
        raise ValueError("Component profiling requires eager FP32 inference")
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    runtime = load_policy(checkpoint, device=device, inference_config=settings)
    load_seconds = time.perf_counter() - started
    raw, noise = generated_inputs(runtime, seed)
    gpu_raw = {key: value.to(device) for key, value in raw.items()}
    gpu_noise = noise.to(device)
    started = time.perf_counter()
    runtime.warmup(raw)
    warmup_seconds = time.perf_counter() - started
    startup_peak = torch.cuda.max_memory_allocated(device) / 2**20
    numerical: list[dict[str, float]] = []
    first_prediction: list[Any] = []
    for sample_seed in range(seed, seed + 8):
        sample, sample_noise = generated_inputs(runtime, sample_seed)
        packed = runtime.packer(sample, device=device)
        with inference_precision(InferenceConfig(precision="fp32")):
            expected = runtime.policy.predict(packed, sample_noise.to(device))
        actual_raw = runtime.predict(sample, sample_noise)
        actual = runtime.packer.normalizer.transform("actions", actual_raw)
        if not bool(torch.isfinite(actual).all()):
            raise RuntimeError("Inference produced nonfinite actions")
        if settings.precision == "fp32":
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        difference = (actual - expected).abs()
        numerical.append(
            {
                "max_abs_normalized": float(difference.max()),
                "rms_normalized": float(difference.square().mean().sqrt()),
            }
        )
        if sample_seed == seed:
            first_prediction = actual_raw.to("cpu").tolist()
    # Inspect runtime launches after capture; graph capture alone does not prove
    # that CUDA graph replay actually occurred.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        for _ in range(3):
            runtime.predict(gpu_raw, gpu_noise)
        torch.cuda.synchronize(device)
    launches = sum(
        event.count for event in profile.key_averages() if "cudaGraphLaunch" in event.key
    )
    graph_count = counters["stats"]["unique_graphs"]
    breaks_before = sum(counters["graph_break"].values())
    torch.cuda.reset_peak_memory_stats(device)
    reports: list[dict[str, Any]] = []
    for _ in range(repetitions):
        device_times: list[float] = []
        total_times: list[float] = []
        event_factory: Any = torch.cuda.Event  # CUDA event APIs lack typed signatures.
        start_event, stop_event = (
            event_factory(enable_timing=True),
            event_factory(enable_timing=True),
        )
        for _ in range(iterations):
            start_event.record()
            runtime.predict(gpu_raw, gpu_noise)
            stop_event.record()
            stop_event.synchronize()
            device_times.append(start_event.elapsed_time(stop_event))
        for _ in range(iterations):
            started = time.perf_counter()
            runtime.predict(raw, noise).to("cpu")
            total_times.append((time.perf_counter() - started) * 1000)
        reports.append({"gpu": _percentiles(device_times), "end_to_end": _percentiles(total_times)})
    recompilations = counters["stats"]["unique_graphs"] - graph_count
    if runtime.compiled and (
        recompilations or breaks_before or sum(counters["graph_break"].values())
    ):
        raise RuntimeError(
            "Compiled benchmark encountered graph breaks or steady-state recompilation"
        )
    if runtime.compiled and settings.mode != "default" and not launches:
        raise RuntimeError("Requested CUDA graph mode did not replay CUDA graphs")
    memory = {
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }
    components = (
        profile_components(runtime, gpu_raw, gpu_noise, iterations) if component_profile else None
    )
    return {
        "checkpoint": checkpoint,
        "sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "settings": settings.model_dump(),
        "device": device,
        "fixture_seed": seed,
        "iterations": iterations,
        "repetitions": reports,
        "load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "startup_peak_allocated_mib": startup_peak,
        **memory,
        "components": components,
        "unique_graphs": graph_count,
        "steady_state_recompilations": recompilations,
        "cuda_graph_launches_in_three_predictions": launches,
        "graph_breaks": sum(counters["graph_break"].values()),
        "numerical_comparisons": numerical,
        "fixed_noise_prediction": first_prediction,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(device),
        "policy": runtime.policy.config.model_dump(),
        "processing": runtime.packer.config.model_dump(),
    }


def run_benchmark_suite(config: InferenceBenchmarkConfig) -> Path:
    """Run isolated measurement workers and upload one aggregate evaluation artifact.

    Args:
        config: Checkpoints, timing settings, and explicit logging destination.

    Returns:
        Local aggregate JSON report, also uploaded when online logging is enabled.
    """
    output_dir = Path(config.logging.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = config.model_dump()
    (output_dir / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    result: dict[str, Any] = {
        "config": resolved,
        "wandb": None,
        "measurements": {},
        "baselines": {
            name: json.loads(Path(path).read_text()) for name, path in config.baselines.items()
        },
    }
    wandb_module, run, identity = None, None, None
    if config.logging.wandb.enabled:
        wandb_module, run, identity = _start_online_run(
            config.logging.wandb, output_dir, resolved, job_type="evaluation"
        )
        result["wandb"] = identity.model_dump()
    success = False
    try:
        for name, checkpoint in config.checkpoints.items():
            variants = [InferenceConfig(compile="disabled")]
            variants.extend(
                InferenceConfig(mode=mode)
                for mode in ("default", "reduce-overhead", "max-autotune")
            )
            variants.extend(
                InferenceConfig(mode=mode, precision="bf16")
                for mode in ("reduce-overhead", "max-autotune")
            )
            for settings in variants:
                label = f"{name}-{settings.precision}-{settings.mode if settings.compile != 'disabled' else 'eager'}"
                path = output_dir / f"{label}.json"
                worker = {
                    "checkpoint": checkpoint,
                    "settings": settings.model_dump(),
                    "device": config.device,
                    "seed": config.seed,
                    "iterations": config.iterations,
                    "repetitions": config.repetitions,
                }
                worker_path = output_dir / f"{label}-config.json"
                worker_path.write_text(json.dumps(worker, indent=2) + "\n")
                print(f"Measuring {label}", flush=True)
                with (output_dir / f"{label}.log").open("w") as stream:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            __name__,
                            "--worker",
                            str(worker_path),
                            "--output",
                            str(path),
                        ],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                measurement = json.loads(path.read_text())
                baseline = result["baselines"].get(name)
                if baseline is not None and (
                    baseline["sha256"] != measurement["sha256"]
                    or baseline["fixture_seed"] != config.seed
                    or baseline["iterations"] != config.iterations
                ):
                    raise ValueError(f"Baseline checkpoint/fixture mismatch for {name}")
                result["measurements"][label] = measurement
                medians = {
                    key: float(
                        np.median([rep["end_to_end"][key] for rep in measurement["repetitions"]])
                    )
                    for key in ("p50_ms", "p95_ms", "p99_ms")
                }
                print(f"{label}: {medians}", flush=True)
                if run is not None:
                    run.log({f"inference/{label}/{key}": value for key, value in medians.items()})
                (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        result["recommendations"] = {}
        for name in config.checkpoints:
            for precision in ("fp32", "bf16"):
                candidates = {
                    label: measurement
                    for label, measurement in result["measurements"].items()
                    if label.startswith(f"{name}-{precision}-")
                }
                latency = {
                    label: float(
                        np.median(
                            [rep["end_to_end"]["p95_ms"] for rep in measurement["repetitions"]]
                        )
                    )
                    for label, measurement in candidates.items()
                }
                fastest = min(latency.values())
                equivalent = [label for label in candidates if latency[label] <= fastest * 1.03]
                winner = min(equivalent, key=lambda label: candidates[label]["warmup_seconds"])
                result["recommendations"][f"{name}-{precision}"] = {
                    "measurement": winner,
                    "settings": candidates[winner]["settings"],
                }
        result_path = output_dir / "results.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        if run is not None:
            assert identity is not None
            _upload_file(
                wandb_module,
                run,
                result_path,
                name=f"robopolicy-inference-{identity.id}",
                artifact_type="evaluation-result",
                aliases=("latest",),
                metadata={"fixture_seed": config.seed},
            )
        success = True
        return result_path
    finally:
        if run is not None:
            run.finish(exit_code=0 if success else 1)


def _worker_main() -> None:
    """Read one internal worker specification; suite orchestration owns logging."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kwargs = json.loads(args.worker.read_text())
    kwargs["settings"] = InferenceConfig.model_validate(kwargs["settings"])
    report = benchmark_checkpoint(**kwargs)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    _worker_main()
