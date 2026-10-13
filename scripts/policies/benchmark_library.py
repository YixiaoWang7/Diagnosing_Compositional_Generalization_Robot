#!/usr/bin/env python3
"""Benchmark finite RoboPolicy loaders with a fixed GPU preprocessing workload.

The benchmark uses the real V2 sampling and collation path. Each batch remains
RGB/BGR uint8 in host memory, is copied to CUDA as uint8, and then follows the
same float conversion, resize, and ImageNet normalization used by DINOv2. It
does not instantiate DINOv2 itself, so reported GPU memory is the data-pipeline
footprint and excludes backbone weights, activations, and optimizer state.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from robopolicy.contracts.schema import FeatureSchema
from robopolicy.datasets.config import DatasetConfig, LoaderConfig
from robopolicy.datasets.factory import build_dataset, build_loader

MIB = 1024.0 * 1024.0


@dataclass
class ResourcePeaks:
    """Peak aggregate resources for the parent and all DataLoader workers."""

    rss_mib: float = 0.0
    pss_mib: float = 0.0
    uss_mib: float = 0.0
    processes: int = 0
    threads: int = 0
    open_files: int = 0


class ProcessTreeMonitor:
    """Periodically sample RAM and process counts for a process tree."""

    def __init__(
        self,
        root_pid: int,
        interval_s: float = 0.2,
        max_pss_mib: float | None = None,
    ) -> None:
        self.root = psutil.Process(int(root_pid))
        self.interval_s = float(interval_s)
        self.max_pss_mib = None if max_pss_mib is None else float(max_pss_mib)
        self.peaks = ResourcePeaks()
        self.history = []
        self._last_history = 0.0
        self._stop_event = threading.Event()
        self._memory_limit_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """Start asynchronous sampling."""
        self._sample()
        self._thread.start()

    def stop(self) -> ResourcePeaks:
        """Stop sampling and return accumulated peaks."""
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._sample()
        return self.peaks

    def snapshot(self) -> ResourcePeaks:
        """Return a one-time aggregate snapshot."""
        return self._collect()

    @property
    def memory_limit_exceeded(self) -> bool:
        """Return whether aggregate PSS reached the configured guard limit."""
        return self._memory_limit_event.is_set()

    def _processes(self) -> list[psutil.Process]:
        processes = [self.root]
        try:
            processes.extend(self.root.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        unique: dict[int, psutil.Process] = {process.pid: process for process in processes}
        return list(unique.values())

    def _collect(self) -> ResourcePeaks:
        rss = 0
        pss = 0
        uss = 0
        process_count = 0
        thread_count = 0
        open_files = 0
        for process in self._processes():
            try:
                memory = process.memory_info()
                rss += int(memory.rss)
                full = process.memory_full_info()
                pss += int(getattr(full, "pss", 0))
                uss += int(getattr(full, "uss", 0))
                thread_count += int(process.num_threads())
                open_files += int(len(process.open_files()))
                process_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
                continue
        return ResourcePeaks(
            rss_mib=rss / MIB,
            pss_mib=pss / MIB,
            uss_mib=uss / MIB,
            processes=process_count,
            threads=thread_count,
            open_files=open_files,
        )

    def _sample(self) -> None:
        sample = self._collect()
        now = time.monotonic()
        if now - self._last_history >= 30:
            self.history.append({"time_s": now, **asdict(sample)})
            self._last_history = now
        for field in asdict(self.peaks):
            setattr(self.peaks, field, max(getattr(self.peaks, field), getattr(sample, field)))
        if self.max_pss_mib is not None and sample.pss_mib >= self.max_pss_mib:
            self._memory_limit_event.set()

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self._sample()


def process_tree_cpu_seconds(root_pid: int) -> float:
    """Return cumulative user+system CPU seconds for the live process tree."""
    root = psutil.Process(int(root_pid))
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    total = 0.0
    for process in processes:
        try:
            times = process.cpu_times()
            total += float(times.user + times.system)
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            continue
    return total


def move_tensor_payload_to_device(value: Any, device: torch.device) -> Any:
    """Move tensors recursively while leaving string metadata on the host."""
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_tensor_payload_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensor_payload_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensor_payload_to_device(item, device) for item in value)
    return value


def virtual_dino_step(
    batch: dict[str, Any],
    *,
    cameras: Sequence[str],
    device: torch.device,
) -> tuple[float, float, int, tuple[int, ...], torch.dtype]:
    """Run actual uint8 H2D transfer and DINOv2 input preprocessing.

    Returns:
        Tuple containing H2D milliseconds, preprocessing milliseconds, checksum,
        packed image shape, and host image dtype.
    """
    host_dtype = batch["images"].dtype
    if host_dtype != torch.uint8:
        raise TypeError(f"Expected host uint8 images, received {host_dtype}.")

    h2d_start = torch.cuda.Event(enable_timing=True)
    h2d_end = torch.cuda.Event(enable_timing=True)
    prep_end = torch.cuda.Event(enable_timing=True)
    h2d_start.record()
    gpu_batch = move_tensor_payload_to_device(batch, device)
    h2d_end.record()

    images = gpu_batch["images"]
    packed_shape = tuple(int(size) for size in images.shape)
    images_flat = images.reshape(-1, *images.shape[-3:])
    normalized = images_flat.to(dtype=torch.float32).div_(255.0)
    normalized = F.interpolate(
        normalized,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )
    mean = normalized.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = normalized.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normalized = (normalized - mean) / std
    checksum_tensor = normalized.mean()
    prep_end.record()
    torch.cuda.synchronize(device)
    checksum = int(float(checksum_tensor.item()) * 1_000_000)
    return (
        float(h2d_start.elapsed_time(h2d_end)),
        float(h2d_end.elapsed_time(prep_end)),
        checksum,
        packed_shape,
        host_dtype,
    )


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a percentile, or zero for an empty input."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def create_loader(args: argparse.Namespace) -> Any:
    """Build a finite library loader from the supplied generic specification."""
    spec = json.loads(args.spec.read_text())
    dataset = build_dataset(
        DatasetConfig.model_validate(spec["dataset"]),
        FeatureSchema.model_validate(spec["features"]),
    )
    loader = build_loader(dataset, LoaderConfig.model_validate(spec["loader"]))
    return loader


class EpochIterator:
    """Benchmark-owned epoch advancement around the finite library loader."""

    def __init__(self, loader: Any) -> None:
        self.loader = loader
        self.iterator = iter(loader)
        self.epoch = 0

    def __next__(self) -> dict[str, Any]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.loader.dataset.set_epoch(self.epoch)
            self.iterator = iter(self.loader)
            return next(self.iterator)

    def _shutdown_workers(self) -> None:
        self.loader.close()


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run startup, warmup, and steady-state measurements."""
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA for the virtual DINOv2 stage.")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gpu_free, gpu_total = torch.cuda.mem_get_info(device)
    gpu_baseline_used = int(gpu_total - gpu_free)

    if args.max_ram_gib is not None and float(args.max_ram_gib) <= 0:
        raise ValueError("max_ram_gib must be > 0 when provided.")
    max_pss_mib = None if args.max_ram_gib is None else float(args.max_ram_gib) * 1024.0
    monitor = ProcessTreeMonitor(os.getpid(), max_pss_mib=max_pss_mib)
    baseline = monitor.snapshot()
    monitor.start()
    loader = create_loader(args)
    iterator = EpochIterator(loader)

    startup_begin = time.perf_counter()
    first_batch = next(iterator)
    first_metrics = virtual_dino_step(
        first_batch,
        cameras=args.cameras,
        device=device,
    )
    startup_s = time.perf_counter() - startup_begin
    gpu_device_peak_used = gpu_baseline_used
    packed_shape = first_metrics[3]
    host_dtype = first_metrics[4]
    del first_batch

    for _ in range(int(args.warmup_batches)):
        batch = next(iterator)
        virtual_dino_step(batch, cameras=args.cameras, device=device)
        free, total = torch.cuda.mem_get_info(device)
        gpu_device_peak_used = max(gpu_device_peak_used, int(total - free))
        del batch

    wait_ms: list[float] = []
    step_ms: list[float] = []
    h2d_ms: list[float] = []
    preprocess_ms: list[float] = []
    checksum = int(first_metrics[2])
    cpu_start = process_tree_cpu_seconds(os.getpid())
    measure_start = time.perf_counter()
    measured_batches = 0
    if args.measure_seconds is not None and float(args.measure_seconds) <= 0:
        raise ValueError("measure_seconds must be > 0 when provided.")
    stop_reason = "duration" if args.measure_seconds is not None else "batches"
    while True:
        if monitor.memory_limit_exceeded:
            stop_reason = "ram_limit"
            break
        elapsed_s = time.perf_counter() - measure_start
        if args.measure_seconds is not None:
            if measured_batches > 0 and elapsed_s >= float(args.measure_seconds):
                break
        elif measured_batches >= int(args.measure_batches):
            break
        step_start = time.perf_counter()
        wait_start = time.perf_counter()
        batch = next(iterator)
        wait_ms.append((time.perf_counter() - wait_start) * 1000.0)
        h2d, preprocess, value, packed_shape, host_dtype = virtual_dino_step(
            batch,
            cameras=args.cameras,
            device=device,
        )
        checksum ^= int(value)
        h2d_ms.append(h2d)
        preprocess_ms.append(preprocess)
        step_ms.append((time.perf_counter() - step_start) * 1000.0)
        free, total = torch.cuda.mem_get_info(device)
        gpu_device_peak_used = max(gpu_device_peak_used, int(total - free))
        del batch
        measured_batches += 1
    measure_s = time.perf_counter() - measure_start
    cpu_s = process_tree_cpu_seconds(os.getpid()) - cpu_start
    peaks = monitor.stop()

    if hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()
    resources = loader.dataset.resources
    selected = loader.dataset.selection
    del iterator, loader
    gc.collect()
    torch.cuda.synchronize(device)

    samples = int(measured_batches) * int(args.batch_size)
    image_frames = samples * len(args.cameras)
    logical_cpus = max(1, int(os.cpu_count() or 1))
    return {
        "backend": str(args.backend),
        "image_color_order": "RGB",
        "data_dir": str(Path(args.data_dir).resolve()),
        "config": {
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "k_passes": int(args.k_passes),
            "active_window_shards": int(args.active_window_shards),
            "locality_block_size": int(args.locality_block_size),
            "max_demos": int(args.max_demos),
            "measure_batches_requested": int(args.measure_batches),
            "measure_seconds_requested": (
                None if args.measure_seconds is None else float(args.measure_seconds)
            ),
            "max_ram_gib": (None if args.max_ram_gib is None else float(args.max_ram_gib)),
            "frame_cache_max_entries": (
                int(args.current_frame_cache) if args.backend == "current" else 0
            ),
            "decoder_cache_max_entries": args.decoder_handles,
            "cameras": list(args.cameras),
        },
        "batch": {
            "packed_image_shape": list(packed_shape),
            "host_dtype": str(host_dtype),
            "raw_image_mib": (int(args.batch_size) * len(args.cameras) * 3 * 256 * 256 / MIB),
        },
        "startup": {
            "first_batch_and_gpu_preprocess_s": float(startup_s),
            "first_h2d_ms": float(first_metrics[0]),
            "first_preprocess_ms": float(first_metrics[1]),
        },
        "throughput": {
            "measurement_s": float(measure_s),
            "measured_batches": int(measured_batches),
            "stop_reason": stop_reason,
            "samples": samples,
            "samples_per_s": samples / measure_s,
            "image_frames_per_s": image_frames / measure_s,
            "batch_ms_mean": float(np.mean(step_ms)),
            "batch_ms_p50": percentile(step_ms, 50),
            "batch_ms_p95": percentile(step_ms, 95),
            "dataloader_wait_ms_mean": float(np.mean(wait_ms)),
            "dataloader_wait_ms_p50": percentile(wait_ms, 50),
            "dataloader_wait_ms_p95": percentile(wait_ms, 95),
            "h2d_ms_mean": float(np.mean(h2d_ms)),
            "preprocess_ms_mean": float(np.mean(preprocess_ms)),
            "cpu_seconds": float(cpu_s),
            "cpu_core_equivalent": float(cpu_s / measure_s),
            "cpu_percent_of_machine": float(100.0 * cpu_s / measure_s / logical_cpus),
        },
        "ram": {
            "baseline_parent_rss_mib": float(baseline.rss_mib),
            "baseline_parent_pss_mib": float(baseline.pss_mib),
            "peak_tree_rss_mib": float(peaks.rss_mib),
            "peak_tree_pss_mib": float(peaks.pss_mib),
            "peak_tree_uss_mib": float(peaks.uss_mib),
            "peak_tree_processes": int(peaks.processes),
            "peak_tree_threads": int(peaks.threads),
            "peak_tree_open_files": int(peaks.open_files),
            "memory_limit_exceeded": bool(monitor.memory_limit_exceeded),
        },
        "gpu": {
            "device": str(device),
            "baseline_device_used_mib": gpu_baseline_used / MIB,
            "peak_device_used_mib": gpu_device_peak_used / MIB,
            "peak_device_delta_mib": (gpu_device_peak_used - gpu_baseline_used) / MIB,
            "peak_torch_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
            "peak_torch_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
            "scope": "uint8 transfer and DINOv2 preprocessing only; model excluded",
        },
        "checksum": checksum,
        "decoder_resources": resources,
        "resource_history": monitor.history,
        "selection": selected,
    }


def parse_args() -> argparse.Namespace:
    """Read a portable dataset specification and measurement duration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--measure-seconds", type=float, default=30)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--max-ram-gib", type=float, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    dataset = DatasetConfig.model_validate(spec["dataset"])
    loader = LoaderConfig.model_validate(spec["loader"])
    features = FeatureSchema.model_validate(spec["features"])
    args.backend = f"{dataset.video.decoder}/{dataset.sampling.strategy}"
    args.data_dir = dataset.root
    args.batch_size = loader.batch_size
    args.num_workers = loader.num_workers
    args.prefetch_factor = loader.prefetch_factor
    args.k_passes = dataset.sampling.k_passes
    args.active_window_shards = dataset.sampling.window_shards
    args.locality_block_size = dataset.sampling.block_size
    args.max_demos = dataset.selection.max_demos
    args.current_frame_cache = 0
    args.decoder_handles = dataset.video.max_handles
    args.cameras = features.cameras
    args.measure_batches = 0
    return args


def main() -> None:
    """Run the benchmark and print a machine-readable result."""
    args = parse_args()
    result = run_benchmark(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered, flush=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
