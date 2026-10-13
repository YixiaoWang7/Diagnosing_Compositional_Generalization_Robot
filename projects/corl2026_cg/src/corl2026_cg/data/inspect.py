"""Bounded visual QA for shared offline and online active-instruction semantics."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
from numpy.typing import NDArray

from corl2026_cg.data.active import read_observations, source_layout, validate_label_dataset
from corl2026_cg.instructions import (
    ActiveInstructionProfile,
    InstructionTracker,
    annotate_instruction,
    instruction_text,
    resolve_profile,
)
from corl2026_cg.tasks import parse_task


def _time_axis(group: h5py.Group, path: str, length: int, fps: float) -> NDArray[Any]:
    """Use recorded timestamps, or the declared uniform frame rate when absent."""
    if path.startswith("/"):
        node = group.file.get(path)
    else:
        node = group.get(path)
    if node is None:
        return np.arange(length, dtype=np.float64) / fps
    values = np.asarray(node, dtype=np.float64)
    if values.shape != (length,) or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError(f"Invalid instruction inspection time axis: {path}")
    return values


def _render_video(
    group: h5py.Group,
    entry: Any,
    labels: NDArray[Any],
    profile: ActiveInstructionProfile,
    destination: Path,
) -> dict[str, Any]:
    """Render the exact step-index alignment used by the native dataset reader.

    Recorded or fabricated clocks remain diagnostics: resampling them here would
    show observations different from those used for training.
    """
    import cv2
    import imageio.v2 as imageio
    from robopolicy.datasets.index import source_path

    cameras: list[dict[str, Any]] = []
    alignment: dict[str, Any] = {"mode": "step_index", "cameras": {}, "warnings": []}
    try:
        for name, video in sorted(entry.videos.items()):
            sidecar = json.loads(source_path(entry.task_root, video["sidecar"]).read_text())
            matches = [item for item in sidecar["segments"] if item["demo_key"] == entry.episode_id]
            if len(matches) != 1:
                raise ValueError(f"Expected one video segment for {entry.identity}")
            segment = matches[0]
            count, start = int(segment["frame_count"]), int(segment["frame_start"])
            fps = float(sidecar.get("fps", 20))
            if count < 1 or start < 0 or start + count > int(sidecar["frames"]) or fps <= 0:
                raise ValueError("Invalid video segment for instruction inspection")
            if count != len(labels):
                raise ValueError("Instruction inspection requires one video frame per numeric step")
            numeric_times = _time_axis(group, "timestamps", len(labels), fps)
            image_times = _time_axis(
                group,
                str(segment.get("timestamps_dataset", f"modalities/images/{name}/timestamps")),
                count,
                fps,
            )
            timestamp_error = float(np.max(np.abs(numeric_times - image_times)))
            alignment["cameras"][name] = {
                "frame_start": start,
                "frame_count": count,
                "playback_fps": fps,
                "numeric_timestamps_source": str(
                    group.attrs.get("timestamps_source", "unspecified")
                ),
                "video_timestamps_source": str(segment.get("timestamps_source", "unspecified")),
                "numeric_median_step_seconds": float(np.median(np.diff(numeric_times)))
                if count > 1
                else None,
                "video_median_step_seconds": float(np.median(np.diff(image_times)))
                if count > 1
                else None,
                "max_timestamp_difference_seconds": timestamp_error,
            }
            if timestamp_error > 1e-6:
                alignment["warnings"].append(
                    f"{name}: numeric/video clocks differ; showing the same step-index alignment as training."
                )
            capture = cv2.VideoCapture(str(source_path(entry.task_root, video["mp4"])))
            if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
                capture.release()
                raise ValueError(f"Cannot open aligned video for camera {name}")
            cameras.append(
                {
                    "capture": capture,
                    "fps": fps,
                }
            )
        if not cameras:
            return alignment
        if any(camera["fps"] != cameras[0]["fps"] for camera in cameras):
            raise ValueError("Instruction inspection cameras must share playback frame rate")
        with imageio.get_writer(
            destination, fps=cameras[0]["fps"], codec="libx264", macro_block_size=1
        ) as writer:
            for token in labels[:, 0]:
                panels = []
                for camera in cameras:
                    ok, bgr = camera["capture"].read()
                    if not ok:
                        raise ValueError(f"Truncated video in {entry.identity}")
                    panels.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if len({panel.shape[0] for panel in panels}) != 1:
                    raise ValueError("Instruction inspection cameras must share image height")
                cast(Any, writer).append_data(
                    annotate_instruction(np.concatenate(panels, axis=1), profile, int(token))
                )
        return alignment
    finally:
        for camera in cameras:
            camera["capture"].release()


def inspect_instructions(
    source: Path,
    output: Path,
    *,
    samples: int = 4,
    seed: int = 0,
    task: str | None = None,
) -> dict[str, Any]:
    """Export sampled instruction videos and phase transitions from raw/prepared data.

    Args:
        source: Small-files dataset root, never modified.
        output: New directory published only after all requested samples succeed.
        samples: Maximum sampled demonstrations, selected reproducibly.
        seed: Independent deterministic inspection sampling seed.
        task: Optional exact concrete task slug.

    Returns:
        JSON-compatible summary also saved as ``summary.json`` beside videos.
    """
    from robopolicy.datasets.index import read_manifest

    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Instruction inspection source and output must be disjoint directories")
    if samples < 1:
        raise ValueError("Instruction inspection samples must be positive")
    if output.exists():
        raise FileExistsError(f"Instruction inspection output exists: {output}")
    entries = [entry for entry in read_manifest(source) if task is None or entry.task_id == task]
    if not entries:
        raise ValueError(f"No matching instruction inspection demonstrations: {task}")
    chosen = np.random.default_rng(seed).choice(
        len(entries), min(samples, len(entries)), replace=False
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent))
    summary: dict[str, Any] = {"source": str(source), "seed": seed, "samples": []}
    try:
        for number, index in enumerate(chosen):
            entry = entries[int(index)]
            concrete = parse_task(entry.task_id)
            visibility = "infer_cont0" if concrete.family == "l4_2s" else "full"
            profile = resolve_profile(concrete.family, visibility, "active_v1")
            assert profile is not None
            with h5py.File(entry.path, "r") as handle:
                group = handle[f"data/{entry.episode_id}"]
                length = int(group["actions"].shape[0])
                layout = source_layout(group, length)
                labels = np.empty((length, 1), dtype=np.int64)
                tracker = InstructionTracker(profile, concrete)
                for start in range(0, length, 4096):
                    stop = min(start + 4096, length)
                    states, entities = read_observations(group, layout, start, stop)
                    labels[start:stop, 0] = [
                        tracker.update(state, poses)
                        for state, poses in zip(states, entities, strict=True)
                    ]
                if profile.path in group:
                    validate_label_dataset(group, profile, length, layout)
                    if not np.array_equal(group[profile.path][()], labels):
                        raise ValueError("Stored active labels disagree with the shared tracker")
                video_name = f"sample-{number:03d}.mp4" if entry.videos else None
                alignment = (
                    _render_video(group, entry, labels, profile, staging / video_name)
                    if video_name is not None
                    else None
                )
                transitions = np.flatnonzero(np.r_[True, np.diff(labels[:, 0]) != 0])
                summary["samples"].append(
                    {
                        "identity": list(entry.identity),
                        "length": length,
                        "profile": profile.to_dict(),
                        "profile_digest": profile.digest,
                        "video": video_name,
                        "alignment": alignment,
                        "phase_counts": np.bincount(labels[:, 0] // 4, minlength=3).tolist(),
                        "transitions": [
                            {
                                "step": int(step),
                                "token": int(labels[step, 0]),
                                "text": instruction_text(profile, int(labels[step, 0])),
                            }
                            for step in transitions
                        ],
                    }
                )
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"Instruction inspection output appeared: {output}")
        staging.rename(output)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)
