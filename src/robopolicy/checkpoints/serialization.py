"""Versioned atomic serialization using a weights-only-compatible payload."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA_VERSION = 1
REQUIRED_FIELDS = {
    "format",
    "schema_version",
    "policy_config",
    "model",
    "statistics",
    "feature_schema",
    "processing",
    "optimizer",
    "scheduler",
    "scaler",
    "completed_updates",
    "data_epoch",
    "rng",
    "dataset_fingerprint",
    "selection",
    "run_config",
    "wandb",
    "segment",
    "skipped_updates",
}


def capture_rng(device: str) -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and the active GPU's random state."""
    numpy_state = np.random.get_state(legacy=True)
    assert isinstance(numpy_state, tuple)
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(torch.device(device))
        if torch.device(device).type == "cuda"
        else None,
    }


def restore_rng(state: dict[str, Any], device: str) -> None:
    """Restore applicable RNG streams after constructing models and optimizers."""
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state[0],
            np.asarray(numpy_state[1], dtype=np.uint32),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )
    torch.set_rng_state(state["torch"].to("cpu"))
    if state["cuda"] is not None and torch.device(device).type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].to("cpu"), torch.device(device))


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically replace a checkpoint after flushing a same-directory temporary.

    A serialization or rename failure leaves the previous checkpoint intact and
    removes the incomplete temporary. Payloads contain tensors and primitives,
    allowing loading with ``weights_only=True``.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read and validate a checkpoint on CPU without data or model construction.

    Returns the original tensor/primitive payload. Strict parameter names and
    shapes are checked by ``load_policy``, not by this metadata reader.
    """
    from robopolicy.config.run import RunConfig
    from robopolicy.contracts.schema import FeatureSchema
    from robopolicy.policies.config import PolicyConfig
    from robopolicy.processing.normalization import Statistics
    from robopolicy.processing.packing import ProcessingConfig

    payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "robopolicy.training"
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("Incompatible RoboPolicy checkpoint schema; no legacy checkpoint support")
    missing = REQUIRED_FIELDS - payload.keys()
    if missing:
        raise ValueError(f"Incomplete checkpoint; missing fields: {sorted(missing)}")
    for key in ("completed_updates", "data_epoch", "skipped_updates"):
        if type(payload[key]) is not int or payload[key] < 0:
            raise ValueError(f"Checkpoint {key} must be a nonnegative integer")
    if not isinstance(payload["segment"], str) or not payload["segment"]:
        raise ValueError("Checkpoint segment must be a nonempty identifier")
    if (
        not isinstance(payload["model"], dict)
        or not payload["model"]
        or any(
            not isinstance(key, str) or not isinstance(value, torch.Tensor)
            for key, value in payload["model"].items()
        )
    ):
        raise ValueError("Checkpoint model must be a nonempty string-to-tensor state dict")
    for key in ("optimizer", "scheduler", "scaler", "rng"):
        if not isinstance(payload[key], dict):
            raise ValueError(f"Checkpoint {key} must be a dictionary")
    if payload["wandb"] is not None and not isinstance(payload["wandb"], dict):
        raise ValueError("Checkpoint wandb must be a dictionary or null")
    config = RunConfig.model_validate(payload["run_config"])
    for key, saved, expected in (
        ("policy_config", PolicyConfig.model_validate(payload["policy_config"]), config.policy),
        (
            "feature_schema",
            FeatureSchema.model_validate(payload["feature_schema"]),
            config.features,
        ),
        ("processing", ProcessingConfig.model_validate(payload["processing"]), config.processing),
    ):
        if saved != expected:
            raise ValueError(f"Checkpoint {key} conflicts with run_config")
    statistics = Statistics.model_validate(payload["statistics"])
    if statistics.feature_schema != config.features:
        raise ValueError("Checkpoint statistics feature schema differs from run_config")
    fingerprint = payload["dataset_fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("Checkpoint dataset_fingerprint must be a SHA-256 hex digest")
    selection = payload["selection"]
    if (
        not isinstance(selection, (list, tuple))
        or not selection
        or any(
            not isinstance(item, (list, tuple))
            or len(item) != 3
            or not isinstance(item[0], str)
            or not item[0]
            or type(item[1]) is not int
            or item[1] < 0
            or not isinstance(item[2], str)
            or not item[2]
            for item in selection
        )
    ):
        raise ValueError("Checkpoint selection must contain (task, shard, episode) identities")
    identities = tuple(tuple(item) for item in selection)
    if len(set(identities)) != len(identities):
        raise ValueError("Checkpoint selection contains duplicate identities")
    if statistics.dataset_fingerprint != fingerprint or statistics.selection != identities:
        raise ValueError("Checkpoint statistics fingerprint or selection differs from checkpoint")
    return payload
