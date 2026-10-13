"""Task-agnostic flow policies with optional integrations loaded only on demand."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robopolicy.checkpoints.inference import InferenceConfig, InferencePolicy, load_policy
    from robopolicy.checkpoints.serialization import read_checkpoint
    from robopolicy.config.run import RunConfig, load_config
    from robopolicy.contracts.batch import PolicyInput
    from robopolicy.contracts.schema import Feature, FeatureSchema
    from robopolicy.datasets.config import DatasetConfig, LoaderConfig
    from robopolicy.datasets.factory import build_dataset, build_loader
    from robopolicy.datasets.identity import DatasetIdentity, dataset_identity
    from robopolicy.datasets.statistics import fit_statistics
    from robopolicy.policies.config import PolicyConfig
    from robopolicy.policies.flow import build_policy
    from robopolicy.processing.packing import InputPacker, ProcessingConfig
    from robopolicy.training.trainer import Trainer

__version__ = "0.2.0a1"

_EXPORTS = {
    "PolicyInput": "contracts.batch",
    "Feature": "contracts.schema",
    "FeatureSchema": "contracts.schema",
    "DatasetConfig": "datasets.config",
    "LoaderConfig": "datasets.config",
    "build_dataset": "datasets.factory",
    "build_loader": "datasets.factory",
    "DatasetIdentity": "datasets.identity",
    "dataset_identity": "datasets.identity",
    "read_checkpoint": "checkpoints.serialization",
    "fit_statistics": "datasets.statistics",
    "PolicyConfig": "policies.config",
    "build_policy": "policies.flow",
    "InputPacker": "processing.packing",
    "ProcessingConfig": "processing.packing",
    "RunConfig": "config.run",
    "load_config": "config.run",
    "Trainer": "training.trainer",
    "load_policy": "checkpoints.inference",
    "InferencePolicy": "checkpoints.inference",
    "InferenceConfig": "checkpoints.inference",
}
__all__ = [
    "PolicyInput",
    "Feature",
    "FeatureSchema",
    "DatasetConfig",
    "LoaderConfig",
    "build_dataset",
    "build_loader",
    "DatasetIdentity",
    "dataset_identity",
    "read_checkpoint",
    "fit_statistics",
    "PolicyConfig",
    "build_policy",
    "InputPacker",
    "ProcessingConfig",
    "RunConfig",
    "load_config",
    "Trainer",
    "load_policy",
    "InferencePolicy",
    "InferenceConfig",
]


def __getattr__(name: str) -> Any:
    """Load a public interface without importing optional backends at package import."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'robopolicy' has no attribute {name!r}")
    value = getattr(import_module(f"robopolicy.{module}"), name)
    globals()[name] = value
    return value
