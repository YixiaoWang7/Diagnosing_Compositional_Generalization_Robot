"""Composition and validation of component-owned settings for one training run."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from robopolicy.contracts.schema import FeatureSchema, StrictConfig
from robopolicy.datasets.config import DatasetConfig, LoaderConfig
from robopolicy.modules.queries import IndexedConditioning
from robopolicy.policies.config import PolicyConfig
from robopolicy.processing.packing import ProcessingConfig
from robopolicy.training.config import TrainConfig
from robopolicy.training.logging import LoggingConfig


class RunConfig(StrictConfig):
    """Resolved, versioned run settings saved as the authority for recovery."""

    version: Literal[1] = 1
    seed: int = 0
    features: FeatureSchema
    dataset: DatasetConfig
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    policy: PolicyConfig
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    training: TrainConfig = Field(default_factory=TrainConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        """Reject feature/architecture mismatches before loading data or weights."""
        self.processing.validate_schema(self.features)
        declarations = {feature.key: feature for feature in self.features.features}
        expected = {"state", "actions"}
        if self.policy.cameras:
            expected.add("images")
        if self.policy.entity_dim:
            expected.update(("entities", "entity_indices"))
        if isinstance(self.policy.conditioning, IndexedConditioning):
            expected.add("task_indices")
        if set(declarations) != expected:
            raise ValueError(
                f"Policy feature keys must be {sorted(expected)}, received {sorted(declarations)}"
            )
        for key, dimension, steps in (
            ("state", self.policy.state_dim, self.policy.obs_steps),
            ("actions", self.policy.action_dim, self.policy.action_steps),
        ):
            feature = declarations[key]
            if (
                feature.shape != (dimension,)
                or feature.window_size != steps
                or feature.dtype != "float32"
            ):
                raise ValueError(f"Feature {key} does not match policy shape, window, or dtype")
        if declarations["state"].future or declarations["actions"].history:
            raise ValueError("State must use history only and actions must use future only")
        if "images" in declarations:
            images = declarations["images"]
            if (
                images.source != "video"
                or len(images.cameras) != self.policy.cameras
                or images.window_size != self.policy.obs_steps
                or images.future
            ):
                raise ValueError("Image camera selection and temporal window must match the policy")
        if isinstance(self.policy.conditioning, IndexedConditioning):
            if declarations["task_indices"].shape != (len(self.policy.conditioning.vocab_sizes),):
                raise ValueError("task_indices width must equal the number of vocabulary slots")
        if "entities" in declarations:
            entities = declarations["entities"]
            if (
                len(entities.shape) != 2
                or entities.shape[-1] != self.policy.entity_dim
                or entities.window_size != self.policy.obs_steps
                or entities.future
            ):
                raise ValueError(
                    "Entity features must match configured (N_entity, D_entity) and history"
                )
            if declarations["entity_indices"].shape != (self.policy.selected_entities,):
                raise ValueError("entity_indices width must match selected_entities")
        return self


def load_config(path: str | Path) -> RunConfig:
    """Read a strict JSON run specification with component defaults resolved."""
    return RunConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))
