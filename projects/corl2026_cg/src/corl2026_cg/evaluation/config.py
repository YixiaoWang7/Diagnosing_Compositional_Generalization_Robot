"""Simulator evaluation settings owned by the CG application."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from robopolicy import InferenceConfig
from robopolicy.training.logging import LoggingConfig, WandbConfig

SIMULATOR_REVISION = "c7e06a0784853d74db57ebf78d7a66fa40c2df99"


class EvaluationConfig(BaseModel):
    """Configure rollout budgets, the external simulator, and result tracking."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    simulator_root: Path = Path("repos/robot_envs")
    simulator_revision: str = SIMULATOR_REVISION
    n_trials_per_task: int = Field(default=25, ge=1)
    num_envs: int = Field(default=8, ge=1)
    horizon: int = Field(default=400, ge=1)
    n_execute: int = Field(default=8, ge=1)
    seed: int = Field(default=0, ge=0)
    device: str = "cuda:0"
    max_videos: int = Field(default=0, ge=0)
    max_attention_videos: int = Field(default=0, ge=0)
    fps: int = Field(default=20, ge=1)
    entity_coordinates: Literal["world", "relative"] = "relative"
    reset_attempts: int = Field(default=5, ge=1)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    logging: LoggingConfig = Field(
        default_factory=lambda: LoggingConfig(wandb=WandbConfig(project="corl2026_cg"))
    )
