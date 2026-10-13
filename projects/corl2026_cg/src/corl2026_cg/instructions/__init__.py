"""CG-only reactive instruction contracts, tracking, and visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from robopolicy import RunConfig

from corl2026_cg.instructions.media import annotate_instruction
from corl2026_cg.instructions.profiles import (
    ActiveInstructionProfile,
    get_profile,
    resolve_profile,
)
from corl2026_cg.instructions.tracker import (
    InstructionTracker,
    container_membership,
    instruction_text,
    label_trajectory,
)

__all__ = [
    "ActiveInstructionProfile",
    "InstructionTracker",
    "annotate_instruction",
    "container_membership",
    "get_profile",
    "instruction_text",
    "label_trajectory",
    "resolve_profile",
    "validate_dataset_profile",
]


def validate_dataset_profile(
    config: RunConfig,
    profile: ActiveInstructionProfile,
    split: Literal["train", "val", "all"] = "train",
) -> None:
    """Validate selected HDF5 label contracts without importing storage on pure use."""
    from corl2026_cg.data.active import validate_dataset_profile as validate

    validate(config, profile, split)
