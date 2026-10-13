"""Immutable, application-owned contracts for reactive categorical instructions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ActiveInstructionProfile:
    """Versioned physical-state interpretation shared by labels and inference.

    Attributes:
        name: Stable identity included in the categorical HDF5 path.
        family: Concrete CG task family.
        instruction: Visible composition semantics, independently of active phase.
        grasp_enter_distance: Maximum end-effector/object distance to acquire grasp.
        grasp_exit_distance: Larger distance that retains an existing grasp.
        grasp_min_separation: Minimum finger separation for object contact.
        grasp_enter_separation: Maximum separation to acquire grasp.
        grasp_exit_separation: Larger separation that retains an existing grasp.
        debounce_observations: Consecutive proposals needed to change active phase.
        bin_half_width: Pinned simulator's axis-aligned bin membership half-width.
        flat_half_height: Vertical membership tolerance for bin and plate.
        mug_radius: Mean of pinned mug inner and outer radii.
        mug_half_height: Pinned mug vertical membership tolerance.
        plate_radius: Pinned plate radius minus its rim width.
    """

    name: str
    family: str
    instruction: str
    grasp_enter_distance: float = 0.04
    grasp_exit_distance: float = 0.05
    grasp_min_separation: float = 0.01
    grasp_enter_separation: float = 0.055
    grasp_exit_separation: float = 0.065
    debounce_observations: int = 2
    bin_half_width: float = 0.03
    flat_half_height: float = 0.06
    mug_radius: float = 0.0325
    mug_half_height: float = 0.04
    plate_radius: float = 0.027

    @property
    def path(self) -> str:
        """Return the checkpoint-visible semantic identity of these labels."""
        return f"cg/active_instruction/{self.name}"

    @property
    def digest(self) -> str:
        """Return a deterministic digest of the complete physical-state contract."""
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Describe all label semantics, including pose layout and phase encoding."""
        return {
            **asdict(self),
            "path": self.path,
            "vocabulary_size": 12,
            "encoding": "phase * 4 + factor",
            "coordinates": "world",
            "entity_order": "cross,cube,cylinder,milk,bin,mug,plate,mug_no_handle",
            "state_layout": "eef_xyz,eef_quat_xyzw,finger_qpos_0,finger_qpos_1",
            "release": "not_grasped",
            "initialization": "immediate",
        }

    def validate_conditioning(
        self, paths: Sequence[str], vocabulary_sizes: Sequence[int] | None = None
    ) -> None:
        """Reject a checkpoint with a different instruction representation."""
        if tuple(paths) != (self.path,):
            raise ValueError(f"Active instruction profile {self.name} requires {(self.path,)}")
        if vocabulary_sizes is not None and tuple(vocabulary_sizes) != (12,):
            raise ValueError(f"Active instruction profile {self.name} requires vocabulary [12]")


_PROFILES = {
    "l5_v1": ActiveInstructionProfile("l5_v1", "l5", "full"),
    "l4_infer_cont0_v1": ActiveInstructionProfile("l4_infer_cont0_v1", "l4_2s", "infer_cont0"),
}


def get_profile(name: str) -> ActiveInstructionProfile:
    """Resolve an exact immutable profile; unknown versions are never migrated."""
    try:
        return _PROFILES[name]
    except KeyError as error:
        raise ValueError(f"Unknown active instruction profile: {name!r}") from error


def resolve_profile(
    family: str, instruction: str, conditioning: str = "static"
) -> ActiveInstructionProfile | None:
    """Resolve active conditioning separately from composition visibility."""
    if conditioning == "static":
        return None
    if conditioning == "active_v1":
        for profile in _PROFILES.values():
            if (profile.family, profile.instruction) == (family, instruction):
                return profile
    raise ValueError(f"Unsupported conditioning {conditioning!r} for {family}/{instruction}")
