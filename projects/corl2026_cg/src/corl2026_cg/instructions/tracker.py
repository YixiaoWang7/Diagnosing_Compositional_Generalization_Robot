"""Pure observation-driven phase tracking, with no simulator or policy dependency."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from corl2026_cg.instructions.profiles import ActiveInstructionProfile
from corl2026_cg.tasks import BUTTONS, CONTAINERS, OBJECTS, CGTask


def container_membership(
    profile: ActiveInstructionProfile,
    object_position: NDArray[Any],
    container_position: NDArray[Any],
    container_id: int,
) -> bool:
    """Match pinned simulator geometry for world-space object/container XYZ vectors.

    The simulator checks body centers in world axes, including for rotated bins.
    This is geometric membership, independently of grasp and episode success.
    """
    relative = object_position - container_position
    if container_id == 0:
        return bool(
            abs(relative[0]) < profile.bin_half_width
            and abs(relative[1]) < profile.bin_half_width
            and abs(relative[2]) < profile.flat_half_height
        )
    radius = profile.plate_radius if container_id == 2 else profile.mug_radius
    height = profile.flat_half_height if container_id == 2 else profile.mug_half_height
    return bool(np.linalg.norm(relative[:2]) < radius and abs(relative[2]) < height)


class InstructionTracker:
    """Track debounced reactive phases for one episode, including physical slips.

    Attributes:
        profile: Frozen label semantics shared with offline preparation.
        task: Full task identity; hidden first-container IDs are never consulted.
        phase: Current phase or None before the first observation.
    """

    def __init__(self, profile: ActiveInstructionProfile, task: CGTask) -> None:
        """Create independent grasp and debounce state for a new episode."""
        if profile.family != task.family:
            raise ValueError(f"Profile {profile.name} does not support task {task.slug}")
        self.profile, self.task = profile, task
        self.phase: int | None = None
        self._grasped: dict[int, bool] = {}
        self._proposal: int | None = None
        self._proposal_count = 0

    def _grasp(self, object_id: int, state: NDArray[Any], entities: NDArray[Any]) -> bool:
        """Update hysteretic contact inferred from XYZ distance and finger opening."""
        retained = self._grasped.get(object_id, False)
        profile = self.profile
        distance = profile.grasp_exit_distance if retained else profile.grasp_enter_distance
        opening = profile.grasp_exit_separation if retained else profile.grasp_enter_separation
        separation = abs(float(state[7] - state[8]))
        grasped = bool(
            profile.grasp_min_separation <= separation <= opening
            and np.linalg.norm(state[:3] - entities[object_id, :3]) <= distance
        )
        self._grasped[object_id] = grasped
        return grasped

    def _placed(self, entities: NDArray[Any], object_id: int, container_id: int) -> bool:
        """Check released placement without latching previous successful poses."""
        return not self._grasped.get(object_id, False) and container_membership(
            self.profile, entities[object_id, :3], entities[4 + container_id, :3], container_id
        )

    def update(self, state: NDArray[Any], world_entities: NDArray[Any]) -> int:
        """Return one current categorical ID after observing physical state.

        Args:
            state: Shape (9,), end-effector XYZ, XYZW, and two finger positions.
            world_entities: Shape (N, 7), world XYZ/XYZW in object/container order.

        Returns:
            Integer in [0, 12), encoding phase * 4 + the phase's requested factor.
        """
        if (
            state.shape != (9,)
            or world_entities.ndim != 2
            or world_entities.shape[0] < 8
            or world_entities.shape[1] != 7
            or not np.isfinite(state).all()
            or not np.isfinite(world_entities).all()
        ):
            raise ValueError("Active instructions require finite state (9,) and world poses (N,7)")
        task = self.task
        grasp0 = self._grasp(task.object0, state, world_entities)
        if task.family == "l5":
            assert task.button is not None
            proposed = (
                2 if self._placed(world_entities, task.object0, task.container0) else int(grasp0)
            )
            factors = (task.object0, task.container0, task.button)
        else:
            assert task.object1 is not None and task.container1 is not None
            grasp1 = self._grasp(task.object1, state, world_entities)
            # The hidden concrete container0 is deliberately absent from this computation.
            first_placed = any(
                self._placed(world_entities, task.object0, container)
                for container in range(4)
                if container != task.container1
            )
            second_placed = self._placed(world_entities, task.object1, task.container1)
            proposed = (2 if grasp1 or second_placed else 1) if first_placed else 0
            factors = (task.object0, task.object1, task.container1)
        if self.phase is None:
            self.phase = proposed
        elif proposed == self.phase:
            self._proposal, self._proposal_count = None, 0
        else:
            if proposed != self._proposal:
                self._proposal, self._proposal_count = proposed, 0
            self._proposal_count += 1
            if self._proposal_count >= self.profile.debounce_observations:
                self.phase = proposed
                self._proposal, self._proposal_count = None, 0
        return self.phase * 4 + factors[self.phase]


def label_trajectory(
    profile: ActiveInstructionProfile,
    task: CGTask,
    states: NDArray[Any],
    world_entities: NDArray[Any],
) -> NDArray[np.int64]:
    """Apply the online tracker sequentially to (T,9) and (T,N,7) observations.

    Returns:
        Integer labels with shape (T,1); no normalization or success-label input.
    """
    if states.ndim != 2 or world_entities.ndim != 3 or len(states) != len(world_entities):
        raise ValueError("Trajectory state and world-entity time axes must align")
    tracker = InstructionTracker(profile, task)
    return np.asarray(
        [
            tracker.update(state, entities)
            for state, entities in zip(states, world_entities, strict=True)
        ],
        dtype=np.int64,
    ).reshape(-1, 1)


def instruction_text(profile: ActiveInstructionProfile, active_id: int) -> str:
    """Resolve the shared human-readable overlay for one valid active token."""
    if not 0 <= active_id < 12:
        raise ValueError(f"Invalid active instruction ID: {active_id}")
    phase, factor = divmod(active_id, 4)
    if profile.family == "l5":
        return (
            f"Pick {OBJECTS[factor]}",
            f"Place in {CONTAINERS[factor]}",
            f"Press {BUTTONS[factor]}",
        )[phase]
    return (
        f"Place first {OBJECTS[factor]} in an allowed container",
        f"Pick second {OBJECTS[factor]}",
        f"Place second in {CONTAINERS[factor]}",
    )[phase]
