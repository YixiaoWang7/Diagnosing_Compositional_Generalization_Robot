"""Physical changes, instruction visibility, and versioned reactive contracts."""

from __future__ import annotations

import numpy as np
import pytest

from corl2026_cg.instructions import (
    InstructionTracker,
    container_membership,
    get_profile,
    instruction_text,
    label_trajectory,
    resolve_profile,
)
from corl2026_cg.tasks import CGTask


def scene() -> tuple[np.ndarray, np.ndarray]:
    """Return a distant end-effector and eight separated world-space entities."""
    state = np.array([0, 0, 1.5, 0, 0, 0, 1, 0.04, -0.04], dtype=np.float32)
    entities = np.zeros((8, 7), dtype=np.float32)
    entities[:, 0] = np.arange(8) * 0.25
    entities[:, 2], entities[:, 6] = 0.8, 1
    return state, entities


def test_l5_grasp_hysteresis_debounce_release_and_slip() -> None:
    profile = get_profile("l5_v1")
    tracker = InstructionTracker(profile, CGTask("l5", 0, 1, button=2))
    state, entities = scene()
    assert tracker.update(state, entities) == 0
    state[:3], state[7:] = entities[0, :3], [0.02, -0.02]
    assert tracker.update(state, entities) == 0
    assert tracker.update(state, entities) == 5
    state[0] += 0.045
    assert tracker.update(state, entities) == 5  # Retain grasp beyond entry threshold.
    state[0] += 0.03
    assert tracker.update(state, entities) == 5
    assert tracker.update(state, entities) == 0
    entities[0, :3] = entities[5, :3]
    state[:3] = entities[0, :3]
    assert tracker.update(state, entities) == 0
    assert tracker.update(state, entities) == 5  # Held in-container is not released.
    state[7:] = [0.04, -0.04]
    assert tracker.update(state, entities) == 5
    assert tracker.update(state, entities) == 10
    entities[0, 0] += 0.15
    assert tracker.update(state, entities) == 10
    assert tracker.update(state, entities) == 0  # Placement loss reverses press phase.


def test_initial_phase_immediate_and_single_observation_flicker_ignored() -> None:
    state, entities = scene()
    entities[0, :3] = entities[4, :3]
    tracker = InstructionTracker(get_profile("l5_v1"), CGTask("l5", 0, 0, button=0))
    assert tracker.update(state, entities) == 8
    entities[0, 0] += 0.2
    assert tracker.update(state, entities) == 8
    entities[0, 0] -= 0.2
    assert tracker.update(state, entities) == 8
    entities[0, 0] += 0.2
    assert tracker.update(state, entities) == 8
    assert tracker.update(state, entities) == 0


def test_inferred_container_hidden_from_tracker_and_first_placement_can_regress() -> None:
    profile = get_profile("l4_infer_cont0_v1")
    tasks = [CGTask("l4_2s", 0, hidden, 1, 3) for hidden in (0, 1, 2)]
    trackers = [InstructionTracker(profile, task) for task in tasks]
    state, entities = scene()

    def observe(expected: int, observations: int = 1) -> None:
        """Require every hidden destination to produce identical visible labels."""
        for _ in range(observations):
            tokens = [tracker.update(state, entities) for tracker in trackers]
        assert tokens == [expected] * 3

    entities[0, :3] = entities[7, :3]
    observe(0)  # First object in the forbidden second destination does not progress.
    entities[0, :3] = entities[5, :3]
    observe(5, 2)
    state[:3], state[7:] = entities[1, :3], [0.02, -0.02]
    observe(11, 2)
    state[:3] += [0, 0, 0.3]
    observe(5, 2)
    entities[1, :3] = entities[7, :3]
    observe(11, 2)  # Released second placement stays in final phase.
    entities[0, 0] += 0.2
    observe(0, 2)


@pytest.mark.parametrize(
    "container,radius,height",
    [(0, 0.03, 0.06), (1, 0.0325, 0.04), (2, 0.027, 0.06), (3, 0.0325, 0.04)],
)
def test_container_bounds_match_simulator(container: int, radius: float, height: float) -> None:
    profile = get_profile("l5_v1")
    center = np.zeros(3)
    assert container_membership(profile, np.array([radius - 1e-5, 0, 0]), center, container)
    assert not container_membership(profile, np.array([radius + 1e-5, 0, 0]), center, container)
    assert not container_membership(profile, np.array([0, 0, height + 1e-5]), center, container)


def test_offline_online_parity_and_invalid_input() -> None:
    profile = get_profile("l5_v1")
    task = CGTask("l5", 1, 2, button=3)
    state, entities = scene()
    states, poses = np.repeat(state[None], 5, axis=0), np.repeat(entities[None], 5, axis=0)
    states[1:3, :3], states[1:3, 7:] = entities[1, :3], [0.02, -0.02]
    poses[3:, 1] = entities[6]
    labels = label_trajectory(profile, task, states, poses)
    tracker = InstructionTracker(profile, task)
    assert labels.dtype == np.dtype("int64") and labels.shape == (5, 1)
    assert labels[:, 0].tolist() == [
        tracker.update(s, p) for s, p in zip(states, poses, strict=True)
    ]
    with pytest.raises(ValueError, match="finite"):
        tracker.update(state[:8], entities)
    with pytest.raises(ValueError, match="does not support"):
        InstructionTracker(profile, CGTask("l4", 0, 1))


def test_profile_selection_and_contract_guards() -> None:
    assert resolve_profile("l4", "full") is None
    profile = resolve_profile("l5", "full", "active_v1")
    assert profile == get_profile("l5_v1") and profile is not None
    profile.validate_conditioning([profile.path], [12])
    for paths, vocab in [(["cg/task/object0"], [12]), ([profile.path], [4])]:
        with pytest.raises(ValueError):
            profile.validate_conditioning(paths, vocab)
    with pytest.raises(ValueError):
        resolve_profile("l4_2s", "full", "active_v1")
    with pytest.raises(ValueError):
        get_profile("l5_v2")
    assert instruction_text(profile, 9) == "Press green"
    with pytest.raises(ValueError):
        instruction_text(profile, 12)


def test_v1_physical_contract_is_frozen() -> None:
    """Any changed label semantics require a new checkpoint-visible profile version."""
    assert (
        get_profile("l5_v1").digest
        == "d3f31785b2425a30b9c611c2caaba9255762e93f79135ab4d26c5f6443c2621e"
    )
    assert (
        get_profile("l4_infer_cont0_v1").digest
        == "c428f79a6952919958f05cbce6b2f9e12b6f2add67d6256f3d623cddc6cc015e"
    )
