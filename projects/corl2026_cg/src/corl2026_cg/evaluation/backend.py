"""Direct integration with a pinned external CG simulator checkout.

Simulator imports happen only when constructing a backend. No simulator test
scripts, copied policy packages, or legacy policy loaders are imported.
"""

from __future__ import annotations

import importlib
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from corl2026_cg.tasks import BUTTONS, CONTAINERS, OBJECTS, CGTask

from .config import EvaluationConfig


def verify_simulator(root: Path, revision: str) -> dict[str, Any]:
    """Validate source revision before importing the external simulator."""
    root = root.expanduser().resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", revision], text=True
    ).strip()
    if actual != expected:
        raise ValueError(f"Simulator revision {actual} differs from pinned {expected}")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if dirty:
        raise ValueError(
            "Simulator has tracked modifications; use a clean checkout of the pinned revision"
        )
    package_root = root / "CG" / "robosuite"
    if not (package_root / "robosuite").is_dir():
        raise FileNotFoundError(f"Missing simulator package under {package_root}")
    loaded = sys.modules.get("robosuite")
    loaded_path = getattr(loaded, "__file__", None)
    if loaded is not None and (
        loaded_path is None or not Path(loaded_path).resolve().is_relative_to(package_root)
    ):
        raise RuntimeError(
            "Another robosuite package is already imported; start a fresh evaluator process"
        )
    sys.path.insert(0, str(package_root))
    return {"root": str(root), "revision": actual}


def task_text(task: CGTask) -> str:
    """Build the simulator instruction used to initialize a concrete task."""
    value = f"place the {OBJECTS[task.object0]} into the {CONTAINERS[task.container0]}"
    if task.family == "l5":
        assert task.button is not None
        value += f" then press the {BUTTONS[task.button]} button"
    return value


def relative_pose(
    pose: np.ndarray, eef_position: np.ndarray, eef_quaternion: np.ndarray
) -> np.ndarray:
    """Match collection coordinates: position delta and canonical relative XYZW."""
    x1, y1, z1, w1 = eef_quaternion * np.array([-1, -1, -1, 1], dtype=np.float32)
    x2, y2, z2, w2 = pose[3:]
    quaternion = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )
    if quaternion[3] < 0:
        quaternion *= -1
    return np.concatenate([pose[:3] - eef_position, quaternion]).astype(np.float32)


class SimulatorBackend:
    """A fixed number of simulator instances sharing batched policy inference."""

    def __init__(
        self,
        family: str,
        config: EvaluationConfig,
        cameras: tuple[str, ...],
        image_size: tuple[int, int],
    ) -> None:
        self.config, self.family, self.cameras = config, family, cameras
        self.provenance = verify_simulator(config.simulator_root, config.simulator_revision)
        controllers = importlib.import_module("robosuite.controllers")
        samplers = importlib.import_module("robosuite.utils.placement_samplers")
        module = "CG_L5" if family == "l5" else "CG_L4"
        environment = getattr(
            importlib.import_module(f"robosuite.environments.manipulation.{module}"), module
        )
        self.randomization_error = importlib.import_module(
            "robosuite.utils.errors"
        ).RandomizationError
        self.envs: list[Any] = []
        self.observations: list[dict[str, np.ndarray]] = []
        try:
            for _ in range(config.num_envs):
                common = dict(
                    rotation=None,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=(0.0, 0.0, 0.8),
                    z_offset=0.01,
                )
                sampler_a = samplers.UniformApartRandomSampler(
                    name="CG_Object_A",
                    x_range=[-0.12 if family == "l5" else -0.18, 0.20],
                    y_range=[-0.20, -0.05],
                    min_distance=0.02,
                    **common,
                )
                sampler_b = samplers.UniformApartRandomSampler(
                    name="CG_Object_B",
                    x_range=[-0.18, 0.20],
                    y_range=[0.05, 0.20],
                    min_distance=0.01 if family == "l5" else 0.005,
                    **common,
                )
                bootstrap = "place the cross into the bin"
                if family == "l5":
                    bootstrap += " then press the red button"
                self.envs.append(
                    environment(
                        robots="Panda",
                        gripper_types="PandaGripper",
                        controller_configs=controllers.load_composite_controller_config(
                            controller="BASIC", robot="Panda"
                        ),
                        task=bootstrap,
                        horizon=config.horizon,
                        hard_reset=False,
                        placement_initializer_A=sampler_a,
                        placement_initializer_B=sampler_b,
                        use_camera_obs=bool(cameras),
                        has_renderer=False,
                        has_offscreen_renderer=bool(cameras),
                        camera_names=list(cameras),
                        camera_heights=image_size[0],
                        camera_widths=image_size[1],
                    )
                )
        except BaseException:
            self.close()
            raise

    def reset(self, tasks: list[CGTask], seeds: list[int]) -> list[dict[str, np.ndarray]]:
        """Reset all instances with recorded seeds and bounded placement retries."""
        self.observations = []
        self.reset_seeds: list[int] = []
        for env, task, seed in zip(self.envs, tasks, seeds, strict=True):
            env.update_task(task_text(task))
            if hasattr(env, "_refresh_task_pointers"):
                env._refresh_task_pointers()
            numpy_state, python_state = np.random.get_state(), random.getstate()
            try:
                for attempt in range(self.config.reset_attempts):
                    effective_seed = int((seed + attempt) % (2**32))
                    np.random.seed(effective_seed)
                    random.seed(effective_seed)
                    try:
                        self.observations.append(env.reset())
                        self.reset_seeds.append(effective_seed)
                        break
                    except self.randomization_error:
                        if attempt + 1 == self.config.reset_attempts:
                            raise
            finally:
                np.random.set_state(numpy_state)
                random.setstate(python_state)
        return self.observe()

    def observe(self) -> list[dict[str, np.ndarray]]:
        """Return canonical state, entity poses and RGB camera frames for every env."""
        result = []
        names = OBJECTS + CONTAINERS
        for env, raw in zip(self.envs, self.observations, strict=True):
            eef, quat = raw["robot0_eef_pos"], raw["robot0_eef_quat"]
            poses = [np.concatenate([raw[f"{name}_pos"], raw[f"{name}_quat"]]) for name in names]
            if self.family == "l5":
                for body_id in env.button_body_ids:
                    position = env.sim.data.body_xpos[body_id]
                    wxyz = env.sim.data.body_xquat[body_id]
                    poses.append(np.concatenate([position, wxyz[[1, 2, 3, 0]]]))
            # Trackers always operate in world coordinates, independently of the
            # policy's entity representation. Schema packing excludes this field.
            world_entities = np.asarray(poses, dtype=np.float32)
            if self.config.entity_coordinates == "relative":
                poses = [relative_pose(pose, eef, quat) for pose in poses]
            observation = {
                "state": np.concatenate([eef, quat, raw["robot0_gripper_qpos"]]).astype(np.float32),
                "entities": np.asarray(poses, dtype=np.float32),
                "world_entities": world_entities,
            }
            if self.cameras:
                # Robosuite images have a bottom-up origin; channel order is already RGB.
                observation["images"] = np.stack(
                    [
                        np.flipud(raw[f"{camera}_image"]).transpose(2, 0, 1)
                        for camera in self.cameras
                    ]
                ).astype(np.uint8)
            result.append(observation)
        return result

    def step(
        self, actions: np.ndarray, active: list[bool]
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray, list[bool]]:
        """Step active episodes only; finished instances retain their last observation."""
        rewards: NDArray[np.float32] = np.zeros(len(self.envs), dtype=np.float32)
        done = [False] * len(self.envs)
        for index, env in enumerate(self.envs):
            if active[index]:
                observation, reward, terminal, _ = env.step(actions[index].copy())
                self.observations[index] = observation
                rewards[index], done[index] = float(reward), bool(terminal)
        return self.observe(), rewards, done

    def set_target(self, index: int, object_id: int, container_id: int) -> None:
        """Retarget L4 without reset; no wrapper success latch is involved."""
        env = self.envs[index]
        env.update_task(f"place the {OBJECTS[object_id]} into the {CONTAINERS[container_id]}")
        if hasattr(env, "_refresh_task_pointers"):
            env._refresh_task_pointers()

    def placement_success(self, index: int, object_id: int, container_id: int) -> bool:
        """Probe a placement target and restore the original simulator task."""
        env = self.envs[index]
        original = env.task
        try:
            self.set_target(index, object_id, container_id)
            return bool(env._check_success())
        finally:
            env.update_task(original)
            if hasattr(env, "_refresh_task_pointers"):
                env._refresh_task_pointers()

    def native_success(self, index: int) -> bool:
        """L5 includes latched placement followed by the requested physical button."""
        return bool(self.envs[index]._check_success())

    def place_stage_success(self, index: int) -> bool:
        """Expose the simulator's L5 placement-stage predicate."""
        env = self.envs[index]
        return bool(env.has_completed_place or env._place_stage_success())

    def close(self) -> None:
        """Release every simulator's offscreen rendering resources."""
        for env in self.envs:
            env.close()
        self.envs.clear()
