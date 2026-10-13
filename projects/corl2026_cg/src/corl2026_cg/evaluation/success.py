"""Instruction visibility determines task success, independently of the policy."""

from dataclasses import dataclass
from typing import Protocol

from corl2026_cg.tasks import CGTask


class SuccessProbe(Protocol):
    """Minimal simulator operations used by the task state machine."""

    def placement_success(self, index: int, object_id: int, container_id: int) -> bool: ...

    def native_success(self, index: int) -> bool: ...

    def place_stage_success(self, index: int) -> bool: ...

    def set_target(self, index: int, object_id: int, container_id: int) -> None: ...


@dataclass
class EpisodeProgress:
    """Latched stage outcomes and concrete choices made under partial instructions."""

    stage0_done: bool = False
    stage1_done: bool = False
    success: bool = False
    same_container_violation: bool = False
    stage0_container: int | None = None
    stage1_container: int | None = None
    chosen_object: int | None = None

    def update(self, env: SuccessProbe, index: int, task: CGTask, instruction: str) -> bool:
        """Update success after one action; return whether the target changed."""
        if self.success:
            return False
        if task.family == "l5":
            self.stage0_done |= env.place_stage_success(index)
            self.success = env.native_success(index)
            self.stage1_done = self.success
            return False
        if task.family == "l4":
            objects = (
                range(4)
                if instruction in {"container_only", "container_latents"}
                else [task.object0]
            )
            containers = range(4) if instruction == "object_only" else [task.container0]
            for object_id in objects:
                for container_id in containers:
                    if env.placement_success(index, object_id, container_id):
                        self.stage0_done = self.success = True
                        self.stage0_container, self.chosen_object = container_id, object_id
                        return False
            return False
        assert task.object1 is not None and task.container1 is not None
        if not self.stage0_done:
            if instruction == "infer_cont0":
                if env.placement_success(index, task.object0, task.container1):
                    self.same_container_violation = True
                containers = [value for value in range(4) if value != task.container1]
            else:
                containers = [task.container0]
            for container_id in containers:
                if env.placement_success(index, task.object0, container_id):
                    self.stage0_done = True
                    self.stage0_container = container_id
                    next_container = task.container1
                    if instruction == "open_cont":
                        next_container = next(
                            value for value in range(4) if value != task.container0
                        )
                    env.set_target(index, task.object1, next_container)
                    return True
        else:
            containers = (
                [value for value in range(4) if value != task.container0]
                if instruction == "open_cont"
                else [task.container1]
            )
            for container_id in containers:
                if env.placement_success(index, task.object1, container_id):
                    self.stage1_done = True
                    self.stage1_container = container_id
                    self.success = not self.same_container_violation
                    break
        return False
