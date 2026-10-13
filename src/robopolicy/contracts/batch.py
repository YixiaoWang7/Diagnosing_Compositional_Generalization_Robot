"""The single observation contract used by training, validation, and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from torch import Tensor


class SampleMetadata(TypedDict):
    """Opaque episode identity retained for auditing sample coverage."""

    task_id: str
    task_name: str
    shard_id: int
    episode_id: str
    step: int


@dataclass(frozen=True)
class PolicyInput:
    """Canonical policy tensors without application-specific interpretation.

    Attributes:
        state: Floating state ``(B, T_obs, D_state)``.
        images: Optional RGB images ``(B, T_obs, N_camera, 3, H, W)``.
        entities: Optional entity features ``(B, T_obs, N_entity, D_entity)``.
        entity_indices: Optional int64 selection indices ``(B, N_selected)``.
        task_indices: Optional int64 categorical IDs ``(B, K)``.
        actions: Optional targets ``(B, T_action, D_action)``.
    """

    state: Tensor
    images: Tensor | None = None
    entities: Tensor | None = None
    entity_indices: Tensor | None = None
    task_indices: Tensor | None = None
    actions: Tensor | None = None

    def to(self, device: str) -> PolicyInput:
        """Copy tensor references to a supported device without mutating inputs.

        Args:
            device: Torch device specifier, such as ``cpu`` or ``cuda:0``.

        Returns:
            A new input with identical tensor shapes and dtypes.
        """

        def move(value: Tensor | None) -> Tensor | None:
            """Move an optional tensor without changing its dtype or shape."""
            return None if value is None else value.to(device, non_blocking=True)

        return PolicyInput(
            state=self.state.to(device, non_blocking=True),
            images=move(self.images),
            entities=move(self.entities),
            entity_indices=move(self.entity_indices),
            task_indices=move(self.task_indices),
            actions=move(self.actions),
        )
