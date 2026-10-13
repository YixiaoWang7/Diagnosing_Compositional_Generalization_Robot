"""Shared lightweight instruction overlay for demonstration and rollout videos."""

from typing import Any

from numpy.typing import NDArray

from corl2026_cg.instructions.profiles import ActiveInstructionProfile
from corl2026_cg.instructions.tracker import instruction_text


def annotate_instruction(
    pixels: NDArray[Any], profile: ActiveInstructionProfile, active_id: int
) -> NDArray[Any]:
    """Copy RGB uint8 (H,W,3) pixels and add the current categorical instruction."""
    import cv2

    result = pixels.copy()
    label = f"{active_id}: {instruction_text(profile, active_id)}"
    scale = min(0.5, max(0.25, pixels.shape[1] / (len(label) * 13)))
    cv2.rectangle(result, (0, 0), (pixels.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(result, label, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)
    return result
