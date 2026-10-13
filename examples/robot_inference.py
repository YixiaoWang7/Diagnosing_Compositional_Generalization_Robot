"""Connect public RoboPolicy inference to application-owned observation/action callbacks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from torch import Tensor

from robopolicy import InferenceConfig, load_policy


def run_control_loop(
    checkpoint: Path,
    read_observation: Callable[[], Mapping[str, Tensor]],
    send_actions: Callable[[Tensor], None],
    keep_running: Callable[[], bool],
    *,
    device: str = "cuda:0",
    inference_config: InferenceConfig | None = None,
) -> None:
    """Warm a policy before repeatedly predicting from real robot observations.

    Args:
        checkpoint: Saved policy and preprocessing state.
        read_observation: Application callback returning raw tensors with B=1.
        send_actions: Callback receiving CPU actions ``(1, T_action, D_action)``.
            The application selects the control horizon and maps actions to its robot.
        keep_running: Application-owned stop condition.
        device: Inference GPU, or an explicitly selected CPU.
        inference_config: Compilation and precision settings; GPU compilation defaults on.
    """
    inference = load_policy(checkpoint, device=device, inference_config=inference_config)
    inference.warmup(read_observation())
    while keep_running():
        observation = read_observation()
        actions = inference.predict(observation)
        send_actions(actions.to("cpu"))
