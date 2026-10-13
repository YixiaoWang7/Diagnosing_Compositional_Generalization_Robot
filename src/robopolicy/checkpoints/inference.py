"""Dataset-free inference reconstruction and an optional compiled prediction runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast

import torch
from pydantic import Field
from torch import Tensor, nn

from robopolicy.checkpoints.serialization import capture_rng, read_checkpoint, restore_rng
from robopolicy.contracts.schema import FeatureSchema, StrictConfig
from robopolicy.policies.config import PolicyConfig
from robopolicy.policies.flow import FlowPolicy, build_policy
from robopolicy.processing.normalization import Statistics
from robopolicy.processing.packing import InputPacker, ProcessingConfig


class InferenceConfig(StrictConfig):
    """Runtime settings, separate from saved architecture and training configuration.

    Attributes:
        version: Supported inference configuration schema.
        compile: Auto compiles on Torch 2.9 CUDA; older runtimes use eager execution.
        mode: Inductor kernel tuning and CUDA graph configuration.
        precision: Neural computation precision; solver state remains FP32.
        num_inference_steps: Optional solver step override; None keeps saved settings.
    """

    version: Literal[1] = 1
    compile: Literal["auto", "enabled", "disabled"] = "auto"
    mode: Literal["default", "reduce-overhead", "max-autotune"] = "max-autotune"
    precision: Literal["fp32", "bf16"] = "fp32"
    num_inference_steps: int | None = Field(default=None, ge=1, strict=True)


@contextmanager
def inference_precision(config: InferenceConfig) -> Iterator[None]:
    """Enforce FP32 arithmetic temporarily, restoring caller backend settings.

    cuDNN permits TF32 convolution by default even for float32 tensors. Different
    eager/compiled kernels then need not agree within FP32 tolerances. Backend
    flags are process-wide; this runtime supports sequential inference, not
    concurrent callers changing precision settings in separate threads.

    Args:
        config: FP32 disables TF32 matmuls and convolutions; BF16 uses autocast.
    """
    convolution_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        if config.precision == "fp32":
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = convolution_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


class _Prediction(nn.Module):
    """Compose shared tensor computations into one capturable inference graph.

    Attributes:
        policy: Original model with unchanged state-dict names.
        packer: Shared preprocessing with nonpersistent device buffers.
        device_type: Device type used for local autocast contexts.
        bf16: Whether neural operations use BF16 autocast.
    """

    def __init__(
        self, policy: FlowPolicy, packer: InputPacker, device_type: str, bf16: bool
    ) -> None:
        """Bind existing components without copying or wrapping model parameters."""
        super().__init__()
        self.policy, self.packer = policy, packer
        self.device_type, self.bf16 = device_type, bf16

    def forward(self, batch: dict[str, Tensor], noise: Tensor) -> Tensor:
        """Transform raw tensors into raw-scale FP32 action trajectories.

        Args:
            batch: Validated raw tensors already on the inference device.
            noise: FP32 initial actions ``(B, T_action, D_action)``.

        Returns:
            FP32 actions ``(B, T_action, D_action)``. Validation, transfers, and
            output ownership are handled by InferencePolicy outside this graph.
        """
        with torch.autocast(self.device_type, enabled=False):
            inputs = self.packer.pack_tensors(batch)
            with torch.autocast(self.device_type, dtype=torch.bfloat16, enabled=self.bf16):
                actions = self.policy.predict_tensors(inputs, noise)
            return self.packer.actions_to_raw(actions)


# Signatures describe device-resident, contiguous inputs rather than caller storage.
_InputSignature = tuple[tuple[str, tuple[int, ...], torch.dtype], ...]


class InferencePolicy:
    """Validate and execute eager or compiled prediction with owned action outputs.

    A compiled instance specializes on its first successful batch. Create a new
    instance for different shapes or precision. A single instance is intended for
    sequential prediction in one robot control loop.

    Attributes:
        policy: Reconstructed eval model, preserving ordinary checkpoint weights.
        packer: The shared input packer and inverse action normalization.
        device: Resolved destination such as ``cuda:0`` or ``cpu``.
        config: Immutable inference runtime settings.
        compiled: Whether this instance uses Inductor compilation.
    """

    def __init__(
        self,
        policy: FlowPolicy,
        packer: InputPacker,
        device: str,
        inference_config: InferenceConfig | None = None,
    ) -> None:
        """Bind the model and prepare buffers, deferring compilation to first use.

        Args:
            policy: Reconstructed model; this runtime places it in evaluation mode.
            packer: Shared raw input processing and saved statistics.
            device: Explicit inference device.
            inference_config: Runtime configuration; CUDA compilation defaults on.
        """
        self.config = inference_config or InferenceConfig()
        destination = torch.device(device)
        compilation_supported = tuple(int(part) for part in torch.__version__.split(".")[:2]) == (
            2,
            9,
        )
        if self.config.compile == "enabled" and not compilation_supported:
            raise ValueError(
                "Compiled inference requires PyTorch 2.9; use compile='auto' or 'disabled' "
                "for eager inference on PyTorch 2.7/2.8"
            )
        if destination.type not in {"cpu", "cuda"}:
            raise ValueError("Inference supports CPU and CUDA devices")
        if destination.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA inference requested but unavailable; select device='cpu'")
        if self.config.compile == "enabled" and destination.type != "cuda":
            raise ValueError("Explicitly enabled inference compilation requires a CUDA device")
        if self.config.precision == "bf16":
            if destination.type != "cuda":
                raise ValueError("BF16 inference requires a CUDA device")
            with torch.cuda.device(destination):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise ValueError("This CUDA device does not support native BF16 inference")
        self.policy = policy.to(device=destination, dtype=torch.float32).eval()
        self.device = str(next(self.policy.parameters()).device)
        self.packer = packer.to(device=self.device, dtype=torch.float32).eval()
        if self.config.num_inference_steps is not None:
            from robopolicy.algorithms import build_algorithm

            algorithm = policy.config.algorithm.model_copy(
                update={"num_inference_steps": self.config.num_inference_steps}
            )
            policy.algorithm = build_algorithm(algorithm)
        self.compiled = (
            compilation_supported
            and self.config.compile != "disabled"
            and destination.type == "cuda"
        )
        self._prediction = _Prediction(
            self.policy, self.packer, destination.type, self.config.precision == "bf16"
        ).eval()
        self._execute: Callable[[dict[str, Tensor], Tensor], Tensor] = self._prediction
        if self.compiled:
            self._execute = torch.compile(
                self._prediction,
                backend="inductor",
                fullgraph=True,
                dynamic=False,
                mode=self.config.mode,
            )
        self._signature: _InputSignature | None = None
        self._input_buffers: dict[str, Tensor] = {}
        self._noise_buffer: Tensor | None = None

    def _prepare(
        self, batch: Mapping[str, Any], noise: Tensor | None, *, zero_noise: bool = False
    ) -> tuple[dict[str, Tensor], Tensor, _InputSignature]:
        """Validate source tensors and place contiguous inputs on the model device.

        Args:
            batch: Raw observations with schema-declared temporal shapes.
            noise: Optional FP32 ``(B, T_action, D_action)`` noise.
            zero_noise: Use fixed zeros for warmup without consuming RNG state.

        Returns:
            Device tensors, FP32 noise, and the specialization signature.
        """
        values = self.packer.validate(batch)
        values.pop("actions", None)  # Targets never enter the inference graph.
        self.policy.encoder.validate(self.packer.canonical_inputs(values))
        signature = tuple((key, tuple(value.shape), value.dtype) for key, value in values.items())
        if self.compiled and self._signature is not None and signature != self._signature:
            raise ValueError("Compiled input signature changed; create a new InferencePolicy")
        if self.policy.training:
            raise ValueError("InferencePolicy requires the model to remain in evaluation mode")
        shape = (
            values["state"].shape[0],
            self.policy.config.action_steps,
            self.policy.config.action_dim,
        )
        if noise is not None and (noise.shape != shape or noise.dtype != torch.float32):
            raise ValueError(
                "Initial noise must be float32 with action shape (B, T_action, D_action)"
            )
        if self.compiled:
            if self._signature is None:
                self._input_buffers = {
                    key: torch.empty(value.shape, dtype=value.dtype, device=self.device)
                    for key, value in values.items()
                }
                self._noise_buffer = torch.empty(shape, dtype=torch.float32, device=self.device)
            # Stable staging avoids new input guards when caller tensors differ in
            # strides, source device, requires_grad, or inference-tensor status.
            for key, value in values.items():
                self._input_buffers[key].copy_(value, non_blocking=True)
            assert self._noise_buffer is not None
            if noise is not None:
                self._noise_buffer.copy_(noise, non_blocking=True)
            elif zero_noise:
                self._noise_buffer.zero_()
            else:
                self._noise_buffer.normal_()
            return self._input_buffers, self._noise_buffer, signature
        moved = {
            key: value.to(self.device, non_blocking=True).contiguous()
            for key, value in values.items()
        }
        if noise is None:
            factory = torch.zeros if zero_noise else torch.randn
            noise = factory(shape, device=self.device, dtype=torch.float32)
        else:
            noise = noise.to(self.device, non_blocking=True).contiguous()
        return moved, noise, signature

    def _run(self, values: dict[str, Tensor], noise: Tensor) -> Tensor:
        """Execute one prediction and return owned ``(B, T_action, D_action)`` actions."""
        if self.compiled:
            cast(Callable[[], None], torch.compiler.cudagraph_mark_step_begin)()
        try:
            with inference_precision(self.config):
                actions = self._execute(values, noise)
        except Exception as error:
            if self.compiled:
                raise RuntimeError(
                    f"Compiled inference failed on {self.device} (mode={self.config.mode}, "
                    f"precision={self.config.precision}); no eager fallback was used"
                ) from error
            raise
        # CUDA graph outputs may be overwritten by the next prediction. Ownership
        # must be established outside the captured graph, including for warmup.
        return actions.clone() if self.compiled else actions

    @torch.inference_mode()
    def predict(self, batch: Mapping[str, Any], noise: Tensor | None = None) -> Tensor:
        """Generate raw-scale FP32 actions from supplied raw observations.

        Args:
            batch: Raw feature tensors in checkpoint-declared shapes and dtypes.
            noise: Optional fixed FP32 ``(B, T_action, D_action)`` initial noise.

        Returns:
            Owned ``(B, T_action, D_action)`` actions on the inference device.
            First use compiles lazily; use warmup before a robot control loop.
        """
        values, initial_noise, signature = self._prepare(batch, noise)
        actions = self._run(values, initial_noise)
        self._signature = signature
        return actions

    @torch.inference_mode()
    def warmup(self, example_batch: Mapping[str, Any], iterations: int = 5) -> None:
        """Compile and warm the deployment shape before entering a control loop.

        Args:
            example_batch: Representative raw tensors with the deployment shapes.
            iterations: Positive number of invocations, including compilation.

        Warmup uses fixed noise and restores Torch RNG state even on failure.
        CUDA work is synchronized before returning. Python and NumPy RNGs are
        untouched by the prediction computation.
        """
        if iterations < 1:
            raise ValueError("Warmup iterations must be positive")
        destination = torch.device(self.device)
        rng = capture_rng(self.device)
        try:
            values, noise, signature = self._prepare(example_batch, None, zero_noise=True)
            for _ in range(iterations):
                self._run(values, noise)
            if destination.type == "cuda":
                torch.cuda.synchronize(destination)
            self._signature = signature
        finally:
            restore_rng(rng, self.device)


def load_policy(
    path: str | Path, *, device: str = "cuda", inference_config: InferenceConfig | None = None
) -> InferencePolicy:
    """Reconstruct inference without data, W&B, or pretrained server access.

    Args:
        path: Versioned RoboPolicy training checkpoint.
        device: Inference destination; CUDA is the default, CPU is explicit.
        inference_config: Runtime compilation and precision settings.

    Returns:
        An eval policy with saved preprocessing. Compilation is lazy until warmup
        or prediction; compiler artifacts are never read from the checkpoint.
    """
    payload = read_checkpoint(path)
    config = PolicyConfig.model_validate(payload["policy_config"])
    schema = FeatureSchema.model_validate(payload["feature_schema"])
    statistics = Statistics.model_validate(payload["statistics"])
    processing = ProcessingConfig.model_validate(payload["processing"])
    policy = build_policy(config, initialize_pretrained=False)
    policy.load_state_dict(payload["model"], strict=True)
    return InferencePolicy(
        policy, InputPacker(schema, statistics, processing), device, inference_config
    )
