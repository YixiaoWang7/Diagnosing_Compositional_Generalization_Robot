# Inference and torch.compile

`load_policy` reconstructs the architecture, weights, feature schema, and normalization
from a checkpoint. Its default device is CUDA. Compilation defaults on for Torch
2.9 CUDA; Torch 2.7/2.8 and CPU use eager execution. Python 3.10/3.11 are supported.

```python
from robopolicy import InferenceConfig, load_policy

inference = load_policy(
    "runs/my-training/latest.pt",
    device="cuda:0",
    inference_config=InferenceConfig(
        compile="auto",
        mode="max-autotune",
        precision="fp32",
    ),
)

# Supply the same raw tensor shapes that the robot will provide.
inference.warmup(first_observation)

# Inside the control loop:
actions = inference.predict(observation)  # FP32, (B, T_action, D_action), on the GPU
robot_actions = actions.to("cpu").numpy()
```

Warmup compiles lazily when compilation is selected, runs five predictions with zero noise, and waits for GPU
completion. It restores Python, NumPy, and Torch random states, including on failure.
First startup may take seconds or minutes, depending on architecture and tuning mode.
Subsequent processes can reuse PyTorch's normal on-disk compiler cache; these artifacts
are device/software-specific and are not part of RoboPolicy checkpoints. Direct
`predict` without warmup works, but its first invocation includes compilation.

## Configuration

`InferenceConfig` is an immutable, version-1 runtime configuration. It is independent
of training configuration and the saved architecture.

| Setting | Values | Default |
| --- | --- | --- |
| `compile` | `auto`, `enabled`, `disabled` | `auto`: Torch 2.9 CUDA compiled, otherwise eager |
| `mode` | `default`, `reduce-overhead`, `max-autotune` | `max-autotune` |
| `precision` | `fp32`, `bf16` | `fp32` |
| `num_inference_steps` | Positive integer or `None` | `None`: saved solver step count |

`enabled` requires Torch 2.9 and CUDA. Select `compile="disabled"` for eager GPU execution, or
`load_policy(path, device="cpu")` for eager CPU execution. Unsupported devices,
unsupported native BF16, and compilation failures produce explicit errors.

`reduce-overhead` uses CUDA graphs to reduce launch overhead. `max-autotune` also
profiles candidate kernels and can take longer at startup. The faster mode depends
on the actual model and GPU; compare the recorded complete-path latency for your
checkpoint. See the [PyTorch 2.9 compile API](https://docs.pytorch.org/docs/2.9/generated/torch.compile.html).

FP32 temporarily disables TF32 for matmuls and convolutions, then restores the
caller's backend settings. This matters because cuDNN otherwise permits TF32
convolutions even for float32 tensors. Solver settings are fixed when constructing
the runtime; use `InferenceConfig(num_inference_steps=10)` for a step override.
Do not replace the runtime's internal algorithm after compilation. BF16 is an
explicit neural-computation option using CUDA autocast; normalization, solver state,
and returned actions remain FP32. Numerical agreement alone does not establish
robot task performance, so validate BF16 in the application before deployment.

The command-line entry point uses the same factory and settings:

```bash
PYTHONPATH=src .venv/bin/python scripts/policies/infer_library.py \
  --checkpoint runs/my-training/latest.pt --input observations.npz \
  --output actions.npy --device cuda:0 --compile auto \
  --compile-mode max-autotune --precision fp32
```

For a single prediction, eager execution avoids compiler startup; compilation is
intended for repeated predictions in a long-lived process.

## Runtime structure

The public boundary validates raw schema, shapes, categorical IDs, and entity
selections before device transfer. GPU staging buffers make input storage stable.
For best latency, supply categorical indices on the CPU so range validation does
not need a GPU-to-host synchronization.

One compiled computation performs deterministic image transforms, normalization,
query construction, vision encoding, fusion, the entire fixed-step flow solver,
and inverse action normalization. The eager and compiled paths use the same
numerical methods. Compiler settings never modify model state-dict names.

A compiled runtime locks shapes and dtypes on its first successful prediction or
warmup. Source tensors may be on CPU or GPU, have different strides, or carry new
values. A different batch size or optional modality requires a new runtime. Keep
the model in evaluation mode, with fixed architecture, device, and precision.
Use an instance sequentially in one control loop; multi-robot scheduling is separate.
PyTorch precision flags are process-wide, so concurrent threads changing those flags
are outside this runtime's supported execution model.

Returned actions own their storage: a later prediction cannot overwrite them.
This ownership copy occurs outside the captured CUDA graph. Attention inspection
continues through `inference.policy.attention_maps(packed_inputs)` using the same
observation encoder. Training remains eager unless the application explicitly
implements a separate training compilation workflow.

## Benchmarking

```bash
PYTHONPATH=src .venv/bin/python scripts/policies/benchmark_inference_library.py \
  --config examples/inference_benchmark.json
```

Set the example specification's checkpoint path to your own; no dataset is accessed.
Each configuration runs in a
fresh **process in the same environment** to isolate compiler caches in memory and
GPU allocations. W&B records one verified evaluation run and the aggregate result.
Compilation logs and intermediate reports are kept under the configured output directory.

Measurements use generated observations at the checkpoint's exact schema shapes,
fixed noise, three repetitions, and 1,000 predictions per repetition. GPU timings
include on-device preprocessing, staging, prediction, and owned output creation.
Complete timings start with pageable CPU observations and end with CPU actions,
including validation and both transfers. Camera acquisition and robot transport
are application costs outside these measurements.

Reports include latency percentiles, startup time, memory, graph replay evidence,
recompilation counts, and numerical errors against eager FP32. Compiler startup
measurements can reuse the normal disk cache and should not be interpreted as
cold-cache installation measurements. Performance results are hardware-specific.
