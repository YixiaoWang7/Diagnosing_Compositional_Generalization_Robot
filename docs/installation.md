# Installation and supported environments

Python 3.10 and 3.11 on Linux are supported. Use Python 3.10 for the paper release
reference environment and reuse an existing `.venv`. The
[root setup guide](../README.md#setup) installs both the library and the separately
packaged CG application. The commands below cover library-only environments. Training supports
CPU or a single CUDA GPU, with zero or multiple DataLoader workers. Distributed
training is not implemented. The reference GPU environment is recorded in
[environment.json](validation/environment.json).

Core dependencies are PyTorch, NumPy, Pydantic, and typing extensions. Importing
`robopolicy` itself loads none of PyTorch, HDF5, video, pretrained models, or W&B.

| Extra | Adds |
| --- | --- |
| `dataset` | HDF5 readers and selected-demo statistics |
| `opencv` | Headless OpenCV video decoding |
| `torchcodec` | TorchCodec 0.9 video decoding; additionally requires PyTorch 2.9 |
| `vision` | Transformers and TorchVision for DINOv2 and optional language components |
| `logging` | Online W&B tracking |
| `dev` | pytest, Ruff, strict mypy, build tools, and benchmark memory sampling |

A fresh CPU installation can use:

```bash
uv venv --python 3.10 .venv
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install -c constraints/training.txt '.[dataset,opencv,logging,dev]'
```

To use TorchCodec, install `.[dataset,torchcodec,logging,dev]`. The 0.9 series is paired
with PyTorch 2.9. FFmpeg shared libraries must be available on Linux. Missing
backend libraries produce a message naming the selected decoder and extra;
the library never falls back to another decoder. State-only data does not import
or instantiate a video decoder. For an explicit local-only training configuration,
set `logging.wandb.enabled=false`; the `logging` extra may then be omitted.

For the reference CUDA installation, explicitly select the CUDA wheels suitable
for the host driver, then install the extras. For example, the validated host
uses PyTorch 2.9.1 and TorchVision 0.24.1 with CUDA 12.8, plus TorchCodec 0.9.1.
The package configuration contains no machine-specific CUDA index selection.
`uv sync` resolves the committed lock from ordinary indexes; selecting a CPU
PyTorch index at installation time remains an independent operation.

```bash
uv pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install -c constraints/training.txt '.[dataset,torchcodec,opencv,vision,logging,dev]'
```

The lightweight patch encoder is part of core and requires no downloads. For
DINOv2, specify architecture dimensions in `policy.vision`. A typical small model
uses `dim=384`, `depth=12`, `heads=6`, `patch_size=14`, and `image_size=518` for its
stored positional embeddings; processing may crop/resize input to 224. Set
`pretrained` to the desired model identifier only when initializing fresh training.
Checkpoint reconstruction always disables pretrained initialization and loads
all weights from the checkpoint. A frozen backbone stays in evaluation mode.

## Separate training and deployment environments

Core supports Torch 2.7–2.9 on Python 3.10/3.11. Torch 2.9 is the full training and
compiled CUDA reference. Torch 2.7/2.8 use eager inference, with state, entity,
patch-image and DINO checkpoint-transfer checks. The coupled vision pairs are
2.7/0.22, 2.8/0.23 and 2.9/0.24 (Torch/TorchVision).

`uv.lock` defines the reference development environment. Do not `uv sync` this
project into a simulator-owned environment. Use an installed wheel or source
package with constraints that preserve that environment's selected runtime:

```bash
# Inside an existing Python 3.11 / Torch 2.7.0 environment:
python -m pip install -c /code/robopolicy/constraints/inference.txt '/code/robopolicy[vision]'
python -m pip check
```

The training profile pins Torch 2.9.1/TorchVision 0.24.1/TorchCodec 0.9.1; the
inference profile pins Torch 2.7.0/TorchVision 0.22.0. These are reproducible
reference pairs, not mandatory patch versions for every application. Keep the
application's own pins when using another compatible pair. TorchCodec combined
with the inference constraints must fail dependency resolution, not upgrade Torch.
Install `dataset` for export-time identity verification; live state inference uses
only core, and DINO inference needs `vision` but no decoder or logging extra.

Projects choose their own code, data and output directories. No sibling checkout
or simulator package is required by RoboPolicy. Version checks and compiler
compatibility live in this library, never in individual task adapters.

Fresh wheel tests must run outside the source tree using isolated environments.
Both wheels and source distributions exclude `projects/` and the old `policies`
package. A wheel must contain `robopolicy/py.typed` and only maintained library
modules. See the release report for executed installation results.
