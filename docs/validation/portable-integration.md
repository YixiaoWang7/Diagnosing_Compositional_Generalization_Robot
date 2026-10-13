# Portable runtime acceptance — 2026-09-14

The compatibility and checkpoint-validation changes were verified against the
reviewed policy-core base `57ba7a1`. Flow-matching numerical code was unchanged.

| Environment / check | Evidence |
| --- | --- |
| Python 3.10.20, Torch 2.9.1+cu128 | Full suite: 235 passed, including CUDA compilation/recovery |
| Python 3.11.15, Torch 2.9.1+cpu, installed wheel | CPU suite: 225 passed; 10 CUDA-only tests excluded explicitly |
| Python 3.11.15, Torch 2.9.1+cu128, installed wheel | Full suite: 235 passed; one expected advisory about deliberately disabled TF32 |
| Python 3.10.20 and 3.11.15, Torch 2.7.0+cpu | Constrained vision wheel installation and four-modality checkpoint transfer passed |
| Python 3.10.20 and 3.11.15, Torch 2.8.0+cpu | Pinned Torch/TorchVision installation and four-modality checkpoint transfer passed |
| Python 3.11.15, Torch 2.7.0+cu128 | Actual CUDA predictions passed for all four modalities; auto compilation selected eager |
| Public imports | Wheel imports worked outside the checkout without importing HDF5, decoders or W&B |
| Quality gates | Ruff check/format, strict mypy (50 source files), lock consistency, wheel/sdist builds and dependency checks passed |

Four tiny state/entity/patch-image/DINO checkpoints were generated through the real
Trainer/save path on Torch 2.9. Training data was then removed. Consumers compared
fixed-noise predictions with `rtol=1e-4`, `atol=1e-5`. Pretrained initialization and
network access are unnecessary during reconstruction.

Reproduce with a fresh output directory and the chosen constrained environments:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest tests -q
python -m tests.core.test_portability write /tmp/portable-checkpoints
# Activate the separately installed consumer environment; do not set PYTHONPATH=src.
HF_HUB_OFFLINE=1 python -m tests.core.test_portability check /tmp/portable-checkpoints
```

CPU-only suite runs use `-m 'not cuda'`. A missing GPU is not a passing CUDA gate.
Sandboxed worker processes stalled on this host, so multiprocessing/CUDA acceptance
was rerun successfully outside the sandbox; the interrupted test processes from
this implementation were terminated before retrying.

An installation dry run combining the inference constraints with the TorchCodec
extra correctly failed because Torch 2.7.0 conflicts with TorchCodec's required
Torch 2.9 range. Installed Torch versions remained pinned after adding vision extras.
The CPU and manual GPU CI workflows now preserve these acceptance combinations.
