# Reproducing the validation artifacts

Use the existing development `.venv` and the dependency/hardware record in
`environment.json`. Generated datasets, checkpoints, and W&B local files remain
ignored. The JSON files in this directory are review artifacts, not library defaults.

## Source checks

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m pytest tests -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
```

CPU-only execution selects `-m 'not cuda'`; a CUDA acceptance runner executes the
whole suite. CI uses Python 3.10 and 3.11 for each fresh-wheel extra combination. The
packaging check script also accepts `--wheel path/to/file.whl` to inspect a final
wheel directly using already-installed dependencies, without installing it again.

See [portable runtime acceptance](portable-integration.md) for the Torch 2.7–2.9
installation matrix and checkpoint-transfer results.

## Candidate loaders and soak

Run each of the four decoder/sampler specs three times, changing the output suffix
from 0 through 2. Each recorded comparison measures 15 seconds after five warm-up
batches, with no concurrent training benchmark competing for resources.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python \
  scripts/policies/benchmark_library.py \
  --spec docs/validation/loader-torchcodec-global_pass.json \
  --device cuda:1 --measure-seconds 15 \
  --output-json /tmp/candidate-torchcodec-global_pass-0.json
```

The other specs are `loader-torchcodec-window_major.json`,
`loader-opencv-window_major.json`, and `loader-opencv-global_pass.json`.
`loader-opencv-handles4.json` reproduces the memory investigation.

Use `pilot-loader.json` and `--measure-seconds 1800` for the thirty-minute soak.
The benchmark deliberately advances finite epochs as their owner. It measures
GPU image transfer/preprocessing for parity with the former harness; it does not
train or evaluate a policy network.

## Preserved baseline

The old source is preserved at dev_2s commit `387bb76`. In a detached checkout,
use `scripts/policies/benchmark_video_dataloader.py`. For matching comparisons,
copy that script to a temporary file and add
`multiprocessing_context="spawn"` to its `DataLoader` construction. The only other
change used for matching runs labels `worker_start_method` as `spawn` in output.
No decoding, sampling, preprocessing, or selection computation is modified.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/tmp/robopolicy-dev2s-baseline/src .venv/bin/python \
  /tmp/robopolicy_baseline_spawn.py --backend global-pass \
  --data-dir /mnt/ssd/yixiao/cg/l4/single_stage_5demo \
  --device cuda:1 --batch-size 16 --num-workers 2 --prefetch-factor 2 \
  --k-passes 1 --active-window-shards 2 --locality-block-size 32 \
  --max-demos 80 --current-frame-cache 0 --decoder-handles 8 \
  --warmup-batches 5 --measure-seconds 15 \
  --output-json /tmp/baseline-spawn-global-pass-0.json
```

Repeat for `current`, `optimized`, and `global-pass`, three times each. The old
harness uses split/subset seed 0 and validation fraction 0.01; the candidate specs
match. `selection-parity.json` verifies exact selected-demo membership. The old
OpenCV handle lifetime is one window, so its effective limit here is four per
worker; its `--decoder-handles` flag applies only to TorchCodec.

## Training and recovery pilots

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python \
  scripts/policies/run_library_pilot.py \
  --config docs/validation/pilot-conv.json --checkpoint-step 250 \
  --output /tmp/pilot-conv-result.json
```

Use the Transformer spec for the second head. Choose a new `logging.out_dir` for
a fresh rerun so its local metric history is easy to distinguish. The script
stops, closes resources, reconstructs the trainer from a checkpoint, continues,
validates, and compares raw-scale predictions exactly for fixed initial noise.

The pretrained DINOv2 pilot uses the recorded small-model architecture in
`pilot-pretrained-dino.json`. Its `pretrained` path identifies the downloaded
snapshot; on another machine supply `facebook/dinov2-small` after authorizing the
download, retaining all architecture fields. Use checkpoint step 2 of 4. The
recorded result additionally compares loaded parameters to the downloaded
reference weights and forbids pretrained loader calls during recovery/inference.

Before a long W&B job, run `.venv/bin/wandb login --verify`. The recorded 0.24.2
SDK verifies the stored credential before hitting a presentation-only banner bug
on null viewer flags. The library supplies silent SDK settings to bypass that
banner, initializes online with the explicit `yixiao_wang` entity, prints the
canonical run URL, and verifies it through the authenticated API. Permission and
strict-resume errors remain fatal; there is no offline or fresh-run fallback.

The completed W&B pilot uses the same command with `pilot-wandb.json`. Its
training and separate evaluation URLs, artifact versions, recovery boundary, and
aggregate metrics are recorded in `wandb-verification.json`.
