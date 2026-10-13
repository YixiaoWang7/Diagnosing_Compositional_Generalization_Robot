# Release acceptance report

Status: **implementation and all release acceptance gates complete, including
verified online W&B checkpoint recovery. The final branch is ready for review.**

The integration branch is `refactor/policy-core`, based on master `efc051e`.
Experiments on `dev_2s` remain at `387bb76`. The maintained package is exclusively
`robopolicy`; old code, experiment collections, simulator entry points, and old
imports have been retired from this branch. No migration wrappers were added.

## Automated verification

| Gate | Executed result |
| --- | --- |
| Original baseline | 115 passed, 6 obsolete-constructor failures |
| Corrected baseline, original assertions | 121 passed |
| Final Python 3.10 CPU/CUDA suite | 172 passed; no skips |
| Ruff lint and formatting | Passed |
| Strict mypy | Passed, 48 maintained source files on Python 3.10 |
| Cold import and dependency boundaries | Passed; root import loads no Torch, decoder, W&B, or pretrained machinery |
| Fresh wheel matrix | Python 3.10: core-only, OpenCV-only, TorchCodec-only all passed |
| Distribution contents | Only `robopolicy` in the wheel; no `projects` or old `policies` in wheel/sdist |
| Public example | 200 CPU updates, recovery from 100, CLI inference passed |

The fresh-install matrix was performed once in temporary environments to verify
packaging. Subsequent work uses the existing Python 3.10 development `.venv`; no
new environments are needed to run the library. CI retains the decoder-extra matrix.
After the final edits, the rebuilt wheel was checked directly in all six
already-installed interpreter/decoder environments, with no further installs.
Every packaged Python file was verified byte-for-byte against the final source. Evidence lives in `tests-python*.txt`, `types-python*.txt`,
`install-*.json`, `final-wheel-*.json`, `distributions.json`, and `example-result.json` here.

Tests use real FFV1 encoded frames with exact RGB identities. They exercise all
four decoder/sampler combinations, repeated frame requests, disjoint exhaustive
partitions including empty partitions, episode windows, worker counts exceeding
shards, persistent-worker epoch changes, early exits, and decoder failures in
worker processes. Tests also cover selected-demo statistics, cache invalidation,
query vocabularies and gradients, masked attention, toy learning, partial-batch
validation, atomic-write failure, complete recovery state, CUDA AMP skips, and
strict W&B identity through a controlled SDK test double.

## Real-data loader comparisons

Primary data: `/mnt/ssd/yixiao/cg/l4/single_stage_5demo`. The subset is 80 demos,
selected once with split/subset seed 0, validation fraction 0.01. Both paths use
batch 16, two **spawn** workers, prefetch factor 2, two-shard windows, block size
32, K=1, sampling seed 42, and the same state/action/camera features. Images remain
RGB uint8 until the identical GPU transfer/preprocessing stage. Decoded-frame
caching is disabled. GPU model inference is excluded from this loader benchmark.

Reference hardware and package versions are in [environment.json](environment.json).
Three steady-state repetitions use five warm-up batches and 15 measured seconds
each. Initial 30-second old-default-fork measurements are retained for provenance,
but are not used for matching comparisons: spawn accounts for Python/PyTorch
process memory differently. PSS sums parent and workers without double-counting
shared pages; RSS, startup, latency percentiles, open files, and resource counts
are also retained in individual JSON artifacts.

| Decoder / sampler | Old samples/s | New samples/s | Old peak PSS MiB | New peak PSS MiB |
| --- | ---: | ---: | ---: | ---: |
| OpenCV / window-major, eight-handle cap | 213.7 | 1,738.8 | 1,913.9 | 2,185.8 |
| OpenCV / global-pass | unavailable | 1,729.3 | unavailable | 2,188.3 |
| TorchCodec / window-major | 722.5 | 1,276.5 | 1,736.0 | 1,735.8 |
| TorchCodec / global-pass | 765.8 | 1,282.3 | 1,724.7 | 1,722.2 |
| OpenCV / window-major, four actual handles | 213.7 | 1,746.1 | 1,913.9 | 1,862.0 |

Values are means across the three runs, including the mean of each run's peak
memory. The original OpenCV implementation clears handles per window, so its
actual bound is four per worker even though the benchmark's eight-handle option
applies to its TorchCodec implementation. The new shared pool retains up to eight
across windows. This explains the **14.2% OpenCV memory increase**. Three additional
runs with `max_handles=4` matched the old actual bound and removed the increase,
with throughput retained. Use this setting when matching the old OpenCV memory
budget. No throughput regression exceeded 10%; TorchCodec memory was comparable.
Old decoder counts are bounded from its implementation and open-file telemetry;
the new implementation reports aggregate live/peak native handles directly.

Candidate specs are `loader-*.json`; old/new results are `baseline-spawn-*.json`
and `candidate-*.json`. The benchmark entry point is
[`benchmark_library.py`](../../scripts/policies/benchmark_library.py).

## Thirty-minute soak

[soak-30min.json](soak-30min.json) records a completed TorchCodec/global-pass soak:

- 1,800.01 measured seconds, 2,310,352 samples, 1,283.5 samples/second.
- Batch latency p50 2.57 ms and p95 45.40 ms; duration completed without a stall.
- Aggregate decoder peak 16, within eight per worker; live count zero after shutdown.
- Peak parent/worker PSS 1,930.2 MiB. First steady sample 1,764.7 MiB, final sample
  1,857.7 MiB: final growth 5.3%, peak growth 9.4% from that steady sample.
- The resource time series retained stable process/thread and open-file bounds.

The soak validates the shared resource-management implementation. Later changes
to metadata fingerprints, configuration checks, and unused neural code do not
change that execution path. This is a bounded operational check, not a claim
about arbitrary datasets or indefinite execution.

## Training, recovery, and inference

Both image heads ran 500 successful optimizer updates on CUDA with AMP, stopping
at 250, restoring state, and continuing at 251. Each completed full validation on
12,588 held-out samples and exact fixed-noise checkpoint prediction parity.

| Image action head | Completed updates | Validation flow loss | Prediction shape |
| --- | ---: | ---: | --- |
| Convolutional | 500 | 1.02831 | (8, 16, 7) |
| Transformer | 500 | 0.68784 | (8, 16, 7) |

See `pilot-conv.json`, `pilot-transformer.json`, and their `*-result.json` files.
These use a small patch vision encoder to check training/recovery behavior.
They do not establish task-success equivalence or compare policy quality.

The additional L5 storage/schema check selected eight demos, verified 2,360
eligible samples, and loaded aligned state, actions, and two cameras. See
[additional-storage.json](additional-storage.json). Task labels are opaque metadata.

A separate real pretrained [DINOv2-small](https://huggingface.co/facebook/dinov2-small)
pilot downloaded 88,249,960 bytes at revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`. It verified exact pretrained-weight
loading, four updates with recovery from two, and exact prediction parity while
forbidding pretrained loader calls during reconstruction. See
[pretrained-dino-result.json](pretrained-dino-result.json). Checkpoints contain the
backbone; fresh training is allowed to download it. Small offline DINOv2
reconstruction is also covered in automated tests.

## Online W&B recovery

The authenticated credential belongs to `yixiao_wang`. The original HTTP 403 was
caused by attempting to create a run under the SDK's unrelated default `models`
entity. The corrected logger always supplies the configured entity and project.
W&B 0.24.2 also has a presentation-only login-banner failure when viewer flags are
null; silent SDK settings bypass that banner while retaining online uploads.
Every initialized URL is checked through the authenticated W&B API.

The 500-update online pilot used [pilot-wandb.json](pilot-wandb.json), stopped at
update 250, and resumed with `resume="must"` into the same
[training run](https://wandb.ai/yixiao_wang/robopolicy-pilot/runs/3844cb75).
The API verified two resume segments, the recovery boundary at 250, final update
500, provenance artifacts for both segments, and checkpoint artifacts at updates
250 and 500. Full validation reproduced loss 1.028310671614434 over 12,588 samples,
and checkpoint-loaded inference retained exact fixed-noise prediction parity.

Aggregate validation metrics and the result artifact were uploaded to a separate
[evaluation run](https://wandb.ai/yixiao_wang/robopolicy-pilot/runs/d5491d85).
Both runs finished successfully. See [pilot-wandb-result.json](pilot-wandb-result.json),
[wandb-verification.json](wandb-verification.json), and the historical
[wandb-attempt.json](wandb-attempt.json).
