# Baseline provenance

Integration branch: `refactor/policy-core`, based on `master` at `efc051e`.
Reusable components and baseline behavior were inspected from `dev_2s` at
`387bb76`. That branch remains unchanged. A detached worktree at
`/tmp/robopolicy-dev2s-baseline` preserves the old execution path for comparison.

The initial suite produced **115 passed, 6 failed** in 7.49 seconds. The six
failures called `FlowMatchingAlgorithm` with obsolete keyword arguments. Each
call was changed to `FlowMatchingAlgorithm(FlowMatchingConfig(...))`, preserving
all assertions. The corrected baseline produced **121 passed** in 7.39 seconds.
The corrected algorithm and convolutional component tests are retained under
`tests/core/`. Dataset tests now assert the new independent decoder/sampler
configuration and strict alignment contract. The former shortest-modality
truncation expectation was deliberately replaced by a mismatch error test.

Real loader baseline JSON files record three repetitions for OpenCV window-major,
TorchCodec window-major, and TorchCodec global-pass with decoded-frame caching
disabled. The initial baseline uses the old Linux default worker start method,
`fork`; the library uses `spawn` for CUDA-safe multiprocessing. These first
measurements are diagnostic; acceptance comparisons must use matching start
methods, selection, schemas, batch sizes, windows, partitions, and block sizes.
