# Dataset format and execution

The runtime reads aligned small-files datasets. The root `dataset_manifest.json`
contains tasks with opaque `task_slug` identifiers and relative `manifest` paths:

```json
{"tasks":[{"task_slug":"task-A","manifest":"task-A/manifest.json"}]}
```

A task manifest lists shards with `shard_id`, relative `hdf5` paths, and `demos`.
Each demo has `demo_key` and optionally `length`. HDF5 groups are
`data/<demo_key>/`; feature paths are relative to that group. Human-readable
`task` strings are retained as metadata. No identifier is interpreted as a task
instruction or category vocabulary.

```json
{"task":"an application-defined label","shards":[{
  "shard_id":0,"hdf5":"shard_000.h5",
  "demos":[{"demo_key":"episode-0","length":100}],
  "videos":{"front":{"mp4":"front.mp4","sidecar":"front.json"}}
}]}
```

Video sidecars require a total `frames` count and episode `segments` containing
`demo_key`, `frame_start`, and `frame_count`. Paths resolve relative to their task
manifest. Each camera's episode length must equal every requested numeric source
and any declared manifest length. Missing sources, shape mismatches, incorrect
camera selection, corrupt video, and length mismatches fail explicitly. Runtime
does not truncate, resample, or align modalities. Prepare aligned data separately.

A numeric feature declares `key`, per-step `shape`, HDF5 `paths`, `dtype`, temporal
`history` and `future`, and explicit `normalization` (`none`, `standard`, or
`minmax`). Multiple numeric paths concatenate along the last feature axis. Integer
categories use `dtype="int64"`, retain that dtype, and cannot be normalized.
`task_indices` and `entity_indices` use current-step windows only. A video feature
uses `source="video"`, `dtype="uint8"`, RGB shape `(3,H,W)`, and ordered `cameras`.

Decoding and sampling are independent:

```json
{"video":{"decoder":"torchcodec"},
 "sampling":{"strategy":"global_pass","k_passes":1}}
```

Both `opencv` and `torchcodec` support `window_major` and `global_pass`. State-only
data requires neither decoder. Each decoder returns RGB uint8 NCHW frames; it
performs no image or numeric normalization.

Selection resolves opaque task filters, deterministic train/validation membership,
and training demo limits once. Limits apply after splitting. Validation membership
does not change when the training demo budget changes. `val_fraction=0` produces
an empty validation set, never a copy of training data. Statistics use the exact
resolved training demos, reading bounded numeric chunks and merging float64
moments. Constant dimensions have unit scale. The JSON cache records schema,
resolved selection, and a content fingerprint; any change invalidates the cache.
Statistics are fixed for validation and inference.

`selection.balance_demos_across_tasks=true` divides `max_demos` into equal
quotas after splitting and per-task caps. Tasks in sorted ID order receive the
remainder slots. A task with fewer available demos leaves its remaining quota
unused; other tasks do not receive those slots. The default is global seeded
selection. Neither mode limits validation demonstrations.

For an episode of length `L`, the eligible anchors are
`0 <= t < L - max(feature.future)`. History repeats the episode's first step.
Future steps are never padded or allowed to cross an episode boundary. Each
finite epoch emits every eligible sample once with `drop_last=False`, including
partial per-worker batches. `K` partitions are disjoint and exhaustive even when
some partitions are empty.

Both strategies use the same per-shard partition calculation and sample assembler.
Window-major visits partitions within one loaded window; global-pass visits all
of a worker's windows for each partition. There is no barrier between workers.
Shards are assigned disjointly. Fixed data, seed, settings, and worker count give
reproducible order. Changing worker count preserves coverage but can change order.
Validation ordering is fixed. The trainer calls `set_epoch`; dataset iterators do
not repeat independently.

The execution layer owns current-window loading, one-window prefetch, one-block
decode prefetch, native decoder handles, and cleanup. Decoded frames live only
for their current/prefetched block; there is no legacy frame cache. Native handles
are bounded per worker by `video.max_handles`. Use loaders as context managers:

```python
from robopolicy import build_dataset, build_loader

with build_loader(build_dataset(dataset_config, feature_schema), loader_config) as loader:
    for epoch in range(2):
        loader.dataset.set_epoch(epoch)
        for batch in loader:
            consume(batch)
```

Memory is the sum of active and prefetched numeric windows, active and prefetched
frame blocks, native decoder buffers, and PyTorch DataLoader queues. Queue size
also depends on worker count, batch size, image/history shapes, and
`prefetch_factor`. Training and validation loaders may both be active. A shard
window count is not a universal RAM limit. The default `spawn` worker method
avoids forking an initialized CUDA runtime; `forkserver` is also supported.
Loader shutdown is centralized and joins prefetch threads before closing decoders.

## Explicit timestamp preparation

`robopolicy.datasets.alignment.align_timestamps` is a preparation utility. Supply
source and target timestamps explicitly; use nearest sampling for categorical
values or linear interpolation for floating features. It rejects unordered
timestamps and extrapolation. Runtime readers never call it automatically.

The OpenCV pilot showed that eight retained decoder handles per worker use more
native memory than the former window-scoped four handles. Setting
`dataset.video.max_handles=4` removed that increase without reducing measured
throughput. The handle limit is independent of shard windows and frame blocks.
`video.threads` controls TorchCodec FFmpeg threads; OpenCV manages codec threads.
