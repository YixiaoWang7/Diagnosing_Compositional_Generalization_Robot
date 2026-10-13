# Training, recovery, and inference

`RunConfig` is a strict JSON composition of `features`, `dataset`, `loader`,
`policy`, `processing`, `training`, and `logging`. Component defaults live beside
their implementations. The schema rejects obsolete fields and incompatible
feature dimensions before constructing a model. Start from
[the minimal example](../examples/minimal_project/README.md).

```bash
uv run python scripts/policies/train_library.py --config examples/state_policy.json
uv run python scripts/policies/train_library.py --resume runs/example/step-100.pt
```

The trainer owns optimizer updates, optional AMP, gradient clipping, learning-rate
scheduling, finite data epochs, validation, checkpoint cadence, and shutdown.
`completed_updates` counts successful optimizer steps. An AMP overflow records a
skip without advancing the scheduler or training-step axis. Repeated overflows
fail clearly instead of causing an unbounded training loop. Validation weights
metrics by sample count, fixes its stochastic evaluation seed, restores training
RNG state, and restores the previous model mode even if evaluation raises.

Checkpoints are written to a same-directory temporary, flushed, and atomically
renamed. A failed write or rename preserves the previous file. Version 1 includes:

- Model architecture and all weights, including the vision backbone.
- Feature schema, image processing settings, normalization, and provenance.
- Optimizer, scheduler, AMP scaler, completed-update count, and data epoch.
- Python, NumPy, CPU Torch, and applicable CUDA random states.
- Dataset content fingerprint, resolved selection, and full run configuration.
- Resolved W&B run identity and recovery-segment metadata.

Recovery uses the saved run configuration. A checkpoint at update 10,000 resumes
with update 10,001. The data iterator starts at the beginning of the next logical
epoch; prefetched batches are not replayed. Consequently recovery is stateful
continuation but can follow a different trajectory from uninterrupted training.
Architecture, feature, selection, or dataset-content mismatches fail clearly.
An unchanged dataset may move, and the supported device may change:

```python
from robopolicy import Trainer

trainer = Trainer.resume(
    "runs/example/step-100.pt",
    dataset_root="/new/location/data",
    device="cpu",
)
trainer.fit()
```

An output directory override is also available for relocating run artifacts.
CUDA RNG/scaler state is restored when applicable to the selected device.
CPU recovery does not require CUDA. Configurations and checkpoints have explicit
schema versions; there is no legacy migration path in this release.

Fine-tuning is a separate operation: `Trainer.finetune(new_config, checkpoint)`
loads matching model weights into a fresh optimizer, scheduler, RNG, normalization
selection, and logging run. It never resumes the old run identity. Use a distinct
output directory for the new training state.

## W&B continuation

Training defaults to online W&B logging with the explicit `yixiao_wang` entity
and a stable project name from the checked-in run configuration. Set
`logging.wandb.enabled=false` explicitly for local-only execution; W&B remains a
lazy optional dependency in that case. Fresh training creates a run, verifies its
canonical URL through the authenticated API, prints that URL, and saves the
server-resolved ID, entity, and project in segment metadata and every checkpoint.
Recovery uses those exact values with `resume="must"` and fails if the run cannot
be resumed, following [W&B's strict recovery behavior](https://docs.wandb.ai/models/runs/resuming).
Online recovery cannot silently become an offline or fresh run.

Optional `logging.wandb.group` assigns an application-chosen W&B group to
training, recovery, and standalone evaluation runs. Group naming has no effect on
run identity or checkpoint recovery.

Training progress uses `train/step` as a [custom metric axis](https://docs.wandb.ai/models/track/log/customize-logging-axes).
The logger never passes the optimizer counter as W&B's internal history step.
If remote events are newer than the checkpoint, recovery restores the checkpoint's
actual update count and appends to the same run. Old events remain intact. Each
segment gets a unique `resume/segment` identifier and a boundary event containing
`resume/checkpoint_step`; repeated training-step values remain visible.

All events also append to local `metrics.jsonl`. A `segment-<id>.json` file records
resolved configuration, run identity, feature schema, dataset fingerprint, and
selected demonstrations. Each segment uploads that provenance file. Checkpoints
are saved locally by default, independently of W&B metrics logging.

To also upload checkpoints, set the following in your run configuration:

```json
{
  "logging": {
    "wandb": {
      "enabled": true,
      "upload_checkpoints": true
    }
  }
}
```

`logging.wandb.upload_checkpoints` defaults to `false`, including when omitted
from older configurations or checkpoints. When enabled, each saved
completed-update checkpoint is uploaded once per logging segment with
`step-<N>` and `latest` aliases; the upload waits for confirmation. This includes
periodic saves and the final checkpoint, with duplicate steps uploaded only once.
Recovery uses the setting saved in the checkpoint. This option does not remove
previously uploaded artifacts or change local checkpoint cadence.

Standalone evaluation uses a separate verified W&B run with aggregate metrics
and a result artifact. Dataset source files are never uploaded.

## Inference

```python
from robopolicy import load_policy

inference = load_policy("runs/example/latest.pt", device="cpu")
actions = inference.predict({"state": raw_state})
```

Raw state has shape `(B,T_obs,D_state)` and the declared float32 dtype. Image
policies additionally accept RGB uint8 images `(B,T_obs,N_camera,3,H,W)`.
Indexed policies require int64 task indices with raw shape `(B,1,K)`; the input
packer removes the current-step axis. Entity policies similarly declare entity
features and selection indices. Actions are not required. Prediction returns
raw-scale actions `(B,T_action,D_action)`.

Inference reconstructs the saved architecture with pretrained initialization
explicitly disabled, loads its complete weights, and enters evaluation mode.
It does not access the training dataset, W&B, or pretrained weight servers. Supply
fixed normalized initial `noise` to compare predictions across checkpoints.
The NPZ command-line entry point is `scripts/policies/infer_library.py`.
