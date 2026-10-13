# Single-stage L4 baseline — launched 2026-09-29

Updated at 2026-10-01T20:27:30.120175+00:00. Training and evaluation are complete:
**77.75% success (311/400)** at 50,000 updates, over 16 tasks with 25 trials each.
All tasks were represented in training; 89 rollouts failed with `stage0_incomplete`.
Local stage records report `complete`; the authenticated W&B API verified both
[training](https://wandb.ai/yixiao_wang/corl2026_cg/runs/28c667fe) and
[evaluation](https://wandb.ai/yixiao_wang/corl2026_cg/runs/1b0cf331) as `finished`.
See the [current results and setup](cg_training_evaluation_setups_20261001.md)
for the final evaluation evidence and other completed baselines. Historical
performance parity remains unverified.

- W&B: https://wandb.ai/yixiao_wang/corl2026_cg/runs/28c667fe
- Job: `runs/l4_1s_baseline_20260929/l4_1s_baseline_20260929_full_afab9aa9e862_seed11/experiment.json`
- Detached pipeline PID at launch: `1576351`; training and subsequent evaluation use GPU 0.
- Source recipe: `dev_2s_1s` commit `7455b6c62434c1cdf197e9d3cbb7d21ab3443cf6`,
  `src/policies/config/1s/l4_1s_full_transformer_cross_masknone_L6_seed11_2token_original.json`.
  The similarly named recipe without `2token_original` uses object-only instructions.

The native `l4_full.json` preset uses full object/container instructions, all 16 tasks,
seed 11, 2,000 training demonstrations, batch size 64, 50,000 updates, learning rate
1e-4, weight decay 1e-4, 1,000 warmup updates and cosine decay. Architecture: frozen
DINOv2-small, two cameras, random 224-pixel crops, four cross-attention fusion layers,
six Transformer head layers, 16-action chunks and ordinary flow matching with ten
Euler inference steps. Validation runs every 5,000 updates; checkpoints every 10,000.

The original `/mnt/ssd/yixiao/cg/l4/single_stage_5demo` contains 8,000 demonstrations
(500 per task despite the directory name). `cg prepare` created an immutable export
at `/mnt/ssd/yixiao/cg/l4/prepared/l4_single_stage_native_v1`, adding explicit task
fields without modifying the source. Runtime uses the current native
`robopolicy.datasets.factory.PolicyDataset`, manifest-indexed HDF5 numeric shards,
MP4 video, finite window-major epochs and bounded block decoding. The legacy loader
and frame cache are not used. Selection validated 2,000 training demos, 70 validation
demos and 356,219 eligible training windows. The exact per-task selection is recorded
in the job's `launch_audit.json` and training provenance.

The detached `cg run` pipeline completed evaluation after successful training:
25 trials per task, 16 tasks (400 episodes), horizon 400, eight parallel environments,
eight executed actions per inference, seed 0, four saved videos, W&B enabled.
All tasks are in-distribution for this full-training baseline.

The new core's sampling, normalization, initialization and RGB-consistent evaluation
remain authoritative. This matches the historical recipe's principal parameters,
not its exact trajectory or a verified historical score. The RobotDecisionMaking
checkout was not located, so equality with its exact loader revision is unverified.
No exact matching historical rollout result was found in the inspected local results.

Monitor `train/metrics.jsonl`, `train.log`, `train_status.json`, then
`evaluate_status.json` and evaluation outputs inside the job directory. The pipeline
does not require this chat session to remain open. Dataset/checkpoint artifacts and
the local reference images remain excluded from Git.
