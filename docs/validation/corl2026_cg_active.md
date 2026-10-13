# Reactive CG instruction validation

Validated on 2026-09-29, implementation commit `8c6ba81` on `corl2026_cg`.
The implementation passes integration checks and both real training pilots
completed, but **neither policy completed a task or its first stage**. These
results do not establish policy quality or convergence.

Exact metrics, profile digests, dataset fingerprints and authenticated W&B
verification are preserved in [the result record](corl2026_cg_active_results.json).
The earlier [application migration report](corl2026_cg.md) remains historical.

## Implemented behavior

Task-specific code stays under `projects/corl2026_cg`; this feature work changes
no core library code. A shared physical-state tracker supplies reactive subgoal
labels offline and during simulator evaluation. Versioned profiles, immutable
dataset exports and checkpoint validation prevent incompatible conditioning from
being mixed. Changed subgoals invalidate the affected environment's queued actions.
Composition membership remains separate from the current active token.

Training and evaluation designs can be selected independently. New full-instruction
L4 designs include minimum pairwise coverage (16 rows), greedy pairwise coverage
(18), minimum threewise coverage (48), and seeded single-factor coverage designs.
Reports expose factor coverage and unseen pairs. These definitions reconstruct
the intended features; they do not claim exact parity with unavailable old patches.
See the [feature guide](../../projects/corl2026_cg/docs/active-instructions.md).

## Executed checks

| Check | Result |
|---|---|
| Complete core suite, including CUDA, vision and real decoders | 275 passed |
| Complete application suite, including real simulator integration | 286 passed, no skips |
| Core / application mypy | 51 / 32 source files passed |
| Ruff lint and formatting | Passed |
| Core and application source/wheel builds | Passed; package boundaries verified |
| External simulator tracked source | Unchanged |
| Six W&B run identities and URLs | Authenticated API verification passed; all finished |

Tests cover tracker regressions and hysteresis, hidden-container invariance,
profile and checkpoint compatibility, atomic preparation, video alignment,
composition coverage, evaluation membership and reporting. The environment was
Python 3.10.19 with two RTX 5080 GPUs and simulator revision
`c7e06a0784853d74db57ebf78d7a66fa40c2df99`.

## Real data and label audit

Both pilots used 256 balanced training demonstrations, seed 11, the image improved
MeanFlow policy, frozen DINOv2-small, batch size 64, learning rate `1e-4`, a cosine
schedule and 100 warmup updates. The model receives two RGB cameras, robot state
and one active token with vocabulary size 12. Active rollout instructions come
from physical simulator state; this is additional supervision when comparing
against static instruction policies.

| Pilot | Source demonstrations | Training compositions | Audited observations | Label mismatches |
|---|---:|---|---:|---:|
| L5 `diag_B4`, `l5_v1` | 18,900 across 63 tasks | 4 visible, 64 demos each | 80,594 | 0 |
| L4 `orthogonal`, `l4_infer_cont0_v1` | 14,400 across 144 tasks | 48 concrete / 16 visible | 101,948 | 0 |

The shared tracker recomputed every selected training label exactly. Every
selected demonstration visited all three phases. Four L5 and seven L4
demonstrations contained reactive returns to earlier phases. Eight annotated
source videos were generated after the alignment correction below; selected
transition frames from four demonstrations were visually reviewed.

Exports are local at `/mnt/ssd/yixiao/cg/l4/prepared/cg_l5_active_v1` and
`/mnt/ssd/yixiao/cg/l4/prepared/cg_l4_infer_cont0_active_v1`. Original numeric
datasets were preserved, and video files were hard-linked. Datasets, checkpoints,
generated videos and the reference images in `tmp_feature/` remain outside Git.
The reference images were never staged or uploaded; `/tmp_feature/` is ignored.

## Training and rollout results

Each pilot trained to update 500, saved a native checkpoint, then resumed through
`cg train --resume` to update 1,000. Both had zero skipped updates, finite model
weights and optimizer states at step 1,000. In each model, 288 of 511 saved tensors
changed between updates 250 and 1,000; the frozen backbone accounts for many
unchanged tensors. W&B retained one training run with separate resume segments.

| Pilot | Completed updates | Training elapsed | ID success | OOD success | Overall success |
|---|---:|---:|---:|---:|---:|
| L5 | 1,000 | 396.60 s | 0/4 | 0/60 | 0/64 |
| L4 infer-cont0 | 1,000 | 398.98 s | 0/16 | 0/32 | 0/48 |

Training elapsed includes startup, checkpointing, resume and logging. Evaluation
used one trial per visible composition, four parallel environments, horizon 400,
and up to eight queued actions. All 112 episodes reached 400 steps, returned zero
reward and failed with `stage0_incomplete`. No policy reached phase 2. Two L5
episodes briefly entered phase 1 and returned to phase 0; L4 remained in phase 0.
For example, L5 `cross_into_mug_press_yellow` changed token `0 -> 5` at step 45
and `5 -> 0` at step 51. These are geometric tracker events, not force-confirmed
grasps.

Video inspection found matching camera order, orientation and RGB conventions,
and no obvious duplicate action scaling or instruction wiring defect. Failed
rollouts showed unsuccessful grasping. Undertraining is plausible, but these
pilots cannot establish its cause. Only four rollout videos per pilot were saved.

L4's 48 evaluation queries deduplicate the hidden first container; one trial
does not cover all 144 concrete initializations. Its visible training design
covers all 44 feasible pairs, with 16/48 visible triples. L5 covers 12/48 pairs
and 4/64 triples; all 60 OOD compositions include unseen pairs. The pilots therefore
do not isolate the same form of compositional generalization.

## Offline losses at update 1,000

Each split uses 500 sampled windows, batch size 50 and metric/sampling seed 0.
Membership uses complete visible compositions, independent of the active token
and the rollout evaluation design. These metrics average sampled windows, not
per-task averages; the sample does not guarantee representation of every available
composition.

| Pilot | Composition split | Sampled / available concrete tasks | iMF objective | Normalized action MSE | Raw action MSE |
|---|---|---:|---:|---:|---:|
| L5 | Trained | 4/4 | 0.991280 | 0.048658 | 0.047426 |
| L5 | Unseen | 2/59 | 0.989958 | 0.043201 | 0.042056 |
| L4 infer-cont0 | Trained | 9/48 | 0.997660 | 0.110772 | 0.110772 |
| L4 infer-cont0 | Unseen | 3/96 | 0.997511 | 0.107171 | 0.107171 |

Sampled task counts were reconstructed from the deterministic native window/block
plan used by loss evaluation. The first 500 windows have particularly narrow
unseen-task coverage; these pilot losses are not full-composition benchmarks.

The adaptive iMF objective divides error by its detached magnitude with default
exponent 1, so values close to one are not evidence of good predictions. Action
MSE is reported separately. These are within-pilot measurements, without a
matched static-policy baseline.

The L5 dataset lacks `milk_into_mug_no_handle_press_blue`: the unseen loss sampling
pool contains 59 of 60 requested compositions. Simulator evaluation still covers all
64 compositions. L4 offline data covers all 48 trained and 96 unseen concrete
compositions, grouped by the visible task for membership.

## Findings corrected during validation

The first pilot startup exposed a DINO positional-grid mismatch: a 224-pixel
declared model grid expected 257 positions, while the pretrained checkpoint stores
1,370. Native CG DINO recipes now declare the pretrained 518-pixel grid and keep
224-pixel input crops. The existing model interpolates positions at runtime.
Failed startup attempts completed zero updates and are excluded from these results.

The legacy L4 converter assigned numeric timestamps at an assumed 50 Hz and video
timestamps at 20 FPS, although the collector saved observations together. The
initial new inspection utility incorrectly matched those conflicting clocks.
It now uses the same `frame_start + step` indexing as training, checks frame counts,
and reports clock differences diagnostically. QA videos were regenerated; training
alignment and source observations were unchanged.

One preexisting distribution uncertainty remains: the current L4 collection script
uses broader placement bounds (`y` to +/-0.28, minimum distance 0.01) than the
migrated evaluator (`y` to +/-0.20, minimum distance 0.02). The source collection's
exact generator revision is unavailable, so collector/evaluator distribution
parity is not established. This does not prove the cause of the failed pilots.

## Artifacts and reproduction

All detailed local artifacts are under `runs/corl2026_cg_validation_20260929/`:
`label_audit.json`, `checkpoint_audit.json`, `loss_sampling_audit.json`,
`wandb_verified.json`, corrected
`instruction_qa/`, `video_diagnostics/`, and generated `report/` JSON/CSV/Markdown.
Successful job directories under `pilots/` are:

- `pilot_l5_active_imf_diag_B4_69730724c5f9_seed11`
- `pilot_l4_2s_active_imf_orthogonal_b2e1cd134244_seed11`

| Pilot | Training | Rollouts | Offline losses |
|---|---|---|---|
| L5 | [cc758d1a](https://wandb.ai/yixiao_wang/corl2026_cg/runs/cc758d1a) | [abd6e1cd](https://wandb.ai/yixiao_wang/corl2026_cg/runs/abd6e1cd) | [5e52a0ec](https://wandb.ai/yixiao_wang/corl2026_cg/runs/5e52a0ec) |
| L4 | [7935d320](https://wandb.ai/yixiao_wang/corl2026_cg/runs/7935d320) | [a8df65aa](https://wandb.ai/yixiao_wang/corl2026_cg/runs/a8df65aa) | [4cb4596d](https://wandb.ai/yixiao_wang/corl2026_cg/runs/4cb4596d) |

The [pilot recipes](../../projects/corl2026_cg/configs/README.md) and feature guide
document dataset preparation and ordinary `cg run` execution. To repeat the
resume check, generate a job with `cg generate`, validate its profile and data,
load its native `RunConfig`, call `Trainer(config).fit(until_step=500)`, then run
`cg train --resume <job-directory>/train/latest.pt`. Keep `max_updates=1000`
through both segments so the schedule remains unchanged. The exact local harness
is `runs/corl2026_cg_validation_20260929/train_resume_pilot.py`.

After training, run:

```bash
cg run --jobs <job-directory>/experiment.json --eval-only
cg evaluate-loss --job <job-directory>/experiment.json --samples 500 --batch-size 50
cg report --jobs <l5-job-directory> <l4-job-directory> --output runs/pilot-report
```

The [earlier report's validation commands](corl2026_cg.md#reproduction) also
apply to the expanded test suite. No full benchmark sweep was run in this feature
validation.
