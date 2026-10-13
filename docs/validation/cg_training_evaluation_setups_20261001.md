# CG training configurations and evaluation setup

This document records the configurations behind the Pick-and-Place (PP), Pick-Place-Press (PPP), Dependent Two-Stage Pick-and-Place (2S-PP), and additional full-instruction two-stage release baselines. The [release README](../../README.md) introduces the tasks and provides setup and reproduction commands. The full-instruction two-stage baseline uses **100,000 updates**, which achieved 62.44% success; its 50,000-update counterpart achieved 23.67%.

All four release baselines and both 50k comparisons have completed training and evaluation. PPP trained on all 64 task combinations. Local stage records and all twelve exact W&B run identities were verified as complete/finished at 2026-10-02T01:47:27.096225+00:00. All percentages below are rollout success rates.

## Baseline results

| Model | Training updates | Training tasks | Training demos | Demos per task | Evaluation success |
| --- | ---: | ---: | ---: | --- | ---: |
| PP | 50,000 | 16 | 2,000 | 110–142, unbalanced | 77.75% (311/400) |
| PPP | 50,000 | 64 | 6,400 | 100, balanced | 71.31% (1,141/1,600) |
| Full-instruction two-stage | 100,000 | 144 | 3,600 | 25, balanced | 62.44% (2,248/3,600) |
| Dependent 2S-PP | 100,000 | 144 concrete / 48 visible | 3,600 | 25 per concrete task, balanced | 45.58% (547/1,200) |

Full-instruction two-stage achieved 81.06% first-stage success. These runs use training seed 11 and are individual seeded results, not averages across training seeds.

## Relationship to the paper

The [paper](https://arxiv.org/abs/2607.29687) calls `l4` / `full` **PP**,
`l5` / `full` **PPP**, and `l4_2s` / `infer_cont0` **Dependent 2S-PP**.
The `l4_2s` / `full` run is an additional full-instruction baseline. These
measurements validate the released implementation under the settings below;
they do not reproduce every paper experiment or figure.

The paper generally reports a mean across three training seeds and 30 trials per
task; its 48-instruction static dependent-task appendix uses five training seeds
and 10 trials per instruction. Here each
baseline uses seed 11 and 25 trials per task or visible instruction. The paper's
main dependent-task study uses 36 visible instructions and a restricted container
vocabulary; this release baseline uses all 48. Training budgets also differ:
these PPP and dependent 2S-PP baselines use 50k and 100k updates, while the paper's
appendices describe 100k and 150k, respectively.

The current `pc_same_container_violation` metric divides violations by **all
rollouts** (85/1,200 = 7.08% for dependent 2S-PP at 100k). The paper defines a
conditional rate among completed first placements, including wrong destinations.
These denominators differ;
the release metric must not be compared directly to that paper metric.
The [paper release audit](paper-release-audit.md#measurement-contracts-to-resolve)
details the missing placement-event accounting, fine-tuning reference splits,
and other requirements for matching the reviewed manuscript.

## Training-budget comparisons and remaining failures

| Instruction | 50k updates | 100k updates | Change |
| --- | ---: | ---: | ---: |
| Full-instruction two-stage | 23.67% (852/3,600) | 62.44% (2,248/3,600) | +38.78 percentage points |
| Dependent 2S-PP | 10.25% (123/1,200) | 45.58% (547/1,200) | +35.33 percentage points |

Differences use unrounded rates. These are individual seed-11 runs evaluated on
the same task sets within each instruction variant; they are not multi-seed estimates.
The infer_cont0 100k run trained from scratch, rather than resuming the 50k checkpoint.
The full and infer_cont0 variants use different instructions and evaluation sets.

The final infer_cont0 result replaces the earlier partial 52.99% over 536 episodes.
At 100k, first-stage success is 60.00%; same-container violations occur in 7.08%
of rollouts (85/1,200). Of 653 failures, 395 are `stage0_incomplete`, 173 are
`stage1_incomplete`, and 85 are `same_container`. All 48 visible instructions
are represented in training, so this is not an unseen-composition result.
The [configuration snapshot](cg_training_evaluation_setups_20261001.json) preserves
final summaries, the two 50k comparisons, and current W&B verification.

## Evaluation tasks and rollout counts

Objects are `cross`, `cube`, `cylinder`, and `milk`. Containers are `bin`, `mug`, `plate`, and `mug_no_handle`. PPP buttons are `red`, `green`, `blue`, and `yellow`.

| Model | Evaluation combinations | Tasks or visible instructions | Rollouts per task or instruction | Total rollouts | Maximum environment steps |
| --- | --- | ---: | ---: | ---: | ---: |
| PP | Object × container | 16 | 25 | 400 | 400 |
| PPP | Object × container × button | 64 | 25 | 1,600 | 400 |
| Full-instruction two-stage | Ordered placements with distinct objects and distinct containers | 144 | 25 | 3,600 | 800 |
| Dependent 2S-PP | First object × distinct second object × second container | 48 | 25 | 1,200 | 800 |

- **PP:** place the specified object in the specified container. All 16 combinations are present in training.
- **PPP:** place the specified object in the specified container, then press the specified button. Training and evaluation both cover all 64 combinations using the `full` task set.
- **Full-instruction two-stage:** place object 0 in container 0, then object 1 in container 1. The objects and containers differ across stages: 4 × 4 × 3 × 3 = 144 combinations. For example, `cross_into_bin__cube_into_mug`. All 144 combinations are in training. The 50k and 100k runs have identical evaluation task sets.
- **Dependent 2S-PP (`infer_cont0`):** the instruction specifies object 0, object 1, and container 1, leaving container 0 unspecified. Place object 0 in any container other than container 1, then object 1 in container 1. There are 4 × 3 × 4 = 48 visible instructions. Each receives **25 total rollouts**, cycling the simulator target for the first placement across its three valid containers (9/8/8 trials). Success accepts any valid first container; placing object 0 in container 1 latches a same-container violation. Training covers all 144 concrete combinations, hence all 48 visible instructions.

All evaluations use eight parallel environments, seed 0, RGB observations from `agentview` and `robot0_eye_in_hand`, and eight executed actions per prediction from a 16-action chunk. Each rollout receives its own reset seed from `numpy.random.SeedSequence(0)`. The flow-matching solver uses 10 Euler steps with fp32 inference and compilation disabled. Episodes end on success, environment termination, or the horizon. Early successes therefore use fewer steps.

The simulator is `repos/robot_envs` at revision `c7e06a0784853d74db57ebf78d7a66fa40c2df99`. Evaluation uses GPU 0, saves up to four rollout videos, and logs separate online W&B evaluation runs under `yixiao_wang/corl2026_cg`.

## Training settings

All four recorded configurations use batch size 64; learning rate 1e-4; weight decay 1e-4; 1,000 warmup updates; cosine decay with minimum learning-rate ratio 1e-4; AMP; gradient clipping at norm 1.0; and training seed 11. Validation runs every 5,000 updates over 10 batches, and checkpoints are saved every 10,000 updates. Dataset validation fraction is 0.01; split and subset seeds are both 0. All four jobs have `finetune_from: null`, so these are fresh training runs.

The shared architecture uses frozen pretrained `facebook/dinov2-small`, two cameras, random 224 × 224 crops with ImageNet RGB normalization, four cross-attention fusion layers (dimension 256, eight heads, feedforward dimension 512, dropout 0.1), and a Transformer flow-matching head (dimension 512, eight heads, feedforward dimension 2048, dropout 0.1). The head has **six layers for PP** and **eight layers for PPP and both two-stage configurations**.

Each policy observes one timestep, has nine state dimensions and seven action dimensions, and predicts 16 actions. Static categorical conditioning contains object/container for PP; object/container/button for PPP; both object/container pairs for full-instruction two-stage; and object 0/object 1/container 1 for infer_cont0. Each categorical vocabulary has size four. Ordinary flow matching uses a linear path, uniform time sampling, and 10 Euler inference steps.

The loader uses eight persistent workers, prefetch factor four, spawn multiprocessing, and batch size 64. Data use the native manifest-indexed dataset, OpenCV video decoding, and window-major sampling (eight passes, sampling seed 42, 10 window shards, block size 64).

| Model | Prepared dataset root |
| --- | --- |
| PP | `/mnt/ssd/yixiao/cg/l4/prepared/l4_single_stage_native_v1` |
| PPP | `/mnt/ssd/yixiao/cg/l4/cg_l5` |
| Full-instruction two-stage and infer_cont0 | `/mnt/ssd/yixiao/cg/l4/prepared/cg_l4_infer_cont0_active_v1` |

The local PPP dataset contains all 64 tasks with 300 episodes each (19,200 total), including training conditioning features and active instruction labels. The release run selects 100 training episodes per task after splitting, for 6,400 training episodes and 210 separate validation episodes.

The dataset directory name does not determine instruction visibility. Full-instruction two-stage explicitly reads all four static task fields; infer_cont0 reads only object 0, object 1, and container 1.

## Configuration files and result artifacts

Use the portable release presets for new jobs:

| Task | Release preset |
| --- | --- |
| PP | [release_pp_50k](../../projects/corl2026_cg/configs/release_pp_50k.json) |
| PPP | [release_ppp_50k](../../projects/corl2026_cg/configs/release_ppp_50k.json) |
| Dependent 2S-PP | [release_2s_pp_100k](../../projects/corl2026_cg/configs/release_2s_pp_100k.json) |
| Full-instruction two-stage | [release_2s_full_100k](../../projects/corl2026_cg/configs/release_2s_full_100k.json) |

These generate one seed-11 job each using the settings above.
Dataset/simulator paths, interpreter paths, output destinations, and generated run
identities are adapted for portability. The snapshot records each current baseline's
resolved job and measurement status. Dataset download and preparation instructions
are in the [release README](../../README.md#datasets-and-checkpoints).

Paths below are local artifact references relative to the original repository root; they are not download links. `run.json` is the resolved native training configuration; `experiment.json` also records the exact training/evaluation task lists, execution settings, and checkpoint path. Local files under `runs/` are excluded from Git. The [configuration snapshot](cg_training_evaluation_setups_20261001.json) preserves the complete resolved experiments, task lists, run identities, and result evidence alongside this document.

### PP

- Training configuration: `runs/l4_1s_baseline_20260929/l4_1s_baseline_20260929_full_afab9aa9e862_seed11/run.json`.
- Experiment and evaluation configuration: `runs/l4_1s_baseline_20260929/l4_1s_baseline_20260929_full_afab9aa9e862_seed11/experiment.json`.
- Final results: `runs/l4_1s_baseline_20260929/l4_1s_baseline_20260929_full_afab9aa9e862_seed11/evaluation/eval_summary.json`.
- W&B train: [https://wandb.ai/yixiao_wang/corl2026_cg/runs/28c667fe](https://wandb.ai/yixiao_wang/corl2026_cg/runs/28c667fe).
- W&B evaluate: [https://wandb.ai/yixiao_wang/corl2026_cg/runs/1b0cf331](https://wandb.ai/yixiao_wang/corl2026_cg/runs/1b0cf331).

### PPP

- Training configuration: `runs/corl2026_cg/ppp_full64_20261001/release_ppp_50k_full_71e63eaff838_seed11/run.json`.
- Experiment and evaluation configuration: `runs/corl2026_cg/ppp_full64_20261001/release_ppp_50k_full_71e63eaff838_seed11/experiment.json`.
- Status: training and evaluation complete; all 64 tasks, 6,400 training episodes; 71.31% (1,141/1,600) success.
- Final results: `runs/corl2026_cg/ppp_full64_20261001/release_ppp_50k_full_71e63eaff838_seed11/evaluation/eval_summary.json`.
- W&B train: [eb26260b](https://wandb.ai/yixiao_wang/corl2026_cg/runs/eb26260b); exact run identity verified.
- W&B evaluate: [1a782b0e](https://wandb.ai/yixiao_wang/corl2026_cg/runs/1a782b0e); exact run identity verified.

### Full-instruction two-stage

- Training configuration: `runs/full_training_20260929/jobs/l4_2s_full_100k_full_caddf1c97aa7_seed11/run.json`.
- Experiment and evaluation configuration: `runs/full_training_20260929/jobs/l4_2s_full_100k_full_caddf1c97aa7_seed11/experiment.json`.
- Final results: `runs/full_training_20260929/jobs/l4_2s_full_100k_full_caddf1c97aa7_seed11/evaluation/eval_summary.json`.
- W&B train: [https://wandb.ai/yixiao_wang/corl2026_cg/runs/8cb425de](https://wandb.ai/yixiao_wang/corl2026_cg/runs/8cb425de).
- W&B evaluate: [https://wandb.ai/yixiao_wang/corl2026_cg/runs/a25e2510](https://wandb.ai/yixiao_wang/corl2026_cg/runs/a25e2510).

### Dependent 2S-PP

- Training configuration: `runs/infer_cont0_100k_20261001_012512/jobs/infer_cont0_full_100k_full_2506943c897e_seed11/run.json`.
- Experiment and evaluation configuration: `runs/infer_cont0_100k_20261001_012512/jobs/infer_cont0_full_100k_full_2506943c897e_seed11/experiment.json`.
- Final results: `runs/infer_cont0_100k_20261001_012512/jobs/infer_cont0_full_100k_full_2506943c897e_seed11/evaluation/eval_summary.json`; 45.58% success (547/1,200), evaluation complete.
- W&B train: [2be8eceb](https://wandb.ai/yixiao_wang/corl2026_cg/runs/2be8eceb).
- W&B evaluate: [f5d39012](https://wandb.ai/yixiao_wang/corl2026_cg/runs/f5d39012).

The final checkpoint for each experiment is `train/latest.pt` under its job directory. Completed summaries verify checkpoint steps of 50,000 for PP and PPP, and 100,000 for both two-stage variants.

The earlier source recipes were `projects/corl2026_cg/configs/l4_full.json`, `projects/corl2026_cg/configs/example_l5_designs.json`, `projects/corl2026_cg/configs/example_l4_2s_full.json`, and `projects/corl2026_cg/configs/l4_infer_cont0_full.json`. Use the release presets above for the measured baseline settings: the earlier recipes can differ in training budgets, task selection, dataset roots, and update counts. In particular, the successful full two-stage run uses 100,000 updates.

## Release documentation and config validation

Checked on 2026-10-01 in the existing Python 3.10.19 / PyTorch 2.9.1+cu128
environment. These checks validate the release presets and documentation; the
PPP preset now additionally selects all 64 tasks with 100 training episodes each.

| Check | Result |
| --- | --- |
| Full core suite, including CUDA, vision, and both decoders | 275 passed |
| Full application suite, with pinned simulator and EGL explicitly enabled | 298 passed, no skips |
| Release preset settings | All four match their completed runs after excluding paths and generated run identities |
| Documented generation commands | Four dry-runs passed; one seed-11 job each, with 400 / 1,600 / 1,200 / 3,600 evaluation episodes for PP / PPP / dependent 2S-PP / full-instruction two-stage |
| Core and application Ruff lint / formatting | Passed |
| Core and application mypy | 51 / 32 source files passed |
| Core and application source / wheel builds | Passed |
| Documentation links, heading anchors, code-block syntax, and diff whitespace | Passed |

The required checks are listed in the [repository contributor guide](../../AGENTS.md)
and [application validation guide](../../projects/corl2026_cg/README.md#validation).
For the application suite, `CG_SIMULATOR_ROOT` pointed to the pinned local checkout
and `MUJOCO_GL=egl` enabled real simulator acceptance. Hugging Face dataset and
checkpoint verification remains open in the [release audit](paper-release-audit.md#public-artifacts).
