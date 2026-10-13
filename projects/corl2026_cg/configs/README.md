# Native CoRL experiment presets

These files are application `ExperimentSpec` JSON containing native RoboPolicy `RunConfig`
objects. Paths are relative to the repository root. Prepare the existing datasets into
`data/corl2026_cg/l4`, `data/corl2026_cg/l4_2s`, and `data/corl2026_cg/l5`, or override
`run.dataset.root`. The image runs use two RGB cameras, current 9-dimensional robot state,
16 seven-dimensional actions, frozen DINOv2-small, and random 224-pixel crops from 256-pixel
images. Every training and evaluation run uses W&B project `corl2026_cg`.

`policy.vision.image_size: 518` describes DINOv2-small's pretrained positional
embedding grid, so its saved weights load exactly. The policy still receives
224-pixel crops (`processing.crop_size`); DINO interpolates that grid during
inference. These settings describe different dimensions and need not match.

## Release baselines

Start with these presets to reproduce the baseline settings advertised in the
[paper release README](../../../README.md#tasks-and-baseline-results). Each generates one
fresh seed-11 job. The [recorded configurations and results](../../../docs/validation/cg_training_evaluation_setups_20261001.md)
are the provenance for these presets; matching measurements also requires the
same dataset contents.

| Preset | Task | Updates | Training demos | Training / evaluation combinations | Evaluation episodes |
| --- | --- | ---: | ---: | --- | ---: |
| [release_pp_50k](release_pp_50k.json) | Pick-and-Place (PP) | 50,000 | 2,000, unbalanced | 16 / 16 | 400 |
| [release_ppp_50k](release_ppp_50k.json) | Pick-Place-Press (PPP) | 50,000 | 6,400, balanced | 64 / 64 | 1,600 |
| [release_2s_pp_100k](release_2s_pp_100k.json) | Dependent Two-Stage Pick-and-Place (2S-PP) | 100,000 | 3,600, balanced | 144 concrete / 48 visible | 1,200 |
| [release_2s_full_100k](release_2s_full_100k.json) | Full-instruction Two-Stage Pick-and-Place | 100,000 | 3,600, balanced | 144 / 144 | 3,600 |

PPP uses `full` and selects 100 training demonstrations from each of all 64 tasks.
The complete dataset contains 300 demonstrations per task; the validation split
is applied before the training budget.
Both two-stage variants train on every concrete combination. Dependent 2S-PP
hides `container0` from the policy and accepts any first destination other than
`container1`; its 25 trials per visible instruction total 1,200 episodes.

Data roots default to `data/corl2026_cg/l4`, `data/corl2026_cg/l5`, and
`data/corl2026_cg/l4_2s`. Both two-stage presets use the latter. The simulator
defaults to `repos/robot_envs` at the pinned revision. Override either path with
`--set run.dataset.root=...` and `--set evaluation.simulator_root=...`.
Jobs are written under `runs/corl2026_cg/release/`, use the invoking Python
interpreter, and train then evaluate serially on GPU 0. W&B remains online under
`yixiao_wang/corl2026_cg`; paths and generated run identities are portable, while
scientific settings match the archived runs.

These are implementation baselines with full training coverage, one training
seed, and 25 evaluation trials per instruction. They are not the sparse-coverage
experiments in the paper. Use the [paper-to-code map](../../../docs/validation/paper-release-audit.md#paper-to-code-map)
to track missing paper configs and results; do not relabel an example or pilot
as paper reproduction.

The other matrices below use different seeds, budgets, or task selections. For example,
`l4_infer_cont0_full` uses seeds 3/4 and 50k updates, and `example_l5_designs`
selects sparse task sets; neither reproduces the measured release baseline settings.

## Experiment matrices

These twelve matrices generate **75 jobs** across task designs, demonstration
budgets, training seeds, and fine-tuning settings.
They preserve earlier experiment recipes, including their differing update and
seed budgets; the paper's full protocol is not encoded by this collection.

| Preset | Jobs | Settings |
| --- | ---: | --- |
| `l4_full` | 3 | Full instructions, seeds 11/22/33, 2,000-demo cap, 50k updates |
| `l4_object_only` | 3 | Object query only, otherwise the same budget |
| `l4_container_only` | 3 | Container query only, otherwise the same budget |
| `l4_container_latents` | 2 | Container plus two constant query slots, seed 11, 50k/100k updates |
| `l4_infer_cont0_full` | 2 | Seeds 3/4, 144 concrete tasks, 3,600 balanced demos, 50k updates |
| `l4_infer_cont0_orthogonal` | 2 | Seeds 3/4, 48 concrete tasks, 3,600 balanced demos, 150k updates |
| `l4_infer_cont0_cov9_B16` | 1 | Seed 0, 48 concrete tasks, 3,600 balanced demos, 50k updates |
| `l5_demo_budgets` | 20 | Full/orthogonal × seeds 44/55 × 64/128/256/512/1,024 demos; 50k updates |
| `l5_orthogonal_smoke_30updates` | 3 | Seeds 11/22/33, 1,800 balanced demos, exactly 30 updates |
| `l5_finetune_full` | 6 | Native final_B10 seed-11 source; seeds 11/22/33 × 64/128 demos; 5k updates |
| `l5_finetune_random_B16` | 15 | Native diag_B4 seed-11 source; task-design seeds 0–4 × training seeds 11/22/33; 16 demos; 5k updates |
| `l5_finetune_random_B32` | 15 | Same matrix with 32 tasks/demos; 5k updates |

The two constant query slots in `l4_container_latents` have vocabulary sizes
`[4, 1, 1]`. These matrices use flow matching with ten Euler sampling steps.
PP uses six Transformer layers; dependent 2S-PP and PPP use eight. Batch size is 64.

## Fine-tuning sources

Run `l5_pretrain_final_B10.json` and `l5_pretrain_diag_B4.json` first. They use seed 11,
1,800 balanced demonstrations, and 100k updates. The fine-tuning matrices refer to the
native `latest.pt` paths resolved by those companion presets. If you change a source
run's dataset root, architecture, budget, or output root, set `finetune_from` to that
new source run's checkpoint. Logging changes such as enabling checkpoint uploads
also change the generated identity. Read the source job's `checkpoint` field
instead of reconstructing its hash or using an old run directory.

`l5_finetune_full` starts from `final_B10`; it does not reproduce Figure 6's
diagonal-to-full adaptation. Add the paper's missing full/orthogonal/scratch
matrices and pretraining-reference evaluation before claiming that comparison.

## Additional examples

`example_l4_design_mask_sweep` covers structured and random L4 designs with `none`/`query`
fusion masks. `example_l4_2s_full` and `example_l4_2s_open_cont` exercise full and partial
stage-set products. `example_l5_designs` includes curated final/logopt, diagonal, and
orthogonal designs. Use the release presets above for the measured baseline settings.

`example_l4_image_improved_mean_flow` and `example_l4_state_improved_mean_flow` show native
one-step improved MeanFlow. The state example requires exported `cg/entities` shaped
`(T, 8, 7)` and the explicit object/container selection fields; entity coordinates must
match evaluation. It has no image features or vision dependency.

## Reactive instruction experiments

The new `pilot_l5_active_imf` and `pilot_l4_2s_active_imf` recipes use 256 balanced
demonstrations and 1,000 image iMF updates with reactive instruction tokens. They
require active-label exports. `example_l4_2s_composition_benchmark` trains four
reconstructed covering/random designs and evaluates the full 144-task universe.
These additions are documented in the [active instruction guide](../docs/active-instructions.md).

`l5_orthogonal_smoke_30updates` runs exactly 30 training updates for a smoke check.
Use the release presets for the 50k/100k baseline runs.
