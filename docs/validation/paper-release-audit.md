# Paper release audit

Reviewed on 2026-10-12 against the 22-page local manuscript
`projects/corl2026_cg/corl_2026_cg-3.pdf`, source branch `corl2026_cg`, and baseline
commit `bfe269a`. The manuscript SHA-256 is
`7479cf9ffa6f0772f790a0f46526107777d5e3eff010465640ca63d2a35964da`.
Figure and appendix references below refer to that manuscript.

The intended publication destination is
[Diagnosing_Compositional_Generalization_Robot](https://github.com/YixiaoWang7/Diagnosing_Compositional_Generalization_Robot).
This review prepares the branch; it does not publish code, models, or website
changes. The experiment map and artifact review below track remaining release
work. The code is an implementation with partial reproduction evidence, not yet
a complete reproduction of the manuscript.

## Paper-to-code map

Source module paths below are relative to
[`projects/corl2026_cg/src/corl2026_cg/`](../../projects/corl2026_cg/src/corl2026_cg/).
Preset names refer to the application's [configs](../../projects/corl2026_cg/configs/README.md).

| Paper experiment | Available implementation | Work remaining |
| --- | --- | --- |
| PP/PPP coverage sweeps, Figure 2; Appendix C | `tasks/l4_designs.py`, `tasks/l5_designs.py`, task selection, matrix generation, per-task rollouts | Publish exact task-design draws, fixed demonstration allocations, seed lists, full run matrix, and native results. `example_*` configs cover only subsets. |
| PPP pairwise coverage, Figures 3, 5, 10, 17, 18 | `TaskSet.coverage()` records factor/pair/triple support; the 16-task orthogonal design is implemented | Add success grouped by 0/1/2/3 seen pairs, sample counts, task heatmaps, and figure/HTML generation from released artifacts. |
| Full versus orthogonal data budgets, Figure 4 | `l5_demo_budgets.json`: 64/128/256/512/1,024 demos | Align 50k updates and two seeds with the paper protocol; publish matched results and selections. |
| Sparse pretraining and fine-tuning, Figure 6; Appendix D | Native Trainer fine-tuning; `l5_pretrain_diag_B4`, `l5_pretrain_final_B10`, full and random-B16/B32 fine-tuning presets | Add diagonal-to-full and orthogonal matrices, B6/B8/B16 pretraining comparisons, scratch controls, and 16/32-demo orthogonal controls. Preserve pretraining membership for evaluation. The existing full-FT source is `final_B10`. |
| Dependent 36-instruction study, Figures 7, 14 | Infer-cont0 instruction and success logic; legacy Cov4/6/9/12 generators | Add the explicit restricted 36-instruction universe and verify exact fixed-B12 task lists. Existing generators and evaluation use four destination values and the 48-instruction universe. |
| Dependent 48-instruction study, Figures 15, 16 | Infer-cont0 `obj0_cont1_cov{k}_B16` and `orthogonal` generators | Complete the Cov6/9/12/16, fixed-B16, 150k-update matrix with five seeds and 10 trials per instruction; publish conditional violation and failure analysis. Current presets mix 50k/150k updates and one/two seeds. |
| GT-stage versus static instruction, Figure 8 | `instructions/`, shared offline/online profiles, active-token conditioning, action-queue invalidation | Add matched ordinary flow-matching configs, full training and evaluation, and figure provenance. Checked-in active presets use improved MeanFlow, 256 demos, 1,000 updates, and one trial per instruction. |
| Trial-count sensitivity and PP omission, Table 1; Appendix E | Configurable trial counts; `l4_full` and `l4_object_only` presets | Publish the 5/10/15/30-trial study and five-seed PP omission comparison. Existing PP matrices have three seeds. |
| Human-demonstration benchmark, Appendix C | No Isaac Lab task, 27-task vocabulary, dataset, or reproduction recipe in this application | Release the environment, 30-demo-per-task data, orthogonal-B9 training, and three-seed/30-trial evaluation, or explicitly exclude this experiment from release scope. |

Ordinary flow matching, a frozen DINOv2 image encoder, instruction-query fusion,
and a Transformer action head are implemented by the core library. Generic
improved MeanFlow, state-policy examples, full-instruction two-stage composition
benchmarks, and the deferred interaction-model archive are additional work;
their existence does not reproduce a paper experiment.

## Measurement contracts to resolve

**Seeds and averaging.** Appendix B specifies a mean of per-seed success rates,
normally three training seeds and 30 trials per task. Appendix E specifies five
seeds and 10 trials for the 48-instruction static comparison. `cg report` records
means and sample standard deviations for compatible run identities. Publish
per-seed counts and explicitly state any SE or pooled-rate calculation used by
paper/website plots. The four [recorded baselines](cg_training_evaluation_setups_20261001.md)
use one training seed and 25 trials, with every visible instruction in training.

**Training and data.** Appendix B describes generated pools of 8,000 PP,
19,200 PPP, and 14,400 two-stage demonstrations; a pool size is not an experiment's
selected training budget. For coverage comparisons, fix the total selected
demonstrations and allocate them uniformly across training tasks. Splitting
occurs before training caps in this implementation; publish actual memberships
and any quota remainders/shortfalls. PPP uses 100k updates and dependent 2S-PP
150k in the paper; native release baselines use 50k and 100k respectively.
Appendix D specifies 100k pretraining and 5k fine-tuning with batch size 64.

**Dependent instruction population.** The policy sees `(object0, object1,
container1)` and may place the first object in any other container. Four
destination values yield 48 visible instructions and 144 concrete training
combinations. Three destination values yield the paper's main 36-instruction
study. Merely reducing the number of training tasks does not restrict the
evaluation universe. Add and version explicit population semantics before
claiming Figure 7 reproduction; do not rename current 48-task designs as 36-task
designs.

**Conditional same-container rate.** In
[`evaluation/runner.py`](../../projects/corl2026_cg/src/corl2026_cg/evaluation/runner.py),
`pc_same_container_violation` is `100 * violations / all_rollouts`. The paper
conditions on completing the first placement, including placing in a wrong
container. In [`evaluation/success.py`](../../projects/corl2026_cg/src/corl2026_cg/evaluation/success.py),
`stage0_done` records a valid first placement, while a violation is latched
separately. Add an explicit first-placement event, disjoint outcomes, numerator
and denominator counts, and tests for no placement, wrong placement, recovery,
and empty populations. Retain the old metric name/meaning for old evidence;
document any new result schema rather than silently relabeling prior rates.

**Fine-tuning membership.** Current rollout validation requires task membership
to agree with the evaluated checkpoint's own training selection. Full-set
fine-tuning therefore makes all 64 tasks ID. Figure 6 compares tasks unseen
during pretraining before and after adaptation. Add a separately named reference
partition with source-checkpoint/task-list provenance; retain actual training
membership and its consistency checks. Reclassifying only report labels would
hide this distinction.

**Offline loss.** Section 5.2 describes losses over 1,000 demonstrations. Current
[`evaluate_losses`](../../projects/corl2026_cg/src/corl2026_cg/experiments/losses.py)
samples windows, defaults to 500 per partition, and uses training demonstrations
partitioned by visible task identity. `--samples 1000` does not select 1,000
distinct demonstrations. Define the intended demo/window weighting, publish
sampled task/demo IDs and seed, verify OOD coverage, and distinguish objective
loss from sampled-action MSE. The [historical 500-sample artifact](../../projects/corl2026_cg/docs/historical/README.md)
is not native reproduction evidence.

**GT indicator evidence.** The [active-instruction guide](../../projects/corl2026_cg/docs/active-instructions.md)
documents shared geometric predicates and two-observation debounce, consistent
with the mechanism described in Appendix E. Its versioned profiles still require
experiment provenance. The [two completed iMF pilots](corl2026_cg_active.md)
had 0/64 and 0/48 rollout successes; they demonstrate integration only. Use
matched flow-matching runs and record oracle inputs and action-chunk behavior.

## Public artifacts

Unauthenticated link checks were performed on 2026-10-12. Reachability does not
establish equivalence to the reviewed manuscript or training data.

| Resource | Observation | Release action |
| --- | --- | --- |
| [arXiv record](https://arxiv.org/abs/2607.29687) and [PDF](https://arxiv.org/pdf/2607.29687) | Record title and authors match; record lists v1 | Publish/identify the version matching the reviewed manuscript. |
| [Project website](https://yixiaowang7.github.io/Diagnosing_Compositional_Generalization_Robot_Page/) | Reachable; Code points to the intended public destination; Dataset and Supplementary use `href="#"` | Replace placeholder targets with released data and supplementary artifacts; update citation and release links. |
| [Website PDF](https://yixiaowang7.github.io/Diagnosing_Compositional_Generalization_Robot_Page/corl_2026_cg.pdf) | 19 pages; SHA-256 `e24b330bd9b1a5fc19d4324019c455bff45c082369093d0e35cb8461204bfd65` | Replace with, or clearly version alongside, the reviewed 22-page manuscript; resolve its Section 5.2 `Figure ??` reference before publication. |
| [Public release repository](https://github.com/YixiaoWang7/Diagnosing_Compositional_Generalization_Robot) | Reachable; selected by maintainer as the future publication destination | Publish the prepared branch, tag a commit, and test cloning/installation without author access. |
| Development origin `YixiaoWang7/robopolicy` | HTTP 404 without authentication | Keep development provenance separate from public clone instructions. |
| [Pinned simulator revision](https://github.com/YixiaoWang7/robot_envs/commit/c7e06a0784853d74db57ebf78d7a66fa40c2df99) | Public commit resolves | Retain the pin; audit collection/evaluation distribution parity separately. |
| Checkpoints and supplementary results | No public native checkpoint links or complete figure-result bundle documented | Publish hashes, model/data/config provenance, result JSON/CSV, and evaluation commands. W&B run links alone are not a public download index. |
| Code license | No repository LICENSE or package license metadata | Maintainer must select and add the license before publication. |

The Hugging Face API confirms all three repositories below are public and
ungated. These are observed revisions, not yet certified paper-data versions:

| Dataset | Revision |
| --- | --- |
| [PP: cg_l4_1s](https://huggingface.co/datasets/yixiiiiii/cg_l4_1s) | `2f21dddc1e39d7d4781e38f8f5d8a42b3c2af1f5` |
| [PPP: cg_l5](https://huggingface.co/datasets/yixiiiiii/cg_l5) | `e8547414dc3ae307a273ac1bad5a500f4205c4e1` |
| [Two-stage: cg_l4_2s](https://huggingface.co/datasets/yixiiiiii/cg_l4_2s) | `3b7659465f0787f8e6af36d98e33a6f30e40445f` |

Verify complete manifests, selected-demo hashes, conversion revisions, active
label-profile digests, and collection bounds. The older active pilot's PPP pool
had 63 tasks; the later baseline records 64. The active pilot report also records
unresolved collection/evaluation placement-bound differences. Neither should be
silently treated as the exact public dataset used by every paper experiment.

## Validation

Use the [maintenance commands](../maintenance.md#paper-release-workflow) for both
packages. The existing Python 3.10.19 environment has PyTorch 2.9.1+cu128,
TorchVision 0.24.1, TorchCodec 0.9.1, Transformers 4.57.5, and two RTX 5080 GPUs.
No environment or runtime packages were replaced during this review.

The initial core gate passed 277 tests including CUDA. The application gate
exposed seven failures: three fine-tuning preset paths retained obsolete
generated identities, and four historical-baseline comparisons treated the new
checkpoint-upload transport field as a scientific setting. The preset paths now
match their companion native pretraining jobs. Baseline comparisons exclude only
the upload flag in addition to their existing operational exclusions; archived
measurements remain unchanged.

Final local checks after those fixes:

| Gate | Result |
| --- | --- |
| Full core pytest suite, including CUDA | 277 passed; one PyTorch TF32 performance warning |
| Full application pytest suite with `CG_SIMULATOR_ROOT` and EGL | 298 passed, including all six static/reactive simulator acceptance cases; none skipped |
| Ruff lint and format, both packages | Passed; 132 Python files formatted |
| Core and application mypy | Passed; 51 core and 32 application source files |
| `uv build` and `uv build projects/corl2026_cg` | Both wheels and source distributions built |
| Distribution contents and metadata | Core excludes projects/legacy policies; type marker, CG entry point, and public project URLs present |
| Extracted-wheel imports and `cg --help` outside the checkout | Passed using the existing runtime; not a fresh-environment install |
| README `cg generate --dry-run`, preset matrix tests, workflow YAML | Passed; no training or W&B run started |
| Relative Markdown links and heading anchors | Checked repository-wide; no unresolved targets |
| Patch whitespace | `git diff --check` passed |

[Test output](paper-release-tests-20261012.txt) preserves the two final pytest
results. The simulator gate used a temporary clean checkout of
`c7e06a0784853d74db57ebf78d7a66fa40c2df99`, preserving the developer's modified
simulator checkout. `AGENTS.md`, the supplied PDF, and existing uncommitted
instruction-guide edits were left unchanged.

CI now includes the application on `corl2026_cg` pushes/PRs and checks its types,
unit tests, and build; the manual CUDA workflow includes its non-simulator tests.
These workflow changes have been inspected locally, not executed on GitHub.
Long training, paper sweeps, online W&B runs, fresh-environment installation
matrices, and new throughput/soak measurements were not run as part of this audit.
Those remain gates for the exact publication commit.
