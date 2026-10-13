# Reactive instructions and composition benchmarks

CG-specific instruction tracking lives in the application. RoboPolicy receives
ordinary categorical features and uses its existing training and checkpoint code.

The [GT indicator implementation guide](gt-indicator.md) details the physical
predicates, L5 and L4_2s phase rules, token encoding, and training/evaluation paths.

## Task composition and active conditioning

`instruction` describes the visible task: `full` for L5, or `infer_cont0` for
two-stage L4. It determines success and ID/OOD membership. `conditioning` controls
what categorical query the policy receives:

- `static` (default): the existing visible composition.
- `active_v1`: one current subgoal ID, encoded as `phase * 4 + factor` in `[0,12)`.

| Family | Phase 0 | Phase 1 | Phase 2 |
|---|---|---|---|
| L5 | Pick object | Place in container | Press button |
| L4 infer-cont0 | Manipulate first object | Pick second object | Place in second container |

The tracker recomputes geometric placement and grasp from world poses and robot
state. It can return to an earlier phase when an object slips. Two consecutive
observations debounce a phase change; reset initializes immediately. This unit is
observations, not seconds. The same implementation labels demonstrations and
runs after every simulator action. Changed instructions discard queued actions
for that environment before the next action executes.

For infer-cont0, the first object may be placed in any container except the second
destination. The hidden first-container ID does not determine the active token.
ID/OOD remains a property of the complete visible composition; seeing an
individual subgoal token does not make a composition in-distribution.

Active evaluation uses physical simulator state to supply a ground-truth
instruction. Image policies receive the token alongside their configured image
and robot-state observations. Report this additional instruction source when
comparing active and static policies.

## Immutable label profiles

Profiles `l5_v1` and `l4_infer_cont0_v1` define geometry, grasp thresholds, entity
order and debounce rules. Prepared labels are `(T,1)` int64 arrays at
`cg/active_instruction/<profile>`. They are current-step, unnormalized categorical
features with vocabulary `[12]`.

The versioned path is stored in native checkpoints. Dataset attributes and
preparation metadata record the complete profile and its digest. Training,
resume, fine-tuning, offline losses and evaluation reject incompatible profiles.
Changing a published profile's semantics requires a new profile name. There is
no separate checkpoint format or automatic conversion of static checkpoints.

The starting grasp thresholds are 4 cm entry / 5 cm retention distance from the
end effector, with finger separation 1–5.5 cm on entry and 1–6.5 cm for retention.
Container membership follows the pinned simulator's world-axis geometry. These
are reconstructed definitions, not claims of exact historical patch parity.

## Prepare and inspect data

Export into a new directory, preferably on the source filesystem so video files
can be hard-linked. Source HDF5 files and existing observations are preserved.

```bash
cg prepare --source /datasets/cg_l5 \
  --destination /datasets/cg_l5_active_v1 --active-profile l5_v1
cg prepare --source /datasets/cg_l4_2s \
  --destination /datasets/cg_l4_infer_cont0_active_v1 \
  --active-profile l4_infer_cont0_v1
cg inspect-instructions --source /datasets/cg_l5_active_v1 \
  --output runs/instruction-inspection/l5 --samples 4 --seed 0
```

Inspection uses the video sidecar's episode frame alignment. It reports label
counts and transitions and overlays the same instruction text as rollout videos.
It requires one frame per numeric step, exactly as the training reader does.
Numeric/video timestamp differences are reported as diagnostics; inspection does
not resample frames using conflicting clocks. In the available L4 collection,
the converter generated numeric timestamps at an assumed 50 Hz and video
timestamps at 20 FPS, although the collector saved state and images together.
Existing prepared semantic fields are validated before adding active labels;
source datasets are never relabeled in place.

## Choose training and evaluation compositions

`sets` selects training designs. Optional `eval_set` independently selects the
evaluation design's compositions. Omission preserves existing preset behavior.
Use `eval_set: "full"` for the full evaluation universe.

Full-instruction two-stage L4 has 144 valid compositions: objects and containers
must differ across stages. Its new designs have explicit coverage guarantees:

| Name | Rows | Guarantee |
|---|---:|---|
| `pairwise_min` | 16 | Minimum coverage of all 88 feasible factor pairs |
| `pairwise_cover` | 18 | Deterministic greedy coverage of all feasible pairs |
| `threewise_cover` | 48 | Minimum coverage of all 192 feasible factor triples |
| `random_coverage_B16_s0` | 16 | Seeded unique rows covering every single factor level |
| `full` | 144 | Complete constrained universe |

Random coverage uses a seeded single-level covering backbone and samples the
remaining rows without replacement. It is not uniform sampling over all
single-level-covering subsets. The budget and design seed are part of the name.

Job metadata records factor counts and pair/triple coverage. The metric
`ood_with_unseen_pairs` counts OOD evaluation compositions containing at least one
factor pair missing from training. This distinguishes higher-order composition
novelty from unseen lower-order combinations. Evaluation identities include the
resolved population and active profile, preventing incompatible comparisons from
being averaged together. The existing overlap runner remains available.

Offline task losses retain their full-family scope: trained compositions versus
all available unseen visible compositions, independently of the rollout's
`eval_set`. Their recorded task lists and dataset fingerprints identify that
measurement explicitly.

## Real-data pilots

`pilot_l5_active_imf.json` and `pilot_l4_2s_active_imf.json` train image improved
MeanFlow policies for 1,000 updates with 256 balanced demonstrations, seed 11, and
a 100-update warmup. Each evaluates one rollout per visible task, horizon 400.
Their frozen DINOv2-small backbone declares `vision.image_size: 518` for its
pretrained positional embedding grid. Preprocessing still crops images to
224 pixels; DINO interpolates the saved grid to the input size at runtime.
Override only the dataset and output locations for local use:

```bash
cg run --config projects/corl2026_cg/configs/pilot_l5_active_imf.json \
  --set run.dataset.root=/datasets/cg_l5_active_v1
```

W&B remains enabled under `yixiao_wang/corl2026_cg`. Pilot results check the real
training/evaluation path and early learning behavior; they are not converged
benchmark estimates. Offline reports explicitly list compositions absent from
the available dataset even when simulator evaluation covers the full universe.

The completed pilots and their limitations are recorded in the
[validation report](../../../docs/validation/corl2026_cg_active.md).
