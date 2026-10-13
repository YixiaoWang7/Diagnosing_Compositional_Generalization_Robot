# Ground-truth subgoal indicator: implementation and use

This guide describes the implementation on `corl2026_cg` at source revision
`bfe269a`. The code calls the feature **active instruction conditioning**, selected
with `conditioning: "active_v1"`. It computes one categorical indicator of the
current subgoal from the task definition, robot state, and object/container poses.

The indicator is supplied both when preparing demonstration labels and during
policy evaluation. The same `InstructionTracker` class implements both paths.
The tracker is a deterministic state machine with grasp hysteresis and phase
debouncing. Its thresholds are fixed by a versioned profile.

## Supported tasks

| Task | Internal family | Visible instruction | Profile | Subgoals |
| --- | --- | --- | --- | --- |
| Pick-Place-Press (L5) | `l5` | `full` | `l5_v1` | Pick object; place in container; press button |
| Dependent two-stage pick-and-place (L4_2s) | `l4_2s` | `infer_cont0` | `l4_infer_cont0_v1` | Place first object in an allowed container; pick second object; place second object |

L4 one-stage (`l4`, sometimes called `l4_1s`) and L4_2s `full` have no active-v1
profile. Requesting these combinations with `active_v1` raises an error.

`instruction` defines which task factors are visible and determines task success
and composition membership. `conditioning` defines the categorical representation
fed to the policy. Its default is `static`; the indicator requires `active_v1`.

## What ground truth means here

The tracker has access to physical world poses from demonstrations or the
simulator. Online evaluation obtains these through `SimulatorBackend.observe()`.
This is additional information supplied to the instruction generator, including
when the action policy uses camera images and robot proprioception.

The phase is computed using geometric rules. A grasp is inferred from proximity
and finger separation; it is not confirmed by contact forces. Placement combines
geometric membership with the inferred release state. Consequently, the indicator
is an oracle subgoal signal under these definitions, with the limitations of those
definitions. It is not a learned phase classifier or an exact physical-contact label.

The image pilot policies receive the resulting token alongside images and robot
state. The evaluator reserves `world_entities` for tracking and rejects that key
as a declared policy input. Independently configured state policies can still use
the separate `entities` feature. Report the simulator-derived instruction source
when comparing active and static conditioning.

## Inputs, coordinates, and vocabulary

Each tracker belongs to one episode and one `CGTask`. Its update takes:

| Input | Shape | Layout |
| --- | --- | --- |
| `state` | `(9,)` | EEF XYZ, EEF quaternion XYZW, two finger joint positions |
| `world_entities` | `(N, 7)`, `N >= 8` | One world XYZ/XYZW pose per entity |

All entries must be finite. Positions and threshold distances use meters.
The phase rules use XYZ and finger positions. Quaternions are carried in the
validated input but do not enter these rules.

The first eight entity rows have a fixed order:

| Factor ID | Object at row `ID` | Container at row `4 + ID` | L5 button ID |
| --- | --- | --- | --- |
| 0 | `cross` | `bin` | `red` |
| 1 | `cube` | `mug` | `green` |
| 2 | `cylinder` | `plate` | `blue` |
| 3 | `milk` | `mug_no_handle` | `yellow` |

The simulator adapter also appends button poses for L5. The tracker needs the
requested button ID to emit its token, but does not use button pose or button
press state to select the phase.

Tracking always uses world coordinates. Changing
`evaluation.entity_coordinates` to `relative` changes the separate policy entity
representation while leaving the tracker's world poses intact.

## Physical predicates

### Grasp with hysteresis

For object `o`, let:

```text
d(o) = Euclidean distance between EEF XYZ and object o's world XYZ
s    = abs(finger_qpos_0 - finger_qpos_1)
```

Each object has its own remembered grasp boolean, initially false. The previous
boolean selects the thresholds for the next observation:

| Previous grasp state | Distance bound | Finger separation bounds |
| --- | --- | --- |
| False: acquire a grasp | `d <= 0.04` | `0.01 <= s <= 0.055` |
| True: retain a grasp | `d <= 0.05` | `0.01 <= s <= 0.065` |

Both conditions must hold. Otherwise the object's grasp boolean becomes false.
The wider retention bounds reduce changes caused by small movements near the
entry boundary. The lower finger-separation bound applies to both cases.

Grasp state updates on every observation. The two-observation debounce described
below applies to the proposed phase, after these grasp booleans have updated.

### Geometric container membership

Let `(dx, dy, dz)` be `object_xyz - container_xyz`, and let
`r = sqrt(dx**2 + dy**2)`. `container_membership()` applies:

| Container | Horizontal condition | Vertical condition |
| --- | --- | --- |
| Bin | `abs(dx) < 0.03` and `abs(dy) < 0.03` | `abs(dz) < 0.06` |
| Mug | `r < 0.0325` | `abs(dz) < 0.04` |
| Plate | `r < 0.027` | `abs(dz) < 0.06` |
| Mug without handle | `r < 0.0325` | `abs(dz) < 0.04` |

These comparisons are strict. Membership uses body centers and world axes,
including for rotated bins; it does not transform the object into a rotated
container frame. These constants belong to the pinned geometry contract.

### Released placement

For each relevant object/container pair:

```text
placed(o, c) = (not grasped(o)) and container_membership(o, c)
```

An object held inside a container does not count as placed while the grasp
predicate remains true. Placement is recomputed from the current observation and
is not latched. There is no additional dwell timer, velocity threshold, or contact
test in this predicate.

## L5: pick, place, then press

The task supplies `object0`, `container0`, and `button`. After updating the grasp
state for `object0`, the tracker proposes:

```python
if placed(object0, container0):
    proposed_phase = 2
elif grasped(object0):
    proposed_phase = 1
else:
    proposed_phase = 0
```

| Phase | Meaning | Factor used in the indicator |
| --- | --- | --- |
| 0 | Pick the requested object | `object0` |
| 1 | Place the held object in the requested container | `container0` |
| 2 | Press the requested button | `button` |

Losing the grasp outside the target container proposes phase 0 again. Losing a
previously released placement also recomputes the phase: a regrasp proposes phase
1; an ungrasped object outside the destination proposes phase 0. These changes
are subject to the debounce rule.

Phase 2 remains the final instruction while the object satisfies placement. The
tracker has no separate terminal token and does not decide whether the button
was successfully pressed. Episode success is checked separately by the simulator.

### Worked L5 example

For `cube_into_mug_press_blue`, the IDs are `object0=1`, `container0=1`,
`button=2`. With the encoding defined below, the usual token sequence is:

```text
1: Pick cube  ->  5: Place in mug  ->  10: Press blue
```

Here is the debounce timing for an illustrative successful sequence:

| Observation | Physical predicate | Proposed phase | Emitted token |
| --- | --- | --- | --- |
| Reset | Cube ungrasped and outside mug | 0 | 1 immediately |
| First grasp observation | Grasp true | 1 | 1 |
| Second consecutive grasp observation | Grasp true | 1 | 5 |
| Held inside mug | Grasp true; placement false | 1 | 5 |
| First released placement observation | Placement true | 2 | 5 |
| Second consecutive released placement observation | Placement true | 2 | 10 |

If a fresh episode starts with the cube already released in the mug, reset emits
token 10 immediately.

## L4_2s: infer the first container

The visible instruction supplies `object0`, `object1`, and `container1`. The
concrete task also has `container0`, but the active tracker never consults it.

Define the set of allowed first destinations:

```text
allowed_first_containers = {0, 1, 2, 3} minus {container1}
```

Both objects' grasp states are updated independently. The phase proposal is:

```python
first_placed = any(placed(object0, c) for c in allowed_first_containers)
second_placed = placed(object1, container1)

if not first_placed:
    proposed_phase = 0
elif grasped(object1) or second_placed:
    proposed_phase = 2
else:
    proposed_phase = 1
```

| Phase | Meaning | Factor used in the indicator |
| --- | --- | --- |
| 0 | Place the first object in an allowed container | `object0` |
| 1 | Pick the second object | `object1` |
| 2 | Place the second object in its destination | `container1` |

Phase 0 covers both picking and placing the first object. Grasping that object
does not introduce another phase. Phase 2 also remains active after the second
object is released in its destination, until the evaluator terminates the episode.

The first placement remains a prerequisite on every observation. If it is lost,
the tracker proposes phase 0 even when the second object is held. If only the
second object slips outside its destination, the tracker proposes phase 1 while
the first placement remains valid.

### Hidden first-container handling

The tracker checks membership against all three allowed containers. For the same
visible task and physical observations, changing the task's hidden `container0`
does not change the token sequence. The active feature contract also rejects
`cg/task/container0` and `cg/entity/container0` in the policy schema.

There is a further property of the current representation: each observation
contains only one phase/factor token. In phase 0, the token identifies the first
object; it does not encode `container1` or enumerate the allowed destinations.
The instruction generator uses `container1` to decide when to advance, while the
policy receives that container ID in phase 2. The token alone therefore does not
specify the full first-stage destination constraint. This matters when interpreting
the information available to the action policy.

### Worked L4_2s example

Suppose the visible task is: place the cross first, then place the cube in the mug
without a handle. The visible IDs are `object0=0`, `object1=1`, `container1=3`.
The first destination may be the bin, mug, or plate.

```text
0: Place first cross in an allowed container
    -> 5: Pick second cube
    -> 11: Place second in mug_no_handle
```

Placing the cross in the mug without a handle leaves the tracker in phase 0.
Placing it in the ordinary mug proposes phase 1. Grasping the cube then proposes
phase 2. If the cube slips before placement, the sequence can return `11 -> 5`.
If the first cross leaves its allowed container, it can return to token 0.
Every change after initialization requires two consecutive matching proposals.

## Debouncing and categorical encoding

The tracker stores the accepted `phase`, a pending proposal, and its count:

1. On the first observation, accept the proposed phase immediately.
2. If the proposal equals the accepted phase, clear any pending change.
3. If it differs, count consecutive occurrences of that proposal. A different
   proposal starts a new count at one.
4. Accept the new phase when its count reaches `debounce_observations=2`.

The unit is observations. Offline, one observation is one recorded timestep;
online, it is reset or the successor of an executed action. The implementation
does not convert this count into seconds. Phases can move backward or jump over
an intermediate phase if the observations support that proposal.

After debouncing:

```text
active_id = 4 * accepted_phase + factors[accepted_phase]

L5 factors:               (object0, container0, button)
L4_2s infer_cont0 factors: (object0, object1, container1)
```

The result is one integer in `[0, 12)`. It combines phase and target identity;
it is not just a phase number or a completion flag.

| ID range | L5 meaning | L4_2s infer_cont0 meaning |
| --- | --- | --- |
| 0–3 | Pick object ID 0–3 | Place first object ID 0–3 in an allowed container |
| 4–7 | Place in container ID 0–3 | Pick second object ID 0–3 |
| 8–11 | Press button ID 0–3 | Place second in container ID 0–3 |

An identical integer has different semantics under the two profiles. For example,
5 means “place in mug” for L5 and “pick second cube” for L4_2s. The profile is part
of the dataset/checkpoint contract and must accompany the vocabulary size.

## Preparing training labels

`data/active.py` bridges HDF5 episodes to the storage-independent tracker:

1. Read robot state from `obs/robot0_eef_pos`, `obs/robot0_eef_quat`, and
   `obs/robot0_gripper_qpos`. If that full set is unavailable, accept a canonical
   `obs/state` array of shape `(T, 9)`.
2. Read world poses from `obs/object`, shaped `(T, N * 7)`, and reshape each block
   to `(T_block, N, 7)`. These must be world poses in the required entity order.
3. Create a fresh tracker for the episode and feed observations in temporal order.
   The implementation reads blocks of up to 4,096 observations while retaining
   tracker state across blocks.
4. Store the emitted labels as an `(T, 1)` `int64` dataset under the episode group:
   `cg/active_instruction/l5_v1` or
   `cg/active_instruction/l4_infer_cont0_v1`.
5. Store `profile_json`, `profile_digest`, and `source_layout` as label attributes.
   Preparation also records the profile and source hashes in `preparation.json`.

Labels use the current and preceding observations through tracker memory. They
do not require future observations, demonstration actions, or success annotations.
The label at timestep `t` conditions the policy input at that timestep.

`prepare_dataset()` copies numeric data into a fresh destination, checks modality
alignment, and publishes the completed export atomically. Videos are hard-linked
when possible and otherwise copied; linked video files must remain immutable.
When a source already contains active labels, preparation checks their metadata
and recomputes every label to reject disagreements with the physical observations.

Training-time `validate_dataset_profile()` checks selected episodes' layouts,
label shapes, dtypes, value ranges, and profile metadata. This startup check does
not recompute the complete trajectories; full recomputation occurs in preparation.

## How the policy consumes the indicator

The active feature contract is:

```json
{
  "key": "task_indices",
  "shape": [1],
  "history": 0,
  "future": 0,
  "paths": ["cg/active_instruction/l5_v1"],
  "dtype": "int64",
  "normalization": "none"
}
```

Use `cg/active_instruction/l4_infer_cont0_v1` for L4_2s. The native policy declares:

```json
{
  "type": "indexed",
  "vocab_sizes": [12]
}
```

That declaration belongs at `run.policy.conditioning`. The experiment-level
`conditioning: "active_v1"` chooses the application semantics. Job resolution
sets the feature path and vocabulary to match the selected profile.

The data flow and shapes are:

```text
Stored labels:       (T, 1), int64
Raw batch feature:   (B, 1, 1), current observation and one query slot
Packed task_indices: (B, 1), temporal axis removed
Embedded query:      (B, 1, D), learned 12-entry embedding table
```

`QueryTokens` learns the embeddings jointly with the policy. For the image pilot
presets, `ObservationEncoder` expands the query over observation timesteps and
cameras, uses it to query visual patch features through cross-attention, and
combines the resulting visual context with robot-state tokens. The action model
then predicts a conditioned action chunk. The image pilots use improved MeanFlow,
with 16 predicted seven-dimensional actions and up to eight executed per query.
These action-model settings are separate from the deterministic indicator rules.

Training uses the stored indicator as an input while optimizing the configured
action-learning objective. There is no additional indicator-prediction head or
classification loss. A current token may condition an action chunk that spans
a future phase transition; the feature carries no future indicator sequence.

## How online evaluation uses the indicator

`run_batch()` keeps one independent tracker and action queue per environment:

1. Reset the environment, create the tracker, and compute the initial token from
   the reset observation before the first policy call.
2. Put the token in `observation["task_indices"]` and construct the input batch
   using the checkpoint's declared feature schema.
3. For an active environment whose queue is empty, predict an action chunk and
   queue its first `n_execute` actions.
4. Execute one action and obtain its successor observation.
5. Update the tracker from that observation even if other queued actions remain.
6. If the emitted token changes, clear that environment's remaining action queue.
   Append the new observation and token to the input history. The next action
   therefore comes from a fresh prediction conditioned on the new token.
7. Apply the independent task-success checks and continue or terminate.

Replanning happens after the debounced token changes. One new-phase proposal by
itself can still leave an old queued action to execute. In batched evaluation,
other environments retain their queues; policy calls preserve the full fixed
batch shape, and predictions are enqueued only for environments needing actions.

Per-episode records contain `active_instruction_ids` for reset and every executed
action's successor, plus `instruction_transitions` with `step`, `from`, and `to`.
Thus an episode with `T` executed actions has `T + 1` recorded online indicators.
These records describe online observations; offline labels have one row per
recorded demonstration timestep. Evaluation also records the profile definition
and digest. Video overlays use the same `instruction_text()` decoder as inspection.

## Indicator phase versus task success

`EpisodeProgress` and the simulator determine success separately from the tracker.
The tracker can react to a lost grasp or placement while the evaluator retains
an earlier completed stage.

For L5, evaluation records placement-stage completion and asks the simulator for
native pick-place-press success. Emitting a press token does not establish that
the requested button has been pressed.

For L4_2s `infer_cont0`, evaluation probes physical placement success for the first
object in an allowed container, records the selected container, and retargets the
simulator to the second object and destination. While the first stage is pending,
placing the first object in the forbidden second destination sets a persistent
`same_container_violation`. After first-stage completion, a successful second
placement completes the task only if that violation is absent.

The indicator itself has no violation latch and does not read `EpisodeProgress`.
It can emit later phases after recovery even when the evaluator has retained a
violation, or return to an earlier phase after the evaluator latched a stage.
Reaching phase 2 is therefore not an episode-success metric.

Composition ID/OOD membership uses the complete visible task: `(object, container,
button)` for L5 and `(object0, object1, container1)` for dependent L4_2s. Familiarity
with individual indicator tokens does not establish familiarity with a composition.

## Versioning and compatibility

Profiles fix the thresholds, entity order, state layout, release definition,
debounce count, and token encoding. Their SHA256 digests are computed from a
deterministically serialized definition. Native checkpoints preserve the versioned
feature path, while prepared data and experiment/evaluation metadata record the
profile definition and digest.

Application entry points validate these contracts before training, resuming,
fine-tuning, loss evaluation, and rollout. Static and active checkpoints cannot
be interchanged through this path. The two active profiles also cannot be
interchanged merely because both use 12 values. Semantic changes require a new
profile name and explicitly prepared data; there is no silent migration.

## Prepare, inspect, and run

Run these commands from the repository root using the existing environment.
Replace `/path/to/...` placeholders with the appropriate native dataset and pinned
simulator paths. Preparation destinations and inspection outputs must be fresh.

```bash
.venv/bin/cg prepare \
  --source /path/to/l5_source \
  --destination /path/to/l5_active_v1 \
  --active-profile l5_v1

.venv/bin/cg prepare \
  --source /path/to/l4_2s_source \
  --destination /path/to/l4_infer_cont0_active_v1 \
  --active-profile l4_infer_cont0_v1

.venv/bin/cg inspect-instructions \
  --source /path/to/l5_active_v1 \
  --output runs/instruction-inspection/l5 --samples 4 --seed 0

.venv/bin/cg inspect-instructions \
  --source /path/to/l4_infer_cont0_active_v1 \
  --output runs/instruction-inspection/l4_2s --samples 4 --seed 0
```

Inspection reports label counts and transitions and overlays decoded instructions
on sampled videos. It uses the sidecar's `frame_start + step` alignment, as training
does, and checks that numeric and video frame counts align. It reports conflicting
timestamp rates instead of resampling those already aligned observations.

The checked-in pilots supply matching model and feature schemas. These commands
resolve their jobs without launching training or evaluation:

```bash
.venv/bin/cg generate \
  --config projects/corl2026_cg/configs/pilot_l5_active_imf.json \
  --set run.dataset.root=/path/to/l5_active_v1 \
  --set evaluation.simulator_root=/path/to/robot_envs \
  --set output_root=runs/corl2026_cg/gt_indicator_l5 \
  --dry-run

.venv/bin/cg generate \
  --config projects/corl2026_cg/configs/pilot_l4_2s_active_imf.json \
  --set run.dataset.root=/path/to/l4_infer_cont0_active_v1 \
  --set evaluation.simulator_root=/path/to/robot_envs \
  --set output_root=runs/corl2026_cg/gt_indicator_l4_2s \
  --dry-run
```

Remove `--dry-run` to write the resolved jobs. Then use the generated path:

```bash
.venv/bin/cg run --jobs /path/to/generated/experiment.json
```

This last command launches training followed by evaluation. Inspect active jobs
and GPU availability before execution. W&B remains enabled under
`yixiao_wang/corl2026_cg` by the supplied presets. These are 1,000-update integration
pilots; their settings are not a demonstrated converged benchmark protocol.

## Verification and known limits

Existing tests cover the algorithm and its connections to training/evaluation:

| Test file | Coverage |
| --- | --- |
| [test_instructions.py](../tests/test_instructions.py) | Grasp hysteresis, debounce, slips, container bounds, hidden-container invariance, offline/online parity, frozen profiles |
| [test_active_data.py](../tests/test_active_data.py) | Immutable preparation, label metadata, corruption rejection, integer batches, frame alignment |
| [test_reactive_evaluation.py](../tests/test_reactive_evaluation.py) | Per-environment replanning, physical regressions independent of success, world-coordinate tracking, checkpoint checks, recorded identities |
| [test_conditioning_guards.py](../tests/test_conditioning_guards.py) | Static/active compatibility and validation before resume/fine-tuning |

The focused CPU checks are:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:projects/corl2026_cg/src \
  .venv/bin/python -m pytest \
  projects/corl2026_cg/tests/test_instructions.py \
  projects/corl2026_cg/tests/test_active_data.py \
  projects/corl2026_cg/tests/test_reactive_evaluation.py \
  projects/corl2026_cg/tests/test_conditioning_guards.py \
  -q -m 'not simulator and not cuda' --strict-markers
```

This command explicitly excludes simulator/CUDA acceptance. The existing
[validation report](../../../docs/validation/corl2026_cg_active.md) records earlier
real-data and simulator checks. It also records that both short pilots had zero
task success; passing the indicator/integration tests does not establish that the
trained action policy solves the task.

The main interpretation limits are the grasp heuristic, center-based container
bounds, observation-count debounce, additional simulator information, and the
single-token information restriction described above for L4_2s. Current profile
definitions reconstruct the intended behavior and do not claim exact equivalence
to unavailable historical implementations.

## Implementation map

| File | Responsibility |
| --- | --- |
| [instructions/tracker.py](../src/corl2026_cg/instructions/tracker.py) | Physical predicates, phase proposal, debounce, indicator encoding and text |
| [instructions/profiles.py](../src/corl2026_cg/instructions/profiles.py) | Immutable profiles, thresholds, vocabulary contract, digest |
| [instructions/contracts.py](../src/corl2026_cg/instructions/contracts.py) | Feature, checkpoint, and fine-tuning compatibility |
| [data/active.py](../src/corl2026_cg/data/active.py) | HDF5 state reading, label generation and validation |
| [data/prepare.py](../src/corl2026_cg/data/prepare.py) | Atomic dataset export and preparation provenance |
| [experiments/config.py](../src/corl2026_cg/experiments/config.py) | Resolve active configuration into concrete job features and query vocabulary |
| [evaluation/backend.py](../src/corl2026_cg/evaluation/backend.py) | Simulator world-state observations and physical success probes |
| [evaluation/rollout.py](../src/corl2026_cg/evaluation/rollout.py) | Per-step tracking, policy input history, action queues and transition logs |
| [evaluation/success.py](../src/corl2026_cg/evaluation/success.py) | Independent stage completion and episode success |
| [processing/packing.py](../../../src/robopolicy/processing/packing.py) | Pack current-step categorical inputs without normalization |
| [modules/queries.py](../../../src/robopolicy/modules/queries.py) | Learned categorical query embeddings |
| [modules/observation.py](../../../src/robopolicy/modules/observation.py) | Condition visual observation encoding on the embedded query |

See [active instructions](active-instructions.md) for the broader composition
benchmark workflow and [the config index](../configs/README.md) for other presets.
