# External CG simulator

The simulator is an external dependency. `simulator.json` pins the tested source
contract to robot_envs commit `c7e06a0784853d74db57ebf78d7a66fa40c2df99`.
Keep its checkout outside the library package. Install its `CG/robosuite` package
and required MuJoCo/offscreen-rendering dependencies into the evaluation Python
environment following that checkout's installation instructions. Install this
project and RoboPolicy into the same environment. Set `evaluation.simulator_root`
to the checkout. The evaluator verifies its HEAD and rejects tracked changes or
an already-imported robosuite from another location. No source is copied into the
simulator and none of its test scripts or old policy loaders are imported.

With the pinned checkout available at `/path/to/robot_envs`, install its simulation
packages into the existing environment (from the RoboPolicy repository root):

```bash
uv pip install --python .venv/bin/python \
  -e /path/to/robot_envs/CG/robosuite \
  -e /path/to/robot_envs/CG/mimicgen \
  'mujoco==3.3.7' 'numpy>=1.26,<3' 'opencv-python>=4.11,<4.12' \
  'imageio>=2.37.3' 'imageio-ffmpeg>=0.6'
```

The application imports these packages directly; the `robot-envs[flow]` extra is
unnecessary and conflicts with the core's model dependency constraints. Keep the
simulator checkout's tracked sources clean. Installation tools can regenerate
tracked egg-info metadata in some external checkout versions; inspect such changes
before restoring installation-generated files.

The reproducible integration check uses the real environments, renderer, native
untrained iMF checkpoints, and both video formats:

```bash
CG_SIMULATOR_ROOT=/path/to/robot_envs MUJOCO_GL=egl \
  .venv/bin/python -m pytest projects/corl2026_cg/tests/test_simulator.py -q
```

Evaluation creates CG_L4 or CG_L5 environments directly with the historical Panda
controller, task placement ranges, and camera dimensions from the checkpoint.
Set the rendering backend (for example `MUJOCO_GL=egl`) in the process environment
as appropriate for the host. Video output needs imageio and an FFmpeg provider;
no video dependencies load when both video limits are zero.

All images entering RoboPolicy are RGB uint8 in the checkpoint's camera order.
Robosuite's bottom-up frames are flipped vertically once. Observation history is
updated after every action, including the intermediate steps of each action
chunk. This is necessary for temporal policies and signed frame differences.
Attention videos overlay the images after saved cropping/resizing, so the maps
align with model input. They show camera rows and raw-image/query-overlay columns.

State policy entity coordinates must match dataset preparation. The default
`entity_coordinates="relative"` matches legacy `obs/object_rel_eef`: positions
are world-frame deltas to the end effector and orientations are canonical XYZW
relative quaternions. Use `"world"` only with world-frame prepared data. Entity
order is the four objects, four containers, and (L5 only) four colored buttons.

The application handles task success. L4 object-only instructions accept any
container; container-only instructions accept any object. Two-stage tasks latch
stage 0 and retarget without resetting. Open-cont excludes the first destination;
infer-cont0 excludes the second destination and records same-container violations.
L5 delegates its placement-then-button predicate to the pinned simulator.

Results include `eval_config_used.json`, incremental `episodes.jsonl`, and
`eval_summary.json` with per-task, ID/OOD, stage and failure summaries. W&B is on
by default with project `corl2026_cg`; explicitly disable it only for local checks.
A directory containing a completed summary is never overwritten. Raw and attention
videos are bounded by their explicit limits. Mock-environment tests validate
control and success semantics; real simulation acceptance additionally requires
the pinned simulator dependencies and a functioning rendering backend.

## Acceptance checks

Run the opt-in simulator tests from the repository root:

```bash
CG_SIMULATOR_ROOT=/absolute/path/to/robot_envs MUJOCO_GL=egl \
  PYTHONPATH=src:projects/corl2026_cg/src .venv/bin/python -m pytest \
  -c projects/corl2026_cg/pyproject.toml projects/corl2026_cg/tests/test_simulator.py \
  -m simulator -q
```

The explicit environment variable enables these tests; a missing simulator or
rendering dependency then fails the acceptance gate instead of being skipped.
Without the variable, ordinary unit-test runs skip simulator acceptance.

The pinned checkout was exercised with MuJoCo 3.3.7, PyTorch 2.9.1 and EGL:
**4 simulator acceptance tests passed**. These check real L4/L5 reset and control
steps, exact RGB camera ordering, native untrained image improved-MeanFlow
checkpoints through the public evaluator, and playable raw/attention MP4 artifacts.
The two-step untrained rollouts validate integration, not task performance.

Checkpoint evaluation verifies the exact ordered instruction feature paths and
vocabulary sizes, plus the configured and actually selected training tasks,
before producing ID/OOD labels. Passing a different checkpoint cannot silently
reuse a task-set definition with incompatible conditioning or training membership.
