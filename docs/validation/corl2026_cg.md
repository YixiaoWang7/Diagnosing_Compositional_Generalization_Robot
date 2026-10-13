# CoRL 2026 CG integration validation

Later reactive-instruction features and real 1,000-update pilot results are
recorded in [the active-instruction validation report](corl2026_cg_active.md).
The measurements below describe the original application migration.

Validated on 2026-09-29 using the existing Python 3.10.19 environment and RTX 5080
CUDA runtime. The branch starts from `refactor/policy-core` at `30b474e`; original
local changes were preserved on `dev_2s_1s` in `7455b6c`. Generic balanced selection
and W&B grouping are committed separately as `346478b`.

## Executed checks

| Check | Result |
|---|---|
| Complete core suite, including CUDA, vision and both real video decoders | 275 passed |
| Complete CG application suite, including real simulator tests | 182 passed |
| Core mypy | 51 source files passed |
| Application mypy | 24 source files passed |
| Ruff lint and formatting | Passed for core and application |
| Core source/wheel build | Passed; no application code in core wheel |
| Application source/wheel build | Passed; data exporter and CLI included |
| Task-generator comparison with preserved sources | 87 L5 and 35 infer-cont0 designs matched |
| External simulator tracked source status after validation | Unchanged |

The first core run exposed a missing TorchCodec installation. Installing the
declared `torchcodec==0.9.1` extra resolved all 13 affected failures; the final
complete core run passed. No tests were skipped in the final application run.

The simulator is robot_envs revision
`c7e06a0784853d74db57ebf78d7a66fa40c2df99`. Tested dependencies include PyTorch
2.9.1+cu128, torchvision 0.24.1, transformers 4.57.5, NumPy 2.2.6, MuJoCo 3.3.7,
imageio 2.38.0 and imageio-ffmpeg 0.6.0. The simulator's model/flow extra is not used.

## Scope of the integration checks

Tests exercise strict task parsing, instruction visibility, hidden-factor ID/OOD,
curated designs, the 75 supported local recipes, balanced selection, immutable
dataset preparation, source/sidecar alignment, content identity, and package
boundaries. Six interaction recipes remain exact archived inputs, outside the
supported native models.

A subprocess CLI test executes prepare → generate → train → checkpoint → offline
loss → report using a small synthetic dataset. Real CPU FM and iMF training tests
exercise the checkpoint/loss path and actual finite sample counts. Failure tests
cover overlapping execution, nonzero child exits, duplicate output protection,
checkpoint/instruction mismatches and invalid evaluation membership. Reports keep
different evaluation settings, checkpoint steps and dataset contents separate.

Four opt-in real simulator tests instantiate and step L4/L5, verify RGB camera
order and state/entity layouts, and run native tiny image iMF checkpoints through
evaluation with raw and attention MP4 artifacts. Those checkpoints are explicitly
untrained and execute only two actions. These are integration tests, not estimates
of policy success or substitutes for a full experimental sweep. No long training
or benchmark sweep was launched. W&B network logging was disabled in local tests;
its core contract was tested with the existing logging tests.

## Reproduction

From the repository root with the application and declared extras installed:

```bash
PYTHONPATH=src:projects/corl2026_cg/src .venv/bin/python -m pytest tests/core -q
PYTHONPATH=src:projects/corl2026_cg/src \
  CG_SIMULATOR_ROOT=/path/to/robot_envs MUJOCO_GL=egl \
  .venv/bin/python -m pytest projects/corl2026_cg/tests -q
ruff check .
ruff format --check .
mypy --python-executable .venv/bin/python
mypy --python-executable .venv/bin/python \
  --config-file projects/corl2026_cg/pyproject.toml projects/corl2026_cg/src
uv build
uv build projects/corl2026_cg
```

The recorded Ruff/mypy runs used existing tooling from
`/tmp/robopolicy-checks312/bin/`, with mypy explicitly targeting the project's
Python environment. Simulator setup is documented in the
[application guide](../../projects/corl2026_cg/docs/simulator.md).
