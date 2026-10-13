# Repository guidelines

RoboPolicy is the task-agnostic library under `src/robopolicy/`. Contracts,
algorithms, neural modules, policies, datasets, processing, training, checkpoints,
and configuration each own their responsibilities. Follow `docs/architecture.md`
and the dependency-boundary tests. Library code never imports projects or simulators.

Keep runnable entry points thin in `scripts/policies/`, tests in `tests/core/`,
and small generic examples in `examples/`. Reserve `projects/` for independently
packaged applications and exclude it from the distribution. Existing experiments
remain on `dev_2s`; do not restore old imports, configs, or checkpoints.

Use the existing `.venv` for work in this checkout. Do not create additional
environments or install packages without a concrete need. Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
uv build
```

The full suite requires the declared decoder and vision extras and a CUDA GPU.
CPU-only checks explicitly select `-m 'not cuda'`. Do not silently skip a missing
runtime when testing its acceptance gate. CI uses Python 3.10.

Follow `coding_styles.md`: explicit types, Google-style docstrings, documented
tensor shapes, four-space indentation, descriptive names, and inferred or supplied
devices. Use composition and keep numerical computations readable. Behavior
changes require meaningful regression tests. Checkpoint and public configuration
schemas are versioned; do not add silent migration paths.

Use focused `[yixiao] action` commits. PR descriptions explain changed behavior,
validation, and material data assumptions. Keep datasets, checkpoints, generated
builds, `runs/`, and `wandb/` out of Git. Preserve benchmark results under
`docs/validation/`; the final PR to master requires every release gate to pass.
See `docs/maintenance.md` for module ownership and release maintenance.

## Cloud experiment tracking

Enable W&B by default for training and standalone evaluation. Use entity
`yixiao_wang`, and give every checked-in run configuration a stable project name.
Online mode is the default; local-only logging requires an explicit configuration.
Before a long job, verify the stored W&B credential. Never place API keys in source,
configuration, documentation, logs, or chat.

Training uploads metrics, resolved configuration, provenance, and checkpoints.
Standalone evaluations use separate runs with aggregate metrics and result
artifacts. Startup must print
`https://wandb.ai/yixiao_wang/<project>/runs/<run-id>`, and the authenticated API
must resolve that exact identity. Treat a missing or malformed URL as a failure.
