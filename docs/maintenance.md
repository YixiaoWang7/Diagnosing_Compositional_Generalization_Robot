# Maintenance and ownership

Yixiao is the accountable maintainer for the `corl2026_cg` paper release branch. Ownership
is explicit by module; contributors should name the responsible owner as reviewer
when changing that module. Additional team members can take over a row through a
reviewed ownership change rather than an implicit handoff.

| Area | Initial owner | Required review focus |
| --- | --- | --- |
| Contracts and configuration | Yixiao | Shapes, categorical dtypes, schema validation |
| Algorithms | Yixiao | Mathematical behavior, solver and loss regression tests |
| Neural modules and policies | Yixiao | Conditioning, gradients, attention, prediction parity |
| Datasets and statistics fitting | Yixiao | Coverage, selection identity, resource bounds, benchmarks |
| Runtime processing | Yixiao | Shared packing, nonmutation, normalization round trips |
| Training, checkpoints, logging | Yixiao | Successful-update counters, full recovery, strict run identity |
| Packaging and release validation | Yixiao | Dependency boundaries, fresh environments, pilot evidence |
| CG application and paper reproduction | Yixiao | Task populations, conditional metrics, exact splits, simulator pin, figure provenance |

Require a behavioral regression test for behavior changes. Keep tests small and
deterministic; longer recovery runs, real-data benchmarks, and soak checks are
release artifacts rather than default unit tests. Preserve benchmark specifications,
selection identities, dependency versions, hardware, and three steady-state
repetitions. Investigate throughput regression or peak-memory growth above 10%
before accepting a release. Compare equivalent worker start methods and disable
the former decoded-frame cache.

Run formatting, linting, strict type checks, and import-boundary tests on every
change. Review dependencies regularly (at least quarterly), and test coupled
PyTorch/TorchVision/TorchCodec updates together. CPU installation must not inherit
a machine's CUDA index configuration. CUDA tests run on the explicitly configured
Linux GPU runner and must fail when that runner is unavailable.

The package currently identifies itself as `0.2.0a1`; configuration and checkpoint
schemas are version 1. Declare the first stable version only after the documented
release gates pass. After that release, version the public API and both schemas
explicitly. Breaking changes require release notes, a documented transition, and
a version change. Do not add silent compatibility paths during ordinary refactors.

`projects/` is reserved for independently packaged applications and excluded from
library distributions. Projects depend only on public interfaces and own task
interpretation, simulators, task-success evaluation, and deployment adapters.
Use the independently packaged example under `examples/minimal_project/` as a
starting point. Library code must never import a project.

## Paper release workflow

The [paper-to-code audit](validation/paper-release-audit.md) tracks remaining
release work. Use it to tie a change to its paper experiment and distinguish measured baselines from
examples, pilots, and archived results. The publication destination is
[Diagnosing_Compositional_Generalization_Robot](https://github.com/YixiaoWang7/Diagnosing_Compositional_Generalization_Robot);
prepare and validate this branch before publishing it there.

Keep paper-specific work in `projects/corl2026_cg/`. Do not change preserved
baseline configs to make their budgets resemble a new paper experiment: create
a separately named preset with explicit task lists, train/design/sampling seeds,
demonstration budget, checkpoint source, and evaluation population. Store actual
per-seed results and provenance under `docs/validation/`; large data, videos, and
checkpoints belong in public artifact storage outside Git. Schema changes require
explicit versioning and documented semantics.

Use the existing `.venv` with the declared development, vision, and both decoder
extras. From the repository root, run all of these checks:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests -q
PYTHONPATH=src:projects/corl2026_cg/src .venv/bin/python -m pytest \
  -c projects/corl2026_cg/pyproject.toml projects/corl2026_cg/tests \
  -q -m 'not simulator' --strict-markers
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/ruff check projects/corl2026_cg
.venv/bin/ruff format --check projects/corl2026_cg
.venv/bin/mypy --config-file projects/corl2026_cg/pyproject.toml projects/corl2026_cg/src
uv build
uv build projects/corl2026_cg
```

On a CPU-only host, explicitly add `-m 'not cuda'` to the core test command and
use `-m 'not cuda and not simulator'` for the application. This is a partial
check, not CUDA acceptance. Run all real simulator checks separately with a
clean checkout of the pinned revision and installed dependencies:

```bash
CG_SIMULATOR_ROOT=/absolute/path/to/robot_envs MUJOCO_GL=egl \
  PYTHONPATH=src:projects/corl2026_cg/src .venv/bin/python -m pytest \
  -c projects/corl2026_cg/pyproject.toml projects/corl2026_cg/tests \
  -q -m simulator --strict-markers
```

A modified simulator checkout must be preserved; use a separate clean checkout
for acceptance. A missing renderer or runtime is a failed gate. Check both
wheels and source distributions, then verify fresh installation and `cg --help`
away from the source tree using the supported-runtime CI matrix. CI checks the
CG application as well as the library; real simulator acceptance remains an
explicit host check. A green unit-test job does not replace paper experiments
or the existing performance gates.

For release evidence runs, verify stored W&B credentials without exposing them,
use `yixiao_wang/corl2026_cg` online, and set
`run.logging.wandb.upload_checkpoints=true` explicitly. The library defaults this
flag to false. Verify the exact printed run URL through the authenticated API
and preserve checkpoint/result artifact identities. Public reproduction users
may override the entity to their own account.

Use focused `[yixiao] action` commits. PR descriptions identify changed behavior,
paper figures/protocols, validation, and material data assumptions. The final
release requires a selected license, working public links and artifacts, and
passing checks for the exact publication commit. Keep historical development
notes in `log.md` out of current setup instructions; follow `AGENTS.md` for agent
conduct without overwriting it during release preparation.
