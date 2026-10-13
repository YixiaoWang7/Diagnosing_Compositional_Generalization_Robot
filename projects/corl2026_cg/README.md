# Compositional generalization application

This is the experiment application for **Diagnosing Compositional Generalization
in Sequential Robot Tasks**. Start with the [paper release README](../../README.md)
for task definitions, baseline results, and installation. This branch is being prepared for
publication in the [public paper repository](https://github.com/YixiaoWang7/Diagnosing_Compositional_Generalization_Robot).

The [paper](https://arxiv.org/abs/2607.29687),
[project website](https://yixiaowang7.github.io/Diagnosing_Compositional_Generalization_Robot_Page/),
and [paper-to-code audit](../../docs/validation/paper-release-audit.md) provide the
scientific context. Runnable presets include measured baselines, preserved
experiment matrices, examples, and integration pilots; they do not yet reproduce
every figure. In particular, paper fine-tuning partitions, conditional violation
metrics, the 36-instruction evaluation space, and Isaac Lab experiments require
additional work.

The `cg` command prepares demonstration data, resolves experiment configurations,
trains policies, and runs simulator evaluations.

## Release baselines

| Paper task | Internal family / instruction | Measured-run preset |
| --- | --- | --- |
| Pick-and-Place (PP) | `l4` / `full` | [release_pp_50k](configs/release_pp_50k.json) |
| Pick-Place-Press (PPP) | `l5` / `full` | [release_ppp_50k](configs/release_ppp_50k.json) |
| Dependent Two-Stage Pick-and-Place (2S-PP) | `l4_2s` / `infer_cont0` | [release_2s_pp_100k](configs/release_2s_pp_100k.json) |
| Full-instruction Two-Stage Pick-and-Place (additional baseline) | `l4_2s` / `full` | [release_2s_full_100k](configs/release_2s_full_100k.json) |

Each release preset generates one seed-11 run with the training and evaluation
settings behind the [recorded results](../../docs/validation/cg_training_evaluation_setups_20261001.md).
Dependent 2S-PP hides the first container and evaluates 48 visible instructions;
the full-instruction baseline exposes both containers and evaluates 144 tasks.
PPP's preset trains on all 64 task combinations and evaluates all 64.
See the [config index](configs/README.md#release-baselines) for budgets and path defaults.

## Install

Use the repository's existing Python environment, from the repository root:

```bash
uv pip install --python .venv/bin/python -c constraints/training.txt \
  -e '.[dataset,vision,logging,opencv]' -e 'projects/corl2026_cg[video]'
.venv/bin/cg --help
```

Rollouts additionally require the pinned external simulator and its dependencies;
see [simulator setup](docs/simulator.md). Training and evaluation may use separate
interpreters with the same core and application installed.

## Prepare data once

```bash
.venv/bin/cg prepare --source /path/to/original/smallfiles --destination /path/to/prepared/l4
```

The exporter copies numeric shards and adds `cg/task/<field>` arrays with shape
`(T, 1)` and dtype int64. Original datasets remain untouched. Existing MP4 files
are hard-linked when possible and copied otherwise; both source and exported
videos must be treated as immutable. Preparation validates episode/video alignment,
records source hashes, and publishes a complete directory atomically.

For state policies with explicit entity poses, add
`--entity-path obs/object_rel_eef --entity-count 8` (L4) or `12` (L5). This exports
`cg/entities` with shape `(T, N_entity, 7)` and semantic `cg/entity/<field>` indices.
The evaluator's default relative coordinates match that source. World-coordinate
data requires `evaluation.entity_coordinates=world`. Input data must already use
the supported small-files manifest format; raw demonstration conversion stays in
the external data collection tooling.

## Resolve, train, evaluate, report

Checked-in [presets](configs/) are compact native configurations. Paths in them
are relative to the invocation directory and are deliberately portable. Override
the prepared dataset root and simulator checkout before running:

```bash
.venv/bin/cg generate --config projects/corl2026_cg/configs/release_pp_50k.json \
  --set run.dataset.root=/path/to/prepared/l4 \
  --set evaluation.simulator_root=/path/to/robot_envs
```

Generation produces an `experiment.json` (task set, evaluation and execution
settings) and a native core `run.json` for each job. `--dry-run` prints these inputs
without creating runs. Existing manifests require explicit `--overwrite`; this
does not delete training or evaluation outputs.

```bash
.venv/bin/cg run --jobs runs/corl2026_cg/release/release_pp_50k_*/experiment.json
# Or train first, then evaluate existing jobs:
.venv/bin/cg run --jobs runs/corl2026_cg/release/release_pp_50k_*/experiment.json --skip-eval
.venv/bin/cg run --jobs runs/corl2026_cg/release/release_pp_50k_*/experiment.json --eval-only
.venv/bin/cg evaluate-loss --job runs/corl2026_cg/release/RUN/experiment.json
.venv/bin/cg report --jobs runs/corl2026_cg/release/release_pp_50k_* \
  --output runs/corl2026_cg_report
```

Use a specific run directory instead of `RUN`; choose either the combined run or
the separate train/evaluate commands, since completed evaluations cannot be
overwritten. Release jobs live under `runs/corl2026_cg/release/`; other presets
default to `runs/corl2026_cg/`. `cg run --config ...` combines
generation and execution. Each stage runs in a separate subprocess, stores logs
and exit status, and propagates failure to the caller. For overlapping training
and evaluation, set `execution.overlap=true`, `execution.train_gpus="0"` and
`execution.eval_gpus="1"`. Both processes then address their assigned GPU as
`cuda:0`. Overlap requires disjoint explicit GPU assignments and bounds the queue
to one evaluation waiting behind the active evaluation. Default execution is serial.

Native `cg train --config run.json`, `cg train --resume latest.pt`, and
`--finetune-from latest.pt` use the core Trainer directly. Fine-tuning requires a
matching native architecture.
Jobs with existing checkpoints refuse a fresh training launch; resume explicitly
or generate a new job. Job fine-tuning sources belong in the experiment specification.

Training seeds share a stable W&B group; data, task-design, architecture, or
fine-tuning-source changes create a different group. W&B is enabled by default
under `yixiao_wang/corl2026_cg`. Explicitly disable it for local tests using
`run.logging.wandb.enabled=false` and `evaluation.logging.wandb.enabled=false`.
Standalone loss evaluation also accepts `--local-only`.

Checkpoint files are saved locally. For release evidence runs, opt into remote
checkpoint artifacts with `--set run.logging.wandb.upload_checkpoints=true` and
verify their upload. Changing source-run settings, including logging settings,
can change the generated directory: use the actual source job's `checkpoint`
value for `finetune_from`. Never substitute a legacy checkpoint.

## Evaluation outputs

Rollout results include per-task success, ID/OOD success based on the **visible
instruction**, stage completion and failure information. Loss evaluation measures
the policy objective separately from normalized/raw sampled-action MSE. It uses
500 samples and batches of 50 by default, records actual sample counts, and marks
an empty unseen-task set as not applicable. These are training-demo samples
partitioned by task identity, not held-out validation demonstrations.

Two paper comparisons need additional metrics: the current same-container rate
divides by all rollouts, and current ID/OOD splits follow the evaluated model's
training tasks rather than a separate pretraining reference. Full-set fine-tuning
therefore has no OOD split. See the [measurement contracts](../../docs/validation/paper-release-audit.md#measurement-contracts-to-resolve)
before using these outputs in paper figures.

The report combines explicit run identities and reports mean and standard deviation
across training seeds. Actual evaluation settings and checkpoint steps have separate
comparison identities, so overrides that change the measurement are not averaged
together. The [release validation report](../../docs/validation/cg_training_evaluation_setups_20261001.md)
documents the completed baseline runs and their differences from the paper's
evaluation protocols. Dataset download links and preparation commands are kept
in the [root README](../../README.md#datasets-and-checkpoints).

| Location | Responsibility |
|---|---|
| `tasks/` | Vocabulary, task parsing, instruction layouts, train/evaluation designs |
| `instructions/` | Versioned reactive profiles, shared physical-state tracker, data contracts |
| `data/` | One-time task conditioning and entity preparation |
| `experiments/` | Matrix resolution, subprocess orchestration, offline metrics, reports |
| `evaluation/` | External simulator adapter, observations, success, videos |
| `configs/` | Runnable native experiment presets |

See [reactive instructions and composition benchmarks](docs/active-instructions.md)
for active conditioning, independent evaluation populations, coverage diagnostics,
dataset inspection, and the two real-data pilot recipes.

## Validation

From the repository root, with the package installed:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests -q
PYTHONPATH=src:projects/corl2026_cg/src .venv/bin/python -m pytest \
  -c projects/corl2026_cg/pyproject.toml projects/corl2026_cg/tests \
  -q -m 'not simulator' --strict-markers
.venv/bin/ruff check projects/corl2026_cg
.venv/bin/ruff format --check projects/corl2026_cg
.venv/bin/mypy --config-file projects/corl2026_cg/pyproject.toml projects/corl2026_cg/src
uv build
uv build projects/corl2026_cg
```

The full core suite requires CUDA, both video decoder extras and vision
dependencies. Simulator smoke tests additionally require the external checkout.
See the [executed integration checks](../../docs/validation/corl2026_cg.md).
Enable the [simulator acceptance gate](docs/simulator.md#acceptance-checks)
explicitly with `CG_SIMULATOR_ROOT`; ordinary test runs otherwise skip it.
Run `-m simulator` across the application tests to include both static and active
instruction smoke tests. The [maintenance guide](../../docs/maintenance.md#paper-release-workflow)
lists the complete release sequence and evidence requirements.
