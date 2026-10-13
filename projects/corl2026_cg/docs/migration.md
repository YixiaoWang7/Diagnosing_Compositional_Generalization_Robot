# Branch integration

`corl2026_cg` starts from `refactor/policy-core` commit
`30b474e74942d3b02aed08523346f01b7de19a3c`. Task features are ported from
`dev_2s` (`ca84589`) and `dev_2s_1s`, including preservation commit `7455b6c`.
The preservation commit contains all previously uncommitted edits and the local
experiment/result files before any branch integration.

This is a feature port into the refactored architecture. It does not merge the
legacy package back into the library. The generic core additions are optional
balanced demonstration selection and optional W&B grouping; their defaults retain
existing behavior. Boundary tests prohibit core imports of this application.

| Feature | New owner |
|---|---|
| L4 one-stage object/container tasks | `tasks/schema.py`, `tasks/sets.py` |
| L4 two-stage, open-cont, infer-cont0 | `tasks/`, `evaluation/success.py` |
| L5 placement/button and curated designs | `tasks/`, pinned simulator |
| Instruction masks and constant learned slots | Explicit prepared fields + native indexed conditioning |
| State/entity inputs and image policies | Native core policy and feature schema |
| Training, checkpoint resume, fine-tuning | Public core Trainer |
| GPU scheduling, matrices and result aggregation | `experiments/` |
| Raw/attention videos | `evaluation/media.py` |
| Label-posterior/future-image interaction model | Deferred archive |

The authoritative task generators take precedence over stale expanded legacy
JSONs. Full two-stage training contains 144 concrete tasks. Open-cont and
infer-cont0 expose 48 visible queries. ID/OOD membership compares visible queries,
so a hidden target never changes the reported task split. The legacy open-cont
L_cor×S design includes its held-out fallback query and keeps its distinct train
and evaluation sets.

Instruction layouts are explicit:

| Layout | Visible categorical fields |
|---|---|
| L4 full | object0, container0 |
| L4 object-only | object0 |
| L4 container-only | container0 |
| L4 container-latents | container0, constant, constant |
| L4 two-stage full | object0, container0, object1, container1 |
| Open-cont | object0, container0, object1 |
| Infer-cont0 | object0, object1, container1 |
| L5 full | object0, container0, button |

The singleton-vocabulary slots in container-latents are learned constant indexed
embeddings. They do not implement the deferred interaction model. Open-cont places
the second object in any container other than the first destination. Infer-cont0
places the first object in any container other than the second destination and
then places the second object in its instructed container.

Preparation materializes categorical arrays, keeping CG parsing out of the data
loader. Balanced selection filters tasks, splits demonstrations, ranks/caps each
task, then assigns sorted equal quotas (with remainder to the first tasks). It
does not redistribute shortages; manifests and fingerprints record actual
selected membership. Validation selection is independent of training caps.

Refactored models, normalization, initialization and checkpoints are authoritative.
There is no legacy checkpoint/config auto-migration and no claim of bitwise
training parity. Re-train prerequisite native checkpoints before using fine-tuning
presets. The evaluator also corrects two legacy input inconsistencies: it consumes
RGB images and updates temporal histories on every simulator action. These changes
must be accounted for when comparing historical scores to new runs.

The earlier easy/medium/hard environment proposal is outside this integration.
No long experiment sweep or new benchmark score is claimed by the migration.
