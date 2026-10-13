# Architecture and public interfaces

| Module | Owns |
| --- | --- |
| `contracts` | Feature declarations, policy tensors, opaque sample metadata |
| `algorithms` | Flow paths, training-time distributions, losses, numerical solvers |
| `modules` | Independent query vocabularies, vision, fusion, and action heads |
| `policies` | Shared observation encoding, loss, prediction, and model construction |
| `datasets` | Manifests, selection, plans, readers, assembly, execution, statistics fitting |
| `processing` | Tensor-only normalization, image transforms, canonical input packing |
| `training` | Optimizer updates, scheduling, validation, logging, resource shutdown |
| `checkpoints` | Atomic serialization, RNG recovery, offline inference construction |
| `config` | Versioned composition and cross-component validation |

Settings live beside their owning components. `RunConfig` composes those settings
without repeating their defaults and rejects unknown configuration fields.
Library code never imports projects or simulators. Neural modules and algorithms
never import datasets or training; datasets never import policies or training;
runtime processing never reads datasets. Automated import-boundary tests enforce
these relationships.

The root public interface exposes feature and configuration types,
`build_dataset`, `build_loader`, `fit_statistics`, `InputPacker`, `build_policy`,
`Trainer`, `read_checkpoint`, `dataset_identity`, and `load_policy`. `FlowPolicy.loss`, `predict`, and `attention_maps`
share the same `ObservationEncoder`. Convolutional FiLM and adaptive-LayerNorm
Transformer heads retain their different computations behind a single flow-field
interface. No general plugin system or application inheritance hierarchy is needed.

Transformer policies can select [improved MeanFlow](improved-mean-flow.md) through
the algorithm configuration. Its interval embedding uses the same AdaLN path;
state remains excluded from action cross-attention.

`PolicyInput` contains:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `state` | `(B, T_obs, D_state)` | Floating proprioception/features |
| `images` | `(B, T_obs, N_camera, 3, H, W)` | Optional processed RGB observations |
| `entities` | `(B, T_obs, N_entity, D_entity)` | Optional explicit entity features |
| `entity_indices` | `(B, N_selected)` | Integer selection into the entity axis |
| `task_indices` | `(B, K)` | Optional integer query vocabulary IDs |
| `actions` | `(B, T_action, D_action)` | Training targets; absent at inference |

Raw image batches are uint8. `InputPacker` converts, optionally crops/resizes, and
normalizes them without modifying caller-owned tensors. `FlowPolicy` consumes the
packed floating image representation. `load_policy(...).predict(raw_batch)` handles
packing and returns actions in their original scale.

`processing.image_representation="current_and_signed_difference"` requires raw
`[previous, current]` RGB with `history=1`. After shared crop/resize and conversion to
[0, 1], packing replaces the first slot with `0.5 + 0.5 * (current - previous)` and
normalizes both slots with the saved image mean/std. Zero change is neutral gray;
the current image and model shape remain unchanged. The default `rgb` retains
ordinary frame history, including for checkpoints saved before this option existed.

Torch 2.9 GPU inference enables `torch.compile` by default through `InferenceConfig`;
Torch 2.7/2.8 automatically use eager inference.
Validation and device transfer surround a shared tensor computation containing
preprocessing, encoding, the full flow solver, and inverse normalization.
Statistics buffers are nonpersistent and reconstructed from the checkpoint.
See [inference configuration and warmup](inference.md) for the robot control-loop API.

`read_checkpoint(path)` returns the existing CPU tensor/primitive payload after
validating schema, configuration, state-container and normalization consistency.
It never accesses datasets or constructs a model. Weight names and shapes remain
the responsibility of `load_policy`. Statistics semantic validation is shared by
the reader and runtime normalization, rather than duplicated by applications.

`dataset_identity(config, schema, split="train")` returns an immutable
`DatasetIdentity(fingerprint, selection)` using the same manifest, selection,
metadata and byte-hashing rules as training. HDF5 is required, but no DataLoader,
video decoder, GPU or W&B is initialized. Verification reads the referenced files
once with bounded memory; applications should perform it at export, not per action.
Copying a dataset with unchanged bytes and internal relative paths preserves its
identity. Saved absolute roots can be replaced for verification or training resume.

Indexed conditioning, for example `{"type":"indexed","vocab_sizes":[4,4,3]}`,
creates three independent embedding tables. There is no separate query-count
setting. Slot meanings belong to the application. Learned conditioning uses
`{"type":"learned","num_tokens":2}` and requires no task IDs. Task names and
identifiers are metadata, never parsed into query indices by the library.

Entity selection uses explicit indices with no object/container offsets. Optional
language-query components accept supplied embeddings or an explicitly configured
HuggingFace encoder. A language-training workflow is not part of this release.
Attention maps come from the same encoding computation used for policy outputs.
Boolean attention masks use `True` for blocked connections; fully masked rows
produce zero attention rather than NaNs or uniform attention to invalid keys.
