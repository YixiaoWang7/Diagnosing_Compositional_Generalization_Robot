# Preserved historical L5 measurements

`l5_checkpoint_losses_500.json` is copied verbatim from preservation commit
`7455b6c:artifacts/l5_checkpoint_losses_500.json`. It contains four legacy checkpoint
measurements using 500 offline samples per seen/unseen partition, together with
rollout summaries joined into the original artifact. The referenced dataset,
checkpoint and evaluation paths are historical provenance, not new run inputs.

These are pre-migration measurements, not validation of the new code or native
checkpoints. The original rollout summaries/checkpoints have not been re-evaluated
as part of this port. New reports read explicit generated job manifests and keep
objective loss distinct from sampled-action MSE.
