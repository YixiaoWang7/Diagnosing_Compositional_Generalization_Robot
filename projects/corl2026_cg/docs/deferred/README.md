# Deferred interaction recipes

These six JSON files are exact archival copies from preservation commit `7455b6c`.
They are not runnable CoRL application configurations and are excluded from experiment generation.

The three `latent_obj` recipes require label-posterior discrete latent plans; the three
`futimg_latent_obj` recipes also require sparse training-only future-image evidence at
offsets 8, 16, 24, and 32. Their implementation exists on the separate `interaction`
branch at `82f46d5` ("save, not validate.") and is outside this migration.
The `factor_vocabs`, `query_factors`, `hidden_factor_vocab`, `interaction_*`, and
`sparse_offsets` fields are not supported by the policy-core run schema.

Ordinary object/container queries and the two constant query slots in `container_latents`
are supported by the application; those do not require this interaction mechanism.
