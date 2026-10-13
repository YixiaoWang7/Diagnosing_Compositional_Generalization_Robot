"""Bind application instruction semantics to ordinary native feature schemas."""

from __future__ import annotations

from collections.abc import Sequence

from robopolicy import FeatureSchema, RunConfig

from corl2026_cg.instructions import ActiveInstructionProfile, get_profile, resolve_profile
from corl2026_cg.tasks import validate_conditioning


def validate_feature_conditioning(
    schema: FeatureSchema,
    vocabulary_sizes: Sequence[int],
    family: str,
    instruction: str,
    conditioning: str = "static",
) -> ActiveInstructionProfile | None:
    """Validate feature meaning before loading weights or allocating a simulator.

    Args:
        schema: Native feature declarations, including current-step task indices.
        vocabulary_sizes: Number of categorical values for each query slot.
        family: CG task family.
        instruction: Visible composition, independently of the active subgoal.
        conditioning: Static composition or the immutable active version-one profile.

    Returns:
        The active profile, or ``None`` for static composition conditioning.
    """
    profile = resolve_profile(family, instruction, conditioning)
    feature = schema["task_indices"]
    if profile is None:
        validate_conditioning(family, instruction, feature.paths, vocabulary_sizes)
        return None
    if (
        feature.paths != (profile.path,)
        or feature.shape != (1,)
        or feature.dtype != "int64"
        or feature.history
        or feature.future
        or tuple(vocabulary_sizes) != (12,)
        or feature.normalization != "none"
    ):
        raise ValueError(
            f"Active conditioning requires current-step int64 task_indices at {profile.path!r} "
            "with shape (1,), vocabulary (12,), and no normalization"
        )
    if instruction == "infer_cont0":
        for declaration in schema.features:
            if (
                "cg/entity/container0" in declaration.paths
                or "cg/task/container0" in declaration.paths
            ):
                raise ValueError("Active infer_cont0 must not expose the hidden first container")
    return profile


def validate_policy_conditioning(
    config: RunConfig, family: str, instruction: str, conditioning: str = "static"
) -> ActiveInstructionProfile | None:
    """Check a native policy's categorical contract against its CG experiment."""
    if config.policy.conditioning.type != "indexed":
        raise ValueError("CG task conditioning requires an indexed policy")
    return validate_feature_conditioning(
        config.features, config.policy.conditioning.vocab_sizes, family, instruction, conditioning
    )


def policy_profile(config: RunConfig) -> ActiveInstructionProfile | None:
    """Resolve a checkpoint's immutable profile from its native feature paths.

    Unknown versions fail explicitly. Ordinary native policies without active
    feature paths remain usable through the application training entry point.
    """
    paths = [path for feature in config.features.features for path in feature.paths]
    active = [path for path in paths if path.startswith("cg/active_instruction/")]
    if not active:
        return None
    if len(active) != 1:
        raise ValueError("Exactly one active instruction feature is required")
    profile = get_profile(active[0].removeprefix("cg/active_instruction/"))
    return validate_policy_conditioning(config, profile.family, profile.instruction, "active_v1")


def validate_finetune_conditioning(target: RunConfig, source: RunConfig) -> None:
    """Reject equally sized categorical vocabularies with different meanings."""
    target_profile, source_profile = policy_profile(target), policy_profile(source)
    if target_profile != source_profile:
        raise ValueError("Fine-tuning requires matching active instruction profiles")
