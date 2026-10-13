"""CG task vocabulary and application-owned instruction semantics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

OBJECTS = ("cross", "cube", "cylinder", "milk")
CONTAINERS = ("bin", "mug", "plate", "mug_no_handle")
BUTTONS = ("red", "green", "blue", "yellow")


@dataclass(frozen=True)
class CGTask:
    """A concrete dataset task, independently of its visible instruction.

    Attributes:
        family: One of ``l4``, ``l4_2s``, or ``l5``.
        object0: First object vocabulary index.
        container0: First destination vocabulary index.
        object1: Second object index for two-stage L4.
        container1: Second destination index for two-stage L4.
        button: Button color index for L5.
    """

    family: str
    object0: int
    container0: int
    object1: int | None = None
    container1: int | None = None
    button: int | None = None

    def __post_init__(self) -> None:
        """Reject invalid categorical IDs and ambiguous task variants."""
        if self.family not in {"l4", "l4_2s", "l5"}:
            raise ValueError(f"Unknown CG task family: {self.family!r}")
        for name, value in semantic_values(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 4:
                raise ValueError(f"Invalid {name} index: {value!r}")
        if self.family == "l4_2s":
            if self.object1 is None or self.container1 is None or self.button is not None:
                raise ValueError("Two-stage L4 requires exactly two object/container pairs")
            if self.object0 == self.object1 or self.container0 == self.container1:
                raise ValueError("Two-stage L4 requires distinct objects and containers")
        elif self.object1 is not None or self.container1 is not None:
            raise ValueError("Only two-stage L4 accepts a second object/container pair")
        if (self.family == "l5") != (self.button is not None):
            raise ValueError("A button color is required only for L5")

    @property
    def slug(self) -> str:
        """Return the canonical concrete dataset identifier."""
        first = f"{OBJECTS[self.object0]}_into_{CONTAINERS[self.container0]}"
        if self.family == "l4_2s":
            assert self.object1 is not None and self.container1 is not None
            return f"{first}__{OBJECTS[self.object1]}_into_{CONTAINERS[self.container1]}"
        if self.family == "l5":
            assert self.button is not None
            return f"{first}_press_{BUTTONS[self.button]}"
        return first


def _parse_pair(slug: str) -> tuple[int, int]:
    """Parse one exact object/container pair; never default unknown words to zero."""
    parts = slug.split("_into_")
    if len(parts) != 2 or parts[0] not in OBJECTS or parts[1] not in CONTAINERS:
        raise ValueError(f"Invalid object/container task slug: {slug!r}")
    return OBJECTS.index(parts[0]), CONTAINERS.index(parts[1])


def parse_task(slug: str) -> CGTask:
    """Parse an existing L4, two-stage L4, or L5 concrete dataset slug.

    Args:
        slug: Exact vocabulary names separated by ``_into_``, ``__``, or ``_press_``.

    Returns:
        A validated task whose ``slug`` property round-trips its input.
    """
    if not isinstance(slug, str) or not slug or slug != slug.strip():
        raise ValueError(f"Expected a nonempty canonical task slug, got {slug!r}")
    if "__" in slug:
        stages = slug.split("__")
        if len(stages) != 2:
            raise ValueError(f"Two-stage tasks require exactly two stages: {slug!r}")
        return CGTask("l4_2s", *_parse_pair(stages[0]), *_parse_pair(stages[1]))
    if "_press_" in slug:
        parts = slug.split("_press_")
        if len(parts) != 2 or parts[1] not in BUTTONS:
            raise ValueError(f"Invalid L5 button task: {slug!r}")
        return CGTask("l5", *_parse_pair(parts[0]), button=BUTTONS.index(parts[1]))
    return CGTask("l4", *_parse_pair(slug))


def semantic_values(task: CGTask) -> dict[str, int]:
    """Expose categorical task fields and an explicit zero-valued constant slot."""
    values = {"object0": task.object0, "container0": task.container0, "constant": 0}
    for name in ("object1", "container1", "button"):
        value = getattr(task, name)
        if value is not None:
            values[name] = value
    return values


def instruction_fields(family: str, instruction: str) -> tuple[str, ...]:
    """Return ordered application semantic fields for a supported instruction mode."""
    fields: dict[str, dict[str, tuple[str, ...]]] = {
        "l4": {
            "full": ("object0", "container0"),
            "object_only": ("object0",),
            "container_only": ("container0",),
            "container_latents": ("container0", "constant", "constant"),
        },
        "l4_2s": {
            "full": ("object0", "container0", "object1", "container1"),
            "open_cont": ("object0", "container0", "object1"),
            "infer_cont0": ("object0", "object1", "container1"),
        },
        "l5": {"full": ("object0", "container0", "button")},
    }
    if family not in fields or instruction not in fields[family]:
        raise ValueError(f"Unsupported instruction {instruction!r} for family {family!r}")
    return fields[family][instruction]


def vocab_sizes(family: str, instruction: str) -> tuple[int, ...]:
    """Return generic core embedding-table sizes, independently of task names."""
    return tuple(
        1 if field == "constant" else 4 for field in instruction_fields(family, instruction)
    )


def validate_conditioning(
    family: str,
    instruction: str,
    paths: Sequence[str],
    vocabulary_sizes: Sequence[int] | None = None,
) -> None:
    """Require saved categorical features to mean the requested CG instruction.

    Args:
        family: Requested task family.
        instruction: Requested visible instruction layout.
        paths: Ordered HDF5 paths from the checkpoint's task_indices feature.
        vocabulary_sizes: Optional saved embedding-table widths. Policy-loading
            boundaries supply these; observation-only adapters can validate paths.

    Raises:
        ValueError: Feature semantics or supplied embedding widths do not match.
    """
    expected = tuple(f"cg/task/{field}" for field in instruction_fields(family, instruction))
    if tuple(paths) != expected:
        raise ValueError(
            f"Checkpoint task_indices paths do not match {family}/{instruction}: "
            f"expected {expected}, received {tuple(paths)}"
        )
    if vocabulary_sizes is not None and tuple(vocabulary_sizes) != vocab_sizes(family, instruction):
        raise ValueError(f"Checkpoint vocabulary sizes do not match {family}/{instruction}")


def instruction_indices(task: CGTask, instruction: str) -> tuple[int, ...]:
    """Project a concrete task onto its policy-visible categorical instruction."""
    values = semantic_values(task)
    return tuple(values[field] for field in instruction_fields(task.family, instruction))
