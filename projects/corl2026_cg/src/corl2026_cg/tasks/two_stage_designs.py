"""Reproducible four-factor designs for distinct-object, distinct-container tasks.

These are reconstructed coverage designs, not aliases for historical generators.
Rows are ``(object0, container0, object1, container1)``. All algorithms use the
same lexicographically ordered, 144-row feasible universe.
"""

from __future__ import annotations

import random
import re
from functools import lru_cache
from itertools import combinations, permutations, product

Row = tuple[int, int, int, int]
Target = tuple[tuple[int, ...], tuple[int, ...]]
DESIGN_NAMES = ("full", "pairwise_cover", "pairwise_min", "threewise_cover")

# Every feasible object0/container0 pair requires a different row, so 16 is a
# lower bound. This fixed cover attains it without a runtime solver dependency.
PAIRWISE_MIN: tuple[Row, ...] = (
    (0, 0, 2, 2),
    (0, 1, 3, 0),
    (0, 2, 1, 3),
    (0, 3, 2, 1),
    (1, 0, 3, 3),
    (1, 1, 2, 0),
    (1, 2, 3, 1),
    (1, 3, 0, 2),
    (2, 0, 1, 1),
    (2, 1, 0, 3),
    (2, 2, 0, 0),
    (2, 3, 3, 2),
    (3, 0, 0, 1),
    (3, 1, 1, 2),
    (3, 2, 2, 3),
    (3, 3, 1, 0),
)


@lru_cache(maxsize=1)
def full_rows() -> tuple[Row, ...]:
    """Return all feasible compositions in lexicographic order."""
    return tuple(
        (object0, container0, object1, container1)
        for object0, container0, object1, container1 in product(range(4), repeat=4)
        if object0 != object1 and container0 != container1
    )


def row_targets(row: tuple[int, ...], strength: int) -> frozenset[Target]:
    """Project one categorical row onto every indexed factor subset of a given size.

    Args:
        row: Ordered categorical factor values.
        strength: Number of factors in each projection, from one to row width.

    Returns:
        Targets containing both factor positions and their categorical values.
    """
    if not 1 <= strength <= len(row):
        raise ValueError("Coverage strength must be between one and the row width")
    return frozenset(
        (indices, tuple(row[index] for index in indices))
        for indices in combinations(range(len(row)), strength)
    )


@lru_cache(maxsize=2)
def _greedy_cover(strength: int) -> tuple[Row, ...]:
    """Cover feasible targets, resolving equal marginal gains lexicographically."""
    candidates = full_rows()
    targets = {row: row_targets(row, strength) for row in candidates}
    uncovered = set().union(*targets.values())
    selected: list[Row] = []
    while uncovered:
        # max returns the first tie; candidates are already lexicographically sorted.
        row = max(candidates, key=lambda candidate: len(targets[candidate] & uncovered))
        selected.append(row)
        uncovered.difference_update(targets[row])
    return tuple(selected)


def _random_coverage(budget: int, seed: int) -> tuple[Row, ...]:
    """Generate a random four-row coverage backbone, then sample remaining rows.

    The backbone independently permutes each first-stage factor and samples
    second-stage derangements uniformly. Additional rows are sampled without
    replacement. This construction guarantees coverage even at budget four;
    it does not claim uniform sampling over all coverage-constrained designs.
    """
    if not 4 <= budget <= len(full_rows()):
        raise ValueError("Four-factor coverage budget must be between 4 and 144")
    generator = random.Random(seed)
    object0 = generator.sample(range(4), 4)
    container0 = generator.sample(range(4), 4)
    object1 = generator.choice(
        [row for row in permutations(range(4)) if all(a != b for a, b in zip(object0, row))]
    )
    container1 = generator.choice(
        [row for row in permutations(range(4)) if all(a != b for a, b in zip(container0, row))]
    )
    backbone = tuple(zip(object0, container0, object1, container1))
    remaining = [row for row in full_rows() if row not in backbone]
    return tuple(sorted((*backbone, *generator.sample(remaining, budget - 4))))


def resolve_rows(name: str) -> tuple[Row, ...]:
    """Resolve a built-in or ``random_coverage_B{budget}_s{seed}`` design.

    Args:
        name: Reconstructed four-factor design name.

    Returns:
        Unique feasible rows. Greedy covers retain selection order; random
        designs are sorted to decouple dataset selection from generation order.
    """
    if name == "full":
        return full_rows()
    if name == "pairwise_min":
        return PAIRWISE_MIN
    if name == "pairwise_cover":
        return _greedy_cover(2)
    if name == "threewise_cover":
        return _greedy_cover(3)
    match = re.fullmatch(r"random_coverage_B(\d+)_s(\d+)", name)
    if match is not None:
        return _random_coverage(int(match[1]), int(match[2]))
    raise ValueError(f"Unknown four-factor two-stage design: {name!r}")
