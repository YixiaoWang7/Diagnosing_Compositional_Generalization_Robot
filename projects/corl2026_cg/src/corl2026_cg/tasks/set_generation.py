"""Pure-Python CG design algorithms, ported from dev_2s:src/policies/tools/set_generation.py."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import combinations, product
from typing import Literal

Row = tuple[int, ...]
LevelPair = tuple[int, int]
StagePair = tuple[int, int]
PairTarget = tuple[int, int, int, int]
DesignMethod = Literal["orthogonal", "greedy", "custom"]


@dataclass(frozen=True)
class DesignResult:
    """Generated row design plus the requirements used to build it."""

    rows: list[Row]
    method: str
    targets: set[PairTarget]
    stage_vectors: list[Row] | None = None
    pair_targets: set[LevelPair] | None = None
    custom_pair_targets: dict[StagePair, set[LevelPair]] | None = None


def is_prime(q: int) -> bool:
    """Check whether q is prime."""
    if q < 2:
        return False
    if q == 2:
        return True
    if q % 2 == 0:
        return False

    d = 3
    while d * d <= q:
        if q % d == 0:
            return False
        d += 2

    return True


def required_dimension(q: int, n_stages: int) -> int:
    """
    Smallest d such that projective directions in F_q^d support n_stages.

    The number of available projective directions is (q^d - 1) / (q - 1).
    This pure-Python implementation assumes q is prime.
    """
    if n_stages <= 0:
        raise ValueError("n_stages must be positive.")

    d = 1
    while (q**d - 1) // (q - 1) < n_stages:
        d += 1
    return d


def canonical_direction(vec: Row, q: int) -> Row:
    """
    Normalize a nonzero vector over F_q into one projective direction.

    Two scalar-multiple vectors represent the same direction. We normalize by
    making the first nonzero coordinate equal to 1.
    """
    vec = tuple(v % q for v in vec)

    if all(v == 0 for v in vec):
        raise ValueError("Zero vector has no projective direction.")

    for v in vec:
        if v != 0:
            inv = pow(v, -1, q)
            return tuple((x * inv) % q for x in vec)

    raise RuntimeError("Unexpected vector normalization failure.")


def projective_directions(q: int, d: int) -> list[Row]:
    """Generate one representative from each one-dimensional direction in F_q^d."""
    seen = set()
    directions = []

    for vec in product(range(q), repeat=d):
        if all(v == 0 for v in vec):
            continue

        direction = canonical_direction(vec, q)
        if direction not in seen:
            seen.add(direction)
            directions.append(direction)

    return directions


def dot_mod(a: Row, z: Row, q: int) -> int:
    """Dot product over F_q, assuming q is prime."""
    return sum(ai * zi for ai, zi in zip(a, z)) % q


def generate_orthogonal_array(
    n_stages: int,
    n_levels: int,
    d: int | None = None,
) -> tuple[list[Row], list[Row]]:
    """
    Generate an OA(q^d, n_stages, q, 2) using finite-field arithmetic.

    This guarantees every pair of stages sees every q^2 ordered level pair
    equally often. The number of rows is q^d.

    This pure-Python version requires n_levels to be prime. For q = 4, 8, 9,
    etc., use a finite-field library such as `galois`.
    """
    q = int(n_levels)

    if not is_prime(q):
        raise ValueError(
            "This pure-Python orthogonal-array generator requires n_levels "
            "to be prime. For prime powers like 4, use finite-field arithmetic "
            "over GF(q), not integer arithmetic modulo q."
        )

    if d is None:
        d = required_dimension(q, n_stages)

    directions = projective_directions(q, int(d))
    if len(directions) < n_stages:
        raise ValueError(
            f"Not enough projective directions: got {len(directions)}, need {n_stages}. Increase d."
        )

    stage_vectors = directions[:n_stages]
    rows = [
        tuple(dot_mod(a, z, q) for a in stage_vectors) for z in product(range(q), repeat=int(d))
    ]

    return rows, stage_vectors


def all_level_pairs(n_levels: int) -> set[LevelPair]:
    """All ordered level pairs for q levels."""
    return {(first, second) for first in range(n_levels) for second in range(n_levels)}


def adjacent_target_pairs(n_levels: int, cyclic: bool = False) -> set[LevelPair]:
    """
    Build adjacent-level pair targets.

    Non-cyclic:
        (0,0), (0,1), (1,1), (1,2), ..., (q-1,q-1)

    Cyclic also includes:
        (q-1,0)
    """
    q = int(n_levels)
    targets = {(a, a) for a in range(q)}
    targets.update((a, a + 1) for a in range(q - 1))

    if cyclic and q > 1:
        targets.add((q - 1, 0))

    return targets


def validate_stage_pair(stage_pair: StagePair, n_stages: int) -> None:
    """Validate that a stage pair is ordered and inside the design width."""
    i, j = stage_pair
    if not (0 <= i < j < n_stages):
        raise ValueError(f"Stage pair must satisfy 0 <= i < j < {n_stages}, got {stage_pair}.")


def validate_level_pair(level_pair: LevelPair, n_levels: int) -> None:
    """Validate that a level pair is inside [0, n_levels)."""
    a, b = level_pair
    if not (0 <= a < n_levels and 0 <= b < n_levels):
        raise ValueError(f"Level pair must be inside [0, {n_levels}), got {level_pair}.")


def all_pair_stage_targets(
    n_stages: int,
    pair_targets: Iterable[LevelPair],
    *,
    n_levels: int | None = None,
) -> set[PairTarget]:
    """
    Apply the same required level-pair set to every stage pair.

    Each target is represented as (i, j, a, b), meaning x_i=a and x_j=b should
    appear in at least one generated row.
    """
    pair_targets = set(pair_targets)
    if n_levels is not None:
        for pair in pair_targets:
            validate_level_pair(pair, int(n_levels))

    targets = set()
    for i, j in combinations(range(n_stages), 2):
        for a, b in pair_targets:
            targets.add((i, j, a, b))
    return targets


def custom_pair_stage_targets(
    n_stages: int,
    selected_pairs_by_stage_pair: dict[StagePair, Iterable[LevelPair]],
    *,
    n_levels: int | None = None,
    default_pair_targets: Iterable[LevelPair] | None = None,
) -> set[PairTarget]:
    """
    Build pairwise targets where each stage pair may request different pairs.

    If default_pair_targets is None, only explicitly listed stage pairs create
    requirements. If provided, unlisted stage pairs use the default set.
    """
    normalized = {
        stage_pair: set(level_pairs)
        for stage_pair, level_pairs in selected_pairs_by_stage_pair.items()
    }

    targets = set()
    for i, j in combinations(range(n_stages), 2):
        stage_pair = (i, j)
        validate_stage_pair(stage_pair, n_stages)

        if stage_pair in normalized:
            pair_targets = normalized[stage_pair]
        elif default_pair_targets is not None:
            pair_targets = set(default_pair_targets)
        else:
            continue

        for pair in pair_targets:
            if n_levels is not None:
                validate_level_pair(pair, int(n_levels))
            a, b = pair
            targets.add((i, j, a, b))

    for stage_pair in normalized:
        validate_stage_pair(stage_pair, n_stages)

    return targets


def targets_covered_by_row(row: Row, targets: Iterable[PairTarget]) -> set[PairTarget]:
    """Return the target set covered by one row."""
    return {
        target for target in targets if row[target[0]] == target[2] and row[target[1]] == target[3]
    }


def validate_candidate_row(row: Row, n_stages: int, n_levels: int) -> None:
    """Validate candidate row width and level range."""
    if len(row) != n_stages:
        raise ValueError(f"Row {row} has width {len(row)}, expected {n_stages}.")
    for value in row:
        if not 0 <= value < n_levels:
            raise ValueError(f"Row {row} contains level {value}, outside [0, {n_levels}).")


def candidate_rows_from_constraint(
    n_stages: int,
    n_levels: int,
    constraint: Callable[[Row], bool] | None = None,
) -> list[Row]:
    """
    Build candidate rows separately from the design algorithm.

    Use this for constraints such as:

        lambda row: row[0] != row[2] and row[1] != row[3]

    Pass the returned rows to generate_design(..., candidate_rows=...).
    """
    rows = product(range(n_levels), repeat=n_stages)
    if constraint is None:
        return list(rows)
    return [row for row in rows if bool(constraint(row))]


def greedy_cover_design(
    n_stages: int,
    n_levels: int,
    targets: Iterable[PairTarget],
    candidate_rows: Iterable[Row] | None = None,
) -> list[Row]:
    """
    Deterministic greedy set-cover design generator.

    At each step, choose the candidate row that covers the most currently
    uncovered pairwise targets. This is not guaranteed to produce the minimum
    number of rows, but it will find a cover if the requested targets are
    coverable by the candidate pool.
    """
    targets = set(targets)
    if not targets:
        return []

    if candidate_rows is None:
        candidate_rows = product(range(n_levels), repeat=n_stages)

    candidate_rows = list(candidate_rows)
    if not candidate_rows:
        raise ValueError("candidate_rows is empty.")

    for row in candidate_rows:
        validate_candidate_row(row, n_stages, n_levels)

    uncovered = set(targets)
    design = []

    while uncovered:
        best_row = None
        best_cover: set[PairTarget] = set()

        for row in candidate_rows:
            cover = targets_covered_by_row(row, uncovered)
            if len(cover) > len(best_cover):
                best_cover = cover
                best_row = row

        if best_row is None or not best_cover:
            sample = next(iter(uncovered))
            raise ValueError(
                "Some targets cannot be covered by the candidate rows. "
                f"Example uncovered target: {sample}."
            )

        design.append(best_row)
        uncovered -= best_cover

    return design


def generate_adjacent_level_design(
    n_stages: int,
    n_levels: int,
    cyclic: bool = False,
    candidate_rows: Iterable[Row] | None = None,
) -> tuple[list[Row], set[LevelPair]]:
    """Compatibility wrapper for adjacent-level greedy designs."""
    pair_targets = adjacent_target_pairs(n_levels, cyclic=cyclic)
    targets = all_pair_stage_targets(n_stages, pair_targets, n_levels=n_levels)
    design = greedy_cover_design(
        n_stages=n_stages,
        n_levels=n_levels,
        targets=targets,
        candidate_rows=candidate_rows,
    )
    return design, pair_targets


def generate_design(
    n_stages: int,
    n_levels: int,
    *,
    method: DesignMethod = "greedy",
    pair_targets: Iterable[LevelPair] | None = None,
    custom_pair_targets: dict[StagePair, Iterable[LevelPair]] | None = None,
    cyclic: bool = False,
    d: int | None = None,
    candidate_rows: Iterable[Row] | None = None,
) -> DesignResult:
    """
    Unified API for staged training-set row generation.

    method="orthogonal":
        Full pairwise orthogonal array. Requires prime n_levels in this
        pure-Python implementation. Ignores pair_targets/custom_pair_targets.

    method="greedy":
        Deterministic greedy cover using the same pair_targets for every stage
        pair. If pair_targets is None, adjacent_target_pairs(n_levels, cyclic)
        is used.

    method="custom":
        Deterministic greedy cover with different requirements per stage pair.
        custom_pair_targets is required. If pair_targets is provided, it is used
        as the default requirement set for unlisted stage pairs.
    """
    n_stages = int(n_stages)
    n_levels = int(n_levels)
    if n_stages <= 0:
        raise ValueError("n_stages must be positive.")
    if n_levels <= 0:
        raise ValueError("n_levels must be positive.")

    if method == "orthogonal":
        if candidate_rows is not None:
            raise ValueError("candidate_rows is not used with method='orthogonal'.")
        rows, stage_vectors = generate_orthogonal_array(
            n_stages=n_stages,
            n_levels=n_levels,
            d=d,
        )
        full_pairs = all_level_pairs(n_levels)
        targets = all_pair_stage_targets(n_stages, full_pairs, n_levels=n_levels)
        return DesignResult(
            rows=rows,
            method=method,
            targets=targets,
            stage_vectors=stage_vectors,
            pair_targets=full_pairs,
        )

    if method == "greedy":
        if custom_pair_targets is not None:
            raise ValueError("Use method='custom' with custom_pair_targets.")
        pair_targets = (
            adjacent_target_pairs(n_levels, cyclic=cyclic)
            if pair_targets is None
            else set(pair_targets)
        )
        targets = all_pair_stage_targets(n_stages, pair_targets, n_levels=n_levels)
        rows = greedy_cover_design(
            n_stages=n_stages,
            n_levels=n_levels,
            targets=targets,
            candidate_rows=candidate_rows,
        )
        return DesignResult(
            rows=rows,
            method=method,
            targets=targets,
            pair_targets=set(pair_targets),
        )

    if method == "custom":
        if custom_pair_targets is None:
            raise ValueError("custom_pair_targets is required with method='custom'.")
        default_pair_targets = None if pair_targets is None else set(pair_targets)
        targets = custom_pair_stage_targets(
            n_stages,
            custom_pair_targets,
            n_levels=n_levels,
            default_pair_targets=default_pair_targets,
        )
        rows = greedy_cover_design(
            n_stages=n_stages,
            n_levels=n_levels,
            targets=targets,
            candidate_rows=candidate_rows,
        )
        return DesignResult(
            rows=rows,
            method=method,
            targets=targets,
            pair_targets=default_pair_targets,
            custom_pair_targets={
                stage_pair: set(level_pairs)
                for stage_pair, level_pairs in custom_pair_targets.items()
            },
        )

    raise ValueError(f"Unknown method: {method!r}.")


def check_pairwise_coverage(
    rows: Iterable[Row],
    n_stages: int,
    n_levels: int,
    targets: Iterable[PairTarget] | None = None,
) -> dict[StagePair, dict[str, int | set[LevelPair]]]:
    """
    Report observed ordered level pairs for every stage pair.

    If targets is provided, the report also includes missing requested targets.
    """
    rows = list(rows)
    all_pairs = all_level_pairs(n_levels)
    targets_by_stage_pair: dict[StagePair, set[LevelPair]] = {}

    if targets is not None:
        for i, j, a, b in targets:
            targets_by_stage_pair.setdefault((i, j), set()).add((a, b))

    result: dict[StagePair, dict[str, int | set[LevelPair]]] = {}
    for i, j in combinations(range(n_stages), 2):
        seen = {(row[i], row[j]) for row in rows}
        entry: dict[str, int | set[LevelPair]] = {
            "seen": seen,
            "missing": all_pairs - seen,
            "n_seen": len(seen),
            "n_missing": len(all_pairs - seen),
        }
        if targets is not None:
            requested = targets_by_stage_pair.get((i, j), set())
            entry["missing_targets"] = requested - seen
            entry["n_missing_targets"] = len(requested - seen)
        result[(i, j)] = entry

    return result


def covers_targets(rows: Iterable[Row], targets: Iterable[PairTarget]) -> bool:
    """Return True if rows cover all requested targets."""
    targets = set(targets)
    covered = set()
    for row in rows:
        covered.update(targets_covered_by_row(row, targets))
    return covered >= targets


def print_rows(rows: Iterable[Row]) -> None:
    """Pretty-print rows as compact strings."""
    for row in rows:
        print("".join(str(x) for x in row))


def print_design(rows: Iterable[Row], factor_names: list[str] | None = None) -> None:
    """Pretty-print rows as a small table."""
    rows = list(rows)
    if not rows:
        print("(empty design)")
        return

    n_stages = len(rows[0])
    if factor_names is None:
        factor_names = [f"F{i + 1}" for i in range(n_stages)]

    header = ["Run"] + factor_names
    body = [[r + 1] + list(row) for r, row in enumerate(rows)]
    widths = [max(len(str(x)) for x in col) for col in zip(header, *body)]

    print(" | ".join(str(h).rjust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))

    for row in body:
        print(" | ".join(str(v).rjust(w) for v, w in zip(row, widths)))
