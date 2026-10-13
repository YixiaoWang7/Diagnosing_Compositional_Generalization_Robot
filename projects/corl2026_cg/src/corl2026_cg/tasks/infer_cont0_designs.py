"""
Task-set design utilities for the L4 two-stage *infer-cont0* variant.

Ported from dev_2s_1s at 7455b6c; current generator definitions are authoritative.

Design space
------------
The policy sees three visible query tokens: ``[obj0, obj1, cont1]``.
``cont0`` is hidden/open: it is used only to materialize a concrete dataset
slug, while evaluation accepts obj0 placed into any container except
``cont1``.

Each concrete row is an integer tuple ``(obj0_idx, obj1_idx, cont1_idx)`` where:

- ``obj0_idx``, ``obj1_idx`` index into :data:`OBJECTS` (4 levels).
- ``cont1_idx`` indexes into :data:`CONTAINERS` (4 levels).
- Constraint: ``obj0_idx != obj1_idx``.

Full visible space: 4 * 3 * 4 = 48 rows.

Design coverage is measured in local factors ``(o, r, c)``:

- ``o`` is the global obj0 index (4 levels).
- ``r`` is obj1's rank among objects other than obj0 (3 levels).
- ``c`` is the global cont1 index (4 levels).

Slug convention
---------------
Two-stage slugs look like ``<obj0>_into_<cont0>__<obj1>_into_<cont1>``.
Because ``cont0`` is not part of the visible query, each visible row has one
canonical representative slug for naming/eval configs and three concrete
training slugs.  The representative prefers ``cont0_idx = obj0_idx`` and falls
back to the next container when that would equal ``cont1``.  For example,
``(0, 1, 2)`` has representative ``(0, 0, 1, 2)`` and training slugs
``(0, 0/1/3, 1, 2)``.
:func:`slug_to_row` discards the ``cont0`` component when parsing back.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

from .schema import CONTAINERS, OBJECTS

Row = tuple[int, ...]
DesignRow = tuple[int, int, int]
FullRow = tuple[int, int, int, int]
Spec = dict[str, Any]

N_FACTORS = 3
N_OBJ_LEVELS = 4
N_CONT_LEVELS = 4


DEFAULT_RANDOM_SEEDS: tuple[int, ...] = (0, 1)
DEFAULT_BUDGETS: tuple[int, ...] = (8, 10, 12, 16, 20, 24, 30, 36, 48)
COVK_B16_LEVELS: tuple[int, ...] = tuple(range(6, N_OBJ_LEVELS * N_CONT_LEVELS + 1))


# ---------------------------------------------------------------------------
# Row / slug helpers
# ---------------------------------------------------------------------------


def canonical_hidden_cont0(row: Row) -> int:
    """Representative hidden cont0 used only to materialize a concrete slug."""
    obj0_idx, _obj1_idx, cont1_idx = row
    cont0_idx = int(obj0_idx) % N_CONT_LEVELS
    if cont0_idx == cont1_idx:
        cont0_idx = (cont0_idx + 1) % N_CONT_LEVELS
    return cont0_idx


def _rank_excluding(value: int, excluded: int, n_levels: int) -> int:
    levels = [i for i in range(n_levels) if i != excluded]
    return levels.index(value)


def _value_from_rank(rank: int, excluded: int, n_levels: int) -> int:
    levels = [i for i in range(n_levels) if i != excluded]
    return levels[rank]


def row_to_design_row(row: Row) -> DesignRow:
    """Convert global ``(obj0, obj1, cont1)`` into local ``(o, r, c)`` factors."""
    obj0_idx, obj1_idx, cont1_idx = row
    return (
        obj0_idx,
        _rank_excluding(obj1_idx, obj0_idx, N_OBJ_LEVELS),
        cont1_idx,
    )


def design_row_to_row(row: DesignRow) -> Row:
    """Convert local ``(o, r, c)`` design factors into global ``(obj0, obj1, cont1)``."""
    obj0_idx, rel_obj1_idx, cont1_idx = row
    return (
        obj0_idx,
        _value_from_rank(rel_obj1_idx, obj0_idx, N_OBJ_LEVELS),
        cont1_idx,
    )


def row_to_full_row(row: Row) -> FullRow:
    """Map visible ``(obj0, obj1, cont1)`` into one concrete ``(obj0, cont0, obj1, cont1)`` task."""
    obj0, obj1, cont1 = row
    return (obj0, canonical_hidden_cont0(row), obj1, cont1)


def row_to_training_full_rows(row: Row) -> list[FullRow]:
    """All concrete training slugs for a visible row, varying hidden cont0."""
    obj0, obj1, cont1 = row
    return [(obj0, cont0, obj1, cont1) for cont0 in range(N_CONT_LEVELS) if cont0 != cont1]


def rows_to_full_rows(rows: list[Row]) -> list[FullRow]:
    return [row_to_full_row(row) for row in rows]


def rows_to_training_full_rows(rows: list[Row]) -> list[FullRow]:
    return [full_row for row in rows for full_row in row_to_training_full_rows(row)]


def full_row_to_slug(row: FullRow) -> str:
    obj0, cont0, obj1, cont1 = row
    return f"{OBJECTS[obj0]}_into_{CONTAINERS[cont0]}__{OBJECTS[obj1]}_into_{CONTAINERS[cont1]}"


def row_to_slug(row: Row) -> str:
    """Return the canonical concrete slug represented by one visible row."""
    return full_row_to_slug(row_to_full_row(row))


def rows_to_slugs(rows: list[Row]) -> list[str]:
    return [row_to_slug(row) for row in rows]


def row_to_training_slugs(row: Row) -> list[str]:
    return [full_row_to_slug(full_row) for full_row in row_to_training_full_rows(row)]


def rows_to_training_slugs(rows: list[Row]) -> list[str]:
    return [slug for row in rows for slug in row_to_training_slugs(row)]


def slug_to_row(slug: str) -> Row:
    """Parse a two-stage slug back to ``(obj0, obj1, cont1)``, ignoring cont0."""
    parts = slug.strip().lower().split("__")
    if len(parts) != 2:
        raise ValueError(f"Expected slug with '__' separator, got: {slug!r}")

    def _parse_stage(stage: str) -> tuple[str, str]:
        if "_into_" not in stage:
            raise ValueError(f"Expected '<obj>_into_<cont>', got: {stage!r}")
        obj, cont = stage.split("_into_", 1)
        return obj.strip(), cont.strip()

    obj0_name, _cont0_name = _parse_stage(parts[0])
    obj1_name, cont1_name = _parse_stage(parts[1])
    return (
        OBJECTS.index(obj0_name),
        OBJECTS.index(obj1_name),
        CONTAINERS.index(cont1_name),
    )


# ---------------------------------------------------------------------------
# Design space
# ---------------------------------------------------------------------------


def all_rows() -> list[Row]:
    """All 48 valid visible rows with obj0 != obj1."""
    return [
        design_row_to_row((o, r, c))
        for o in range(N_OBJ_LEVELS)
        for r in range(N_OBJ_LEVELS - 1)
        for c in range(N_CONT_LEVELS)
    ]


def is_valid(row: Row) -> bool:
    return row[0] != row[1]


def validate_budget(budget: int) -> int:
    budget = int(budget)
    max_budget = len(all_rows())
    if not 0 < budget <= max_budget:
        raise ValueError(f"budget must be in [1, {max_budget}], got {budget}.")
    return budget


# ---------------------------------------------------------------------------
# Coverage & metrics
# ---------------------------------------------------------------------------


def factor_counts(rows: list[Row]) -> list[list[int]]:
    """counts[i][v] = number of rows with factor i at level v."""
    levels_per_factor = [N_OBJ_LEVELS, N_OBJ_LEVELS - 1, N_CONT_LEVELS]
    counts: list[list[int]] = []
    for i in range(N_FACTORS):
        counter = Counter(row_to_design_row(row)[i] for row in rows)
        counts.append([counter[v] for v in range(levels_per_factor[i])])
    return counts


def covers_all_single_levels(rows: list[Row]) -> bool:
    """True if every local design level appears for o, r, and c."""
    levels_per_factor = [N_OBJ_LEVELS, N_OBJ_LEVELS - 1, N_CONT_LEVELS]
    return all(
        {row_to_design_row(row)[i] for row in rows} == set(range(levels_per_factor[i]))
        for i in range(N_FACTORS)
    )


def pairwise_counts(rows: list[Row]) -> defaultdict[tuple[int, int, int, int], int]:
    counts: defaultdict[tuple[int, int, int, int], int] = defaultdict(int)
    for row in rows:
        design_row = row_to_design_row(row)
        for i in range(N_FACTORS):
            for j in range(i + 1, N_FACTORS):
                counts[(i, j, design_row[i], design_row[j])] += 1
    return counts


def unique_pairwise_coverage(rows: list[Row]) -> int:
    return sum(1 for v in pairwise_counts(rows).values() if v > 0)


def pairwise_coverage_by_pair(rows: list[Row]) -> dict[str, dict[str, int]]:
    counts = pairwise_counts(rows)
    pair_names = {
        (0, 1): "o_r",
        (0, 2): "o_c",
        (1, 2): "r_c",
    }
    levels_per_factor = [N_OBJ_LEVELS, N_OBJ_LEVELS - 1, N_CONT_LEVELS]
    out: dict[str, dict[str, int]] = {}
    for (i, j), name in pair_names.items():
        covered = sum(
            1 for key, value in counts.items() if key[0] == i and key[1] == j and value > 0
        )
        out[name] = {
            "covered": covered,
            "total": levels_per_factor[i] * levels_per_factor[j],
        }
    return out


def feasible_pairwise_targets() -> set[tuple[int, int, int, int]]:
    """All pairwise targets in local ``(o, r, c)`` coordinates."""
    targets: set[tuple[int, int, int, int]] = set()
    for row in all_rows():
        design_row = row_to_design_row(row)
        for i in range(N_FACTORS):
            for j in range(i + 1, N_FACTORS):
                targets.add((i, j, design_row[i], design_row[j]))
    return targets


def max_pairwise_coverage() -> int:
    return len(feasible_pairwise_targets())


def one_way_imbalance(rows: list[Row]) -> float:
    levels_per_factor = [N_OBJ_LEVELS, N_OBJ_LEVELS - 1, N_CONT_LEVELS]
    total = 0.0
    for i, fc in enumerate(factor_counts(rows)):
        target = len(rows) / levels_per_factor[i]
        total += sum((c - target) ** 2 for c in fc)
    return total


def design_score(rows: list[Row], *, w_one: float = 0.1, w_unique: float = -0.1) -> float:
    return w_one * one_way_imbalance(rows) + w_unique * unique_pairwise_coverage(rows)


def design_metrics(rows: list[Row]) -> dict[str, Any]:
    return {
        "budget": len(rows),
        "concrete_task_count": len(rows_to_full_rows(rows)),
        "training_concrete_task_count": len(rows_to_training_full_rows(rows)),
        "covers_all_single_levels": covers_all_single_levels(rows),
        "factor_counts": factor_counts(rows),
        "unique_pairwise_coverage": unique_pairwise_coverage(rows),
        "max_pairwise_coverage": max_pairwise_coverage(),
        "pairwise_coverage_by_pair": pairwise_coverage_by_pair(rows),
        "one_way_imbalance": one_way_imbalance(rows),
        "design_score": design_score(rows),
    }


# ---------------------------------------------------------------------------
# Row generators
# ---------------------------------------------------------------------------


def full_rows() -> list[Row]:
    return all_rows()


def random_coverage_rows(
    budget: int,
    *,
    seed: int | None = None,
    max_trials: int = 100_000,
) -> list[Row]:
    """Randomly sample rows ensuring every factor level appears."""
    budget = validate_budget(budget)
    if budget < N_OBJ_LEVELS:
        raise ValueError(f"budget={budget} is too small to cover all {N_OBJ_LEVELS} levels.")

    rng = random.Random(seed)
    universe = all_rows()
    for _ in range(max_trials):
        rows = rng.sample(universe, budget)
        if covers_all_single_levels(rows):
            return rows

    raise RuntimeError("Failed to sample a random coverage design; increase max_trials or budget.")


def orthogonal_rows() -> list[Row]:
    """
    Minimal 16-row pairwise-coverage design over local factors (o, r, c).

    It covers:
      - 12 (o, r) pairs
      - 16 (o, c) pairs
      - 12 (r, c) pairs
    Total feasible targets: 40.
    """
    # In local coordinates, r = (o + c) mod 3 gives all r levels for every
    # fixed o and all r levels for every fixed c.  The 16 rows are minimal
    # because covering all global (o, c) pairs already requires 4 * 4 rows.
    design = [
        design_row_to_row((o, (o + c) % (N_OBJ_LEVELS - 1), c))
        for o in range(N_OBJ_LEVELS)
        for c in range(N_CONT_LEVELS)
    ]
    if any(not is_valid(row) for row in design):
        raise AssertionError(f"Invalid row in orthogonal design: {design}")
    if unique_pairwise_coverage(design) != max_pairwise_coverage():
        raise AssertionError("Orthogonal design does not cover all feasible pairwise targets.")
    return design


def _rows_from_design_rows(design_rows: list[DesignRow]) -> list[Row]:
    rows = [design_row_to_row(row) for row in design_rows]
    if any(not is_valid(row) for row in rows):
        raise AssertionError(f"Invalid row in design: {rows}")
    return rows


def obj0_cont1_covk_b16_rows(k: int) -> list[Row]:
    """
    16-row fixed-budget design using exactly ``k`` global ``(o, c)`` pairs.

    The set keeps full ``(o, r)`` and ``(r, c)`` pairwise coverage while
    intentionally reducing only ``(o, c)`` coverage.  One-way balance is not
    enforced.  ``k=6`` is the minimum because five ``(o, c)`` pairs can hold at
    most 15 unique rows (three ``r`` levels each).
    """
    k = int(k)
    min_k = 6
    max_k = N_OBJ_LEVELS * N_CONT_LEVELS
    if not min_k <= k <= max_k:
        raise ValueError(f"k must be in [{min_k}, {max_k}], got {k}.")
    if k == max_k:
        return _validate_covk_b16_rows(orthogonal_rows(), k)

    rng = random.Random(10_000 + k)
    all_pairs = [(o, c) for o in range(N_OBJ_LEVELS) for c in range(N_CONT_LEVELS)]
    full_pair_targets = {
        "o_r": N_OBJ_LEVELS * (N_OBJ_LEVELS - 1),
        "r_c": (N_OBJ_LEVELS - 1) * N_CONT_LEVELS,
    }

    for _ in range(100_000):
        pairs = rng.sample(all_pairs, k)
        if {o for o, _c in pairs} != set(range(N_OBJ_LEVELS)):
            continue
        if {c for _o, c in pairs} != set(range(N_CONT_LEVELS)):
            continue

        design_rows: list[DesignRow] = []
        for o, c in pairs:
            ranks = list(range(N_OBJ_LEVELS - 1))
            rng.shuffle(ranks)
            design_rows.append((o, ranks[0], c))

        used = set(design_rows)
        candidates = [
            (o, r, c) for o, c in pairs for r in range(N_OBJ_LEVELS - 1) if (o, r, c) not in used
        ]
        if len(candidates) < 16 - len(design_rows):
            continue
        design_rows.extend(rng.sample(candidates, 16 - len(design_rows)))

        rows = _rows_from_design_rows(sorted(design_rows))
        metrics = design_metrics(rows)
        pair_cov = metrics["pairwise_coverage_by_pair"]
        if (
            metrics["covers_all_single_levels"]
            and pair_cov["o_c"]["covered"] == k
            and pair_cov["o_r"]["covered"] == full_pair_targets["o_r"]
            and pair_cov["r_c"]["covered"] == full_pair_targets["r_c"]
        ):
            return _validate_covk_b16_rows(rows, k)

    raise RuntimeError(f"Failed to generate cov{k}_B16 design; increase max_trials.")


def _validate_covk_b16_rows(rows: list[Row], k: int) -> list[Row]:
    metrics = design_metrics(rows)
    pair_cov = metrics["pairwise_coverage_by_pair"]
    if len(rows) != 16:
        raise AssertionError(f"Expected 16 rows for cov{k}_B16, got {len(rows)}.")
    if pair_cov["o_c"]["covered"] != k:
        raise AssertionError(f"Expected {k}/16 (o,c) coverage, got {pair_cov['o_c']}.")
    if not metrics["covers_all_single_levels"]:
        raise AssertionError(f"cov{k}_B16 does not cover all single levels.")
    if pair_cov["o_r"]["covered"] != pair_cov["o_r"]["total"]:
        raise AssertionError(f"cov{k}_B16 does not cover all (o,r) pairs.")
    if pair_cov["r_c"]["covered"] != pair_cov["r_c"]["total"]:
        raise AssertionError(f"cov{k}_B16 does not cover all (r,c) pairs.")
    return rows


def obj0_cont1_cov6_b16_rows() -> list[Row]:
    """16-row fair-budget set with 6/16 global ``(o,c)`` coverage."""
    return obj0_cont1_covk_b16_rows(6)


def obj0_cont1_cov9_b16_rows() -> list[Row]:
    """16-row fair-budget set with 9/16 global ``(o,c)`` coverage."""
    return obj0_cont1_covk_b16_rows(9)


def obj0_cont1_cov12_b16_rows() -> list[Row]:
    """16-row fair-budget set with 12/16 global ``(o,c)`` coverage."""
    return obj0_cont1_covk_b16_rows(12)


def obj0_cont1_cov4_rows() -> list[Row]:
    """12-row set with full (o,r)/(r,c) coverage and 4/16 (o,c) coverage."""
    return _rows_from_design_rows(
        [
            (0, 0, 0),
            (0, 1, 0),
            (0, 2, 0),
            (1, 0, 1),
            (1, 1, 1),
            (1, 2, 1),
            (2, 0, 2),
            (2, 1, 2),
            (2, 2, 2),
            (3, 0, 3),
            (3, 1, 3),
            (3, 2, 3),
        ]
    )


def obj0_cont1_cov6_rows() -> list[Row]:
    """12-row set with full (o,r)/(r,c) coverage and 6/16 (o,c) coverage."""
    return _rows_from_design_rows(
        [
            (0, 0, 0),
            (0, 1, 0),
            (0, 2, 0),
            (1, 0, 1),
            (1, 1, 1),
            (1, 2, 1),
            (2, 0, 2),
            (2, 1, 3),
            (2, 2, 2),
            (3, 0, 3),
            (3, 1, 2),
            (3, 2, 3),
        ]
    )


def obj0_cont1_cov12_rows() -> list[Row]:
    """Legacy 12-row set: full (o,r)/(r,c) coverage and 12/16 (o,c) coverage."""
    return _rows_from_design_rows(
        [
            (0, 0, 1),
            (0, 1, 2),
            (0, 2, 3),
            (1, 0, 2),
            (1, 1, 3),
            (1, 2, 0),
            (2, 0, 3),
            (2, 1, 0),
            (2, 2, 1),
            (3, 0, 0),
            (3, 1, 1),
            (3, 2, 2),
        ]
    )


def obj0_cont1_cov9_rows() -> list[Row]:
    """12-row set with full (o,r)/(r,c) coverage and 9/16 (o,c) coverage."""
    return _rows_from_design_rows(
        [
            (0, 0, 0),
            (0, 1, 1),
            (0, 2, 2),
            (1, 0, 1),
            (1, 1, 2),
            (1, 2, 1),
            (2, 0, 2),
            (2, 1, 3),
            (2, 2, 3),
            (3, 0, 3),
            (3, 1, 0),
            (3, 2, 0),
        ]
    )


# ---------------------------------------------------------------------------
# Registry & generate_rows
# ---------------------------------------------------------------------------


def generate_rows(name: str, spec: Spec) -> list[Row]:
    method = spec["method"]

    if method == "full":
        return full_rows()
    elif method == "random_coverage":
        return random_coverage_rows(
            int(spec["budget"]),
            seed=spec.get("seed"),
            max_trials=int(spec.get("max_trials", 100_000)),
        )
    elif method == "orthogonal":
        return orthogonal_rows()
    elif method == "obj0_cont1_cov4":
        return obj0_cont1_cov4_rows()
    elif method == "obj0_cont1_cov6":
        return obj0_cont1_cov6_rows()
    elif method == "obj0_cont1_cov12":
        return obj0_cont1_cov12_rows()
    elif method == "obj0_cont1_cov9":
        return obj0_cont1_cov9_rows()
    elif method == "obj0_cont1_cov_b16":
        return obj0_cont1_covk_b16_rows(int(spec["coverage"]))
    else:
        raise ValueError(f"Unknown method: {method!r}")


def _build_budgeted_registry() -> dict[str, Spec]:
    registry: dict[str, Spec] = {}
    for seed in DEFAULT_RANDOM_SEEDS:
        for budget in DEFAULT_BUDGETS:
            registry[f"random_coverage_B{budget}_s{seed}"] = {
                "method": "random_coverage",
                "seed": seed,
                "budget": budget,
            }
    return registry


CORE_SET_REGISTRY: dict[str, Spec] = {
    "full": {"method": "full"},
    "orthogonal": {"method": "orthogonal"},
    "obj0_cont1_cov4": {"method": "obj0_cont1_cov4"},
    "obj0_cont1_cov6": {"method": "obj0_cont1_cov6"},
    "obj0_cont1_cov9": {"method": "obj0_cont1_cov9"},
    "obj0_cont1_cov12": {"method": "obj0_cont1_cov12"},
    **{
        f"obj0_cont1_cov{k}_B16": {
            "method": "obj0_cont1_cov_b16",
            "coverage": k,
        }
        for k in COVK_B16_LEVELS
    },
}

SET_REGISTRY: dict[str, Spec] = {
    **CORE_SET_REGISTRY,
    **_build_budgeted_registry(),
}
