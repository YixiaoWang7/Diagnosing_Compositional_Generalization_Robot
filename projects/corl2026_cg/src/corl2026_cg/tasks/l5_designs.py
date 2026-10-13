"""Pure-Python CG design algorithms, ported from dev_2s:src/policies/tools/l5_task_designs.py."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from itertools import combinations, product
from typing import Any

from . import set_generation as sg
from .schema import BUTTONS as BUTTON_COLORS
from .schema import CONTAINERS, OBJECTS

Row = tuple[int, ...]
Spec = dict[str, Any]

N_FACTORS = 3
N_LEVELS = 4

# Budgeted sets are registered only for these sizes. Add to this tuple when a
# new training budget should become a first-class set name.
DEFAULT_BUDGETS: tuple[int, ...] = (6, 7, 8, 10, 12, 16, 24, 30)
DEFAULT_RANDOM_SEEDS: tuple[int, ...] = (0, 1)


def row_to_slug(row: Row) -> str:
    return f"{OBJECTS[row[0]]}_into_{CONTAINERS[row[1]]}_press_{BUTTON_COLORS[row[2]]}"


def rows_to_slugs(rows: list[Row]) -> list[str]:
    return [row_to_slug(row) for row in rows]


def slug_to_row(slug: str) -> Row:
    lhs, color = slug.rsplit("_press_", 1)
    obj, container = lhs.split("_into_", 1)
    return (
        OBJECTS.index(obj),
        CONTAINERS.index(container),
        BUTTON_COLORS.index(color),
    )


def all_rows(q: int = N_LEVELS, d: int = N_FACTORS) -> list[Row]:
    """Full q^d row space."""
    return list(product(range(q), repeat=d))


def validate_budget(budget: int, q: int = N_LEVELS, d: int = N_FACTORS) -> int:
    budget = int(budget)
    max_budget = q**d
    if not 0 < budget <= max_budget:
        raise ValueError(f"budget must be in [1, {max_budget}], got {budget}.")
    return budget


def factor_counts(rows: list[Row], q: int = N_LEVELS, d: int = N_FACTORS) -> list[list[int]]:
    """counts[i][v] = number of rows with factor i at level v."""
    counts = []
    for i in range(d):
        counter = Counter(row[i] for row in rows)
        counts.append([counter[v] for v in range(q)])
    return counts


def pairwise_counts(
    rows: list[Row], d: int = N_FACTORS
) -> defaultdict[tuple[int, int, int, int], int]:
    """Count observed pairwise relation frequencies."""
    counts: defaultdict[tuple[int, int, int, int], int] = defaultdict(int)
    for row in rows:
        for i in range(d):
            for j in range(i + 1, d):
                counts[(i, j, row[i], row[j])] += 1
    return counts


def unique_pairwise_coverage(rows: list[Row], d: int = N_FACTORS) -> int:
    """Number of unique pairwise relations covered by rows."""
    return sum(1 for value in pairwise_counts(rows, d=d).values() if value > 0)


def covers_all_single_levels(
    rows: list[Row],
    q: int = N_LEVELS,
    d: int = N_FACTORS,
) -> bool:
    return all({row[i] for row in rows} == set(range(q)) for i in range(d))


def one_way_imbalance(rows: list[Row], q: int = N_LEVELS, d: int = N_FACTORS) -> float:
    """Single-factor balance objective; lower is better."""
    target = len(rows) / q
    return sum(
        (count - target) ** 2
        for factor_count in factor_counts(rows, q=q, d=d)
        for count in factor_count
    )


def pairwise_marginal_loss(
    rows: list[Row],
    q: int = N_LEVELS,
    d: int = N_FACTORS,
) -> float:
    """Pairwise marginal matching objective; lower is better."""
    target = len(rows) / (q * q)
    counts = pairwise_counts(rows, d=d)
    loss = 0.0
    for i in range(d):
        for j in range(i + 1, d):
            for a in range(q):
                for b in range(q):
                    loss += (counts[(i, j, a, b)] - target) ** 2
    return loss


def design_score(
    rows: list[Row],
    q: int = N_LEVELS,
    d: int = N_FACTORS,
    *,
    w_pair: float = 1.0,
    w_one: float = 0.1,
    w_unique: float = -0.1,
) -> float:
    """Combined design objective for choosing extra rows; lower is better."""
    return (
        w_pair * pairwise_marginal_loss(rows, q=q, d=d)
        + w_one * one_way_imbalance(rows, q=q, d=d)
        + w_unique * unique_pairwise_coverage(rows, d=d)
    )


def design_metrics(rows: list[Row], q: int = N_LEVELS, d: int = N_FACTORS) -> dict[str, Any]:
    return {
        "budget": len(rows),
        "covers_all_single_levels": covers_all_single_levels(rows, q=q, d=d),
        "factor_counts": factor_counts(rows, q=q, d=d),
        "unique_pairwise_coverage": unique_pairwise_coverage(rows, d=d),
        "max_pairwise_coverage": min(len(rows) * (d * (d - 1) // 2), (d * (d - 1) // 2) * q * q),
        "one_way_imbalance": one_way_imbalance(rows, q=q, d=d),
        "pairwise_marginal_loss": pairwise_marginal_loss(rows, q=q, d=d),
        "design_score": design_score(rows, q=q, d=d),
    }


def full_rows() -> list[Row]:
    return all_rows()


def manual_rows(rows: list[Row]) -> list[Row]:
    return [tuple(row) for row in rows]


def greedy_pairwise_rows(spec: Spec) -> list[Row]:
    pt_key = spec.get("pair_targets", "adjacent")
    if isinstance(pt_key, list):
        pair_targets = set(tuple(pair) for pair in pt_key)
    elif pt_key == "all":
        pair_targets = sg.all_level_pairs(N_LEVELS)
    elif pt_key == "adjacent":
        pair_targets = sg.adjacent_target_pairs(
            N_LEVELS,
            cyclic=bool(spec.get("cyclic", False)),
        )
    else:
        raise ValueError(f"Unknown pair_targets value: {pt_key!r}")

    result = sg.generate_design(N_FACTORS, N_LEVELS, method="greedy", pair_targets=pair_targets)
    if not sg.covers_targets(result.rows, result.targets):
        raise AssertionError(f"Generated rows do not cover all pair targets for spec={spec}")
    return result.rows


def xor_oa_4x4x4() -> list[Row]:
    """
    Strength-2 orthogonal array for 4 x 4 x 4.

    The construction c = object XOR container gives 16 rows and covers each
    pairwise relation exactly once.
    """
    return [
        (obj, container, obj ^ container)
        for obj in range(N_LEVELS)
        for container in range(N_LEVELS)
    ]


def optimal_oa_subset(budget: int) -> list[Row]:
    """
    Choose the most balanced subset of the 16-row XOR OA.

    Inside the OA no pairwise relation repeats, so for budget <= 16 every
    subset has maximal unique pairwise coverage. The tie-breakers optimize
    one-way balance first, then pairwise marginal loss.
    """
    budget = validate_budget(budget)
    oa = xor_oa_4x4x4()
    if budget > len(oa):
        raise ValueError("Use oa_rows for budgets above 16.")

    best_rows: list[Row] | None = None
    best_key: tuple[float, float, int] | None = None
    for idxs in combinations(range(len(oa)), budget):
        rows = [oa[i] for i in idxs]
        key = (
            one_way_imbalance(rows),
            pairwise_marginal_loss(rows),
            -unique_pairwise_coverage(rows),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_rows = rows

    if best_rows is None:
        raise RuntimeError("Failed to select an OA subset.")
    return best_rows


def greedy_add_extra_rows(rows: list[Row], budget: int) -> list[Row]:
    """For budgets above the OA size, greedily add rows by design score."""
    selected = list(rows)
    universe = all_rows()

    while len(selected) < budget:
        candidates = [row for row in universe if row not in selected]
        selected.append(min(candidates, key=lambda row: design_score(selected + [row])))

    return selected


def oa_rows(budget: int) -> list[Row]:
    """OA-based design: balanced OA subset up to 16, then greedy extras."""
    budget = validate_budget(budget)
    oa = xor_oa_4x4x4()
    if budget <= len(oa):
        return optimal_oa_subset(budget)
    return greedy_add_extra_rows(oa, budget)


def random_coverage_rows(
    budget: int,
    *,
    seed: int | None = None,
    max_trials: int = 100000,
) -> list[Row]:
    """Randomly sample rows while requiring every factor level to appear."""
    budget = validate_budget(budget)
    if budget < N_LEVELS:
        raise ValueError(
            f"budget={budget} is too small to cover all {N_LEVELS} levels in each factor."
        )

    rng = random.Random(seed)
    universe = all_rows()
    for _ in range(max_trials):
        rows = rng.sample(universe, budget)
        if covers_all_single_levels(rows):
            return rows

    raise RuntimeError("Failed to sample a random coverage design; increase max_trials.")


def opt_inv_rows(budget: int) -> list[Row]:
    """
    Two-diagonal design: identity diagonal D0 plus a shifted diagonal D1.

    D0 = {(a, a, ..., a) : a in [q]}
    D1 = {(a, a+1, a+2, ...) mod q : a in [q]}   (shift = [0, 1, ..., n-1] mod q)

    Takes min(B, q) rows from D0 then the remainder from D1.
    Max supported budget is 2*q.
    """
    q, n = N_LEVELS, N_FACTORS
    max_b = 2 * q
    budget = validate_budget(budget)
    if budget > max_b:
        raise ValueError(f"opt_inv supports at most 2*q={max_b} rows, got budget={budget}.")

    shift = [i % q for i in range(n)]
    d0 = [tuple(a for _ in range(n)) for a in range(q)]
    d1 = [tuple((a + shift[i]) % q for i in range(n)) for a in range(q)]

    num_from_d0 = min(budget, q)
    return d0[:num_from_d0] + d1[: budget - num_from_d0]


def extend_rows(rows: list[Row], budget: int) -> list[Row]:
    """Trim to budget or append deterministic unseen rows from the full space."""
    budget = validate_budget(budget)
    selected = list(rows[:budget])
    if len(selected) >= budget:
        return selected

    used = set(selected)
    for row in all_rows():
        if len(selected) >= budget:
            break
        if row not in used:
            selected.append(row)
            used.add(row)
    return selected


def diagonal_rows(budget: int) -> list[Row]:
    """Correlated diagonal-style baseline."""
    base: list[Row] = [
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2),
        (3, 3, 3),
        (0, 0, 1),
        (1, 1, 2),
        (2, 2, 3),
        (3, 3, 0),
        (0, 1, 0),
        (1, 2, 1),
        (2, 3, 2),
        (3, 0, 3),
        (1, 0, 0),
        (2, 1, 1),
        (3, 2, 2),
        (0, 3, 3),
    ]
    return extend_rows(base, budget)


def adjacent_chain_rows(budget: int) -> list[Row]:
    """Adjacent-level local transition baseline."""
    base: list[Row] = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 1),
        (1, 2, 1),
        (2, 1, 1),
        (2, 2, 2),
        (2, 3, 2),
        (3, 2, 2),
        (3, 3, 3),
        (3, 0, 3),
        (0, 3, 0),
        (0, 2, 1),
        (1, 3, 2),
        (2, 0, 3),
        (3, 1, 0),
    ]
    return extend_rows(base, budget)


def random_walk_rows(budget: int, *, seed: int | None = None) -> list[Row]:
    """Random-walk local transition baseline."""
    budget = validate_budget(budget)
    rng = random.Random(seed)

    row = tuple(rng.randrange(N_LEVELS) for _ in range(N_FACTORS))
    selected = [row]
    used = {row}

    for _ in range(10000):
        if len(selected) >= budget:
            return selected

        row_list = list(selected[-1])
        factor = rng.randrange(N_FACTORS)
        row_list[factor] = (row_list[factor] + rng.choice([-1, 1])) % N_LEVELS
        new_row = tuple(row_list)
        if new_row not in used:
            selected.append(new_row)
            used.add(new_row)

    universe = all_rows()
    rng.shuffle(universe)
    for row in universe:
        if len(selected) >= budget:
            break
        if row not in used:
            selected.append(row)
            used.add(row)
    return selected


def typical_rows(kind: str, budget: int, *, seed: int | None = None) -> list[Row]:
    """Unified interface for structured typical baselines."""
    if kind == "diagonal":
        return diagonal_rows(budget)
    if kind == "adjacent":
        return adjacent_chain_rows(budget)
    if kind == "random_walk":
        return random_walk_rows(budget, seed=seed)
    raise ValueError(f"Unknown typical design kind: {kind!r}")


def assert_budget(name: str, spec: Spec, rows: list[Row]) -> None:
    """Enforce explicit budgets on all budgeted specs."""
    if "budget" not in spec:
        return
    budget = validate_budget(int(spec["budget"]))
    if len(rows) != budget:
        raise AssertionError(f"Set {name!r} expected budget {budget}, generated {len(rows)} rows.")


def generate_rows(name: str, spec: Spec) -> list[Row]:
    method = spec["method"]

    if method == "manual":
        rows = manual_rows(spec["rows"])
    elif method == "full":
        rows = full_rows()
    elif method == "greedy":
        rows = greedy_pairwise_rows(spec)
    elif method == "oa":
        rows = oa_rows(int(spec["budget"]))
    elif method == "random_coverage":
        rows = random_coverage_rows(
            int(spec["budget"]),
            seed=spec.get("seed"),
            max_trials=int(spec.get("max_trials", 100000)),
        )
    elif method == "opt_inv":
        rows = opt_inv_rows(int(spec["budget"]))
    elif method == "typical":
        rows = typical_rows(
            str(spec.get("kind", "adjacent")),
            int(spec["budget"]),
            seed=spec.get("seed"),
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    assert_budget(name, spec, rows)
    return rows


DETERMINISTIC_BUDGETED_APPROACHES: tuple[tuple[str, Spec], ...] = (
    ("oa", {"method": "oa"}),
    ("opt_inv", {"method": "opt_inv", "max_budget": 2 * N_LEVELS}),
    ("typical_diagonal", {"method": "typical", "kind": "diagonal"}),
    ("typical_adjacent", {"method": "typical", "kind": "adjacent"}),
)

SEEDED_BUDGETED_APPROACHES: tuple[tuple[str, Spec], ...] = (
    ("random_coverage", {"method": "random_coverage"}),
    ("typical_random_walk", {"method": "typical", "kind": "random_walk"}),
)


def budgeted_set_name(approach: str, budget: int) -> str:
    validate_budget(budget)
    return f"{approach}_B{budget}"


def build_budgeted_registry(budgets: tuple[int, ...] = DEFAULT_BUDGETS) -> dict[str, Spec]:
    registry: dict[str, Spec] = {}
    for approach, base_spec in DETERMINISTIC_BUDGETED_APPROACHES:
        max_b = base_spec.get("max_budget")
        for budget in budgets:
            if max_b is not None and budget > max_b:
                continue
            spec = {k: v for k, v in base_spec.items() if k != "max_budget"}
            registry[budgeted_set_name(approach, budget)] = {
                **spec,
                "budget": budget,
            }
    for approach, base_spec in SEEDED_BUDGETED_APPROACHES:
        for seed in DEFAULT_RANDOM_SEEDS:
            for budget in budgets:
                registry[budgeted_set_name(f"{approach}_s{seed}", budget)] = {
                    **base_spec,
                    "seed": seed,
                    "budget": budget,
                }
    return registry


CORE_SET_REGISTRY: dict[str, Spec] = {
    "full": {
        "method": "full",
    },
    "orthogonal": {
        "method": "greedy",
        "pair_targets": "all",
    },
    "adjacent": {
        "method": "greedy",
        "pair_targets": "adjacent",
    },
    "Smin": {
        "method": "manual",
        "budget": 7,
        "rows": [
            (0, 0, 1),
            (0, 1, 0),
            (1, 1, 2),
            (1, 2, 1),
            (2, 2, 3),
            (2, 3, 2),
            (3, 3, 3),
        ],
    },
    "logopt_B8": {
        "method": "manual",
        "budget": 8,
        "rows": [
            (0, 0, 0),
            (1, 1, 1),
            (1, 2, 3),
            (1, 3, 2),
            (2, 1, 3),
            (2, 2, 1),
            (3, 2, 2),
            (3, 3, 3),
        ],
    },
    "logopt_B10": {
        "method": "manual",
        "budget": 10,
        "rows": [
            (0, 0, 0),
            (0, 2, 2),
            (1, 0, 2),
            (1, 1, 1),
            (1, 3, 3),
            (2, 1, 2),
            (2, 2, 1),
            (3, 0, 3),
            (3, 2, 0),
            (3, 3, 1),
        ],
    },
    "logopt_B6": {
        "method": "manual",
        "budget": 6,
        "rows": [
            (0, 0, 0),
            (1, 1, 2),
            (1, 2, 3),
            (2, 3, 1),
            (3, 1, 3),
            (3, 2, 2),
        ],
    },
    "logopt_B7": {
        "method": "manual",
        "budget": 7,
        "rows": [
            (0, 0, 0),
            (1, 1, 2),
            (1, 2, 1),
            (1, 3, 3),
            (2, 1, 1),
            (3, 2, 3),
            (3, 3, 2),
        ],
    },
    "final_B6": {
        "method": "manual",
        "budget": 6,
        "rows": [
            (0, 0, 1),
            (1, 1, 1),
            (1, 3, 2),
            (2, 3, 0),
            (3, 2, 1),
            (3, 3, 3),
        ],
    },
    "final_B7": {
        "method": "manual",
        "budget": 7,
        "rows": [
            (0, 3, 1),
            (1, 0, 3),
            (1, 1, 1),
            (1, 2, 0),
            (1, 3, 2),
            (2, 3, 0),
            (3, 3, 3),
        ],
    },
    "final_B8": {
        "method": "manual",
        "budget": 8,
        "rows": [
            (0, 0, 2),
            (0, 1, 1),
            (0, 3, 3),
            (1, 2, 1),
            (1, 3, 0),
            (2, 3, 1),
            (3, 0, 1),
            (3, 3, 2),
        ],
    },
    "final_B9": {
        "method": "manual",
        "budget": 9,
        "rows": [
            (0, 3, 1),
            (1, 0, 3),
            (1, 1, 1),
            (1, 3, 2),
            (2, 0, 1),
            (2, 3, 0),
            (3, 0, 2),
            (3, 2, 1),
            (3, 3, 3),
        ],
    },
    "final_B10": {
        "method": "manual",
        "budget": 10,
        "rows": [
            (0, 0, 0),
            (0, 3, 1),
            (1, 0, 3),
            (1, 1, 1),
            (2, 0, 1),
            (2, 3, 0),
            (3, 0, 2),
            (3, 1, 0),
            (3, 2, 1),
            (3, 3, 3),
        ],
    },
    "final_B12": {
        "method": "manual",
        "budget": 12,
        "rows": [
            (0, 0, 0),
            (0, 3, 1),
            (1, 0, 3),
            (1, 1, 1),
            (1, 2, 0),
            (1, 3, 2),
            (2, 0, 1),
            (2, 3, 0),
            (3, 0, 2),
            (3, 1, 0),
            (3, 2, 1),
            (3, 3, 3),
        ],
    },
    "final_B16": {
        "method": "manual",
        "budget": 16,
        "rows": [
            (0, 0, 0),
            (0, 1, 2),
            (0, 2, 3),
            (0, 3, 1),
            (1, 0, 3),
            (1, 1, 1),
            (1, 2, 0),
            (1, 3, 2),
            (2, 0, 1),
            (2, 1, 3),
            (2, 2, 2),
            (2, 3, 0),
            (3, 0, 2),
            (3, 1, 0),
            (3, 2, 1),
            (3, 3, 3),
        ],
    },
    "final_B20": {
        "method": "manual",
        "budget": 20,
        "rows": [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 2),
            (0, 2, 3),
            (0, 3, 1),
            (1, 0, 3),
            (1, 1, 0),
            (1, 1, 1),
            (1, 2, 0),
            (1, 3, 2),
            (2, 0, 1),
            (2, 1, 3),
            (2, 2, 2),
            (2, 2, 3),
            (2, 3, 0),
            (3, 0, 2),
            (3, 1, 0),
            (3, 2, 1),
            (3, 3, 2),
            (3, 3, 3),
        ],
    },
    "final_B22": {
        "method": "manual",
        "budget": 22,
        "rows": [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 2),
            (0, 2, 3),
            (0, 3, 1),
            (1, 0, 3),
            (1, 1, 0),
            (1, 1, 1),
            (1, 2, 0),
            (1, 3, 2),
            (2, 0, 1),
            (2, 1, 3),
            (2, 2, 2),
            (2, 2, 3),
            (2, 3, 0),
            (2, 3, 1),
            (3, 0, 2),
            (3, 1, 0),
            (3, 2, 0),
            (3, 2, 1),
            (3, 3, 2),
            (3, 3, 3),
        ],
    },
    "final_B24": {
        "method": "manual",
        "budget": 24,
        "rows": [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (0, 3, 1),
            (1, 0, 2),
            (1, 0, 3),
            (1, 1, 0),
            (1, 1, 1),
            (1, 2, 0),
            (1, 3, 2),
            (2, 0, 1),
            (2, 1, 3),
            (2, 2, 2),
            (2, 2, 3),
            (2, 3, 0),
            (2, 3, 1),
            (3, 0, 2),
            (3, 1, 0),
            (3, 2, 0),
            (3, 2, 1),
            (3, 3, 2),
            (3, 3, 3),
        ],
    },
    "logopt_B9": {
        "method": "manual",
        "budget": 9,
        "rows": [
            (0, 1, 1),
            (0, 2, 3),
            (1, 0, 2),
            (1, 3, 0),
            (2, 0, 3),
            (2, 2, 1),
            (2, 3, 2),
            (3, 1, 0),
            (3, 3, 1),
        ],
    },
    "logopt_B12": {
        "method": "manual",
        "budget": 12,
        "rows": [
            (0, 0, 1),
            (0, 1, 0),
            (0, 3, 2),
            (1, 1, 2),
            (1, 2, 3),
            (1, 3, 0),
            (2, 0, 2),
            (2, 1, 3),
            (2, 2, 1),
            (3, 0, 3),
            (3, 2, 0),
            (3, 3, 1),
        ],
    },
    "diag_B4": {
        "method": "manual",
        "budget": 4,
        "rows": [
            (0, 0, 0),
            (1, 1, 1),
            (2, 2, 2),
            (3, 3, 3),
        ],
    },
    "adj1_B8": {
        "method": "manual",
        "budget": 8,
        "rows": [
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 2),
            (1, 2, 3),
            (2, 2, 2),
            (2, 3, 3),
            (3, 3, 0),
        ],
    },
    "adj2_B8": {
        "method": "manual",
        "budget": 8,
        "rows": [
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (3, 1, 2),
            (1, 2, 3),
            (2, 2, 2),
            (2, 3, 3),
            (3, 3, 0),
        ],
    },
    # Backward-compatible alias for the B10 adjacent-chain baseline.
    "adjacent_B10": {
        "method": "typical",
        "kind": "adjacent",
        "budget": 10,
    },
    "diag_03": {
        "method": "greedy",
        "pair_targets": [(0, 0), (1, 1), (2, 2), (3, 3), (0, 3)],
    },
    "diag_02": {
        "method": "greedy",
        "pair_targets": [(0, 0), (1, 1), (2, 2), (3, 3), (0, 2)],
    },
    "row_col_0": {
        "method": "greedy",
        "pair_targets": [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0)],
    },
    "row_col_1": {
        "method": "greedy",
        "pair_targets": [(1, 0), (1, 1), (1, 2), (1, 3), (0, 1), (2, 1), (3, 1)],
    },
}

SET_REGISTRY: dict[str, Spec] = {
    **CORE_SET_REGISTRY,
    **build_budgeted_registry(),
}
