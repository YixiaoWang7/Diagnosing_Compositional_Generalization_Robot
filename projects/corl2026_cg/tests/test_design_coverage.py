"""Coverage guarantees, reproducibility, and independent evaluation populations."""

from __future__ import annotations

from hashlib import sha256
from itertools import combinations

import pytest

from corl2026_cg.tasks import build_task_set, instruction_indices, parse_task, registered_task_sets
from corl2026_cg.tasks.two_stage_designs import full_rows, resolve_rows, row_targets


@pytest.mark.parametrize(
    "name,count,strength,fingerprint",
    [
        (
            "pairwise_cover",
            18,
            2,
            "c9a75d334cc7c594f34304361e616cca3b93f47557e8339451dc98dbea2762d0",
        ),
        ("pairwise_min", 16, 2, "f73733f067ad9f0c147385b8d07c0f921cd1f9b01657e58e516e542644a5a51a"),
        (
            "threewise_cover",
            48,
            3,
            "81375012af88f0bd3782f4076126217151edba93564ac0c5419158af6603487f",
        ),
    ],
)
def test_structured_covers_are_feasible_complete_and_reproducible(
    name: str, count: int, strength: int, fingerprint: str
) -> None:
    """Freeze exact task populations and verify their advertised coverage."""
    rows = resolve_rows(name)
    assert len(rows) == len(set(rows)) == count
    assert set(rows) <= set(full_rows())
    expected_targets = set().union(*(row_targets(row, strength) for row in full_rows()))
    assert len(expected_targets) == {2: 88, 3: 192}[strength]
    assert set().union(*(row_targets(row, strength) for row in rows)) == expected_targets
    assert sha256(repr(rows).encode()).hexdigest() == fingerprint


@pytest.mark.parametrize(
    "name,strength,bound", [("pairwise_min", 2, 16), ("threewise_cover", 3, 48)]
)
def test_minimum_designs_attain_projection_lower_bounds(
    name: str, strength: int, bound: int
) -> None:
    """Each row covers only one assignment for any fixed factor subset."""
    lower_bound = max(
        len({tuple(row[index] for index in indices) for row in full_rows()})
        for indices in combinations(range(4), strength)
    )
    assert lower_bound == bound == len(resolve_rows(name))


@pytest.mark.parametrize("budget", [4, 8, 24, 144])
@pytest.mark.parametrize("seed", [0, 7, 1989])
def test_random_coverage_guarantees_every_factor_level(budget: int, seed: int) -> None:
    """Small budgets need guaranteed coverage rather than retry-limit luck."""
    name = f"random_coverage_B{budget}_s{seed}"
    rows = resolve_rows(name)
    assert len(rows) == len(set(rows)) == budget
    assert set(rows) <= set(full_rows())
    assert all({row[index] for row in rows} == set(range(4)) for index in range(4))
    assert rows == resolve_rows(name) == tuple(sorted(rows))


def test_random_algorithm_is_seeded_and_stable() -> None:
    """Protect seeded population identities across refactors."""
    rows = resolve_rows("random_coverage_B8_s7")
    assert sha256(repr(rows).encode()).hexdigest() == (
        "99a4c8489dd22b2b29f62720ba1065685193c59059cf0694f081d15113bced59"
    )
    assert rows != resolve_rows("random_coverage_B8_s8")


@pytest.mark.parametrize(
    "name",
    ["random_coverage_B3_s0", "random_coverage_B145_s0", "random_coverage_B8_s-1", "missing"],
)
def test_invalid_design_names_and_budgets_fail(name: str) -> None:
    """Reject impossible coverage budgets and malformed identifiers."""
    with pytest.raises(ValueError):
        resolve_rows(name)


def test_explicit_evaluation_uses_selected_rows_and_preserves_legacy_defaults() -> None:
    """Selecting an eval design never substitutes that design's historical universe."""
    legacy = build_task_set("l4_2s", "s0_diag__s1_diag")
    independent = build_task_set("l4_2s", "s0_diag__s1_diag", eval_set="full")
    assert len(legacy.eval_slugs) == 12
    assert len(independent.eval_slugs) == 144
    assert independent.train_slugs == legacy.train_slugs
    assert sum(independent.is_id(slug) for slug in independent.eval_slugs) == 12
    assert "eval_set" not in legacy.to_dict()
    assert independent.to_dict()["eval_set"] == "full"
    restricted = build_task_set("l4", "full", eval_set="diag")
    assert len(restricted.eval_slugs) == 4
    assert restricted.eval_slugs == build_task_set("l4", "diag").train_slugs
    assert len(build_task_set("l5", "full", eval_set="diag_B4").eval_slugs) == 4


@pytest.mark.parametrize("instruction", ["open_cont", "infer_cont0"])
def test_explicit_masked_evaluation_deduplicates_selected_visible_queries(instruction: str) -> None:
    """Hidden destinations change concrete tasks without multiplying evaluations."""
    result = build_task_set("l4_2s", "pairwise_min", instruction, eval_set="full")
    assert len(result.train_slugs) == 16
    assert len(result.eval_slugs) == 48
    visible = {instruction_indices(parse_task(slug), instruction) for slug in result.eval_slugs}
    assert len(visible) == 48
    selected = build_task_set("l4_2s", "full", instruction, eval_set="pairwise_min")
    assert len(selected.eval_slugs) == 16


def test_infer_design_namespace_and_cartesian_names_are_preserved() -> None:
    """Legacy three-factor random designs retain their hidden-container expansion."""
    infer = build_task_set("l4_2s", "random_coverage_B8_s0", "infer_cont0")
    full = build_task_set("l4_2s", "random_coverage_B8_s0")
    assert len(infer.train_slugs) == 24
    assert len(full.train_slugs) == 8
    assert "s0_diag__s1_diag" in registered_task_sets("l4_2s")
    assert "pairwise_min" in registered_task_sets("l4_2s", "infer_cont0")


def test_coverage_distinguishes_unseen_compositions_from_unseen_pairs() -> None:
    """Pairwise-complete subsets have OOD compositions but no unseen factor pairs."""
    complete = build_task_set("l4_2s", "pairwise_min", eval_set="full")
    metrics = complete.coverage()
    assert len(complete.to_dict()["ood_slugs"]) == 128
    assert metrics["ood_with_unseen_pairs"] == 0
    assert metrics["pairwise"]["covered"] == metrics["pairwise"]["total"] == 88
    assert metrics["threewise"]["covered"] == 64
    assert metrics["threewise"]["total"] == 192
    assert all(counts == [4, 4, 4, 4] for counts in metrics["factor_counts"].values())
    incomplete = build_task_set("l4_2s", "s0_diag__s1_diag", eval_set="full")
    metrics = incomplete.coverage()
    assert metrics["ood_with_unseen_pairs"] == 132
    assert set(metrics["ood_with_unseen_pairs_slugs"]) == set(incomplete.to_dict()["ood_slugs"])


def test_coverage_counts_visible_queries_without_constants_or_hidden_multiplicity() -> None:
    """Three hidden first destinations count as one infer-cont0 composition."""
    metrics = build_task_set("l4_2s", "orthogonal", "infer_cont0").coverage()
    assert metrics["concrete_train_count"] == 48
    assert metrics["visible_train_count"] == 16
    assert metrics["visible_universe_count"] == 48
    assert "container0" not in metrics["factor_counts"]
    assert all(sum(counts) == 16 for counts in metrics["factor_counts"].values())
    constants = build_task_set("l4", "diag", "container_latents").coverage()
    assert constants["factor_counts"] == {"container0": [1, 1, 1, 1]}
    assert constants["pairwise"]["total"] == 0
    assert constants["pairwise"]["fraction"] is None
