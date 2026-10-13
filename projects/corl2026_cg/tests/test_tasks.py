"""Regressions for CG task projections, coverage, and hidden-factor semantics."""

from __future__ import annotations

import json

import pytest

from corl2026_cg.tasks import (
    CGTask,
    build_task_set,
    infer_cont0_designs,
    instruction_fields,
    instruction_indices,
    l5_designs,
    parse_task,
    semantic_values,
    vocab_sizes,
)


@pytest.mark.parametrize(
    "slug,family,indices",
    [
        ("milk_into_mug_no_handle", "l4", (3, 3)),
        ("cross_into_bin__milk_into_mug_no_handle", "l4_2s", (0, 0, 3, 3)),
        ("cube_into_plate_press_yellow", "l5", (1, 2, 3)),
    ],
)
def test_parse_round_trip_and_independent_vocabularies(slug, family, indices):
    """Multiword container names and button slots preserve explicit categorical IDs."""
    task = parse_task(slug)
    assert task.family == family
    assert task.slug == slug
    assert instruction_indices(task, "full") == indices
    assert vocab_sizes(family, "full") == (4,) * len(indices)
    assert semantic_values(task)["constant"] == 0


@pytest.mark.parametrize(
    "slug",
    [
        "unknown_into_bin",
        "cross_into_unknown",
        "cross_into_bin_press_purple",
        "cross_into_bin__cube_into_bin",
        "cross_into_bin__cross_into_mug",
        "cross_into_bin__cube_into_mug__milk_into_plate",
        " cross_into_bin",
        "",
    ],
)
def test_invalid_slugs_never_silently_map_to_first_category(slug):
    """Reject malformed vocabulary and infeasible two-stage task combinations."""
    with pytest.raises(ValueError):
        parse_task(slug)


def test_partial_queries_and_constant_queries():
    """Container-only and learned constants do not reveal the omitted object."""
    task = parse_task("milk_into_plate")
    assert instruction_indices(task, "object_only") == (3,)
    assert instruction_indices(task, "container_only") == (2,)
    assert instruction_indices(task, "container_latents") == (2, 0, 0)
    assert vocab_sizes("l4", "container_latents") == (4, 1, 1)
    assert instruction_fields("l4", "container_latents") == ("container0", "constant", "constant")
    with pytest.raises(ValueError):
        instruction_indices(task, "infer_cont0")


def test_two_stage_layouts_are_distinct():
    """The hidden destination differs between the two three-query variants."""
    task = parse_task("cube_into_bin__milk_into_plate")
    assert instruction_indices(task, "full") == (1, 0, 3, 2)
    assert instruction_indices(task, "open_cont") == (1, 0, 3)
    assert instruction_indices(task, "infer_cont0") == (1, 3, 2)


@pytest.mark.parametrize(
    "name,train_count,eval_count",
    [
        ("diag", 4, 16),
        ("full", 16, 16),
        ("S", 7, 16),
        ("cross_all_cube_full", 8, 8),
        ("cross_all_cube_none", 4, 8),
        ("cross_all_cube_rand_s2_B3", 7, 8),
        ("cross_all_cube_rand_K2_s0_B1", 3, 4),
        ("cross_all_cube_rand_K3_s0_B1", 4, 6),
        ("cross_rand_K4_objs3_B1_s0", 7, 16),
        ("cross_rand_K3_objs2_Bs1_s0", 5, 9),
        ("rand_cov_B6_s0", 6, 16),
    ],
)
def test_l4_designs_preserve_restricted_evaluation_worlds(name, train_count, eval_count):
    """Training budgets and restricted evaluation universes are independent."""
    result = build_task_set("l4", name)
    assert len(result.train_slugs) == train_count
    assert len(result.eval_slugs) == eval_count
    assert result == build_task_set("l4", name)
    assert len(set(result.train_slugs)) == train_count
    assert set(result.train_slugs) <= set(result.eval_slugs)
    assert all(result.is_id(slug) for slug in result.train_slugs)


def test_partial_query_id_ignores_hidden_categories():
    """ID/OOD follows what the policy sees rather than concrete dataset membership."""
    exact = build_task_set("l4", "diag")
    assert not exact.is_id("cross_into_mug")
    assert build_task_set("l4", "diag", "object_only").is_id("cross_into_mug")
    assert build_task_set("l4", "diag", "container_only").is_id("cross_into_mug")


def test_cartesian_and_open_container_eval_are_not_the_same_expansion():
    """Preserve the historical OOD open-cont query introduced by placeholder fallback."""
    full = build_task_set("l4_2s", "full")
    assert len(full.train_slugs) == len(full.eval_slugs) == 144
    open_full = build_task_set("l4_2s", "full", "open_cont")
    assert len(open_full.train_slugs) == 144
    assert len(open_full.eval_slugs) == 48
    partial = build_task_set("l4_2s", "s0_L_cor__s1_S", "open_cont")
    assert len(partial.train_slugs) == 28
    assert len(partial.eval_slugs) == 21
    assert sum(partial.is_id(slug) for slug in partial.eval_slugs) == 20
    ood = "cross_into_mug_no_handle__milk_into_bin"
    assert ood in partial.eval_slugs
    assert not partial.is_id(ood)
    assert partial.is_id("cross_into_bin__cube_into_plate")


def test_infer_cont0_expands_all_hidden_destinations_and_uses_visible_id():
    """Full and orthogonal designs use the current 48-visible-query universe."""
    full = build_task_set("l4_2s", "full", "infer_cont0")
    orthogonal = build_task_set("l4_2s", "orthogonal", "infer_cont0")
    assert len(full.train_slugs) == 144
    assert len(full.eval_slugs) == 48
    assert len(orthogonal.train_slugs) == 48
    assert sum(orthogonal.is_id(slug) for slug in orthogonal.eval_slugs) == 16
    first = parse_task(orthogonal.train_slugs[0])
    for container0 in range(4):
        if container0 != first.container1:
            task = CGTask("l4_2s", first.object0, container0, first.object1, first.container1)
            assert orthogonal.is_id(task.slug)
    payload = json.loads(json.dumps(orthogonal.to_dict()))
    assert len(payload["id_slugs"]) == 16
    assert len(payload["ood_slugs"]) == 32


@pytest.mark.parametrize("coverage", range(6, 17))
def test_infer_cont0_coverage_ablation_holds_other_pairwise_coverage_fixed(coverage):
    """The fixed-budget ablation changes only the selected object/container pair coverage."""
    name = f"obj0_cont1_cov{coverage}_B16"
    result = build_task_set("l4_2s", name, "infer_cont0")
    rows = list(dict.fromkeys(infer_cont0_designs.slug_to_row(slug) for slug in result.train_slugs))
    metrics = infer_cont0_designs.design_metrics(rows)["pairwise_coverage_by_pair"]
    assert len(rows) == 16
    assert len(result.train_slugs) == 48
    assert metrics["o_c"]["covered"] == coverage
    assert metrics["o_r"]["covered"] == 12
    assert metrics["r_c"]["covered"] == 12


@pytest.mark.parametrize("name", tuple(l5_designs.CORE_SET_REGISTRY))
def test_all_l5_curated_and_structured_designs_are_feasible(name):
    """Retain all curated dev_2s designs with the full 64-task evaluation universe."""
    result = build_task_set("l5", name)
    assert len(result.eval_slugs) == 64
    assert set(result.train_slugs) <= set(result.eval_slugs)
    spec = l5_designs.CORE_SET_REGISTRY[name]
    if "budget" in spec:
        assert len(result.train_slugs) == spec["budget"]


def test_l5_curated_final_b10_table_and_dynamic_large_budget():
    """Protect a curated design exactly and accept new seeds/budgets beyond registry defaults."""
    result = build_task_set("l5", "final_B10")
    expected = (
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
    )
    assert (
        tuple(instruction_indices(parse_task(slug), "full") for slug in result.train_slugs)
        == expected
    )
    random_set = build_task_set("l5", "random_coverage_s4_B32")
    assert len(random_set.train_slugs) == 32
    assert random_set == build_task_set("l5", "random_coverage_s4_B32")
    for slot in range(3):
        assert {
            instruction_indices(parse_task(slug), "full")[slot] for slug in random_set.train_slugs
        } == set(range(4))


@pytest.mark.parametrize(
    "family,name,instruction",
    [
        ("unknown", "full", "full"),
        ("l4", "missing", "full"),
        ("l4", "rand_cov_B3_s0", "full"),
        ("l5", "missing", "full"),
        ("l4_2s", "obj0_cont1_cov5_B16", "infer_cont0"),
        ("l4_2s", "orthogonal", "full"),
    ],
)
def test_invalid_design_requests_fail_clearly(family, name, instruction):
    """Do not silently fall back to a different dataset task design."""
    with pytest.raises(ValueError):
        build_task_set(family, name, instruction)


@pytest.mark.parametrize(
    "name",
    [
        "oa_B6",
        "oa_B16",
        "opt_inv_B6",
        "opt_inv_B8",
        "typical_diagonal_B12",
        "typical_adjacent_B24",
        "typical_random_walk_s3_B10",
        "random_coverage_s1_B8",
    ],
)
def test_l5_budgeted_algorithm_families_are_deterministic(name):
    """Each ported generator family preserves explicit budgets and seeded ordering."""
    result = build_task_set("l5", name)
    budget = int(name.rsplit("_B", 1)[1])
    assert len(result.train_slugs) == budget
    assert len(set(result.train_slugs)) == budget
    assert result == build_task_set("l5", name)
