"""Deterministic CG train/evaluation task designs and visible-query membership."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from . import infer_cont0_designs, l4_designs, l5_designs, two_stage_designs
from .schema import CGTask, instruction_fields, instruction_indices, parse_task

L4_BASELINES: dict[str, tuple[str, ...]] = {
    "full": tuple(f"{obj}{container}" for obj in range(4) for container in range(4)),
    "Sfull": ("00", "01", "11", "12", "22", "23", "33", "30"),
    "S": ("00", "01", "11", "12", "22", "23", "33"),
    "L_cor": ("00", "01", "02", "03", "10", "20", "30"),
    "L_mid": ("01", "10", "11", "12", "13", "21", "31"),
    "diag_cor": ("00", "11", "22", "33", "03"),
    "diag_mid": ("00", "11", "22", "33", "02"),
    "diag": ("00", "11", "22", "33"),
}


@dataclass(frozen=True)
class TaskSet:
    """Resolved concrete training tasks and evaluation representatives.

    Attributes:
        family: CG task family.
        name: Design identifier, including any budget and random seed.
        instruction: Visible query layout and success-rule variant.
        train_slugs: Concrete dataset tasks used to select demonstrations.
        eval_slugs: Concrete representatives to evaluate.
        eval_set: Explicit independent evaluation design, or legacy default behavior.
    """

    family: str
    name: str
    instruction: str
    train_slugs: tuple[str, ...]
    eval_slugs: tuple[str, ...]
    eval_set: str | None = None
    _train_keys: frozenset[tuple[int, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate tasks once and cache visible training membership."""
        instruction_fields(self.family, self.instruction)
        for label, slugs in (("training", self.train_slugs), ("evaluation", self.eval_slugs)):
            if not slugs or len(slugs) != len(set(slugs)):
                raise ValueError(f"{label.capitalize()} task slugs must be nonempty and unique")
            if any(parse_task(slug).family != self.family for slug in slugs):
                raise ValueError(f"{label.capitalize()} tasks do not match family {self.family!r}")
        object.__setattr__(
            self,
            "_train_keys",
            frozenset(
                instruction_indices(parse_task(slug), self.instruction) for slug in self.train_slugs
            ),
        )

    def is_id(self, slug: str) -> bool:
        """Classify by visible instruction, ignoring hidden concrete destinations."""
        task = parse_task(slug)
        if task.family != self.family:
            raise ValueError(f"Task family {task.family!r} does not match {self.family!r}")
        return instruction_indices(task, self.instruction) in self._train_keys

    def to_dict(self) -> dict[str, Any]:
        """Return portable JSON-compatible resolved task provenance."""
        result = {
            "family": self.family,
            "name": self.name,
            "instruction": self.instruction,
            "train_slugs": list(self.train_slugs),
            "eval_slugs": list(self.eval_slugs),
            "id_slugs": [slug for slug in self.eval_slugs if self.is_id(slug)],
            "ood_slugs": [slug for slug in self.eval_slugs if not self.is_id(slug)],
        }
        if self.eval_set is not None:
            result["eval_set"] = self.eval_set
        return result

    def coverage(self) -> dict[str, Any]:
        """Measure visible composition coverage against the full feasible universe.

        Factor counts count distinct visible queries, so expanding a hidden
        destination does not multiply counts. Pair/triple totals include only
        feasible assignments. OOD with unseen pairs requires at least one
        previously unobserved pair of visible factor values.

        Returns:
            JSON-compatible coverage counts, fractions, per-factor counts, and
            the evaluation representatives containing unseen training pairs.
        """
        instruction_layout = instruction_fields(self.family, self.instruction)
        fields = tuple(field for field in instruction_layout if field != "constant")
        positions = tuple(instruction_layout.index(field) for field in fields)
        train = {tuple(row[index] for index in positions) for row in self._train_keys}
        universe = {
            tuple(
                instruction_indices(parse_task(slug), self.instruction)[index]
                for index in positions
            )
            for slug in build_task_set(self.family, "full", self.instruction).train_slugs
        }
        metrics: dict[str, Any] = {
            "concrete_train_count": len(self.train_slugs),
            "visible_train_count": len(train),
            "visible_universe_count": len(universe),
            "factor_counts": {
                field: [sum(row[index] == value for row in train) for value in range(4)]
                for index, field in enumerate(fields)
            },
        }
        train_pairs: set[two_stage_designs.Target] = set()
        for strength, label in ((2, "pairwise"), (3, "threewise")):
            by_factors: dict[str, dict[str, int | float]] = {}
            covered_total, feasible_total = 0, 0
            for indices in combinations(range(len(fields)), strength):
                covered = {tuple(row[index] for index in indices) for row in train}
                feasible = {tuple(row[index] for index in indices) for row in universe}
                by_factors["|".join(fields[index] for index in indices)] = {
                    "covered": len(covered),
                    "total": len(feasible),
                    "fraction": len(covered) / len(feasible),
                }
                covered_total += len(covered)
                feasible_total += len(feasible)
                if strength == 2:
                    train_pairs.update((indices, values) for values in covered)
            metrics[label] = {
                "covered": covered_total,
                "total": feasible_total,
                "fraction": covered_total / feasible_total if feasible_total else None,
                "by_factors": by_factors,
            }
        unseen_pairs = [
            slug
            for slug in self.eval_slugs
            if len(fields) >= 2
            and not self.is_id(slug)
            and not two_stage_designs.row_targets(
                tuple(
                    instruction_indices(parse_task(slug), self.instruction)[index]
                    for index in positions
                ),
                2,
            ).issubset(train_pairs)
        ]
        metrics["ood_with_unseen_pairs"] = len(unseen_pairs)
        metrics["ood_with_unseen_pairs_slugs"] = unseen_pairs
        return metrics


def _single_stage(name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve baseline or seeded L4 train/evaluation universes."""
    if name in L4_BASELINES:
        train_codes, eval_codes = L4_BASELINES[name], L4_BASELINES["full"]
    else:
        generated_train, generated_eval = l4_designs.resolve_set(name)
        train_codes, eval_codes = tuple(generated_train), tuple(generated_eval)
    return tuple(CGTask("l4", int(code[0]), int(code[1])).slug for code in train_codes), tuple(
        CGTask("l4", int(code[0]), int(code[1])).slug for code in eval_codes
    )


def _stage_sets(name: str) -> tuple[tuple[CGTask, ...], tuple[CGTask, ...]]:
    """Resolve stage-specific L4 task selections for a Cartesian design."""
    if name == "full":
        name = "s0_full__s1_full"
    match = re.fullmatch(r"s0_(.+)__s1_(.+)", name)
    if match is None:
        raise ValueError(f"Expected 's0_<set>__s1_<set>' or 'full', got {name!r}")
    return tuple(parse_task(slug) for slug in _single_stage(match[1])[0]), tuple(
        parse_task(slug) for slug in _single_stage(match[2])[0]
    )


def _two_stage(name: str, instruction: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Preserve concrete training constraints and open-container eval coverage."""
    if name != "full" and not name.startswith("s0_"):
        train = tuple(CGTask("l4_2s", *row).slug for row in two_stage_designs.resolve_rows(name))
        if instruction == "infer_cont0":
            return train, tuple(infer_cont0_designs.rows_to_slugs(infer_cont0_designs.full_rows()))
        return train, _visible_representatives(train, instruction)
    stage0, stage1 = _stage_sets(name)
    train = tuple(
        CGTask("l4_2s", first.object0, first.container0, second.object0, second.container0).slug
        for first in stage0
        for second in stage1
        if first.object0 != second.object0 and first.container0 != second.container0
    )
    if instruction == "full":
        return train, train
    if instruction == "infer_cont0":
        return train, tuple(infer_cont0_designs.rows_to_slugs(infer_cont0_designs.full_rows()))
    evaluation = []
    stage1_objects = tuple(dict.fromkeys(task.object0 for task in stage1))
    for first in stage0:
        for object1 in stage1_objects:
            if first.object0 == object1:
                continue
            candidates = [
                task.container0
                for task in stage1
                if task.object0 == object1 and task.container0 != first.container0
            ]
            # A valid placeholder can be outside stage1's training set. This preserves
            # the legacy open-cont OOD query cross/mug_no_handle/milk in L_cor x S.
            container1 = next(
                iter(candidates), next(index for index in range(4) if index != first.container0)
            )
            evaluation.append(
                CGTask("l4_2s", first.object0, first.container0, object1, container1).slug
            )
    return train, tuple(evaluation)


def _infer_cont0(name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Expand visible three-factor designs over all valid hidden destinations."""
    spec = infer_cont0_designs.SET_REGISTRY.get(name)
    match = re.fullmatch(r"random_coverage_B(\d+)_s(\d+)", name)
    if spec is None and match is not None:
        spec = {"method": "random_coverage", "budget": int(match[1]), "seed": int(match[2])}
    if spec is None:
        raise ValueError(f"Unknown infer-cont0 design: {name!r}")
    rows = infer_cont0_designs.generate_rows(name, spec)
    return tuple(infer_cont0_designs.rows_to_training_slugs(rows)), tuple(
        infer_cont0_designs.rows_to_slugs(infer_cont0_designs.full_rows())
    )


def _l5(name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve exact curated L5 designs and arbitrary supported seeded budgets."""
    spec = l5_designs.SET_REGISTRY.get(name)
    match = re.fullmatch(r"(random_coverage|typical_random_walk)_s(\d+)_B(\d+)", name)
    if spec is None and match is not None:
        spec = {"method": "random_coverage", "seed": int(match[2]), "budget": int(match[3])}
        if match[1] == "typical_random_walk":
            spec.update(method="typical", kind="random_walk")
    if spec is None:
        raise ValueError(f"Unknown L5 design: {name!r}")
    return tuple(l5_designs.rows_to_slugs(l5_designs.generate_rows(name, spec))), tuple(
        l5_designs.rows_to_slugs(l5_designs.full_rows())
    )


def _visible_representatives(slugs: tuple[str, ...], instruction: str) -> tuple[str, ...]:
    """Keep the first selected concrete representative of each visible query."""
    representatives: dict[tuple[int, ...], str] = {}
    for slug in slugs:
        representatives.setdefault(instruction_indices(parse_task(slug), instruction), slug)
    return tuple(representatives.values())


def _resolve_design(
    family: str, name: str, instruction: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve concrete selected rows and the historical evaluation population."""
    if family == "l4":
        return _single_stage(name)
    if family == "l5":
        return _l5(name)
    if (
        instruction == "infer_cont0"
        and not name.startswith("s0_")
        and name not in two_stage_designs.DESIGN_NAMES[1:]
    ):
        return _infer_cont0(name)
    return _two_stage(name, instruction)


def build_task_set(
    family: str, name: str, instruction: str = "full", *, eval_set: str | None = None
) -> TaskSet:
    """Resolve a named CG design without files, simulator imports, or side effects.

    Args:
        family: ``l4``, ``l4_2s``, or ``l5``.
        name: A baseline, structured design, or seeded budget name.
        instruction: Policy-visible query layout for this task family.
        eval_set: Independent design whose selected rows are evaluated. Hidden
            factors are deduplicated; omission preserves historical defaults.

    Returns:
        Concrete training tasks and evaluation representatives with visible ID/OOD rules.
    """
    instruction_fields(family, instruction)
    train, evaluation = _resolve_design(family, name, instruction)
    if eval_set is not None:
        selected, _ = _resolve_design(family, eval_set, instruction)
        evaluation = _visible_representatives(selected, instruction)
    return TaskSet(family, name, instruction, train, evaluation, eval_set)


def registered_task_sets(family: str, instruction: str = "full") -> tuple[str, ...]:
    """List built-in designs; seeded dynamic names are additionally accepted."""
    instruction_fields(family, instruction)
    if family == "l4":
        return tuple(L4_BASELINES) + tuple(l4_designs.SET_REGISTRY)
    if family == "l5":
        return tuple(l5_designs.SET_REGISTRY)
    if instruction == "infer_cont0":
        return tuple(infer_cont0_designs.SET_REGISTRY) + two_stage_designs.DESIGN_NAMES[1:]
    return two_stage_designs.DESIGN_NAMES + tuple(
        f"s0_{first}__s1_{second}" for first in L4_BASELINES for second in L4_BASELINES
    )
