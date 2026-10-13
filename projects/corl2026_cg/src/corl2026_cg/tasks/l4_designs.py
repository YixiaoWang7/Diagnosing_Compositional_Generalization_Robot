"""Seeded L4 task designs, ported from dev_2s_1s at 7455b6c."""

from __future__ import annotations

import random as _random
import re

# Fixed cross tasks (object_idx=0, all containers)
_CROSS_CODES = ["00", "01", "02", "03"]
# Cube tasks (object_idx=1, all containers)
_CUBE_CODES = ["10", "11", "12", "13"]
# All cross+cube tasks
_CROSS_CUBE_CODES = _CROSS_CODES + _CUBE_CODES

# Static registry
SET_REGISTRY: dict[str, list[str]] = {
    "cross_all_cube_full": _CROSS_CUBE_CODES,
    "cross_all_cube_none": _CROSS_CODES,
}

# Dynamic pattern (legacy, K=4 implicit): cross_all_cube_rand_s<seed>_B<N>
_DYNAMIC_RE = re.compile(r"^cross_all_cube_rand_s(\d+)_B(\d+)$")
# Dynamic pattern with explicit container budget K:
#   cross_all_cube_rand_K<K>_s<seed>_B<B>
# - K of 4 containers are sampled (seeded) to form a "restricted world".
# - B cube tasks are sampled (same seed) from those K containers.
# - Training tasks: K cross + B cube (all within the K-container subset).
# - Evaluation tasks: 2K (cross+cube within the K-container subset);
#   the (K - B) held-out cube tasks are the OOD set.
_DYNAMIC_K_RE = re.compile(r"^cross_all_cube_rand_K(\d+)_s(\d+)_B(\d+)$")
# Generalized multi-object form: cross_rand_K<K>_objs<C>_B<B>_s<seed>
# - K containers sampled from {bin, mug, plate, mug_no_handle}.
# - C non-cross objects sampled from {cube, cylinder, milk}.
# - B containers per non-cross object used for training (sampled independently per object).
# - Train: K cross + C*B non-cross. Eval: K*(C+1).
# Shared-B variant: cross_rand_K<K>_objs<C>_Bs<B>_s<seed>
# - Same as above but all non-cross objects share the same B containers for training.
_NON_CROSS_OBJ_IDXS = [1, 2, 3]  # cube, cylinder, milk
_GENERALIZED_RE = re.compile(r"^cross_rand_K(\d+)_objs(\d+)_(Bs?)(\d+)_s(\d+)$")
# Random coverage: rand_cov_B<budget>_s<seed>
# Randomly sample `budget` tasks from all 16, ensuring all 4 objects and
# all 4 containers appear at least once.  Eval on all 16.
_RAND_COV_RE = re.compile(r"^rand_cov_B(\d+)_s(\d+)$")
_ALL_CODES = [f"{oi}{ci}" for oi in range(4) for ci in range(4)]


def resolve_set(set_name: str) -> tuple[list[str], list[str]]:
    """Resolve a set name to (train_codes, eval_codes).

    eval_codes are the codes that should appear in the eval config's
    `eval_tasks` field. For legacy / static sets this is all 8 cross+cube.
    For the K-form it is the 2K codes within the selected K-container world.
    """
    if set_name in SET_REGISTRY:
        return list(SET_REGISTRY[set_name]), list(_CROSS_CUBE_CODES)

    m = _DYNAMIC_K_RE.fullmatch(set_name)
    if m is not None:
        k = int(m.group(1))
        seed = int(m.group(2))
        b = int(m.group(3))
        if not 2 <= k <= 4:
            raise ValueError(f"K must be 2..4, got {k} in {set_name!r}")
        if not 1 <= b <= k:
            raise ValueError(f"B must be 1..K (1..{k}), got {b} in {set_name!r}")

        rng = _random.Random(seed)
        container_idxs = sorted(rng.sample(range(4), k))
        cube_container_idxs = sorted(rng.sample(container_idxs, b))
        train_codes = [f"0{ci}" for ci in container_idxs] + [f"1{ci}" for ci in cube_container_idxs]
        eval_codes = [f"0{ci}" for ci in container_idxs] + [f"1{ci}" for ci in container_idxs]
        return train_codes, eval_codes

    m = _GENERALIZED_RE.fullmatch(set_name)
    if m is not None:
        k = int(m.group(1))
        c = int(m.group(2))
        shared = m.group(3) == "Bs"
        b = int(m.group(4))
        seed = int(m.group(5))
        if not 2 <= k <= 4:
            raise ValueError(f"K must be 2..4, got {k} in {set_name!r}")
        if not 1 <= c <= 3:
            raise ValueError(f"C (objs) must be 1..3, got {c} in {set_name!r}")
        if not 1 <= b <= k:
            raise ValueError(f"B must be 1..K (1..{k}), got {b} in {set_name!r}")

        rng = _random.Random(seed)
        container_idxs = sorted(rng.sample(range(4), k))
        obj_idxs = sorted(rng.sample(_NON_CROSS_OBJ_IDXS, c))

        train_codes = [f"0{ci}" for ci in container_idxs]
        eval_codes = list(train_codes)
        if shared:
            shared_ci = sorted(rng.sample(container_idxs, b))
        for oi in obj_idxs:
            train_ci = shared_ci if shared else sorted(rng.sample(container_idxs, b))
            train_codes += [f"{oi}{ci}" for ci in train_ci]
            eval_codes += [f"{oi}{ci}" for ci in container_idxs]
        return train_codes, eval_codes

    m = _RAND_COV_RE.fullmatch(set_name)
    if m is not None:
        budget = int(m.group(1))
        seed = int(m.group(2))
        if not 4 <= budget <= 16:
            raise ValueError(f"rand_cov budget must be 4..16, got {budget} in {set_name!r}")

        rng = _random.Random(seed)
        for _ in range(100_000):
            codes = rng.sample(_ALL_CODES, budget)
            objs_seen = {c[0] for c in codes}
            conts_seen = {c[1] for c in codes}
            if len(objs_seen) == 4 and len(conts_seen) == 4:
                return sorted(codes), list(_ALL_CODES)
        raise RuntimeError(
            f"Failed to sample a random-coverage design for {set_name!r}; "
            "increase max_trials or budget."
        )

    m = _DYNAMIC_RE.fullmatch(set_name)
    if m is None:
        raise ValueError(
            f"Unknown set {set_name!r}. Static sets: {sorted(SET_REGISTRY)}. "
            "Dynamic formats: cross_all_cube_rand_s<seed>_B<N> (N=1..4); "
            "cross_all_cube_rand_K<K>_s<seed>_B<B> (K=2..4, B=1..K); "
            "cross_rand_K<K>_objs<C>_B<B>_s<seed> (K=2..4, C=1..3, B=1..K per obj); "
            "cross_rand_K<K>_objs<C>_Bs<B>_s<seed> (same, shared B across objs); "
            "rand_cov_B<budget>_s<seed> (budget=4..16, random coverage of all 16 tasks)."
        )
    seed = int(m.group(1))
    n = int(m.group(2))
    if not 1 <= n <= 4:
        raise ValueError(f"Budget N must be 1..4, got {n} in {set_name!r}")

    rng = _random.Random(seed)
    chosen_cube = sorted(rng.sample(_CUBE_CODES, n))
    return _CROSS_CODES + chosen_cube, list(_CROSS_CUBE_CODES)
