"""Knapsack instance generators: 0/1, multidimensional, subset-sum."""

from __future__ import annotations

import numpy as np

from algofinder.instances.base import annotate_best_known, save_manifest
from algofinder.problems.base import Instance
from algofinder.problems.registry import get_problem
from algofinder.solvers.base import all_solvers
import algofinder.solvers.registry  # noqa: F401  (run solver registration)


def generate_knapsack_instances(
    subproblem: str,
    item_count: int | tuple[int, int],
    *,
    count: int,
    seed: int,
    split: str = "test",
) -> list[Instance]:
    if subproblem not in ("0-1", "multidimensional", "subset-sum"):
        raise ValueError(f"unknown knapsack subproblem: {subproblem}")
    if isinstance(item_count, tuple):
        lo, hi = item_count
    else:
        lo = hi = item_count

    rng = np.random.default_rng(seed)
    records: list[Instance] = []
    for index in range(count):
        n = int(rng.integers(lo, hi + 1)) if lo != hi else lo
        if n < 1:
            raise ValueError("item_count must be at least one")

        if subproblem == "subset-sum":
            weights = rng.integers(1, 200, size=(n, 1))
            target = int(0.5 * int(weights.sum()))
            data = {
                "weights": weights.tolist(),
                "values": weights.tolist(),
                "capacities": [target],
            }
        elif subproblem == "multidimensional":
            dims = 3
            weights = rng.integers(1, 60, size=(n, dims))
            capacities = (0.5 * weights.sum(axis=0)).astype(int)
            data = {
                "weights": weights.tolist(),
                "values": rng.integers(1, 100, size=(n,)).tolist(),
                "capacities": capacities.tolist(),
            }
        else:
            weights = rng.integers(1, 100, size=(n, 1))
            values = rng.integers(1, 100, size=(n,))
            capacity = int(0.5 * int(weights.sum()))
            data = {
                "weights": weights.tolist(),
                "values": values.tolist(),
                "capacities": [capacity],
            }

        name = f"knapsack:{subproblem}:n{n}-s{seed}-i{index}"
        records.append(
            Instance(
                name=name,
                problem="knapsack",
                subproblem=subproblem,
                family=subproblem,
                seed=seed,
                data=data,
                best_known=None,
                split=split,
            )
        )
    return records


def generate_knapsack_manifests(
    output_dir: str = "public/data",
    *,
    best_known_budget_seconds: float | None = 60.0,
) -> None:
    all_instances: list[Instance] = []
    for subproblem in ("0-1", "multidimensional", "subset-sum"):
        all_instances.extend(
            generate_knapsack_instances(
                subproblem, (18, 26), count=4, seed=4101, split="test"
            )
        )
        all_instances.extend(
            generate_knapsack_instances(
                subproblem, (34, 44), count=4, seed=4201, split="test"
            )
        )

    problem = get_problem("knapsack")
    exact = next(s for s in all_solvers().values() if s.id == "knapsack-bnb-exact")
    all_instances = annotate_best_known(
        all_instances, exact, problem, budget_seconds=best_known_budget_seconds
    )
    save_manifest(all_instances, f"{output_dir}/knapsack_manifests.json")
