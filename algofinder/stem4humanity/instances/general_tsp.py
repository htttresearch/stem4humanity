"""General symmetric TSP instance generator (no Euclidean structure)."""

from __future__ import annotations

import numpy as np

from stem4humanity.instances.base import annotate_best_known, save_manifest
from stem4humanity.problems.base import Instance
from stem4humanity.problems.registry import get_problem
from stem4humanity.solvers.base import all_solvers
import stem4humanity.solvers.registry  # noqa: F401  (run solver registration)


def generate_general_instances(
    city_count: int | tuple[int, int],
    *,
    count: int,
    seed: int,
    split: str = "test",
) -> list[Instance]:
    """Random symmetric distance matrices with arbitrary magnitudes."""
    if isinstance(city_count, tuple):
        lo, hi = city_count
    else:
        lo = hi = city_count

    rng = np.random.default_rng(seed)
    records: list[Instance] = []
    for index in range(count):
        n = int(rng.integers(lo, hi + 1)) if lo != hi else lo
        if n < 3:
            raise ValueError("city_count must be at least three")
        matrix = rng.random((n, n))
        matrix = (matrix + matrix.T) / 2.0
        np.fill_diagonal(matrix, 0.0)
        name = f"tsp:general:random-n{n}-s{seed}-i{index}"
        records.append(
            Instance(
                name=name,
                problem="tsp",
                subproblem="general",
                family="random",
                seed=seed,
                data={"distances": matrix.tolist()},
                best_known=None,
                split=split,
            )
        )
    return records


def generate_general_tsp_manifests(
    output_dir: str = "public/data",
    *,
    best_known_budget_seconds: float | None = None,
) -> None:
    all_instances: list[Instance] = []
    all_instances.extend(
        generate_general_instances((9, 13), count=4, seed=3101, split="test")
    )
    all_instances.extend(
        generate_general_instances((18, 24), count=4, seed=3201, split="test")
    )

    problem = get_problem("tsp")
    exact = next(s for s in all_solvers().values() if s.id == "incremental-exact")
    all_instances = annotate_best_known(all_instances, exact, problem, budget_seconds=best_known_budget_seconds)
    save_manifest(all_instances, f"{output_dir}/tsp_general_manifests.json")
