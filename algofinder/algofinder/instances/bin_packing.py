"""Bin-packing instance generators: vector packing and large items.

``vector`` instances have 2-3 resource dimensions with sizes drawn from
``U(0, 1]`` per dimension (unit-capacity bins); ``large-items``
instances are one-dimensional with every size above a third of the bin,
the class where the blossom matching is optimal. Best-known values come
from the exact DFS solver (vector) and the blossom solver (large items).
"""

from __future__ import annotations

import numpy as np

from algofinder.instances.base import annotate_best_known, save_manifest
from algofinder.problems.base import Instance
from algofinder.problems.registry import get_problem
from algofinder.solvers.base import all_solvers
import algofinder.solvers.registry  # noqa: F401  (run solver registration)


def generate_vector_bin_packing_instances(
    n_items: int | tuple[int, int],
    dims: int | tuple[int, int] = (2, 3),
    *,
    count: int,
    seed: int,
    split: str = "test",
) -> list[Instance]:
    """Random multi-dimensional items with unit-capacity bins."""
    if isinstance(n_items, tuple):
        lo, hi = n_items
    else:
        lo = hi = n_items
    if isinstance(dims, tuple):
        dim_lo, dim_hi = dims
    else:
        dim_lo = dim_hi = dims

    rng = np.random.default_rng(seed)
    records: list[Instance] = []
    for index in range(count):
        n = int(rng.integers(lo, hi + 1)) if lo != hi else lo
        d = int(rng.integers(dim_lo, dim_hi + 1)) if dim_lo != dim_hi else dim_lo
        if n < 1 or d < 1:
            raise ValueError("n_items and dims must be positive")
        sizes = rng.uniform(0.05, 1.0, size=(n, d))
        name = f"bin-packing:vector:n{n}d{d}-s{seed}-i{index}"
        records.append(
            Instance(
                name=name,
                problem="bin-packing",
                subproblem="vector",
                family="random",
                seed=seed,
                data={"items": sizes.tolist()},
                best_known=None,
                split=split,
            )
        )
    return records


def generate_large_items_instances(
    n_items: int | tuple[int, int],
    *,
    count: int,
    seed: int,
    split: str = "test",
) -> list[Instance]:
    """One-dimensional items drawn from ``U(1/3, 1]`` (max two per bin)."""
    if isinstance(n_items, tuple):
        lo, hi = n_items
    else:
        lo = hi = n_items

    rng = np.random.default_rng(seed)
    records: list[Instance] = []
    for index in range(count):
        n = int(rng.integers(lo, hi + 1)) if lo != hi else lo
        if n < 1:
            raise ValueError("n_items must be positive")
        sizes = rng.uniform(0.34, 1.0, size=n)
        name = f"bin-packing:large-items:n{n}-s{seed}-i{index}"
        records.append(
            Instance(
                name=name,
                problem="bin-packing",
                subproblem="large-items",
                family="random",
                seed=seed,
                data={"items": sizes.tolist()},
                best_known=None,
                split=split,
            )
        )
    return records


def generate_bin_packing_manifests(
    output_dir: str = "public/data",
    *,
    best_known_budget_seconds: float | None = None,
) -> None:
    all_instances: list[Instance] = []
    for sizes in ((10, 13), (13, 16)):
        all_instances.extend(
            generate_vector_bin_packing_instances(
                sizes, count=4, seed=7001, split="test"
            )
        )
    all_instances.extend(
        generate_vector_bin_packing_instances(
            (24, 34), count=6, seed=7101, split="train"
        )
    )
    for sizes in ((12, 15), (15, 20)):
        all_instances.extend(
            generate_large_items_instances(sizes, count=4, seed=7201, split="test")
        )
    all_instances.extend(
        generate_large_items_instances((30, 40), count=6, seed=7301, split="train")
    )

    problem = get_problem("bin-packing")
    exact = next(
        solver
        for solver in all_solvers().values()
        if solver.id == "bp-large-items-exact"
    )
    large_items = [i for i in all_instances if i.subproblem == "large-items"]
    annotated = {
        instance.name: instance
        for instance in annotate_best_known(large_items, exact, problem, budget_seconds=best_known_budget_seconds)
    }
    all_instances = [
        annotated.get(instance.name, instance) for instance in all_instances
    ]

    exact = next(solver for solver in all_solvers().values() if solver.id == "bp-exact-dfs")
    vectors = [i for i in all_instances if i.subproblem == "vector"]
    annotated = {
        instance.name: instance
        for instance in annotate_best_known(vectors, exact, problem, budget_seconds=best_known_budget_seconds)
    }
    all_instances = [
        annotated.get(instance.name, instance) for instance in all_instances
    ]
    save_manifest(all_instances, f"{output_dir}/bin_packing_manifests.json")
