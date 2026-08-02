"""Parallel-scheduling instance generators: general DAGs and unit out-forests.

``general`` instances are sparse random DAGs (predecessors always point
to lower-numbered tasks, so acyclicity is by construction) with integer
processing times; ``forest`` instances are unit-time out-forests, the
class where Hu's level algorithm is optimal. Best-known values come from
the exact DFS solver (general) and Hu's algorithm (forest).
"""

from __future__ import annotations

import numpy as np

from stem4humanity.instances.base import annotate_best_known, save_manifest
from stem4humanity.problems.base import Instance
from stem4humanity.problems.registry import get_problem
from stem4humanity.solvers.base import all_solvers
import stem4humanity.solvers.registry  # noqa: F401  (run solver registration)


def generate_general_instances(
    n_tasks: int | tuple[int, int],
    machines: int | tuple[int, int] = (2, 4),
    *,
    count: int,
    seed: int,
    split: str = "test",
) -> list[Instance]:
    """Sparse random DAGs with integer processing times."""
    if isinstance(n_tasks, tuple):
        lo, hi = n_tasks
    else:
        lo = hi = n_tasks
    if isinstance(machines, tuple):
        machine_lo, machine_hi = machines
    else:
        machine_lo = machine_hi = machines

    rng = np.random.default_rng(seed)
    records: list[Instance] = []
    for index in range(count):
        n = int(rng.integers(lo, hi + 1)) if lo != hi else lo
        m = (
            int(rng.integers(machine_lo, machine_hi + 1))
            if machine_lo != machine_hi
            else machine_lo
        )
        if n < 1 or m < 1:
            raise ValueError("n_tasks and machines must be positive")
        predecessors: list[list[int]] = [[] for _ in range(n)]
        for task in range(1, n):
            if rng.random() < 0.35:
                predecessors[task].append(int(rng.integers(0, task)))
            if rng.random() < 0.25 and len(predecessors[task]) == 0:
                predecessors[task].append(int(rng.integers(0, task)))
        times = rng.integers(1, 7, size=n)
        name = f"parallel-scheduling:general:n{n}m{m}-s{seed}-i{index}"
        records.append(
            Instance(
                name=name,
                problem="parallel-scheduling",
                subproblem="general",
                family="random",
                seed=seed,
                data={
                    "node_count": n,
                    "machines": m,
                    "times": times.tolist(),
                    "predecessors": predecessors,
                },
                best_known=None,
                split=split,
            )
        )
    return records


def generate_forest_instances(
    n_tasks: int | tuple[int, int],
    machines: int | tuple[int, int] = (2, 4),
    *,
    count: int,
    seed: int,
    split: str = "test",
) -> list[Instance]:
    """Random unit-time out-forests: every node has at most one predecessor."""
    if isinstance(n_tasks, tuple):
        lo, hi = n_tasks
    else:
        lo = hi = n_tasks
    if isinstance(machines, tuple):
        machine_lo, machine_hi = machines
    else:
        machine_lo = machine_hi = machines

    rng = np.random.default_rng(seed)
    records: list[Instance] = []
    for index in range(count):
        n = int(rng.integers(lo, hi + 1)) if lo != hi else lo
        m = (
            int(rng.integers(machine_lo, machine_hi + 1))
            if machine_lo != machine_hi
            else machine_lo
        )
        if n < 1 or m < 1:
            raise ValueError("n_tasks and machines must be positive")
        predecessors: list[list[int]] = [[] for _ in range(n)]
        for task in range(1, n):
            if rng.random() < 0.8:
                predecessors[task].append(int(rng.integers(0, task)))
        name = f"parallel-scheduling:forest:n{n}m{m}-s{seed}-i{index}"
        records.append(
            Instance(
                name=name,
                problem="parallel-scheduling",
                subproblem="forest",
                family="random",
                seed=seed,
                data={
                    "node_count": n,
                    "machines": m,
                    "times": [1] * n,
                    "predecessors": predecessors,
                },
                best_known=None,
                split=split,
            )
        )
    return records


def generate_parallel_scheduling_manifests(
    output_dir: str = "data/instances",
) -> None:
    all_instances: list[Instance] = []
    for sizes in ((8, 11), (11, 14)):
        all_instances.extend(
            generate_general_instances(sizes, count=4, seed=7401, split="test")
        )
    all_instances.extend(
        generate_general_instances((20, 30), count=6, seed=7501, split="train")
    )
    for sizes in ((10, 13), (13, 16)):
        all_instances.extend(
            generate_forest_instances(sizes, count=4, seed=7601, split="test")
        )
    all_instances.extend(
        generate_forest_instances((20, 30), count=6, seed=7701, split="train")
    )

    problem = get_problem("parallel-scheduling")
    exact = next(
        solver for solver in all_solvers().values() if solver.id == "ps-hu"
    )
    forests = [i for i in all_instances if i.subproblem == "forest"]
    annotated = {
        instance.name: instance
        for instance in annotate_best_known(forests, exact, problem)
    }
    all_instances = [
        annotated.get(instance.name, instance) for instance in all_instances
    ]

    exact = next(
        solver for solver in all_solvers().values() if solver.id == "ps-exact"
    )
    generals = [i for i in all_instances if i.subproblem == "general"]
    annotated = {
        instance.name: instance
        for instance in annotate_best_known(generals, exact, problem)
    }
    all_instances = [
        annotated.get(instance.name, instance) for instance in all_instances
    ]
    save_manifest(all_instances, f"{output_dir}/parallel_scheduling_manifests.json")
