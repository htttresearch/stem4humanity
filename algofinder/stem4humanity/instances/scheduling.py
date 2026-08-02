"""Scheduling instance generators: job shop and two-machine flow shop."""

from __future__ import annotations

import numpy as np

from stem4humanity.instances.base import annotate_best_known, save_manifest
from stem4humanity.problems.base import Instance
from stem4humanity.problems.registry import get_problem
from stem4humanity.solvers.base import all_solvers
import stem4humanity.solvers.registry  # noqa: F401  (run solver registration)


def generate_flow_shop_instances(
    job_count: int | tuple[int, int],
    *,
    count: int,
    seed: int,
    split: str = "test",
) -> list[Instance]:
    """Two-machine flow shops with integer processing times."""
    if isinstance(job_count, tuple):
        lo, hi = job_count
    else:
        lo = hi = job_count

    rng = np.random.default_rng(seed)
    records: list[Instance] = []
    for index in range(count):
        n = int(rng.integers(lo, hi + 1)) if lo != hi else lo
        if n < 1:
            raise ValueError("job_count must be at least one")
        times = rng.integers(1, 31, size=(n, 2))
        name = f"scheduling:flow-shop-2:n{n}-s{seed}-i{index}"
        records.append(
            Instance(
                name=name,
                problem="scheduling",
                subproblem="flow-shop-2",
                family="random",
                seed=seed,
                data={"processing_times": times.tolist()},
                best_known=None,
                split=split,
            )
        )
    return records


def generate_job_shop_instances(
    n_jobs: int | tuple[int, int],
    n_machines: int | tuple[int, int],
    *,
    count: int,
    seed: int,
    split: str = "test",
) -> list[Instance]:
    """Tiny job shops: each job visits every machine in a random order."""
    if isinstance(n_jobs, tuple):
        job_lo, job_hi = n_jobs
    else:
        job_lo = job_hi = n_jobs
    if isinstance(n_machines, tuple):
        machine_lo, machine_hi = n_machines
    else:
        machine_lo = machine_hi = n_machines

    rng = np.random.default_rng(seed)
    records: list[Instance] = []
    for index in range(count):
        jobs = int(rng.integers(job_lo, job_hi + 1)) if job_lo != job_hi else job_lo
        machines = (
            int(rng.integers(machine_lo, machine_hi + 1))
            if machine_lo != machine_hi
            else machine_lo
        )
        if jobs < 1 or machines < 1:
            raise ValueError("n_jobs and n_machines must be positive")
        operations = []
        for _ in range(jobs):
            route = rng.permutation(machines)
            operations.append(
                [[int(machine), int(rng.integers(1, 21))] for machine in route]
            )
        name = f"scheduling:job-shop:j{jobs}m{machines}-s{seed}-i{index}"
        records.append(
            Instance(
                name=name,
                problem="scheduling",
                subproblem="job-shop",
                family="random",
                seed=seed,
                data={
                    "n_jobs": jobs,
                    "n_machines": machines,
                    "operations": operations,
                },
                best_known=None,
                split=split,
            )
        )
    return records


def generate_scheduling_manifests(
    output_dir: str = "public/data",
    *,
    best_known_budget_seconds: float | None = None,
) -> None:
    all_instances: list[Instance] = []
    for sizes in ((6, 10), (12, 18), (22, 28)):
        all_instances.extend(
            generate_flow_shop_instances(
                sizes, count=4, seed=6101, split="test"
            )
        )
    all_instances.extend(
        generate_flow_shop_instances(
            (8, 12), count=8, seed=6201, split="train"
        )
    )
    for n_jobs, n_machines in ((3, 2), (3, 3), (4, 2), (4, 3)):
        all_instances.extend(
            generate_job_shop_instances(
                n_jobs, n_machines, count=3, seed=6301, split="test"
            )
        )
    all_instances.extend(
        generate_job_shop_instances(
            (3, 4), (2, 3), count=4, seed=6401, split="train"
        )
    )

    problem = get_problem("scheduling")
    exact = next(s for s in all_solvers().values() if s.id == "johnson-flow-shop")
    flow_shops = [i for i in all_instances if i.subproblem == "flow-shop-2"]
    annotated = {
        instance.name: instance
        for instance in annotate_best_known(flow_shops, exact, problem, budget_seconds=best_known_budget_seconds)
    }
    all_instances = [
        annotated.get(instance.name, instance) for instance in all_instances
    ]

    exact = next(s for s in all_solvers().values() if s.id == "jobshop-bnb-exact")
    job_shops = [i for i in all_instances if i.subproblem == "job-shop"]
    annotated = {
        instance.name: instance
        for instance in annotate_best_known(job_shops, exact, problem, budget_seconds=best_known_budget_seconds)
    }
    all_instances = [
        annotated.get(instance.name, instance) for instance in all_instances
    ]
    save_manifest(all_instances, f"{output_dir}/scheduling_manifests.json")
