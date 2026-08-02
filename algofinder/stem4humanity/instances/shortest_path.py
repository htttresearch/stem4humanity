"""Shortest-path instance generators: general graphs and DAGs."""

from __future__ import annotations

import numpy as np

from stem4humanity.instances.base import annotate_best_known, save_manifest
from stem4humanity.problems.base import Instance
from stem4humanity.problems.registry import get_problem
from stem4humanity.solvers.base import all_solvers
import stem4humanity.solvers.registry  # noqa: F401  (run solver registration)


def generate_shortest_path_instances(
    subproblem: str,
    node_count: int | tuple[int, int],
    *,
    count: int,
    seed: int,
    split: str = "test",
) -> list[Instance]:
    """Random directed graphs; DAGs are indexed so tails precede heads."""
    if subproblem not in ("general", "dag"):
        raise ValueError(f"unknown shortest-path subproblem: {subproblem}")
    if isinstance(node_count, tuple):
        lo, hi = node_count
    else:
        lo = hi = node_count

    rng = np.random.default_rng(seed)
    records: list[Instance] = []
    for index in range(count):
        n = int(rng.integers(lo, hi + 1)) if lo != hi else lo
        if n < 2:
            raise ValueError("node_count must be at least two")
        arcs: list[list[float]] = []
        if subproblem == "dag":
            for tail in range(n):
                for head in range(tail + 1, n):
                    if rng.random() < 0.3:
                        arcs.append(
                            [tail, head, float(rng.integers(1, 101))]
                        )
        else:
            for tail in range(n):
                for head in range(n):
                    if tail != head and rng.random() < 0.22:
                        arcs.append(
                            [tail, head, float(rng.integers(1, 101))]
                        )
        name = f"shortest-path:{subproblem}:n{n}-s{seed}-i{index}"
        records.append(
            Instance(
                name=name,
                problem="shortest-path",
                subproblem=subproblem,
                family="random",
                seed=seed,
                data={
                    "node_count": n,
                    "source": 0,
                    "arcs": arcs,
                },
                best_known=None,
                split=split,
            )
        )
    return records


def generate_shortest_path_manifests(
    output_dir: str = "public/data",
    *,
    best_known_budget_seconds: float | None = None,
) -> None:
    all_instances: list[Instance] = []
    for subproblem in ("general", "dag"):
        all_instances.extend(
            generate_shortest_path_instances(
                subproblem, (10, 14), count=4, seed=5101, split="test"
            )
        )
        all_instances.extend(
            generate_shortest_path_instances(
                subproblem, (20, 26), count=4, seed=5201, split="test"
            )
        )
    all_instances.extend(
        generate_shortest_path_instances(
            "dag", (10, 14), count=8, seed=5301, split="train"
        )
    )

    problem = get_problem("shortest-path")
    exact = next(s for s in all_solvers().values() if s.id == "dijkstra")
    all_instances = annotate_best_known(all_instances, exact, problem, budget_seconds=best_known_budget_seconds)
    save_manifest(all_instances, f"{output_dir}/shortest_path_manifests.json")
