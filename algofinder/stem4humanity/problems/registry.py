"""Problem matrix registry.

The matrix: 8-10 interesting problems, each with 2-3 subproblems. Implemented
problems can build states; the remaining ones are declared for the roadmap and
raise ``NotImplementedError`` until their subproblems are wired (see
``docs/adding-problems.md``).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from stem4humanity.problems.base import Instance, Problem, ProblemState
from stem4humanity.problems.bin_packing import BinPackingState
from stem4humanity.problems.knapsack import KnapsackState
from stem4humanity.problems.parallel_scheduling import ParallelSchedulingState
from stem4humanity.problems.scheduling import FlowShopState, JobShopState
from stem4humanity.problems.shortest_path import ShortestPathState
from stem4humanity.problems.tsp import (
    EuclideanTravellingSalespersonProblem,
    TravellingSalespersonProblem,
)
from stem4humanity.problems.unit_commitment import (
    StorageArbitrageState,
    UnitCommitmentState,
)

_PROBLEMS: dict[str, Problem] = {}


def register_problem(cls: type[Problem]) -> type[Problem]:
    instance = cls()
    if instance.id in _PROBLEMS:
        raise ValueError(f"duplicate problem id: {instance.id}")
    _PROBLEMS[instance.id] = instance
    return cls


@register_problem
class TSPProblem(Problem):
    id = "tsp"
    display = "Traveling Salesperson Problem"
    subproblems = ("general", "euclidean", "clustered")

    def build_state(self, instance: Instance) -> ProblemState:
        subproblem = instance.subproblem
        if subproblem == "general":
            return TravellingSalespersonProblem(
                instance.name, instance.data["distances"]
            )
        if subproblem in ("euclidean", "clustered"):
            points = np.asarray(instance.data["points"], dtype=float)
            return EuclideanTravellingSalespersonProblem(instance.name, points)
        raise ValueError(f"unknown tsp subproblem: {subproblem}")


@register_problem
class KnapsackProblem(Problem):
    id = "knapsack"
    display = "Knapsack (multi-dimensional)"
    subproblems = ("0-1", "multidimensional", "subset-sum")
    minimize = False

    def build_state(self, instance: Instance) -> ProblemState:
        subproblem = instance.subproblem
        weights = np.asarray(instance.data["weights"], dtype=float)
        values = np.asarray(instance.data["values"], dtype=float)
        capacities = np.asarray(instance.data["capacities"], dtype=float)
        if subproblem == "0-1" and weights.ndim == 1:
            weights = weights.reshape(-1, 1)
        if subproblem == "subset-sum":
            values = weights[:, 0] if weights.ndim == 2 else weights
        return KnapsackState(instance.name, weights, values, capacities)


class _PlannedProblem(Problem):
    """A declared-but-unimplemented problem for the roadmap."""

    implemented: ClassVar[bool] = False

    def build_state(self, instance: Instance) -> ProblemState:
        raise NotImplementedError(
            f"{self.id} is declared in the matrix but not implemented yet; "
            "see docs/adding-problems.md"
        )


@register_problem
class BinPackingProblem(Problem):
    id = "bin-packing"
    display = "Bin Packing (vector / large items)"
    subproblems = ("vector", "large-items")

    def build_state(self, instance: Instance) -> ProblemState:
        items = np.asarray(instance.data["items"], dtype=float)
        if items.ndim == 1:
            items = items.reshape(-1, 1)
        capacities = instance.data.get("capacities")
        if capacities is not None:
            capacities = np.asarray(capacities, dtype=float)
        return BinPackingState(instance.name, items, capacities)


@register_problem
class VRPProblem(_PlannedProblem):
    id = "vrp"
    display = "Vehicle Routing Problem"
    subproblems = ("cvrp", "vrptw", "open-euclidean")


@register_problem
class ColoringProblem(_PlannedProblem):
    id = "coloring"
    display = "Graph Coloring"
    subproblems = ("general", "planar", "interval")


@register_problem
class CliqueProblem(_PlannedProblem):
    id = "max-clique"
    display = "Maximum Clique / Independent Set"
    subproblems = ("general", "sparse", "dense")


@register_problem
class SchedulingProblem(Problem):
    id = "scheduling"
    display = "Scheduling (job shop / two-machine flow shop)"
    subproblems = ("job-shop", "flow-shop-2")

    def build_state(self, instance: Instance) -> ProblemState:
        subproblem = instance.subproblem
        if subproblem == "flow-shop-2":
            return FlowShopState(
                instance.name, instance.data["processing_times"]
            )
        if subproblem == "job-shop":
            return JobShopState(
                instance.name,
                int(instance.data["n_jobs"]),
                int(instance.data["n_machines"]),
                instance.data["operations"],
            )
        raise ValueError(f"unknown scheduling subproblem: {subproblem}")


@register_problem
class ShortestPathProblem(Problem):
    id = "shortest-path"
    display = "Shortest Path (single source)"
    subproblems = ("general", "dag")

    def build_state(self, instance: Instance) -> ProblemState:
        return ShortestPathState(
            instance.name,
            int(instance.data["node_count"]),
            int(instance.data["source"]),
            instance.data["arcs"],
        )


@register_problem
class ParallelSchedulingProblem(Problem):
    id = "parallel-scheduling"
    display = "Parallel Scheduling (DAG on identical machines)"
    subproblems = ("general", "forest")

    def build_state(self, instance: Instance) -> ProblemState:
        return ParallelSchedulingState(
            instance.name,
            int(instance.data["node_count"]),
            int(instance.data["machines"]),
            instance.data["times"],
            instance.data["predecessors"],
        )


@register_problem
class UnitCommitmentProblem(Problem):
    id = "unit-commitment"
    display = "Unit Commitment (thermal / storage arbitrage)"
    subproblems = ("classic", "storage")

    def build_state(self, instance: Instance) -> ProblemState:
        if instance.subproblem == "storage":
            return StorageArbitrageState(
                instance.name,
                int(instance.data["periods"]),
                instance.data["prices"],
                float(instance.data["capacity"]),
                float(instance.data["rate"]),
                float(instance.data["efficiency"]),
                float(instance.data["initial_energy"]),
                instance.data.get("target_energy"),
            )
        if instance.subproblem == "classic":
            return UnitCommitmentState(
                instance.name,
                int(instance.data["hours"]),
                instance.data["generators"],
                instance.data["demand"],
                instance.data["reserve"],
            )
        raise ValueError(
            f"unknown unit-commitment subproblem: {instance.subproblem}"
        )


@register_problem
class SATProblem(_PlannedProblem):
    id = "sat"
    display = "SAT / MaxSAT"
    subproblems = ("3-sat", "horn-sat", "max-2-sat")


@register_problem
class SteinerProblem(_PlannedProblem):
    id = "steiner"
    display = "Steiner Tree"
    subproblems = ("graph", "euclidean", "rectilinear")


@register_problem
class PartitionProblem(_PlannedProblem):
    id = "partition"
    display = "Graph Partitioning"
    subproblems = ("bisection", "k-way", "mesh")


def all_problems() -> dict[str, Problem]:
    return dict(_PROBLEMS)


def get_problem(problem_id: str) -> Problem:
    try:
        return _PROBLEMS[problem_id]
    except KeyError:
        raise ValueError(f"unknown problem: {problem_id}") from None


def matrix_table() -> str:
    """Render the problem matrix as a markdown table."""
    lines = [
        "| Problem | Subproblems | Status |",
        "|---|---|---|",
    ]
    for problem in _PROBLEMS.values():
        status = "planned" if isinstance(problem, _PlannedProblem) else "implemented"
        lines.append(
            f"| {problem.display} | {', '.join(problem.subproblems)} | {status} |"
        )
    return "\n".join(lines)
