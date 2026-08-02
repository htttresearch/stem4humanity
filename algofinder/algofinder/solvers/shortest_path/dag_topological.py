"""Topological-order relaxation: the linear specialized DAG solver."""

from __future__ import annotations

from collections import deque
from time import perf_counter

import numpy as np

from algofinder.problems.shortest_path import ShortestPathState
from algofinder.solvers.base import (
    InapplicableError,
    Solver,
    SolverResult,
    register_solver,
)


def topological_order(state: ShortestPathState) -> list[int]:
    """A topological ordering of the graph, or None if it contains a cycle."""
    node_count = state.node_count
    in_degree = [0] * node_count
    for node in range(node_count):
        for head in state.out_edges(node):
            in_degree[head] += 1
    ready = deque(node for node in range(node_count) if in_degree[node] == 0)
    order: list[int] = []
    while ready:
        node = ready.popleft()
        order.append(node)
        for head in state.out_edges(node):
            in_degree[head] -= 1
            if in_degree[head] == 0:
                ready.append(head)
    return order if len(order) == node_count else []


@register_solver
class DagTopologicalSolver(Solver):
    id = "dag-topological-relaxation"
    display = "Topological-order relaxation (linear, DAG subclass only)"
    tags = frozenset({"exact"})
    applies_to = frozenset({"shortest-path:dag"})

    def solve(
        self,
        state: ShortestPathState,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        started = perf_counter()
        order = topological_order(state)
        if not order:
            raise InapplicableError(
                "topological relaxation requires an acyclic graph"
            )
        node_count = state.node_count
        source = state.source
        distances = np.full(node_count, np.inf)
        distances[source] = 0.0
        parents = [-1] * node_count
        for node in order:
            if not np.isfinite(distances[node]):
                continue
            for head, weight in state.out_arcs(node):
                improved = distances[node] + weight
                if improved < distances[head]:
                    distances[head] = improved
                    parents[head] = node
        solution = [
            parents[node] if np.isfinite(distances[node]) else -1
            for node in range(node_count)
        ]
        cost = float(distances[np.isfinite(distances)].sum())
        return SolverResult(
            solution=solution,
            cost=cost,
            exact=True,
            wall_seconds=perf_counter() - started,
            metadata={
                "algorithm": "dag-topological-relaxation",
                "reachable_nodes": int(np.isfinite(distances).sum()),
            },
        )


__all__ = ["DagTopologicalSolver", "topological_order"]
