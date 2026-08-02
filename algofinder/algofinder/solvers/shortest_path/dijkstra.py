"""Dijkstra's algorithm: the classic general single-source solver."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from algofinder.problems.shortest_path import ShortestPathState
from algofinder.solvers.base import Solver, SolverResult, register_solver


@register_solver
class DijkstraSolver(Solver):
    id = "dijkstra"
    display = "Dijkstra shortest-path tree (array-based, any non-negative graph)"
    tags = frozenset({"exact"})
    applies_to = frozenset({"shortest-path:general", "shortest-path:dag"})

    def solve(
        self,
        state: ShortestPathState,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        started = perf_counter()
        node_count = state.node_count
        source = state.source
        distances = np.full(node_count, np.inf)
        distances[source] = 0.0
        parents = [-1] * node_count
        settled = [False] * node_count
        for _ in range(node_count):
            unsettled = np.flatnonzero(~np.asarray(settled))
            candidates = distances[unsettled]
            if not candidates.size:
                break
            pivot = int(unsettled[np.argmin(candidates)])
            if not np.isfinite(distances[pivot]):
                break
            settled[pivot] = True
            for head, weight in state.out_arcs(pivot):
                improved = distances[pivot] + weight
                if improved < distances[head]:
                    distances[head] = improved
                    parents[head] = pivot
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
                "algorithm": "dijkstra",
                "reachable_nodes": int(np.isfinite(distances).sum()),
            },
        )


__all__ = ["DijkstraSolver"]
