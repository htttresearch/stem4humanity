"""Greedy density heuristic for multi-dimensional knapsack."""

from __future__ import annotations

import time

import numpy as np

from stem4humanity.problems.knapsack import KnapsackState
from stem4humanity.solvers.base import Solver, SolverResult, register_solver


@register_solver
class GreedyDensity(Solver):
    id = "knapsack-greedy-density"
    display = "Greedy by value/weight density"
    tags = frozenset({"heuristic"})
    applies_to = frozenset(
        {"knapsack:0-1", "knapsack:multidimensional", "knapsack:subset-sum"}
    )

    def solve(
        self, state: KnapsackState, *, budget_seconds: float | None = None
    ) -> SolverResult:
        weights = state.weights
        values = state.values
        density = values / (weights + 1e-12).max(axis=1)
        order = np.argsort(density)[::-1]

        remaining = state.capacities.copy()
        selected: list[int] = []
        start = time.perf_counter()
        for j in order:
            item_weight = weights[j]
            if np.all(remaining >= item_weight - 1e-9):
                selected.append(int(j))
                remaining -= item_weight
        solution = tuple(selected)
        cost = float(state.objective_value(solution))
        return SolverResult(
            solution=solution,
            cost=cost,
            exact=False,
            wall_seconds=time.perf_counter() - start,
            metadata={},
        )
