"""Exact dynamic-programming solver for single-dimension knapsacks (0/1, subset-sum)."""

from __future__ import annotations

import time

import numpy as np

from stem4humanity.problems.knapsack import KnapsackState
from stem4humanity.solvers.base import InapplicableError, Solver, SolverResult, register_solver


@register_solver
class DynamicProgrammingExact(Solver):
    id = "knapsack-dp-exact"
    display = "Dynamic programming over capacity (exact)"
    tags = frozenset({"exact"})
    applies_to = frozenset({"knapsack:0-1", "knapsack:subset-sum"})

    def solve(
        self, state: KnapsackState, *, budget_seconds: float | None = None
    ) -> SolverResult:
        if state.dims != 1:
            raise InapplicableError("DP exact requires a single weight dimension")
        weights = state.weights[:, 0]
        values = state.values
        if not np.allclose(weights, np.round(weights)) or not np.allclose(
            values, np.round(values)
        ):
            raise InapplicableError("DP exact requires integral weights and values")
        capacity = int(round(state.capacities[0]))

        start = time.perf_counter()
        items = sorted(
            enumerate(zip(weights, values)),
            key=lambda item: float(item[1][0]),
        )
        n = len(items)
        # best[k][c] = max value with first k items (processed by weight) and cap c
        best = np.zeros(capacity + 1, dtype=np.float64)
        choice = np.zeros((n, capacity + 1), dtype=bool)
        for k, (index, (weight, value)) in enumerate(items):
            w = int(round(weight))
            if w > capacity:
                continue
            before = best[: capacity - w + 1] + value
            window = best[w : capacity + 1]
            improved = before > window
            best[w : capacity + 1] = np.where(improved, before, window)
            choice[k, w : capacity + 1] = improved

        selected: list[int] = []
        c = capacity
        for k in range(n - 1, -1, -1):
            if choice[k, c]:
                selected.append(items[k][0])
                c -= int(round(items[k][1][0]))
        selected.reverse()
        solution = tuple(selected)
        cost = float(state.objective_value(solution))
        elapsed = time.perf_counter() - start
        return SolverResult(
            solution=solution,
            cost=cost,
            exact=True,
            wall_seconds=elapsed,
            metadata={"dp_cells": (n + 1) * (capacity + 1)},
        )
