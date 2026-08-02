"""Rolling-threshold heuristic for storage arbitrage.

Compares each price with the rolling mean of the preceding prices and
charges when the price is below the mean minus a margin, discharges when
it is above the mean plus the margin, then repairs the terminal energy.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from stem4humanity.problems.unit_commitment import StorageArbitrageState
from stem4humanity.solvers.base import Solver, SolverResult, register_solver
from stem4humanity.solvers.unit_commitment.common import repair_terminal, simulate_actions


@register_solver
class UCStorageThresholdSolver(Solver):
    id = "uc-storage-threshold"
    display = "Rolling-price threshold arbitrage"
    tags = frozenset({"heuristic"})
    applies_to = frozenset({"unit-commitment:storage"})

    def __init__(self, window: int | None = None, margin: float = 0.03) -> None:
        self.window = window
        self.margin = margin

    def solve(
        self,
        state: StorageArbitrageState,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        started = perf_counter()
        window = self.window or max(4, state.periods // 4)
        actions: list[int] = []
        prices = state.prices
        for period in range(state.periods):
            lower = max(0, period - window)
            mean = float(prices[lower : period + 1].mean())
            if prices[period] < mean * (1 - self.margin):
                actions.append(0)
            elif prices[period] > mean * (1 + self.margin):
                actions.append(2)
            else:
                actions.append(1)
        solution = repair_terminal(state, simulate_actions(state, actions))
        return SolverResult(
            solution=solution,
            cost=state.objective_value(solution),
            exact=False,
            wall_seconds=perf_counter() - started,
            metadata={
                "algorithm": "rolling-threshold",
                "window": window,
                "margin": self.margin,
            },
        )


__all__ = ["UCStorageThresholdSolver"]
