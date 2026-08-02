"""Johnson's rule: the O(n log n) exact two-machine flow-shop solver."""

from __future__ import annotations

from time import perf_counter

from algofinder.problems.scheduling import FlowShopState
from algofinder.solvers.base import Solver, SolverResult, register_solver


def johnson_order(state: FlowShopState) -> list[int]:
    """Optimal job permutation for the two-machine flow shop."""
    first: list[tuple[float, int]] = []
    second: list[tuple[float, int]] = []
    for job in state.jobs:
        first_time, second_time = state.times[job]
        if first_time <= second_time:
            first.append((first_time, job))
        else:
            second.append((second_time, job))
    first.sort()
    second.sort(reverse=True)
    return [job for _, job in first] + [job for _, job in second]


@register_solver
class JohnsonFlowShopSolver(Solver):
    id = "johnson-flow-shop"
    display = "Johnson's rule (exact, two-machine flow shop)"
    tags = frozenset({"exact"})
    applies_to = frozenset({"scheduling:flow-shop-2"})

    def solve(
        self,
        state: FlowShopState,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        started = perf_counter()
        order = johnson_order(state)
        cost = state.objective_value(order)
        return SolverResult(
            solution=order,
            cost=cost,
            exact=True,
            wall_seconds=perf_counter() - started,
            metadata={"algorithm": "johnson-rule"},
        )


__all__ = ["JohnsonFlowShopSolver", "johnson_order"]
