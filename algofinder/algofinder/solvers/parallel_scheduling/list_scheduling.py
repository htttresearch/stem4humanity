"""Longest-processing-time list scheduling with critical-path tie-break."""

from __future__ import annotations

from time import perf_counter

from algofinder.problems.parallel_scheduling import ParallelSchedulingState
from algofinder.solvers.base import Solver, SolverResult, register_solver
from algofinder.solvers.parallel_scheduling.common import critical_paths, list_schedule


@register_solver
class PSListSchedulingSolver(Solver):
    id = "ps-list-scheduling"
    display = "List scheduling (LPT)"
    tags = frozenset({"heuristic"})
    applies_to = frozenset(
        {"parallel-scheduling:general", "parallel-scheduling:forest"}
    )

    def solve(
        self,
        state: ParallelSchedulingState,
        *,
        budget_seconds: float | None = None,
    ) -> SolverResult:
        started = perf_counter()
        longest = critical_paths(state)
        priority = lambda task: (-float(state.times[task]), -float(longest[task]), task)
        solution = list_schedule(state, priority)
        return SolverResult(
            solution=solution,
            cost=state.objective_value(solution),
            exact=False,
            wall_seconds=perf_counter() - started,
            metadata={"algorithm": "list-scheduling-lpt"},
        )


__all__ = ["PSListSchedulingSolver"]
