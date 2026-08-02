"""Critical-path list scheduling: available tasks ranked by longest path."""

from __future__ import annotations

from time import perf_counter

from stem4humanity.problems.parallel_scheduling import ParallelSchedulingState
from stem4humanity.solvers.base import Solver, SolverResult, register_solver
from stem4humanity.solvers.parallel_scheduling.common import critical_paths, list_schedule


@register_solver
class PSCriticalPathListSolver(Solver):
    id = "ps-cp-list"
    display = "Critical-path list scheduling"
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
        priority = lambda task: (-float(longest[task]), task)
        solution = list_schedule(state, priority)
        return SolverResult(
            solution=solution,
            cost=state.objective_value(solution),
            exact=False,
            wall_seconds=perf_counter() - started,
            metadata={"algorithm": "critical-path-list-scheduling"},
        )


__all__ = ["PSCriticalPathListSolver"]
